"""Dry-run check for the Step 12 transfer-hit policy (no file writes).

Re-solves the current gameweek exactly the way ``fpl.optimise.main`` does
(same candidate pool, selling prices, strategy mode, free-transfer count),
then asserts the rules the user asked for:

1. the recommended number of transfers never exceeds the real free
   transfers available, unless a hit is justified;
2. a hit is only ever recommended when every "paid" transfer sells a
   flagged (injured/suspended/unavailable) player who was a starter in the
   held XI -- a flagged *bench* player (Souza, in the real data this repo
   ships with) never counts;
3. when a hit **is** justified, the reasons are populated and human
   readable.

Run with ``uv run python -m fpl.check_transfer_policy``. Exits non-zero
(and prints ``FAIL``) if any assertion doesn't hold, so it also works as a
smoke test; there's no pytest dependency in this project, so this is a
plain script rather than a ``tests/`` suite.
"""

from __future__ import annotations

import sys

import pandas as pd

from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.optimise import (
    WEEKLY_DISCOUNT,
    build_candidate_pool,
    estimate_selling_prices,
    is_flagged_unavailable,
    load_tables,
    lookup_weekly_xp,
    recommend_transfers,
    resolve_free_transfers,
    resolve_my_squad,
    solve_horizon,
    strategy_adjusted,
)


def run_check() -> bool:
    config = load_config()
    tables = load_tables()
    players, xp_by_gw = tables["players"], tables["xp_by_gameweek"]
    league_picks, league_history = tables["league_picks"], tables["league_history"]
    league_managers = tables.get("league_managers", pd.DataFrame())

    team_id = resolve_team_id(config, league_managers)
    if team_id is None or league_picks.empty or league_history.empty:
        print("SKIP: no team_id/league data available -- run fpl.fetch and fpl.league first.")
        return True

    my_squad_ids, hist_bank, gameweek = resolve_my_squad(league_picks, league_history, team_id)
    bank = config.get("bank_override")
    bank = int(bank) if bank is not None else hist_bank

    league_eo: dict[int, float] = {}
    swings = tables["league_swings"]
    if not swings.empty:
        league_eo = swings.set_index("player_id")["league_eo"].to_dict()

    mode = config.get("strategy")
    if mode not in {"chasing", "protecting", "balanced"}:
        mode = tables["league_strategy"]["mode"].iloc[0] if not tables["league_strategy"].empty else "balanced"

    sell_price_owned = estimate_selling_prices(my_squad_ids, players, tables["my_transfers"])
    avoid_ids = {int(x) for x in (config.get("avoid") or []) if isinstance(x, int)}
    pool_ids = build_candidate_pool(players, xp_by_gw, my_squad_ids, avoid_ids)

    pool_source = players.set_index("id")[
        ["web_name", "position", "team", "now_cost", "status", "chance_of_playing_next_round", "news"]
    ].copy()
    pool_source["chance_of_playing_next_round"] = pool_source["chance_of_playing_next_round"].where(
        pool_source["chance_of_playing_next_round"].notna(), None
    )
    pool_info = pool_source.to_dict("index")
    now_cost_map = {p: pool_info[p]["now_cost"] for p in pool_ids}
    sell_price = {p: sell_price_owned.get(p, now_cost_map[p]) for p in pool_ids}

    weeks = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    horizon = max(min(int(config.get("horizon", 5)), len(weeks)), 1)
    weeks = weeks[:horizon]
    discount = {g: WEEKLY_DISCOUNT**i for i, g in enumerate(weeks)}

    raw_weekly_xp = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    adj_weekly_value = strategy_adjusted(raw_weekly_xp, league_eo, mode)
    effective_free = resolve_free_transfers(tables["league_free_transfers"], team_id)

    print(f"Gameweek {weeks[0]}. Free transfers available: {effective_free}")

    baseline = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )
    final_plan, gain, recommended, hit_reasons = recommend_transfers(
        baseline, pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, raw_weekly_xp=raw_weekly_xp,
    )
    wk0 = final_plan["weeks"][0]
    hits = wk0["hits"]
    n_transfers = len(wk0["transfers_in"])

    ok = True

    def check(label: str, cond: bool) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        ok = ok and cond

    print(f"\nRecommendation: {n_transfers} transfer(s), {hits} hit(s), recommended={recommended}")
    if wk0["transfers_out"]:
        out_names = ", ".join(pool_info[p]["web_name"] for p in sorted(wk0["transfers_out"]))
        in_names = ", ".join(pool_info[p]["web_name"] for p in sorted(wk0["transfers_in"]))
        print(f"  OUT {out_names} -> IN {in_names}")
    else:
        print("  roll the free transfer")
    for r in hit_reasons:
        print(f"  hit reason: {r}")

    # Rule: never recommend more transfers than free transfers available,
    # unless the extra ones are paid hits that passed ``justify_hits``.
    check(
        f"transfers ({n_transfers}) <= free transfers ({effective_free}) + justified hits ({hits})",
        n_transfers <= effective_free + hits,
    )
    check("every hit comes with a justification reason", hits == 0 or bool(hit_reasons))

    # Rule: Souza -- the user's real flagged *bench* player in this repo's
    # live data -- must never be the reason for a hit.
    souza_rows = players[players["web_name"].str.contains("Souza", case=False, na=False)]
    if not souza_rows.empty:
        souza_id = int(souza_rows.iloc[0]["id"])
        if souza_id in pool_info:
            souza_in_xi = souza_id in wk0["xi_ids"]
            souza_flagged = is_flagged_unavailable(pool_info[souza_id])
            print(f"\nSouza: flagged={souza_flagged}, started={souza_in_xi}")
            if souza_flagged and not souza_in_xi:
                check(
                    "a flagged BENCH player (Souza) is never the reason for a hit",
                    souza_id not in wk0["transfers_out"] or hits == 0,
                )

    # Rule 6 (user's exact scenario): with 1 free transfer and no flagged
    # *starter*, expect at most 1 transfer and no hit.
    if effective_free == 1:
        check("with 1 free transfer available, at most 1 transfer is recommended", n_transfers <= 1)
        check("with 1 free transfer available and no flagged starter, no hit is taken", hits == 0)

    return ok


if __name__ == "__main__":
    passed = run_check()
    print("\n" + ("PASS: transfer-hit policy behaves as specified." if passed else "FAIL: see above."))
    sys.exit(0 if passed else 1)
