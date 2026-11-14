"""Formation check: the optimiser picks its XI under FPL's formation rules
only (exactly 1 goalkeeper, at least 3 defenders, at least 2 midfielders,
at least 1 forward -- ``fpl.optimise.FORMATION_BOUNDS``), not a fixed 4-4-2.

Proof by experiment, on the real data: solve this gameweek with the squad
held as it is (so only the starting XI can change), then, one at a time,
boost a *benched* outfield player's projected points for this gameweek so
starting him should win, re-solve, and check the formation changes to
accommodate him -- e.g. a benched midfielder pushes a 4-4-2 to 3-5-2 or
4-5-1. The boost lives only in an in-memory copy of the inputs for that
one solve; nothing is written.

Run with ``uv run python -m fpl.check_formation``. Exits non-zero on failure.
"""

from __future__ import annotations

import sys

import pandas as pd

from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.optimise import (
    FORMATION_BOUNDS,
    WEEKLY_DISCOUNT,
    build_candidate_pool,
    estimate_selling_prices,
    load_tables,
    lookup_weekly_xp,
    resolve_free_transfers,
    resolve_my_squad,
    solve_horizon,
    strategy_adjusted,
)

BOOST = 15.0  # projected points given to the benched player for this one test solve


def formation(xi_ids, pool_info) -> str:
    counts = {pos: sum(pool_info[p]["position"] == pos for p in xi_ids) for pos in ("DEF", "MID", "FWD")}
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def legal(xi_ids, pool_info) -> bool:
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        n = sum(pool_info[p]["position"] == pos for p in xi_ids)
        if not lo <= n <= hi:
            return False
    return len(xi_ids) == 11


def main() -> None:
    config = load_config()
    tables = load_tables()
    players, xp_by_gw = tables["players"], tables["xp_by_gameweek"]
    league_managers = tables.get("league_managers", pd.DataFrame())
    team_id = resolve_team_id(config, league_managers)
    my_squad_ids, bank, _ = resolve_my_squad(tables["league_picks"], tables["league_history"], team_id)

    swings = tables["league_swings"]
    league_eo = swings.set_index("player_id")["league_eo"].to_dict() if not swings.empty else {}
    mode = tables["league_strategy"]["mode"].iloc[0] if not tables["league_strategy"].empty else "balanced"
    sell_owned = estimate_selling_prices(my_squad_ids, players, tables["my_transfers"])
    pool_ids = build_candidate_pool(players, xp_by_gw, my_squad_ids, set())
    pool_info = players.set_index("id")[["web_name", "position", "team", "now_cost"]].to_dict("index")
    sell_price = {p: sell_owned.get(p, pool_info[p]["now_cost"]) for p in pool_ids}
    weeks = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))[:5]
    discount = {g: WEEKLY_DISCOUNT**i for i, g in enumerate(weeks)}
    raw = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    free = resolve_free_transfers(tables["league_free_transfers"], team_id)
    g0 = weeks[0]

    def solve(values):
        plan = solve_horizon(pool_ids, values, pool_info, my_squad_ids, sell_price, bank, free, weeks, discount,
                             held_weeks=frozenset({g0}))
        return plan["weeks"][0]

    print(f"FORMATION_BOUNDS = {FORMATION_BOUNDS}  (FPL: 1 GK, >=3 DEF, >=2 MID, >=1 FWD)")
    base = solve(strategy_adjusted(raw, league_eo, mode))
    base_form = formation(base["xi_ids"], pool_info)
    ok = legal(base["xi_ids"], pool_info)
    print(f"GW{g0}, squad held: optimiser picks {base_form} (legal: {ok})")

    bench = sorted(base["squad_ids"] - base["xi_ids"], key=lambda p: pool_info[p]["position"])
    for p in bench:
        if pool_info[p]["position"] == "GKP":
            continue
        boosted_raw = dict(raw)
        boosted_raw[(p, g0)] = BOOST  # in-memory only, for this one solve
        wk = solve(strategy_adjusted(boosted_raw, league_eo, mode))
        form = formation(wk["xi_ids"], pool_info)
        started = p in wk["xi_ids"]
        passed = started and legal(wk["xi_ids"], pool_info) and form != base_form
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] boost benched {pool_info[p]['position']} {pool_info[p]['web_name']} "
              f"to {BOOST:.0f} -> starts: {started}, formation {base_form} -> {form}")
    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
