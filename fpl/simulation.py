"""Step 11 Part B.4 extension 4 -- Monte Carlo P(finish 1st).

Runs thousands of simulated draws of **next gameweek's** realised points,
for every manager in the league, and reports how often each of *my*
candidate plans comes out on top. This directly answers "recommend the
plan with the best odds, not the best average", by testing candidates
against the same simulated rival outcomes rather than only comparing
their expected xP.

**Scope, stated plainly:** this simulates *one* gameweek's rank, not a
season-long league finish -- "P(finish 1st)" below means "P(top of the
GW table this week)". A true season-long P(finish 1st overall) would need
simulating every remaining fixture for every player and every rival's
future transfers, which is out of scope here; this single-week version
already answers the only question a weekly routine can act on (this
week's captain/transfer choice), and is flagged as a future extension in
the module docstring rather than silently conflated with the season race.

**My candidate plans.** Re-solves Step 9's MILP twice for the upcoming
gameweek: once normally (``baseline``, may include a transfer) and once
constrained to hold the current squad (``held``, rolling the free
transfer) -- the same two alternatives ``fpl.optimise.recommend_transfers``
already compares by expected value; here they're compared by win
probability instead. For each, every plausible captain is tested: the
top ``N_CAPTAIN_CANDIDATES`` starters by raw xP, which always includes
the "average-xP plan" (the highest-xP captain, i.e. what the site's
captain board recommends by default) -- so that plan is always shown
alongside whichever option has the best simulated odds.

**Rival plans.** There's no live picks endpoint for this private league
(see ``fpl.league_snapshot``'s module docstring), so each rival's squad
is carried forward unchanged from the most recent ``data/manual_snapshots/
<gw>.json`` archive: same XI, same captain. This is a documented
simplification -- we have no visibility into a rival's actual transfer or
captaincy decision for the gameweek being simulated, so "their last known
choice, unchanged" is the most defensible guess available, and is far
better than ignoring rivals altogether (the whole point of this league is
relative rank, not an absolute score).

**Points model.** Each player's realised points ~ Poisson(xp) -- a simple,
honest approximation, not a full re-derivation of FPL's scoring
distribution. It understates real variance (no fat tail for a captain
haul, no correlation between a striker's goals and his bonus), so these
probabilities should be read as "which plan looks better", not taken as
calibrated percentages. Flagged here, not hidden.

A player who appears in more than one squad (e.g. both me and a rival own
Haaland) gets exactly one simulated outcome per draw, shared across every
team that owns him -- draw once per distinct player, not once per team,
or the simulation would (wrongly) treat shared picks as independent.

Run with `uv run python -m fpl.simulation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.league_snapshot import build_league_tables
from fpl.optimise import (
    HIT_COST,
    build_candidate_pool,
    estimate_selling_prices,
    load_tables,
    lookup_weekly_xp,
    resolve_my_squad,
    solve_horizon,
    strategy_adjusted,
)

ROOT = Path(__file__).resolve().parent.parent
MANUAL_SNAPSHOTS_DIR = ROOT / "data" / "manual_snapshots"
TABLES_DIR = ROOT / "data" / "tables"
OUTPUT_PATH = TABLES_DIR / "simulation_results.json"

N_DRAWS = 20_000
N_CAPTAIN_CANDIDATES = 4
# Fixed, not security-sensitive -- just makes a reported run reproducible.
SEED = 11


def _latest_archive_gameweek() -> int | None:
    gws = [int(p.stem) for p in MANUAL_SNAPSHOTS_DIR.glob("*.json") if p.stem.isdigit()]
    return max(gws) if gws else None


def load_rival_plans(players: pd.DataFrame, next_gw: int, xp_by_gw: pd.DataFrame) -> list[dict[str, Any]]:
    """Every *other* manager's squad for ``next_gw``, carried forward from
    their last-known archive (see module docstring). Returns one dict per
    rival: ``team``, ``manager``, ``xi_ids``, ``captain_id``."""
    archive_gw = _latest_archive_gameweek()
    if archive_gw is None:
        return []
    snapshot = json.loads((MANUAL_SNAPSHOTS_DIR / f"{archive_gw}.json").read_text())
    league_managers, league_picks, _, _, my_manager_id, _ = build_league_tables(players, snapshot)
    team_by_manager = league_managers.set_index("manager_id")["entry_name"]
    name_by_manager = league_managers.set_index("manager_id")["player_name"]

    xp_col = f"gw{next_gw}_xp"
    xp_map = xp_by_gw.set_index("player_id")[xp_col].to_dict() if xp_col in xp_by_gw.columns else {}

    plans = []
    for manager_id, picks in league_picks.groupby("manager_id"):
        if manager_id == my_manager_id:
            continue
        xi = picks[picks["multiplier"] > 0]
        captain_rows = xi[xi["multiplier"] == 2]
        captain_id = int(captain_rows["player_id"].iloc[0]) if not captain_rows.empty else None
        if captain_id is None and not xi.empty:
            # Fallback (shouldn't trigger -- league_snapshot already elevates
            # a benched captain's vice): captain the highest-xp starter.
            captain_id = int(max(xi["player_id"], key=lambda p: xp_map.get(p, 0.0)))
        plans.append(
            {
                "team": team_by_manager.get(manager_id, f"manager {manager_id}"),
                "manager": name_by_manager.get(manager_id, ""),
                "xi_ids": sorted(int(p) for p in xi["player_id"]),
                "captain_id": captain_id,
                "source_gameweek": archive_gw,
            }
        )
    return plans


def _my_candidate_plans(
    pool_ids: list[int],
    adj_weekly_value: dict[tuple[int, int], float],
    raw_weekly_xp: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    my_squad_ids: set[int],
    sell_price: dict[int, int],
    bank: int,
    effective_free: int,
    weeks: list[int],
    discount: dict[int, float],
) -> list[dict[str, Any]]:
    """The two squad scenarios (transfer vs hold) crossed with every
    plausible captain, as plain dicts ready for the simulator. Mirrors
    ``fpl.optimise.recommend_transfers``'s own ``baseline``/``held`` solve,
    just keeping *both* results instead of picking one."""
    week0 = weeks[0]
    baseline = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )
    held = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, held_weeks=frozenset({week0}),
    )

    scenarios = []
    for label, plan in (("transfer plan", baseline), ("hold (roll free transfer)", held)):
        wk0 = plan["weeks"][0]
        xi_ids = sorted(wk0["xi_ids"])
        by_xp = sorted(xi_ids, key=lambda p: raw_weekly_xp.get((p, week0), 0.0), reverse=True)
        captain_candidates = by_xp[:N_CAPTAIN_CANDIDATES]
        average_xp_captain = by_xp[0] if by_xp else None
        transfers_in = sorted(wk0["transfers_in"])
        transfers_out = sorted(wk0["transfers_out"])
        # Derive this week's *actual* hit cost from the transfers really
        # made, rather than trusting ``wk0["hits"]`` directly: the MILP
        # solves the whole horizon jointly, and can legitimately report a
        # nonzero "hits" in a held week purely to bank an extra free
        # transfer for a later week (see ``solve_horizon``'s free-transfer
        # recursion) -- a correct multi-week bookkeeping artifact, but not
        # a real cost incurred *this* gameweek, which is all that matters
        # for a single-gameweek simulation.
        hits = max(0, len(transfers_out) - effective_free)
        for captain_id in captain_candidates:
            scenarios.append(
                {
                    "scenario": label,
                    "xi_ids": xi_ids,
                    "captain_id": captain_id,
                    "is_average_xp_captain": captain_id == average_xp_captain,
                    "hits": hits,
                    "transfers_in": transfers_in,
                    "transfers_out": transfers_out,
                }
            )
    return scenarios


def _team_points_matrix(
    xi_ids: list[int], captain_id: int | None, draws: np.ndarray, index_of: dict[int, int]
) -> np.ndarray:
    """``draws`` is (N_DRAWS, n_unique_players); sum this team's XI columns,
    doubling the captain's."""
    cols = [index_of[p] for p in xi_ids if p in index_of]
    total = draws[:, cols].sum(axis=1) if cols else np.zeros(draws.shape[0])
    if captain_id is not None and captain_id in index_of:
        total = total + draws[:, index_of[captain_id]]
    return total


def run_simulation(
    my_plans: list[dict[str, Any]],
    rival_plans: list[dict[str, Any]],
    xp_map: dict[int, float],
    pool_info: dict[int, dict[str, Any]],
    n_draws: int = N_DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Draws one shared Poisson(xp) outcome per distinct player (so a
    player owned by several teams gets a correlated, not independent,
    outcome -- see module docstring), then tallies how often each of my
    plans finishes outright top (ties for the max count as 1st, shared)."""
    universe: set[int] = set()
    for plan in my_plans:
        universe.update(plan["xi_ids"])
        if plan["captain_id"] is not None:
            universe.add(plan["captain_id"])
    for plan in rival_plans:
        universe.update(plan["xi_ids"])
        if plan["captain_id"] is not None:
            universe.add(plan["captain_id"])
    universe_ids = sorted(universe)
    index_of = {p: i for i, p in enumerate(universe_ids)}

    rng = np.random.default_rng(seed)
    lambdas = np.clip(np.array([xp_map.get(p, 0.0) for p in universe_ids]), 0.0, None)
    draws = rng.poisson(lambdas, size=(n_draws, len(universe_ids)))

    rival_totals = np.column_stack(
        [_team_points_matrix(p["xi_ids"], p["captain_id"], draws, index_of) for p in rival_plans]
    ) if rival_plans else np.zeros((n_draws, 0))
    rival_best = rival_totals.max(axis=1) if rival_totals.shape[1] else np.zeros(n_draws)

    rows = []
    for plan in my_plans:
        # Hit cost is deterministic (a flat -4/transfer), not drawn --
        # subtract it from every draw so a hits-heavy transfer plan is
        # judged on its real net score, not just its XI's gross points.
        mine = _team_points_matrix(plan["xi_ids"], plan["captain_id"], draws, index_of)
        mine = mine - HIT_COST * plan["hits"]
        win = mine >= rival_best
        captain_name = pool_info.get(plan["captain_id"], {}).get("web_name", "?")
        rows.append(
            {
                "scenario": plan["scenario"],
                "captain": captain_name,
                "is_average_xp_captain": plan["is_average_xp_captain"],
                "hits": plan["hits"],
                "mean_points": round(float(mine.mean()), 2),
                "p_finish_1st": round(float(win.mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("p_finish_1st", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=N_DRAWS, help="Monte Carlo sample size")
    args = parser.parse_args()
    n_draws = args.draws

    config = load_config()
    tables = load_tables()
    players, xp_by_gw = tables["players"], tables["xp_by_gameweek"]
    league_picks, league_history = tables["league_picks"], tables["league_history"]
    league_managers = tables.get("league_managers", pd.DataFrame())

    team_id = resolve_team_id(config, league_managers)
    if team_id is None or league_picks.empty:
        print("No team_id/league data resolvable -- nothing to simulate.")
        return

    my_squad_ids, hist_bank, gameweek = resolve_my_squad(league_picks, league_history, team_id)
    bank = config.get("bank_override")
    bank = int(bank) if bank is not None else hist_bank

    swings = tables["league_swings"]
    league_eo: dict[int, float] = {}
    if not swings.empty:
        if "weighted_league_eo" in swings.columns:
            effective = swings["weighted_league_eo"].fillna(swings["league_eo"])
            league_eo = pd.Series(effective.values, index=swings["player_id"]).to_dict()
        else:
            league_eo = swings.set_index("player_id")["league_eo"].to_dict()
    mode = config.get("strategy")
    if mode not in {"chasing", "protecting", "balanced"}:
        mode = tables["league_strategy"]["mode"].iloc[0] if not tables["league_strategy"].empty else "balanced"

    sell_price_owned = estimate_selling_prices(my_squad_ids, players, tables["my_transfers"])
    avoid_ids = {int(x) for x in (config.get("avoid") or []) if isinstance(x, int)}
    pool_ids = build_candidate_pool(players, xp_by_gw, my_squad_ids, avoid_ids)
    pool_info = players.set_index("id")[["web_name", "position", "team", "now_cost"]].to_dict("index")
    now_cost_map = {p: pool_info[p]["now_cost"] for p in pool_ids}
    sell_price = {p: sell_price_owned.get(p, now_cost_map[p]) for p in pool_ids}

    weeks_all = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    horizon = max(min(int(config.get("horizon", 5)), len(weeks_all)), 1)
    weeks = weeks_all[:horizon]
    next_gw = weeks[0]
    discount = {g: 0.9**i for i, g in enumerate(weeks)}

    raw_weekly_xp = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    adj_weekly_value = strategy_adjusted(raw_weekly_xp, league_eo, mode)
    effective_free = max(int(config.get("free_transfers", 1)) - int(config.get("transfers_made", 0)), 0)

    print(f"Simulating gameweek {next_gw} ({n_draws:,} draws)...")
    my_plans = _my_candidate_plans(
        pool_ids, adj_weekly_value, raw_weekly_xp, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )
    rival_plans = load_rival_plans(players, next_gw, xp_by_gw)
    if not rival_plans:
        print("No data/manual_snapshots/<gw>.json archive found -- can't resolve rival squads, "
              "simulating against myself only (P(finish 1st) will read as ~1.0 and isn't meaningful).")

    xp_map = xp_by_gw.set_index("player_id")[f"gw{next_gw}_xp"].to_dict()
    results = run_simulation(my_plans, rival_plans, xp_map, pool_info, n_draws=n_draws)

    results.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"  wrote {OUTPUT_PATH} ({len(results)} rows)")
    print(f"\nP(finish 1st) in gameweek {next_gw}, by plan (vs {len(rival_plans)} rivals' last-known squads):")
    print(results.to_string(index=False))
    if not results.empty:
        best = results.iloc[0]
        avg_plan = results[results["is_average_xp_captain"]].sort_values("p_finish_1st", ascending=False)
        avg_best = avg_plan.iloc[0] if not avg_plan.empty else best
        print(
            f"\nBest win-probability plan: {best['scenario']}, captain {best['captain']} "
            f"-> P(1st) {best['p_finish_1st']:.1%}, mean {best['mean_points']} pts"
        )
        print(
            f"Average-xP plan (site's default captain): {avg_best['scenario']}, captain {avg_best['captain']} "
            f"-> P(1st) {avg_best['p_finish_1st']:.1%}, mean {avg_best['mean_points']} pts"
        )


if __name__ == "__main__":
    main()
