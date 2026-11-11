"""Step 11 Part B.4 extension 4 -- Monte Carlo P(finish 1st). Step 12
extends this with a season-long version (see below).

Runs thousands of simulated draws of **next gameweek's** realised points,
for every manager in the league, and reports how often each of *my*
candidate plans comes out on top. This directly answers "recommend the
plan with the best odds, not the best average", by testing candidates
against the same simulated rival outcomes rather than only comparing
their expected xP.

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

**Rival plans.** FPL hides a rival's picks for the *next* gameweek until
its deadline passes (see ``fpl.league``'s module docstring), so each
rival's squad here is their most recent confirmed one, straight from the
live-fetched ``league_picks.parquet`` (``fpl.fetch``/``fpl.league_snapshot``
keep this current every run regardless of mode) -- same XI, same
captain, carried forward unchanged. This is a documented simplification
-- we have no visibility into a rival's actual transfer or captaincy
decision for the gameweek being simulated, so "their last known choice,
unchanged" is the most defensible guess available, and is far better
than ignoring rivals altogether (the whole point of this league is
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

**Season-long P(finish 1st by GW19) -- Step 12.** ``simulate_season_
standings`` answers a different, longer-horizon question: not "who wins
this gameweek" but "who's likely to be top of the table at the GW19
halfway/chip-reset checkpoint". It does *not* re-run the squad-level
xP/Poisson model for every remaining week for every manager (that would
need forecasting 9 rivals' future transfers and captaincy, which this
project has no basis to predict -- see ``fpl.rival_transfers``'s module
docstring on why a transfer *habit* is learnable but a specific future
pick isn't). Instead it treats each manager's own historical weekly
score (``league_history.points``) as a Normal(mean, std) draw and sums
``target_gw - current_gw`` independent draws onto their current total --
a coarser, team-level model than the GW-level simulation above, but an
honest one: it only assumes "recent form continues, with recent
variance", not a specific future squad. Flagged as coarser, not hidden,
same convention as the GW-level model above.

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
from fpl.optimise import (
    HIT_COST,
    build_candidate_pool,
    estimate_selling_prices,
    load_tables,
    lookup_weekly_xp,
    resolve_free_transfers,
    resolve_my_squad,
    solve_horizon,
    strategy_adjusted,
)

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
OUTPUT_PATH = TABLES_DIR / "simulation_results.json"
SEASON_OUTPUT_PATH = TABLES_DIR / "season_simulation.json"

N_DRAWS = 20_000
N_CAPTAIN_CANDIDATES = 4
# Fixed, not security-sensitive -- just makes a reported run reproducible.
SEED = 11

# Step 12: the season's halfway/chip-reset checkpoint (see fpl.league's
# CHIP_HALF_BOUNDARY) -- the natural "where do we stand" milestone for a
# season-long projection, since every manager's chip budget resets there
# too.
SEASON_SIM_TARGET_GW = 19
SEASON_SIM_DRAWS = 20_000
# Floor for a manager's week-to-week std when too few recorded gameweeks
# exist to estimate one reliably (or it comes out as exactly 0, e.g. a
# manager who has played only a single gameweek so far) -- without a
# floor, a single lucky/unlucky manager would simulate as having *zero*
# variance, which silently overstates how sure this projection is.
SEASON_SIM_MIN_STD = 8.0


def load_rival_plans(
    league_picks: pd.DataFrame, league_managers: pd.DataFrame, team_id: int, xp_map: dict[int, float]
) -> list[dict[str, Any]]:
    """Every *other* manager's most recently confirmed squad, carried
    forward unchanged as their guess for next gameweek (see module
    docstring). Works identically in live-API and manual-snapshot mode,
    since both already write the same shape to ``league_picks.parquet``/
    ``league_managers.parquet``. Returns one dict per rival: ``team``,
    ``manager``, ``xi_ids``, ``captain_id``, ``source_gameweek``."""
    if league_picks.empty:
        return []
    source_gameweek = int(league_picks["gameweek"].max())
    latest = league_picks[league_picks["gameweek"] == source_gameweek]
    team_by_manager = league_managers.set_index("manager_id")["entry_name"]
    name_by_manager = league_managers.set_index("manager_id")["player_name"]

    plans = []
    for manager_id, picks in latest.groupby("manager_id"):
        if manager_id == team_id:
            continue
        xi = picks[picks["multiplier"] > 0]
        captain_rows = xi[xi["is_captain"]] if "is_captain" in xi.columns else xi[xi["multiplier"] >= 2]
        captain_id = int(captain_rows["player_id"].iloc[0]) if not captain_rows.empty else None
        if captain_id is None and not xi.empty:
            # Fallback (shouldn't trigger -- every fetched gameweek has a captain): captain the highest-xp starter.
            captain_id = int(max(xi["player_id"], key=lambda p: xp_map.get(p, 0.0)))
        plans.append(
            {
                "team": team_by_manager.get(manager_id, f"manager {manager_id}"),
                "manager": name_by_manager.get(manager_id, ""),
                "xi_ids": sorted(int(p) for p in xi["player_id"]),
                "captain_id": captain_id,
                "source_gameweek": source_gameweek,
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


def simulate_season_standings(
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    team_id: int,
    gameweek: int,
    target_gw: int = SEASON_SIM_TARGET_GW,
    n_draws: int = SEASON_SIM_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """P(finish 1st by GW``target_gw``) for every manager, from each
    manager's own historical weekly-score distribution (see module
    docstring for why this is a coarser, team-level model rather than a
    repeat of the squad-level GW simulation above). If ``gameweek`` is
    already at or past ``target_gw``, there's nothing left to simulate --
    returns the actual (settled) standings instead, flagged as such."""
    totals_now = league_history[league_history["gameweek"] == gameweek].set_index("manager_id")["total_points"]
    if team_id not in totals_now.index:
        return {"available": False, "reason": f"manager {team_id} has no league_history row at gameweek {gameweek}"}

    manager_ids = totals_now.index.tolist()
    weeks_remaining = max(target_gw - gameweek, 0)

    if weeks_remaining == 0:
        finals = np.tile(totals_now.loc[manager_ids].values, (n_draws, 1))
        settled = True
    else:
        stats = league_history.groupby("manager_id")["points"].agg(["mean", "std"])
        stats["std"] = stats["std"].fillna(0.0).clip(lower=SEASON_SIM_MIN_STD)
        means = stats.loc[manager_ids, "mean"].values
        stds = stats.loc[manager_ids, "std"].values
        rng = np.random.default_rng(seed)
        draws = rng.normal(
            means[None, :, None], stds[None, :, None], size=(n_draws, len(manager_ids), weeks_remaining)
        ).clip(min=0)
        finals = totals_now.loc[manager_ids].values[None, :] + draws.sum(axis=2)
        settled = False

    names = league_managers.set_index("manager_id")
    per_manager = []
    for i, manager_id in enumerate(manager_ids):
        mine = finals[:, i]
        others = np.delete(finals, i, axis=1)
        best_other = others.max(axis=1) if others.shape[1] else np.full(n_draws, -np.inf)
        per_manager.append(
            {
                "manager_id": int(manager_id),
                "entry_name": names["entry_name"].get(manager_id),
                "player_name": names["player_name"].get(manager_id),
                "current_total": int(totals_now.loc[manager_id]),
                "mean_points_per_gw": round(float(league_history.loc[league_history["manager_id"] == manager_id, "points"].mean()), 1),
                "projected_total_by_gw19": round(float(mine.mean()), 1),
                "p_finish_1st_by_gw19": round(float((mine >= best_other).mean()), 4),
            }
        )
    per_manager.sort(key=lambda r: r["p_finish_1st_by_gw19"], reverse=True)

    return {
        "available": True,
        "settled": settled,
        "gameweek": gameweek,
        "target_gameweek": target_gw,
        "weeks_remaining": weeks_remaining,
        "team_id": team_id,
        "standings": per_manager,
    }


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
    effective_free = resolve_free_transfers(tables["league_free_transfers"], team_id)

    print(f"Simulating gameweek {next_gw} ({n_draws:,} draws)...")
    my_plans = _my_candidate_plans(
        pool_ids, adj_weekly_value, raw_weekly_xp, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )
    xp_map = xp_by_gw.set_index("player_id")[f"gw{next_gw}_xp"].to_dict()
    rival_plans = load_rival_plans(league_picks, league_managers, team_id, xp_map)
    if not rival_plans:
        print("No other managers found in league_picks.parquet -- can't resolve rival squads, "
              "simulating against myself only (P(finish 1st) will read as ~1.0 and isn't meaningful).")

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

    # Step 12: season-long "where will I actually finish" projection --
    # separate from the single-gameweek squad simulation above (see module
    # docstring).
    if not league_history.empty and not league_managers.empty:
        season = simulate_season_standings(league_history, league_managers, team_id, gameweek)
        SEASON_OUTPUT_PATH.write_text(json.dumps(season, indent=2))
        print(f"  wrote {SEASON_OUTPUT_PATH}")
        if season["available"]:
            mine = next((r for r in season["standings"] if r["manager_id"] == team_id), None)
            if mine:
                status = "(season already settled)" if season["settled"] else f"({season['weeks_remaining']} GWs left)"
                print(
                    f"\nP(finish 1st by GW{season['target_gameweek']}) {status}: "
                    f"{mine['p_finish_1st_by_gw19']:.1%}, projected total {mine['projected_total_by_gw19']}"
                )
        else:
            print(f"  season simulation unavailable: {season['reason']}")


if __name__ == "__main__":
    main()
