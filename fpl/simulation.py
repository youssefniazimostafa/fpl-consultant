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

from fpl import goal_calls
from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.optimise import (
    HIT_COST,
    WEEKLY_DISCOUNT,
    build_squad_plan_table,
    build_transfers_table,
    gate_planned_weeks,
    recommend_transfers,
    rule_inputs,
    build_candidate_pool,
    estimate_selling_prices,
    load_tables,
    lookup_weekly_xp,
    resolve_free_transfers,
    resolve_my_squad,
    solve_horizon,
)

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
OUTPUT_PATH = TABLES_DIR / "simulation_results.json"
# Step 15: "Your options this week" (Analytics) -- a separate output, never
# read by the decision, so the recommendation can't change because of it.
OPTIONS_PATH = TABLES_DIR / "simulation_options.json"
N_OPTIONS = 3
SEASON_OUTPUT_PATH = TABLES_DIR / "season_simulation.json"
HOLD_SQUAD_PATH = TABLES_DIR / "simulation_hold_squad.parquet"
HOLD_TRANSFERS_PATH = TABLES_DIR / "simulation_hold_transfers.parquet"

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
                "manager_id": int(manager_id),
                "team": team_by_manager.get(manager_id, f"manager {manager_id}"),
                "manager": name_by_manager.get(manager_id, ""),
                "xi_ids": sorted(int(p) for p in xi["player_id"]),
                "captain_id": captain_id,
                "source_gameweek": source_gameweek,
            }
        )
    return plans


TRANSFER_LABEL = "transfer plan"
HOLD_LABEL = "hold (roll free transfer)"


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
    rules: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """The squad scenarios crossed with every plausible captain, as plain
    dicts ready for the simulator, plus the full plans themselves.

    The "transfer plan" is exactly what ``fpl.optimise`` recommends -- it
    goes through ``recommend_transfers`` the same way ``optimise.main``
    does, so an unjustified hit is re-solved away here too. (Before this,
    the raw ``baseline`` solve was simulated instead: in GW11 that was a
    3-transfer, 2-hit plan the optimiser had already rejected, so the
    simulation compared "hold" against a plan nobody recommended -- the
    source of "simulation says hold, verdict says transfer".) When the
    optimiser itself recommends holding, only the hold scenario exists.

    Returns (scenarios, plans by label, meta) where meta carries the
    optimiser's own horizon gain / recommended flag / hit reasons."""
    week0 = weeks[0]
    rules = rules or {}
    baseline = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, **rules,
    )
    recommended_plan, gain, transfer_recommended, hit_reasons = recommend_transfers(
        baseline, pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, raw_weekly_xp=raw_weekly_xp, **rules,
    )
    held = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, held_weeks=frozenset({week0}), **rules,
    )
    plans: dict[str, dict[str, Any]] = {HOLD_LABEL: held}
    if transfer_recommended and recommended_plan["weeks"][0]["transfers_in"]:
        plans = {TRANSFER_LABEL: recommended_plan, HOLD_LABEL: held}

    scenarios = []
    for label, plan in plans.items():
        wk0 = plan["weeks"][0]
        xi_ids = sorted(wk0["xi_ids"])
        by_xp = sorted(xi_ids, key=lambda p: raw_weekly_xp.get((p, week0), 0.0), reverse=True)
        captain_candidates = by_xp[:N_CAPTAIN_CANDIDATES]
        # always test the captain the plan itself would use
        if wk0["captain_id"] is not None and wk0["captain_id"] not in captain_candidates:
            captain_candidates.append(wk0["captain_id"])
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
                    "is_plan_captain": captain_id == wk0["captain_id"],
                    "hits": hits,
                    "transfers_in": transfers_in,
                    "transfers_out": transfers_out,
                }
            )
    meta = {"gain": gain, "transfer_recommended": transfer_recommended, "hit_reasons": hit_reasons}
    return scenarios, plans, meta


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
                "is_plan_captain": plan.get("is_plan_captain", False),
                "hits": plan["hits"],
                "transfers_in": [pool_info.get(p, {}).get("web_name", "?") for p in plan.get("transfers_in", [])],
                "transfers_out": [pool_info.get(p, {}).get("web_name", "?") for p in plan.get("transfers_out", [])],
                "mean_points": round(float(mine.mean()), 2),
                "p_finish_1st": round(float(win.mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("p_finish_1st", ascending=False).reset_index(drop=True)


def simulate_options(
    options: list[dict[str, Any]],
    rival_plans: list[dict[str, Any]],
    xp_map: dict[int, float],
    leader_id: int | None,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """For each option (``xi_ids``, ``captain_id``, ``hits``): mean points,
    a bad week and a good week (10th / 90th percentile of the same shared
    Poisson draws as ``run_simulation``), the chance of the top score in the
    league, and the chance of outscoring the leader this week."""
    universe: set[int] = set()
    for plan in options + rival_plans:
        universe.update(plan["xi_ids"])
        if plan.get("captain_id") is not None:
            universe.add(plan["captain_id"])
    ids = sorted(universe)
    index_of = {p: i for i, p in enumerate(ids)}
    rng = np.random.default_rng(seed)
    draws = rng.poisson(np.clip(np.array([xp_map.get(p, 0.0) for p in ids]), 0.0, None), size=(n_draws, len(ids)))
    rival_tot = {p["manager_id"]: _team_points_matrix(p["xi_ids"], p["captain_id"], draws, index_of) for p in rival_plans}
    best = np.column_stack(list(rival_tot.values())).max(axis=1) if rival_tot else np.zeros(n_draws)
    leader = rival_tot.get(leader_id)
    out = []
    for o in options:
        mine = _team_points_matrix(o["xi_ids"], o["captain_id"], draws, index_of) - HIT_COST * o.get("hits", 0)
        out.append({
            "mean_points": round(float(mine.mean()), 1),
            "bad_week": int(np.percentile(mine, 10)),
            "good_week": int(np.percentile(mine, 90)),
            "p_top_score": round(float((mine >= best).mean()), 3),
            "p_beat_leader": round(float((mine > leader).mean()), 3) if leader is not None else None,
        })
    return out


def build_options(
    full_plans: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    solver_args: tuple,
    rules: dict[str, Any],
    pool_info: dict[int, dict[str, Any]],
    raw_weekly_xp: dict[tuple[int, int], float],
    xp_map: dict[int, float],
    rival_plans: list[dict[str, Any]],
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    team_id: int,
    keep: list | None = None,
) -> dict[str, Any]:
    """Analytics' "Your options this week": the top captain choices for the
    recommended plan and the top transfer choices (the optimiser's move, the
    next best ones with its incoming player(s) left out, and holding),
    each with its simulated range and rival context. Options, not advice:
    the consultant's pick is only marked."""
    pool_ids, weekly_value, info, my_squad, sell, bank, free, weeks, discount = solver_args
    g0 = weeks[0]
    name = lambda p: pool_info.get(p, {}).get("web_name", "?")  # noqa: E731
    latest = league_picks[league_picks["gameweek"] == league_picks["gameweek"].max()] if not league_picks.empty else league_picks
    rivals = latest[latest["manager_id"] != team_id]
    n_rivals = int(rivals["manager_id"].nunique()) if not rivals.empty else 0
    rivals_captaining = rivals[rivals["is_captain"]].groupby("player_id").size().to_dict() if not rivals.empty else {}
    rivals_owning = rivals.groupby("player_id").size().to_dict() if not rivals.empty else {}
    leader_id, leader_name = None, None
    if not league_history.empty:
        lh = league_history[league_history["gameweek"] == league_history["gameweek"].max()]
        lh = lh[lh["manager_id"] != team_id].sort_values("total_points", ascending=False)
        if not lh.empty:
            leader_id = int(lh["manager_id"].iloc[0])
            names = league_managers.set_index("manager_id")["entry_name"].to_dict() if not league_managers.empty else {}
            leader_name = names.get(leader_id)

    def five_weeks(plan: dict[str, Any]) -> float:
        tot = 0.0
        for wk in plan["weeks"]:
            g = wk["gameweek"]
            tot += sum(raw_weekly_xp.get((p, g), 0.0) for p in wk["xi_ids"])
            if wk["captain_id"] is not None:
                tot += raw_weekly_xp.get((wk["captain_id"], g), 0.0)
            tot -= HIT_COST * wk["hits"]
        return round(tot, 1)

    rec_label = TRANSFER_LABEL if meta.get("transfer_recommended") and TRANSFER_LABEL in full_plans else HOLD_LABEL
    rec = full_plans[rec_label]
    w0 = rec["weeks"][0]
    xi = sorted(w0["xi_ids"])

    # captain choices: the recommended plan's XI, highest expected points first
    caps = sorted(xi, key=lambda p: -xp_map.get(p, 0.0))[:N_OPTIONS]
    if w0["captain_id"] is not None and w0["captain_id"] not in caps:
        caps = caps[: N_OPTIONS - 1] + [w0["captain_id"]]
    cap_opts = [{"xi_ids": xi, "captain_id": c, "hits": w0["hits"]} for c in caps]
    cap_sims = simulate_options(cap_opts, rival_plans, xp_map, leader_id)
    captains = [{
        "player_id": c, "web_name": name(c), "projected_points": round(2 * xp_map.get(c, 0.0), 1),
        "team_projected": sim["mean_points"], "bad_week": sim["bad_week"], "good_week": sim["good_week"],
        "p_beat_leader": sim["p_beat_leader"], "p_top_score": sim["p_top_score"],
        "rivals_captaining": int(rivals_captaining.get(c, 0)), "is_consultant_pick": c == w0["captain_id"],
    } for c, sim in zip(caps, cap_sims)]

    # transfer choices
    plans: list[tuple[dict[str, Any], bool]] = []
    if rec_label == TRANSFER_LABEL:
        plans.append((rec, True))
    banned: set[int] = set(rec["weeks"][0]["transfers_in"])
    for _ in range(N_OPTIONS - len(plans)):
        ids = [p for p in pool_ids if p not in banned or p in my_squad]
        alt = solve_horizon(ids, weekly_value, info, my_squad, sell, bank, free, weeks, discount,
                            no_hit_weeks=frozenset({g0}), **rules)
        if alt["status"] != "Optimal" or not alt["weeks"][0]["transfers_in"]:
            break
        # later weeks through the same transfer bar as the recommendation
        alt = gate_planned_weeks(alt, *solver_args, raw_weekly_xp=raw_weekly_xp, **rules)
        plans.append((alt, False))
        banned |= set(alt["weeks"][0]["transfers_in"])
    hold = full_plans[HOLD_LABEL]
    if rec_label != HOLD_LABEL:  # the recommendation itself is already gated
        hold = gate_planned_weeks(hold, *solver_args, raw_weekly_xp=raw_weekly_xp, **rules)
    plans.append((hold, rec_label == HOLD_LABEL))
    if keep is not None:  # the same transfer options for fpl.goal_calls
        keep.extend(plans)
    t_opts = [{"xi_ids": sorted(p["weeks"][0]["xi_ids"]), "captain_id": p["weeks"][0]["captain_id"],
               "hits": p["weeks"][0]["hits"]} for p, _ in plans]
    t_sims = simulate_options(t_opts, rival_plans, xp_map, leader_id)
    transfers = []
    for (plan, is_rec), sim in zip(plans, t_sims):
        wk = plan["weeks"][0]
        ins, outs = sorted(wk["transfers_in"]), sorted(wk["transfers_out"])
        transfers.append({
            "transfers_in": [name(p) for p in ins], "transfers_out": [name(p) for p in outs],
            "hold": not ins, "captain": name(wk["captain_id"]), "hits": wk["hits"],
            "projected_points": sim["mean_points"], "five_week_points": five_weeks(plan),
            "bad_week": sim["bad_week"], "good_week": sim["good_week"],
            "p_beat_leader": sim["p_beat_leader"], "p_top_score": sim["p_top_score"],
            "rivals_owning_in": [int(rivals_owning.get(p, 0)) for p in ins], "is_consultant_pick": is_rec,
        })
    return {"available": True, "gameweek": g0, "rivals": n_rivals, "leader_name": leader_name,
            "captains": captains, "transfers": transfers}


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
    # same fields as fpl.optimise.main's pool_info -- recommend_transfers'
    # hit check reads status/chance_of_playing_next_round/news
    pool_source = players.set_index("id")[
        ["web_name", "position", "team", "now_cost", "status", "chance_of_playing_next_round", "news"]
    ].copy()
    pool_source["chance_of_playing_next_round"] = pool_source["chance_of_playing_next_round"].where(
        pool_source["chance_of_playing_next_round"].notna(), None
    )
    pool_info = pool_source.to_dict("index")
    now_cost_map = {p: pool_info[p]["now_cost"] for p in pool_ids}
    sell_price = {p: sell_price_owned.get(p, now_cost_map[p]) for p in pool_ids}

    weeks_all = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    horizon = max(min(int(config.get("horizon", 5)), len(weeks_all)), 1)
    weeks = weeks_all[:horizon]
    next_gw = weeks[0]
    discount = {g: WEEKLY_DISCOUNT**i for i, g in enumerate(weeks)}

    raw_weekly_xp = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    adj_weekly_value, rules = rule_inputs(raw_weekly_xp, league_eo, mode, weeks, team_id=team_id)
    effective_free = resolve_free_transfers(tables["league_free_transfers"], team_id)

    print(f"Simulating gameweek {next_gw} ({n_draws:,} draws)...")
    my_plans, full_plans, meta = _my_candidate_plans(
        pool_ids, adj_weekly_value, raw_weekly_xp, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, rules,
    )
    xp_map = xp_by_gw.set_index("player_id")[f"gw{next_gw}_xp"].to_dict()
    rival_plans = load_rival_plans(league_picks, league_managers, team_id, xp_map)
    if not rival_plans:
        print("No other managers found in league_picks.parquet -- can't resolve rival squads, "
              "simulating against myself only (P(finish 1st) will read as ~1.0 and isn't meaningful).")

    results = run_simulation(my_plans, rival_plans, xp_map, pool_info, n_draws=n_draws)

    results.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"  wrote {OUTPUT_PATH} ({len(results)} rows)")
    option_plans: list = []
    try:
        options = build_options(
            full_plans, meta,
            (pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price, bank, effective_free, weeks, discount),
            rules, pool_info, raw_weekly_xp, xp_map, rival_plans, league_picks, league_history, league_managers, team_id,
            keep=option_plans,
        )
    except Exception as e:  # noqa: BLE001 -- an Analytics extra must never stop the pipeline
        options = {"available": False, "reason": f"options not computed: {e}"}
    OPTIONS_PATH.write_text(json.dumps(options, indent=2, default=float))
    # Step 17: today's close calls scored on your league goals (display,
    # and the rule's own pick when fpl.goals.USE_GOAL_AWARE is on)
    try:
        solver = {"xp": xp_map, "raw": raw_weekly_xp, "discount": discount, "free": effective_free,
                  "ft_value": rules.get("ft_value", 0.0), "info": pool_info, "next_gw": next_gw}
        calls = goal_calls.build(full_plans, option_plans, solver, rival_plans, league_history, team_id)
    except Exception as e:  # noqa: BLE001 -- a display extra must never stop the pipeline
        calls = {"available": False, "reason": f"goal chances not computed: {e}"}
    goal_calls.OUTPUT_PATH.write_text(json.dumps(calls, indent=2, default=goal_calls._np_default))
    if calls.get("available"):
        for label, c in calls["captain"].items():
            print(f"  goal calls ({calls['goals_text']}), {label}: captain {c['verdict']}: {c.get('text') or c['why']}")
    else:
        print(f"  goal calls: {calls.get('reason')}")
    print(f"  wrote {OPTIONS_PATH.name} ({len(options.get('captains', []))} captain, {len(options.get('transfers', []))} transfer options)")
    # The hold plan's full squad/transfer tables (same shape as
    # optimise_squad/optimise_transfers), so fpl.publish can switch the
    # one recommendation to holding if its decision rule says so.
    held = full_plans[HOLD_LABEL]
    build_squad_plan_table(held, pool_info, weeks, raw_weekly_xp).to_parquet(HOLD_SQUAD_PATH, index=False)
    build_transfers_table(held, pool_info, 0.0, False).to_parquet(HOLD_TRANSFERS_PATH, index=False)
    print(f"  wrote {HOLD_SQUAD_PATH.name}, {HOLD_TRANSFERS_PATH.name}")
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
