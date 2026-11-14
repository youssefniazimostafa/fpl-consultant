"""Publish -- exports ``site/data/latest.json``, the one file the static
``site/index.html`` fetches to render every panel. This is the last stage
of ``fpl.pipeline`` (fetch -> external data -> features -> model -> league
-> optimise -> **publish**), so it only *reads* tables other steps already
wrote; it never calls the FPL API or re-derives a number another module
owns.

**Panels, and which step's tables they come from:**

* ``verdict`` / ``xi`` / ``captain_board`` / ``captaincy_risk`` /
  ``transfer_plan`` / ``chips.yours`` -- Step 9's ``optimise_*.parquet``.
  Needs ``team_id`` set in ``config.json`` *or* resolvable from the manual
  snapshot (Step F; see below), and for the transfer/chip panels,
  ``league_picks``/``league_history`` from Step 8 too.
* ``threats`` / ``eo_threats`` / ``differentials`` / ``league_table`` /
  ``ownership_heatmap`` / ``vs_leader`` / ``chips.rivals`` -- Step 8's
  ``league_*.parquet``. Needs ``league_id`` set *or* a manual snapshot at
  ``data/manual/league_snapshot.json`` (Step F: ``fpl.league`` builds and
  persists these tables from whichever source is configured, so this
  module never needs to know which one it was).
* ``fixture_ticker`` -- Step 5/6's ``features.parquet``: one row per team
  per upcoming fixture, with ``source`` ("odds" when The Odds API covered
  that match, else whichever of "elo+xg" / "elo_only" / "xg_only" /
  "league_avg" actually fed the fallback lambda -- truthful even when Elo
  or Understat themselves fell back to a stale last-good table) so the
  site can show *which* model is driving each number.
* ``source_health`` -- ``data/tables/source_health.json`` (``fpl.source_health``):
  per-source (fpl_api / odds / elo / understat) status ("ok" / "stale" /
  "failed", plus "fallback_derived" for elo -- see fpl.elo), last
  successful fetch time, row count, and a short error --
  so the site can show a warning banner when a fallback is in use.
* ``model_accuracy`` -- Step 7's ``fpl/backtest_results_log.csv``: our
  model's MAE/rank-correlation against FPL's own ``ep_next`` and two naive
  baselines, one row per backtest run.
* ``top_xp`` -- Step 6's ``xp_by_gameweek.parquet`` alone, with no
  team/league config needed at all, so the site always has *something* to
  show even before ``config.json`` is filled in.
* ``live_accuracy`` / ``news_flags`` -- Step 11's ``data/accuracy/*`` and
  ``p_start_overrides_applied.json``: this season's real track record
  (distinct from the one-off historical ``model_accuracy`` backtest above)
  and which press-conference overrides are new since the last run.
* ``simulation`` -- Step 11 Part B.4 extension 4's
  ``data/tables/simulation_results.json`` (``fpl.simulation``): P(finish
  1st) this gameweek per candidate plan.

``team_id`` in the exported ``config`` is always the *resolved* value
(``fpl.league.resolve_team_id``) -- ``config.json``'s own ``team_id`` in
live-API mode, or the manual snapshot's "you" row in manual mode -- so the
site never has to know which source produced it.

Every panel that needs config which isn't set, or a table a step hasn't
written yet, is simply ``null`` in the JSON with a short ``reason`` --
"add panels as each step lands", not "crash until every step has run".

Run with `uv run python -m fpl.publish`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.check_consistency import check as consistency_errors
from fpl.consultant import build_section as build_consultant_section
from fpl.consultant import freeze_recommendation
from fpl.decision import ROLL, build_decision, choose_plan, rival_captaincy
from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.optimise import resolve_my_squad

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
SITE_DATA_DIR = ROOT / "site" / "data"
BACKTEST_LOG_PATH = ROOT / "fpl" / "backtest_results_log.csv"

# A player owned by fewer than this share of your league's rivals counts
# as a differential -- the whole point being he's *not* template, so a
# haul swings the table in your favour almost regardless of his EO.
DIFFERENTIAL_OWNERSHIP_MAX_PCT = 20.0
TOP_XP_LIMIT = 15
DIFFERENTIAL_LIMIT = 10
OWNERSHIP_HEATMAP_LIMIT = 15
EO_THREATS_LIMIT = 10


# ---------------------------------------------------------------------------
# loading (every table is optional -- a step that hasn't run yet, or a
# config field that isn't set, just means that panel is missing, not a
# crash)
# ---------------------------------------------------------------------------


def load_optional(name: str) -> pd.DataFrame:
    path = TABLES_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_backtest_log() -> pd.DataFrame:
    return pd.read_csv(BACKTEST_LOG_PATH) if BACKTEST_LOG_PATH.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def build_top_xp(xp_by_gameweek: pd.DataFrame) -> dict[str, Any]:
    if xp_by_gameweek.empty:
        return {"available": False, "reason": "xp_by_gameweek.parquet not found -- run fpl.xp first."}
    weeks = sorted(int(c[2:-3]) for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp"))
    next_gw = weeks[0] if weeks else None
    cols = ["player_id", "web_name", "position", "team_short", "now_cost", "xp_total"]
    if next_gw is not None:
        cols.insert(-1, f"gw{next_gw}_xp")
    top = xp_by_gameweek[cols].sort_values("xp_total", ascending=False).head(TOP_XP_LIMIT)
    if next_gw is not None:
        top = top.rename(columns={f"gw{next_gw}_xp": "next_gw_xp"})
    return {"available": True, "next_gameweek": next_gw, "horizon_weeks": weeks, "players": records(top)}


def build_verdict(
    optimise_transfers: pd.DataFrame, optimise_squad: pd.DataFrame
) -> dict[str, Any]:
    if optimise_transfers.empty or optimise_squad.empty:
        return {
            "available": False,
            "reason": "optimise_*.parquet not found -- set team_id in config.json and "
            "run fpl.optimise first.",
        }
    gw0 = int(optimise_transfers["gameweek"].min())
    wk0_moves = optimise_transfers[optimise_transfers["gameweek"] == gw0]
    recommended = bool(wk0_moves["recommended"].iloc[0])
    moves = wk0_moves.dropna(subset=["player_in", "player_out"], how="all")

    squad0 = optimise_squad[optimise_squad["gameweek"] == gw0]
    captain_row = squad0[squad0["role"] == "captain"]
    captain = records(captain_row)[0] if not captain_row.empty else None

    hits_taken = int(wk0_moves["hits_taken"].sum())
    hit_reason = wk0_moves["hit_reason"].iloc[0] if "hit_reason" in wk0_moves.columns else None
    hit_reason = hit_reason if isinstance(hit_reason, str) and hit_reason else None
    if not (recommended and not moves.empty):
        # holding: a held week's "hits" is only multi-week free-transfer
        # bookkeeping in the solver (see fpl.simulation), never a real cost
        hits_taken, hit_reason = 0, None

    if recommended and not moves.empty:
        pairs = ", ".join(
            f"{r.player_out or '?'} -> {r.player_in or '?'}" for r in moves.itertuples()
        )
        headline = f"Transfer: {pairs} ({hits_taken} hit(s))"
    else:
        headline = "Hold -- roll the free transfer."

    return {
        "available": True,
        "gameweek": gw0,
        "headline": headline,
        "transfer_recommended": recommended,
        "horizon_gain_xp": None if wk0_moves["horizon_gain_xp"].isna().all()
        else float(wk0_moves["horizon_gain_xp"].iloc[0]),
        "captain": captain,
        "hits_taken": hits_taken,
        "hit_reason": hit_reason,
    }


def build_xi(optimise_squad: pd.DataFrame) -> dict[str, Any]:
    if optimise_squad.empty:
        return {"available": False, "reason": "optimise_squad.parquet not found -- run fpl.optimise first."}
    gw0 = int(optimise_squad["gameweek"].min())
    squad0 = optimise_squad[optimise_squad["gameweek"] == gw0].sort_values(
        ["role", "position"], key=lambda s: s if s.name != "role" else s.map({"captain": 0, "xi": 1, "bench": 2})
    )
    return {"available": True, "gameweek": gw0, "players": records(squad0)}


# Step 12: FPL's own "unavailable" statuses -- a transfer out of one of
# these (or a player flagged at a flat 0% chance of playing) is forced,
# not a model preference.
TRANSFER_URGENCY_FORCED_STATUSES = {"i", "s", "u"}


def _player_urgency(web_name: str | None, players: pd.DataFrame) -> dict[str, Any]:
    """Step 12: classify why an outgoing player is being transferred --
    ``forced`` (injury/suspension/unavailable, or a flat 0% chance of
    playing -- sell today), ``flagged`` (doubtful, a live
    ``chance_of_playing_next_round`` below 100% -- a risk, not certain),
    or ``model_edge`` (no injury/news signal at all, a pure
    points-optimisation swap -- nice to have, not urgent). Looked up by
    ``web_name`` since ``optimise_transfers.parquet`` only stores display
    names, not ids."""
    empty = {"urgency": "model_edge", "status": None, "chance_of_playing_next_round": None, "news": None}
    if not web_name or players.empty:
        return empty
    match = players[players["web_name"] == web_name]
    if match.empty:
        return empty
    row = match.iloc[0]
    status = row.get("status")
    chance = row.get("chance_of_playing_next_round")
    chance = None if pd.isna(chance) else float(chance)
    if status in TRANSFER_URGENCY_FORCED_STATUSES or chance == 0:
        urgency = "forced"
    elif status == "d" or (chance is not None and chance < 100):
        urgency = "flagged"
    else:
        urgency = "model_edge"
    news = row.get("news")
    return {
        "urgency": urgency,
        "status": status,
        "chance_of_playing_next_round": chance,
        "news": news if isinstance(news, str) and news else None,
    }


def build_transfer_plan(optimise_transfers: pd.DataFrame, players: pd.DataFrame) -> dict[str, Any]:
    """Step 12: each week's recommended transfer(s), annotated with an
    urgency tier per outgoing player (see ``_player_urgency``) -- a
    ``forced`` transfer should read as must-do even to someone skimming
    the dashboard, while a ``model_edge`` swap is take-it-or-leave-it."""
    if optimise_transfers.empty:
        return {"available": False, "reason": "optimise_transfers.parquet not found -- run fpl.optimise first."}
    weeks = records(optimise_transfers)
    for row in weeks:
        row.update(_player_urgency(row.get("player_out"), players))
    return {"available": True, "weeks": weeks}


# Step 12: any of these statuses (or news text) means the player card on
# the "Recommended Team" pitch should show the red flagged treatment --
# deliberately broader than ``TRANSFER_URGENCY_FORCED_STATUSES`` above,
# since a doubtful ('d') player is worth a visual warning on the pitch
# even when he isn't "forced out" grounds for a hit.
PITCH_FLAGGED_STATUSES = {"d", "i", "s", "u"}

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
# Step 12: FPL's own bench labelling convention -- the GK is always
# unlabelled-by-number, the three outfield subs are numbered in playing-
# priority order, each tagged with their real position.
BENCH_LABELS = {0: "GKP", 1: "1. {pos}", 2: "2. {pos}", 3: "3. {pos}"}


def _next_fixture_by_player(features: pd.DataFrame, gw0: int) -> dict[int, dict[str, Any]]:
    if features.empty:
        return {}
    upcoming = features[features["event"] >= gw0].sort_values(["player_id", "event", "kickoff_time"])
    first = upcoming.drop_duplicates(subset="player_id", keep="first")
    out: dict[int, dict[str, Any]] = {}
    for row in first.itertuples():
        out[int(row.player_id)] = {
            "event": int(row.event),
            "opponent_short": row.opponent_short,
            "is_home": bool(row.is_home),
        }
    return out


def _recommended_team_card(
    player_id: int,
    players_by_id: dict[int, dict[str, Any]],
    fixtures_by_player: dict[int, dict[str, Any]],
    xp_next_by_player: dict[int, float] | None = None,
) -> dict[str, Any]:
    info = players_by_id.get(player_id, {})
    status = info.get("status")
    chance = info.get("chance_of_playing_next_round")
    chance = None if pd.isna(chance) else float(chance) if chance is not None else None
    news = info.get("news")
    fixture = fixtures_by_player.get(player_id)
    now_cost = info.get("now_cost")
    projected = (xp_next_by_player or {}).get(player_id)
    return {
        "player_id": player_id,
        "web_name": info.get("web_name"),
        "team_code": None if pd.isna(info.get("team_code")) else int(info.get("team_code")),
        "team_short": info.get("team_short_name"),
        "status": status,
        "chance_of_playing_next_round": chance,
        "flagged": status in PITCH_FLAGGED_STATUSES,
        "news": news if isinstance(news, str) and news else None,
        "fixture": fixture,
        # desktop app: price in £m and projected points for the next gameweek
        "price": None if now_cost is None or pd.isna(now_cost) else round(float(now_cost) / 10, 1),
        "projected_points": None if projected is None or pd.isna(projected) else round(float(projected), 2),
    }


def _xp_next_by_player(xp_by_gameweek: pd.DataFrame, gw0: int) -> dict[int, float]:
    col = f"gw{gw0}_xp"
    if xp_by_gameweek.empty or col not in xp_by_gameweek.columns:
        return {}
    return {int(pid): float(v) for pid, v in zip(xp_by_gameweek["player_id"], xp_by_gameweek[col]) if pd.notna(v)}


def _match_player_out(name: str | None, position: str | None, players: pd.DataFrame, exclude: set[int]) -> int | None:
    """optimise_transfers names players by web_name; resolve the outgoing
    one to an id, preferring the incoming player's position (FPL transfers
    are like-for-like) and skipping anyone already in the new squad."""
    if not name or players.empty:
        return None
    hits = players[(players["web_name"] == name) & (~players["id"].isin(exclude))]
    if position is not None and "position" in hits.columns and len(hits) > 1:
        same_pos = hits[hits["position"] == position]
        hits = same_pos if not same_pos.empty else hits
    return int(hits["id"].iloc[0]) if not hits.empty else None


def build_recommended_team(
    optimise_squad: pd.DataFrame,
    optimise_transfers: pd.DataFrame,
    players: pd.DataFrame,
    features: pd.DataFrame,
    league_free_transfers: pd.DataFrame,
    team_id: int | None,
    xp_by_gameweek: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Step 12: the full 15-man squad *after* the recommended transfer(s)
    (or the current squad unchanged if the recommendation is to roll),
    shaped for the pitch-view "Recommended Team" tab -- starting XI
    grouped by position row, bench in real bench order, captain/vice
    captain, and a one-line transfer summary with any hit warning."""
    if optimise_squad.empty:
        return {"available": False, "reason": "optimise_squad.parquet not found -- run fpl.optimise first."}
    if players.empty:
        return {"available": False, "reason": "players.parquet not found -- run fpl.fetch first."}

    gw0 = int(optimise_squad["gameweek"].min())
    squad0 = optimise_squad[optimise_squad["gameweek"] == gw0].copy()
    players_by_id = players.set_index("id").to_dict("index")
    fixtures_by_player = _next_fixture_by_player(features, gw0)
    xp_next = _xp_next_by_player(xp_by_gameweek if xp_by_gameweek is not None else pd.DataFrame(), gw0)

    def card(row) -> dict[str, Any]:
        base = _recommended_team_card(int(row.player_id), players_by_id, fixtures_by_player, xp_next)
        base.update(
            {
                "position": row.position,
                "is_captain": row.role == "captain",
                "is_vice_captain": bool(row.is_vice_captain),
                "transferred_in": bool(row.transferred_in),
            }
        )
        return base

    xi_rows = squad0[squad0["role"].isin(["xi", "captain"])].copy()
    xi_rows["position_order"] = xi_rows["position"].map(POSITION_ORDER)
    xi_rows = xi_rows.sort_values(["position_order", "player_id"])
    starting_xi = [card(r) for r in xi_rows.itertuples()]

    bench_rows = squad0[squad0["role"] == "bench"].copy()
    bench_rows["bench_order"] = bench_rows["bench_order"].fillna(0).astype(int)
    bench_rows = bench_rows.sort_values("bench_order")
    bench = []
    for r in bench_rows.itertuples():
        entry = card(r)
        order = int(r.bench_order)
        label = BENCH_LABELS.get(order, "{pos}").format(pos=entry["position"])
        entry["bench_label"] = label
        bench.append(entry)

    captain = next((p for p in starting_xi if p["is_captain"]), None)
    vice_captain = next((p for p in starting_xi if p["is_vice_captain"]), None)

    # Transfer summary + hit warning, from gw0's row in optimise_transfers
    # (hit_reason/hits_taken are populated by fpl.optimise's hit-policy,
    # see ``fpl.optimise.justify_hits``).
    hits_taken = 0
    hit_reason = None
    summary = "Rolling the free transfer -- no transfer recommended this week."
    # desktop app: each recommended move as full out/in player cards
    transfers: list[dict[str, Any]] = []
    if not optimise_transfers.empty:
        wk0 = optimise_transfers[optimise_transfers["gameweek"] == gw0]
        if not wk0.empty:
            recommended = bool(wk0["recommended"].iloc[0])
            hits_taken = int(wk0["hits_taken"].sum())
            hit_reason = wk0["hit_reason"].iloc[0] if "hit_reason" in wk0.columns else None
            hit_reason = hit_reason if isinstance(hit_reason, str) and hit_reason else None
            moves = wk0.dropna(subset=["player_in", "player_out"], how="all")
            if recommended and not moves.empty:
                pairs = ", ".join(
                    f"OUT: {r.player_out or '?'} -> IN: {r.player_in or '?'}" for r in moves.itertuples()
                )
                summary = pairs
                squad_ids = {int(x) for x in squad0["player_id"]}
                incoming = {p["web_name"]: p for p in starting_xi + bench if p["transferred_in"]}
                for r in moves.itertuples():
                    in_card = incoming.get(r.player_in)
                    out_id = _match_player_out(
                        r.player_out, in_card["position"] if in_card else None, players, squad_ids
                    )
                    out_card = None
                    if out_id is not None:
                        out_card = _recommended_team_card(out_id, players_by_id, fixtures_by_player, xp_next)
                        out_card["position"] = players_by_id.get(out_id, {}).get("position")
                    transfers.append(
                        {
                            "player_out": out_card or {"web_name": r.player_out},
                            "player_in": in_card or {"web_name": r.player_in},
                        }
                    )
            if not transfers:
                # holding (see build_verdict): no real hit this gameweek
                hits_taken, hit_reason = 0, None

    free_transfers_available = None
    if not league_free_transfers.empty and team_id is not None:
        match = league_free_transfers[league_free_transfers["manager_id"] == team_id]
        if not match.empty:
            free_transfers_available = int(match["free_transfers"].iloc[0])

    return {
        "available": True,
        "gameweek": gw0,
        "summary": summary,
        "transfers": transfers,
        "free_transfers_available": free_transfers_available,
        "hits_taken": hits_taken,
        "hit_reason": hit_reason,
        "captain": captain,
        "vice_captain": vice_captain,
        "starting_xi": starting_xi,
        "bench": bench,
    }


def decision_captain_tables(
    decision: dict[str, Any],
    xp_by_gameweek: pd.DataFrame,
    players: pd.DataFrame,
    league_swings: pd.DataFrame,
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    team_id: int | None,
    mode: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The site's captain board and captaincy-risk tables, rebuilt for the
    decision's own XI and captain (same functions fpl.optimise uses), so
    they always highlight the same captain as the app -- including when the
    decision switched to the hold plan. ``is_optimiser_captain`` here marks
    the decision's captain."""
    from fpl.optimise import build_captain_board as optimise_captain_board
    from fpl.optimise import build_captaincy_risk as optimise_captaincy_risk

    if not decision.get("available") or not decision.get("captain") or players.empty:
        return None
    gw = int(decision["gameweek"])
    col = f"gw{gw}_xp"
    if xp_by_gameweek.empty or col not in xp_by_gameweek.columns:
        return None
    xi_ids = [int(p["player_id"]) for p in decision["starting_xi"]]
    xp = xp_by_gameweek.set_index("player_id")[col].to_dict()
    raw = {(p, gw): float(xp.get(p, 0.0)) for p in xi_ids}
    pool_info = players.set_index("id")[["web_name", "position"]].to_dict("index")
    week0 = {"gameweek": gw, "xi_ids": set(xi_ids), "captain_id": int(decision["captain"]["player_id"])}
    raw_eo: dict[int, float] = {}
    eo: dict[int, float] = {}
    if not league_swings.empty:
        raw_eo = league_swings.set_index("player_id")["league_eo"].to_dict()
        if "weighted_league_eo" in league_swings.columns:
            eff = league_swings["weighted_league_eo"].fillna(league_swings["league_eo"])
            eo = pd.Series(eff.values, index=league_swings["player_id"]).to_dict()
        else:
            eo = dict(raw_eo)
    board = optimise_captain_board(week0, raw, pool_info, mode or "balanced", eo, raw_eo)
    risk = pd.DataFrame()
    if team_id is not None and not league_picks.empty and not league_history.empty:
        _, _, current_gw = resolve_my_squad(league_picks, league_history, team_id)
        risk = optimise_captaincy_risk(week0, raw, pool_info, league_picks, league_history, league_managers, team_id, current_gw)
    return board, risk


def build_captain_board(optimise_captain_board: pd.DataFrame) -> dict[str, Any]:
    if optimise_captain_board.empty:
        return {"available": False, "reason": "optimise_captain_board.parquet not found -- run fpl.optimise first."}
    return {"available": True, "players": records(optimise_captain_board)}


def build_chips(
    optimise_chip_plan: pd.DataFrame,
    league_chips_grid: pd.DataFrame,
    season_simulation: dict[str, Any],
    team_id: int | None,
) -> dict[str, Any]:
    """Step 12: ``season_context`` gives the chip planner its "why now"
    grounding -- your current P(finish 1st by GW``target_gameweek``) from
    ``fpl.simulation.simulate_season_standings``, alongside the weeks
    left to use it in -- so a chip recommendation isn't shown purely as
    "biggest estimated_gain this week" with no sense of the bigger
    picture it's meant to serve."""
    chips: dict[str, Any] = {}
    chips["yours"] = (
        {"available": True, "plan": records(optimise_chip_plan)}
        if not optimise_chip_plan.empty
        else {"available": False, "reason": "optimise_chip_plan.parquet not found -- run fpl.optimise first."}
    )
    chips["rivals"] = (
        {"available": True, "grid": records(league_chips_grid)}
        if not league_chips_grid.empty
        else {"available": False, "reason": "league_chips_grid.parquet not found -- set league_id and run fpl.league first."}
    )
    season_context = None
    if season_simulation.get("available") and team_id is not None:
        mine = next((r for r in season_simulation["standings"] if r["manager_id"] == team_id), None)
        if mine is not None:
            season_context = {
                "target_gameweek": season_simulation["target_gameweek"],
                "weeks_remaining": season_simulation["weeks_remaining"],
                "settled": season_simulation["settled"],
                "p_finish_1st_by_target": mine["p_finish_1st_by_gw19"],
            }
    chips["season_context"] = season_context
    return chips


def build_threats(league_squad_overlap: pd.DataFrame, league_rival_window: pd.DataFrame) -> dict[str, Any]:
    """Step 12: restricted to rivals inside your rival window when that
    table is available and non-empty for this gameweek -- squad overlap
    against someone who can't realistically catch or be caught isn't a
    threat worth showing. Falls back to every rival in the league
    otherwise."""
    if league_squad_overlap.empty:
        return {
            "available": False,
            "reason": "league_squad_overlap.parquet not found -- set league_id and run fpl.league first.",
        }
    restricted = league_squad_overlap
    restricted_to_window = False
    if not league_rival_window.empty:
        window_ids = set(league_rival_window.loc[league_rival_window["in_window"], "manager_id"])
        candidate = league_squad_overlap[league_squad_overlap["manager_id"].isin(window_ids)]
        if not candidate.empty:
            restricted = candidate
            restricted_to_window = True
    return {"available": True, "restricted_to_rival_window": restricted_to_window, "rivals": records(restricted)}


def build_differentials(
    league_swings: pd.DataFrame,
    xp_by_gameweek: pd.DataFrame,
    config: dict[str, Any],
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
) -> dict[str, Any]:
    """Low-ownership players you don't currently own, ranked by
    next-gameweek xP -- the picks whose haul swings the table almost
    regardless of their EO. Step 12: ownership is judged against your
    *rival window* (``rival_window_ownership_pct`` -- see
    ``fpl.league.build_rival_window_swings_table``) when that column is
    available, since a player nobody in your realistic title fight owns
    is a differential that actually matters, even if he's template
    further down the league; falls back to whole-league ownership
    otherwise (e.g. manual-snapshot mode, or a rival window with nobody
    in it). ``config.json``'s ``differential_exclude_managers`` (a list of
    manager names) also excludes anyone those specific rivals already
    own -- a low-ownership player owned by whoever you're actually
    chasing/protecting against isn't really a differential against
    *them*, even if the rest of the league hasn't caught on."""
    if league_swings.empty or xp_by_gameweek.empty:
        return {
            "available": False,
            "reason": "league_swings.parquet or xp_by_gameweek.parquet not found -- "
            "run fpl.xp and fpl.league first.",
        }
    team_id = resolve_team_id(config, league_managers)
    my_squad_ids: set[int] = set()
    if team_id is not None and not league_picks.empty and not league_history.empty:
        try:
            my_squad_ids, _, _ = resolve_my_squad(league_picks, league_history, team_id)
        except ValueError:
            pass

    excluded_owned_ids: set[int] = set()
    exclude_names = config.get("differential_exclude_managers") or []
    if exclude_names and not league_picks.empty and not league_managers.empty:
        excluded_manager_ids = league_managers[league_managers["player_name"].isin(exclude_names)]["manager_id"]
        gw = int(league_picks["gameweek"].max())
        excluded_owned_ids = set(
            league_picks[
                (league_picks["gameweek"] == gw) & (league_picks["manager_id"].isin(excluded_manager_ids))
            ]["player_id"]
        )

    weeks = sorted(int(c[2:-3]) for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp"))
    next_gw = weeks[0] if weeks else None
    xp_col = xp_by_gameweek[["player_id", f"gw{next_gw}_xp"]].rename(
        columns={f"gw{next_gw}_xp": "next_gw_xp"}
    ) if next_gw is not None else xp_by_gameweek[["player_id"]].assign(next_gw_xp=0.0)

    ownership_col = (
        "rival_window_ownership_pct" if "rival_window_ownership_pct" in league_swings.columns
        and league_swings["rival_window_ownership_pct"].notna().any()
        else "league_ownership_pct"
    )

    candidates = league_swings.merge(xp_col, on="player_id", how="left")
    candidates = candidates[
        (candidates[ownership_col] < DIFFERENTIAL_OWNERSHIP_MAX_PCT)
        & (~candidates["player_id"].isin(my_squad_ids))
        & (~candidates["player_id"].isin(excluded_owned_ids))
    ]
    candidates = candidates.sort_values("next_gw_xp", ascending=False).head(DIFFERENTIAL_LIMIT)
    return {
        "available": True,
        "next_gameweek": next_gw,
        "ownership_column": ownership_col,
        "players": records(candidates),
    }


def build_eo_threats(league_swings: pd.DataFrame) -> dict[str, Any]:
    """Players with the highest EO that you *don't* own -- these are the
    ones who hurt you most if they haul. Step 12: prefers
    ``rival_window_eo`` (EO computed from just the managers inside your
    rival window -- see ``fpl.league.build_rival_window_swings_table``)
    over the whole-league rank-weighted EO, since a threat from a manager
    who can't realistically catch or be caught by you isn't really a
    threat; falls back to the rank-weighted EO, then the plain league-wide
    one, when the window column isn't available."""
    if league_swings.empty:
        return {
            "available": False,
            "reason": "league_swings.parquet not found -- set league_id "
            "(or add data/manual/league_snapshot.json) and run fpl.league first.",
        }
    if "rival_window_eo" in league_swings.columns and league_swings["rival_window_eo"].notna().any():
        eo_col = "rival_window_eo"
    elif "weighted_league_eo" in league_swings.columns:
        eo_col = "weighted_league_eo"
    else:
        eo_col = "league_eo"
    not_owned = league_swings[league_swings["your_multiplier"] == 0]
    top = not_owned.sort_values(eo_col, ascending=False).head(EO_THREATS_LIMIT)
    return {"available": True, "eo_column": eo_col, "players": records(top)}


def build_captaincy_risk(optimise_captaincy_risk: pd.DataFrame) -> dict[str, Any]:
    if optimise_captaincy_risk.empty:
        return {
            "available": False,
            "reason": "optimise_captaincy_risk.parquet not found -- run fpl.optimise first.",
        }
    rival_cols = [c for c in optimise_captaincy_risk.columns if c.startswith("swing_vs_")]
    return {
        "available": True,
        "rivals": [c[len("swing_vs_"):] for c in rival_cols],
        "players": records(optimise_captaincy_risk),
    }


def build_league_table(
    league_history: pd.DataFrame, league_managers: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    if league_history.empty or league_managers.empty:
        return {
            "available": False,
            "reason": "league_history.parquet/league_managers.parquet not found -- set league_id "
            "(or add data/manual/league_snapshot.json) and run fpl.league first.",
        }
    team_id = resolve_team_id(config, league_managers)
    gw = int(league_history["gameweek"].max())
    latest = (
        league_history[league_history["gameweek"] == gw]
        .merge(league_managers, on="manager_id")
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )
    latest["league_rank"] = latest.index + 1
    leader_points = int(latest["total_points"].iloc[0])
    latest["gap_to_leader"] = latest["total_points"] - leader_points
    latest["is_you"] = latest["manager_id"] == team_id
    cols = ["league_rank", "manager_id", "entry_name", "player_name", "total_points", "gap_to_leader", "is_you"]
    return {"available": True, "gameweek": gw, "standings": records(latest[cols])}


def build_ownership_heatmap(
    league_picks: pd.DataFrame, league_managers: pd.DataFrame, league_swings: pd.DataFrame
) -> dict[str, Any]:
    """The top ``OWNERSHIP_HEATMAP_LIMIT`` players by league ownership,
    each manager's multiplier on them (0 bench, 1 start, 2 captain, 3
    triple captain) -- the site colours this in as a heatmap, which is
    "who's template in your league" (Step 8's first metric) at a glance."""
    if league_picks.empty or league_managers.empty or league_swings.empty:
        return {
            "available": False,
            "reason": "league_picks.parquet/league_managers.parquet/league_swings.parquet not found -- "
            "set league_id (or add data/manual/league_snapshot.json) and run fpl.league first.",
        }
    gw = int(league_picks["gameweek"].max())
    gw_picks = league_picks[league_picks["gameweek"] == gw]
    top = league_swings.sort_values("league_ownership_pct", ascending=False).head(OWNERSHIP_HEATMAP_LIMIT)
    top_ids = top["player_id"].tolist()
    pivot = (
        gw_picks[gw_picks["player_id"].isin(top_ids)]
        .pivot_table(index="player_id", columns="manager_id", values="multiplier", fill_value=0)
    )
    manager_ids = sorted(league_managers["manager_id"].tolist())
    manager_labels = league_managers.set_index("manager_id")["entry_name"].to_dict()

    rows = []
    for row in top.itertuples():
        entry = {
            "player_id": int(row.player_id),
            "web_name": row.web_name,
            "position": row.position,
            "league_ownership_pct": float(row.league_ownership_pct),
        }
        for m in manager_ids:
            entry[f"m_{m}"] = int(pivot.loc[row.player_id, m]) if row.player_id in pivot.index and m in pivot.columns else 0
        rows.append(entry)
    return {
        "available": True,
        "gameweek": gw,
        "managers": [{"manager_id": m, "entry_name": manager_labels.get(m, f"manager {m}")} for m in manager_ids],
        "players": rows,
    }


def build_vs_leader(
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    league_squad_overlap: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """How far off top you are, and -- if you're not already there -- what
    you share/don't share with whoever is, since closing a points gap
    mostly comes down to differing from exactly that squad."""
    if league_history.empty or league_managers.empty:
        return {
            "available": False,
            "reason": "league_history.parquet/league_managers.parquet not found -- set league_id "
            "(or add data/manual/league_snapshot.json) and run fpl.league first.",
        }
    team_id = resolve_team_id(config, league_managers)
    if team_id is None:
        return {
            "available": False,
            "reason": "no team_id resolvable -- set team_id in config.json, or set league_id and "
            "add data/manual/league_snapshot.json.",
        }
    gw = int(league_history["gameweek"].max())
    latest = league_history[league_history["gameweek"] == gw].merge(league_managers, on="manager_id")
    leader = latest.sort_values("total_points", ascending=False).iloc[0]
    you = latest[latest["manager_id"] == team_id]
    if you.empty:
        return {"available": False, "reason": f"manager_id {team_id} not found in league_history for gameweek {gw}."}
    you = you.iloc[0]

    if int(leader["manager_id"]) == team_id:
        return {"available": True, "gameweek": gw, "is_leader": True, "gap_to_leader": 0, "leader": None, "overlap": None}

    overlap = None
    if not league_squad_overlap.empty:
        row = league_squad_overlap[league_squad_overlap["manager_id"] == int(leader["manager_id"])]
        if not row.empty:
            overlap = records(row)[0]
    return {
        "available": True,
        "gameweek": gw,
        "is_leader": False,
        "gap_to_leader": int(leader["total_points"] - you["total_points"]),
        "leader": {
            "manager_id": int(leader["manager_id"]),
            "entry_name": leader["entry_name"],
            "player_name": leader["player_name"],
            "total_points": int(leader["total_points"]),
        },
        "overlap": overlap,
    }


SOURCE_HEALTH_PATH = TABLES_DIR / "source_health.json"


def build_source_health() -> dict[str, Any]:
    """Per-source status from ``fpl.source_health`` (fpl_api / odds / elo /
    understat), plus a `warnings` list of plain-English messages for any
    source that isn't "ok", so the site can show a banner without having
    to know each source's quirks itself."""
    if not SOURCE_HEALTH_PATH.exists():
        return {
            "available": False,
            "reason": "data/tables/source_health.json not found -- run fpl.fetch / fpl.odds / fpl.elo / "
            "fpl.understat first.",
        }
    records_ = json.loads(SOURCE_HEALTH_PATH.read_text())

    warnings = []
    for rec in records_:
        if rec["status"] == "ok":
            continue
        source = rec["source"]
        if rec["status"] == "fallback_derived":
            n = rec.get("matches_applied", 0)
            msg = (f"ClubElo unavailable: using its ratings from {rec.get('seed_date', 'an unknown date')}, "
                   f"updated with {n} result{'s' if n != 1 else ''} since.")
            if rec.get("unseeded_teams"):
                msg += (f" Not in that ClubElo snapshot, so started at its lowest rating: "
                        f"{', '.join(rec['unseeded_teams'])}.")
            warnings.append(msg)
        elif rec["status"] == "stale":
            age = "an unknown age"
            if rec.get("last_success_at"):
                age_hours = (pd.Timestamp.now("UTC") - pd.Timestamp(rec["last_success_at"])).total_seconds() / 3600
                age = f"{age_hours:.0f}h old" if age_hours < 48 else f"{age_hours / 24:.1f} days old"
            warnings.append(
                f"{source}: using a fallback table ({age}) -- {rec['error']}"
            )
        else:
            warnings.append(f"{source}: failed, no fallback available -- {rec['error']}")

    return {
        "available": True,
        "sources": records_,
        "all_ok": all(rec["status"] == "ok" for rec in records_),
        "warnings": warnings,
    }


def build_fixture_ticker(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"available": False, "reason": "features.parquet not found -- run fpl.features first."}
    cols = [
        "event", "team_id", "team_short", "opponent_id", "opponent_short", "is_home",
        "lambda_for", "lambda_against", "clean_sheet_prob", "source",
    ]
    ticker = features[cols].drop_duplicates(subset=["event", "team_id"]).sort_values(["event", "team_short"])
    return {"available": True, "fixtures": records(ticker)}


def build_model_accuracy(backtest_log: pd.DataFrame) -> dict[str, Any]:
    if backtest_log.empty:
        return {
            "available": False,
            "reason": "fpl/backtest_results_log.csv not found -- run fpl.backtest first "
            "(a one-off validation step, not part of the scheduled pipeline).",
        }
    runs = records(backtest_log.sort_values("run_at"))
    # The headline panel compares every model on the same gameweeks, so it
    # uses the latest matched-scope run (FPL-xP-valid gameweeks only). Runs
    # logged before the 'scope' column existed mixed samples, so they're
    # only used if no matched run exists yet.
    matched = [r for r in runs if r.get("scope") == "fpl_xp_valid_gws"]
    return {"available": True, "runs": runs, "latest": matched[-1] if matched else runs[-1]}


ACCURACY_DIR = ROOT / "data" / "accuracy"


def build_live_accuracy() -> dict[str, Any]:
    """Step 11's real, this-season track record (distinct from
    ``model_accuracy`` above, which is the one-off historical backtest):
    cumulative MAE/bias per gameweek from ``fpl.scoring``'s
    ``data/accuracy/accuracy_log.csv``, plus the most recent gameweek's
    full component breakdown, league results and override-log outcomes
    from its matching ``data/accuracy/gw<N>.json``."""
    log_path = ACCURACY_DIR / "accuracy_log.csv"
    if not log_path.exists():
        return {
            "available": False,
            "reason": "data/accuracy/accuracy_log.csv not found -- run fpl.scoring after a "
            "gameweek finishes (the scheduled post-gameweek scoring run) to populate this panel.",
        }
    cumulative = pd.read_csv(log_path).sort_values("gameweek")

    latest_gw = int(cumulative["gameweek"].max())
    detail_path = ACCURACY_DIR / f"gw{latest_gw}.json"
    detail = json.loads(detail_path.read_text()) if detail_path.exists() else None

    return {
        "available": True,
        "latest_gameweek": latest_gw,
        "cumulative": records(cumulative),
        "latest_detail": detail,
    }


def build_simulation(decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Part B.4 extension 4's Monte Carlo results (``fpl.simulation``,
    ``data/tables/simulation_results.json``): P(finish 1st) this gameweek
    per candidate plan, the win-probability pick and the average-xP pick
    shown side by side."""
    path = TABLES_DIR / "simulation_results.json"
    if not path.exists():
        return {
            "available": False,
            "reason": "data/tables/simulation_results.json not found -- run `uv run python -m "
            "fpl.simulation` (needs a resolved team_id/league data, see fpl.optimise).",
        }
    rows = json.loads(path.read_text())
    if not rows:
        return {"available": True, "plans": [], "best": None, "average_xp": None}

    best = rows[0]  # already sorted by p_finish_1st desc when written
    avg_candidates = [r for r in rows if r.get("is_average_xp_captain")]
    average_xp = max(avg_candidates, key=lambda r: r["p_finish_1st"]) if avg_candidates else best
    # The one recommendation (fpl/decision.py): the row with the decision's
    # plan (transfer or hold) and captain. The site highlights this row,
    # not the noisiest-best one, so it can't contradict the app.
    recommended = None
    if decision and decision.get("available") and decision.get("captain"):
        from fpl.simulation import HOLD_LABEL, TRANSFER_LABEL

        label = TRANSFER_LABEL if decision["action"] == "transfer" else HOLD_LABEL
        cap = decision["captain"]["web_name"]
        recommended = next((r for r in rows if r["scenario"] == label and r["captain"] == cap), None)
    for r in rows:
        r["is_recommended"] = r is recommended
    return {"available": True, "plans": rows, "best": best, "average_xp": average_xp, "recommended": recommended}


STRATEGY_EXPLANATIONS = {
    "chasing": (
        "You're trailing the leader by more than a typical week's swing per gameweek "
        "remaining -- lean into upside (differentials, riskier captaincy) to close the gap."
    ),
    "protecting": (
        "You're ahead of second place by more than a typical week's swing per gameweek "
        "remaining -- lean into safety (template picks, fewer hits) to protect the lead."
    ),
    "balanced": (
        "Your gap to first isn't large relative to the gameweeks left -- no strong lean "
        "either way; play the straight xP picks."
    ),
}


def build_strategy(league_strategy: pd.DataFrame) -> dict[str, Any]:
    """Step 12's "strategy dial" panel: chasing/protecting/balanced (see
    ``fpl.league.determine_strategy``), plus a plain-English explanation
    of *why* -- the dial is meaningless to a reader who hasn't memorised
    the rule it's built on."""
    if league_strategy.empty:
        return {
            "available": False,
            "reason": "league_strategy.parquet not found -- set league_id and run fpl.league first.",
        }
    row = {k: _to_native(v) for k, v in league_strategy.iloc[0].to_dict().items()}
    return {"available": True, **row, "explanation": STRATEGY_EXPLANATIONS.get(row.get("mode"), "")}


def build_free_transfers(league_free_transfers: pd.DataFrame) -> dict[str, Any]:
    """Step 12: every manager's free-transfer count right now, computed
    by walking their whole transfer history (``fpl.league.build_free_transfers_table``)
    instead of anyone having to type it in -- replaces the old
    hand-maintained ``config.json`` fields."""
    if league_free_transfers.empty:
        return {
            "available": False,
            "reason": "league_free_transfers.parquet not found -- set league_id and run fpl.league first.",
        }
    return {"available": True, "managers": records(league_free_transfers)}


def build_rival_transfer_forecast(
    league_rival_transfer_forecast: pd.DataFrame, league_rival_habits: pd.DataFrame
) -> dict[str, Any]:
    """Step 12: ``fpl.rival_transfers``' next-gameweek predicted in/out per
    rival (from their own haul-chasing/flagged-selling habits), alongside
    the habit rates themselves so the site can show *why* a prediction was
    made, not just the prediction."""
    if league_rival_transfer_forecast.empty:
        return {
            "available": False,
            "reason": "league_rival_transfer_forecast.parquet not found -- set league_id and "
            "run fpl.rival_transfers first.",
        }
    merged = league_rival_transfer_forecast
    if not league_rival_habits.empty:
        habit_cols = ["manager_id", "gameweeks_played", "chase_rate", "hit_rate", "sells_flagged_rate"]
        merged = league_rival_transfer_forecast.merge(
            league_rival_habits[habit_cols], on="manager_id", how="left"
        )
    return {"available": True, "forecasts": records(merged)}


def build_season_simulation() -> dict[str, Any]:
    """Step 12: ``fpl.simulation.simulate_season_standings``'s season-long
    P(finish 1st by GW``target_gameweek``) per manager -- a coarser,
    team-level projection from each manager's own weekly-score
    distribution (not a full squad/xP re-simulation; see
    ``fpl.simulation``'s module docstring for the stated limitation).
    Already shaped with an ``available`` key by the writer, so this just
    passes the file through."""
    path = TABLES_DIR / "season_simulation.json"
    if not path.exists():
        return {
            "available": False,
            "reason": "data/tables/season_simulation.json not found -- run `uv run python -m "
            "fpl.simulation` (needs a resolved team_id/league data, see fpl.optimise).",
        }
    return json.loads(path.read_text())


def build_status(
    gameweeks: pd.DataFrame, league_free_transfers: pd.DataFrame, team_id: int | None, generated_at: str
) -> dict[str, Any]:
    """Step 12's status strip: current/next gameweek, the next deadline
    (UTC ISO -- the site converts to US Eastern client-side with
    ``Intl.DateTimeFormat``, which handles the EST/EDT switch
    automatically instead of us hand-rolling a fixed offset), this run's
    ``generated_at``, and your free-transfer count right now -- the
    handful of facts that answer "is this dashboard showing me something
    current" at a glance. Staleness itself isn't computed here: if a
    pipeline run fails, ``site/data/latest.json`` simply isn't
    regenerated (see ``fpl.pipeline``'s module docstring), so the site
    compares this ``generated_at`` against the viewer's own clock to flag
    it, rather than this module ever claiming to know "now" is stale."""
    status: dict[str, Any] = {"generated_at": generated_at}
    if not gameweeks.empty:
        current = gameweeks[gameweeks["is_current"]]
        nxt = gameweeks[gameweeks["is_next"]]
        status["current_gameweek"] = int(current["gameweek"].iloc[0]) if not current.empty else None
        status["next_gameweek"] = int(nxt["gameweek"].iloc[0]) if not nxt.empty else None
        status["next_deadline_utc"] = nxt["deadline_time"].iloc[0] if not nxt.empty else None
    else:
        status["current_gameweek"] = None
        status["next_gameweek"] = None
        status["next_deadline_utc"] = None
    status["your_free_transfers"] = None
    if team_id is not None and not league_free_transfers.empty:
        row = league_free_transfers[league_free_transfers["manager_id"] == team_id]
        if not row.empty:
            status["your_free_transfers"] = int(row["free_transfers"].iloc[0])
    return status


def build_news_flags() -> dict[str, Any]:
    """Part B.2's "flags any player whose news changed since the last
    run": every ``p_start_overrides`` entry this run actually matched and
    applied (``fpl.features.apply_p_start_overrides``), newest first, with
    whether its ``date`` is more recent than the *previous* run's logged
    prediction snapshot -- i.e. genuinely new news, not a stale override
    left in config from a prior week."""
    path = TABLES_DIR / "p_start_overrides_applied.json"
    if not path.exists():
        return {"available": False, "reason": "no overrides applied -- add p_start_overrides to config.json."}
    payload = json.loads(path.read_text())
    applied = [o for o in payload.get("overrides", []) if o.get("matched")]
    if not applied:
        return {"available": True, "flags": [], "generated_at": payload.get("generated_at")}

    predictions_dir = ROOT / "data" / "predictions"
    previous_runs = sorted(predictions_dir.glob("*.parquet")) if predictions_dir.exists() else []
    previous_logged_at = (
        pd.Timestamp(
            previous_runs[-2].stem[:10] + "T" + previous_runs[-2].stem[11:13] + ":" + previous_runs[-2].stem[13:15],
            tz="UTC",
        )
        if len(previous_runs) >= 2
        else None
    )
    flags = []
    for o in applied:
        try:
            is_new = previous_logged_at is None or pd.Timestamp(o["date"], tz="UTC") > previous_logged_at
        except (KeyError, ValueError):
            is_new = True
        flags.append({**o, "new_since_last_run": bool(is_new)})
    return {"available": True, "flags": flags, "generated_at": payload.get("generated_at")}


# ---------------------------------------------------------------------------
# JSON-safety
# ---------------------------------------------------------------------------


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``df.to_dict("records")``, but with every numpy/pandas scalar (and
    NaN) turned into something ``json.dumps`` can actually write."""
    return [{k: _to_native(v) for k, v in row.items()} for row in df.to_dict("records")]


def _to_native(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(payload: dict[str, Any]) -> Path:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = SITE_DATA_DIR / "latest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()

    xp_by_gameweek = load_optional("xp_by_gameweek")
    features = load_optional("features")
    gameweeks = load_optional("gameweeks")
    players = load_optional("players")
    league_picks = load_optional("league_picks")
    league_history = load_optional("league_history")
    league_managers = load_optional("league_managers")
    league_swings = load_optional("league_swings")
    league_chips_grid = load_optional("league_chips_grid")
    league_squad_overlap = load_optional("league_squad_overlap")
    league_strategy = load_optional("league_strategy")
    league_free_transfers = load_optional("league_free_transfers")
    league_rival_window = load_optional("league_rival_window")
    league_rival_transfer_forecast = load_optional("league_rival_transfer_forecast")
    league_rival_habits = load_optional("league_rival_habits")
    optimise_squad = load_optional("optimise_squad")
    optimise_transfers = load_optional("optimise_transfers")
    optimise_chip_plan = load_optional("optimise_chip_plan")
    optimise_captain_board = load_optional("optimise_captain_board")
    optimise_captaincy_risk = load_optional("optimise_captaincy_risk")
    backtest_log = load_backtest_log()

    # Step F: team_id may come straight from config.json (live-API mode) or
    # be derived from the manual snapshot's is_me flag -- resolve it once
    # here so the panels below, and the page's header, see "you" either way.
    team_id = resolve_team_id(config, league_managers)
    generated_at = pd.Timestamp.now("UTC").isoformat()
    season_simulation = build_season_simulation()
    strategy_mode = league_strategy["mode"].iloc[0] if not league_strategy.empty else config.get("strategy")

    # ONE decision for the upcoming gameweek (see fpl/decision.py): choose
    # transfer vs hold with a single rule, then build every "what to do"
    # panel -- verdict, xi, recommended_team, transfer_plan and the app's
    # `decision` -- from that same chosen squad, so they can't disagree.
    sim_path = TABLES_DIR / "simulation_results.json"
    sim_rows = json.loads(sim_path.read_text()) if sim_path.exists() else []
    gw0_moves = (
        optimise_transfers[optimise_transfers["gameweek"] == optimise_transfers["gameweek"].min()]
        if not optimise_transfers.empty else optimise_transfers
    )
    optimiser_transfer = bool(not gw0_moves.empty and gw0_moves["recommended"].iloc[0])
    horizon_gain = (
        float(gw0_moves["horizon_gain_xp"].iloc[0])
        if not gw0_moves.empty and pd.notna(gw0_moves["horizon_gain_xp"].iloc[0]) else None
    )
    plan_choice = choose_plan(strategy_mode, optimiser_transfer, sim_rows)
    squad_src, transfers_src = optimise_squad, optimise_transfers
    if plan_choice["overridden"] and plan_choice["choice"] == ROLL:
        hold_squad = load_optional("simulation_hold_squad")
        hold_transfers = load_optional("simulation_hold_transfers")
        if not hold_squad.empty and not hold_transfers.empty:
            squad_src, transfers_src = hold_squad, hold_transfers
        else:  # can't switch without the hold plan's squad -- keep the optimiser's
            plan_choice = {**plan_choice, "choice": plan_choice["optimiser_choice"], "rule": "projected_points", "overridden": False}
    print(
        f"Decision: {plan_choice['choice']} (rule: {plan_choice['rule']}, mode: {plan_choice['mode']}, "
        f"optimiser: {plan_choice['optimiser_choice']}, simulation: {plan_choice['simulation_choice']}, "
        f"win chance transfer/hold: {plan_choice['win_chance_transfer']}/{plan_choice['win_chance_hold']})"
    )

    recommended_team = build_recommended_team(
        squad_src, transfers_src, players, features, league_free_transfers, team_id, xp_by_gameweek
    )
    transfer_plan = build_transfer_plan(transfers_src, players)
    chips = build_chips(optimise_chip_plan, league_chips_grid, season_simulation, team_id)
    gw_cols = sorted(int(c[2:-3]) for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp")) \
        if not xp_by_gameweek.empty else []
    grid_row = None
    if not league_chips_grid.empty and team_id is not None:
        match = league_chips_grid[league_chips_grid["manager_id"] == team_id]
        grid_row = records(match)[0] if not match.empty else None
    strategy = build_strategy(league_strategy)
    decision = build_decision(
        recommended_team,
        transfer_plan,
        plan_choice,
        horizon_gain,
        len(gw_cols) or 5,
        records(optimise_chip_plan) if not optimise_chip_plan.empty else [],
        grid_row,
        rival_captaincy(league_picks, team_id),
        _xp_next_by_player(xp_by_gameweek, recommended_team.get("gameweek", gw_cols[0] if gw_cols else 0)),
        players.set_index("id")["web_name"].to_dict() if not players.empty else {},
        strategy,
        chips.get("season_context"),
    )

    captain_board_src, captaincy_risk_src = optimise_captain_board, optimise_captaincy_risk
    rebuilt = decision_captain_tables(
        decision, xp_by_gameweek, players, league_swings, league_picks, league_history, league_managers,
        team_id, strategy_mode,
    )
    if rebuilt is not None:
        captain_board_src, captaincy_risk_src = rebuilt

    payload = {
        "generated_at": generated_at,
        "status": build_status(gameweeks, league_free_transfers, team_id, generated_at),
        "config": {
            "team_id": team_id,
            "league_id": config.get("league_id"),
            "strategy_mode": strategy_mode,
        },
        "decision": decision,
        "verdict": build_verdict(transfers_src, squad_src),
        "xi": build_xi(squad_src),
        "recommended_team": recommended_team,
        "captain_board": build_captain_board(captain_board_src),
        "captaincy_risk": build_captaincy_risk(captaincy_risk_src),
        "transfer_plan": transfer_plan,
        "chips": chips,
        "strategy": strategy,
        "free_transfers": build_free_transfers(league_free_transfers),
        "rival_transfer_forecast": build_rival_transfer_forecast(
            league_rival_transfer_forecast, league_rival_habits
        ),
        "season_simulation": season_simulation,
        "threats": build_threats(league_squad_overlap, league_rival_window),
        "eo_threats": build_eo_threats(league_swings),
        "differentials": build_differentials(
            league_swings, xp_by_gameweek, config, league_picks, league_history, league_managers
        ),
        "league_table": build_league_table(league_history, league_managers, config),
        "rival_window": (
            {"available": True, "rivals": records(league_rival_window)}
            if not league_rival_window.empty
            else {"available": False, "reason": "league_rival_window.parquet not found -- set league_id and run fpl.league first."}
        ),
        "ownership_heatmap": build_ownership_heatmap(league_picks, league_managers, league_swings),
        "vs_leader": build_vs_leader(league_history, league_managers, league_squad_overlap, config),
        "fixture_ticker": build_fixture_ticker(features),
        "source_health": build_source_health(),
        "model_accuracy": build_model_accuracy(backtest_log),
        "live_accuracy": build_live_accuracy(),
        "simulation": build_simulation(decision),
        "news_flags": build_news_flags(),
        "top_xp": build_top_xp(xp_by_gameweek),
        "consultant_vs_you": build_consultant_section(),
    }

    # Fail loudly *before* writing if the page would contradict itself
    # (fpl/check_consistency.py), so a bad latest.json is never published.
    problems = consistency_errors(payload)
    if problems:
        raise RuntimeError("decision consistency check failed:\n  " + "\n  ".join(problems))
    path = write_json(payload)
    frozen = freeze_recommendation(decision, gameweeks)
    print(f"  {'froze recommendation to ' + str(frozen.relative_to(ROOT)) if frozen else 'recommendation not frozen (deadline passed or no decision)'}")
    available = [k for k, v in payload.items() if isinstance(v, dict) and v.get("available")]
    missing = [k for k, v in payload.items() if isinstance(v, dict) and v.get("available") is False]
    print(f"Checkpoint: wrote {path}")
    print(f"  panels available: {', '.join(available) or 'none'}")
    if missing:
        print(f"  panels missing (step not run / config not set): {', '.join(missing)}")


if __name__ == "__main__":
    main()
