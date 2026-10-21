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
  that match, "elo_xg_blend" when it fell back to Elo + xG) so the site can
  show *which* model is driving each number -- the "odds vs Elo" view.
* ``model_accuracy`` -- Step 7's ``fpl/backtest_results_log.csv``: our
  model's MAE/rank-correlation against FPL's own ``ep_next`` and two naive
  baselines, one row per backtest run.
* ``top_xp`` -- Step 6's ``xp_by_gameweek.parquet`` alone, with no
  team/league config needed at all, so the site always has *something* to
  show even before ``config.json`` is filled in.

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

    if recommended and not moves.empty:
        pairs = ", ".join(
            f"{r.player_out or '?'} -> {r.player_in or '?'}" for r in moves.itertuples()
        )
        headline = f"Transfer: {pairs} ({int(wk0_moves['hits_taken'].sum())} hit(s))"
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
    }


def build_xi(optimise_squad: pd.DataFrame) -> dict[str, Any]:
    if optimise_squad.empty:
        return {"available": False, "reason": "optimise_squad.parquet not found -- run fpl.optimise first."}
    gw0 = int(optimise_squad["gameweek"].min())
    squad0 = optimise_squad[optimise_squad["gameweek"] == gw0].sort_values(
        ["role", "position"], key=lambda s: s if s.name != "role" else s.map({"captain": 0, "xi": 1, "bench": 2})
    )
    return {"available": True, "gameweek": gw0, "players": records(squad0)}


def build_transfer_plan(optimise_transfers: pd.DataFrame) -> dict[str, Any]:
    if optimise_transfers.empty:
        return {"available": False, "reason": "optimise_transfers.parquet not found -- run fpl.optimise first."}
    return {"available": True, "weeks": records(optimise_transfers)}


def build_captain_board(optimise_captain_board: pd.DataFrame) -> dict[str, Any]:
    if optimise_captain_board.empty:
        return {"available": False, "reason": "optimise_captain_board.parquet not found -- run fpl.optimise first."}
    return {"available": True, "players": records(optimise_captain_board)}


def build_chips(optimise_chip_plan: pd.DataFrame, league_chips_grid: pd.DataFrame) -> dict[str, Any]:
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
    return chips


def build_threats(league_squad_overlap: pd.DataFrame) -> dict[str, Any]:
    if league_squad_overlap.empty:
        return {
            "available": False,
            "reason": "league_squad_overlap.parquet not found -- set league_id and run fpl.league first.",
        }
    return {"available": True, "rivals": records(league_squad_overlap)}


def build_differentials(
    league_swings: pd.DataFrame,
    xp_by_gameweek: pd.DataFrame,
    config: dict[str, Any],
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
) -> dict[str, Any]:
    """Low league-ownership players (below ``DIFFERENTIAL_OWNERSHIP_MAX_PCT``
    of rivals) you don't currently own, ranked by next-gameweek xP -- the
    picks whose haul swings the table almost regardless of their EO.
    ``config.json``'s ``differential_exclude_managers`` (a list of manager
    names) also excludes anyone those specific rivals already own -- a
    low aggregate-ownership player owned by whoever you're actually
    chasing/protecting against isn't really a differential against *them*,
    even if the rest of the league hasn't caught on."""
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

    candidates = league_swings.merge(xp_col, on="player_id", how="left")
    candidates = candidates[
        (candidates["league_ownership_pct"] < DIFFERENTIAL_OWNERSHIP_MAX_PCT)
        & (~candidates["player_id"].isin(my_squad_ids))
        & (~candidates["player_id"].isin(excluded_owned_ids))
    ]
    candidates = candidates.sort_values("next_gw_xp", ascending=False).head(DIFFERENTIAL_LIMIT)
    return {"available": True, "next_gameweek": next_gw, "players": records(candidates)}


def build_eo_threats(league_swings: pd.DataFrame) -> dict[str, Any]:
    """Step D: players with the highest league EO that you *don't* own --
    these are the ones who hurt you most if they haul, ranked by the
    rank-weighted EO (close rivals dominate) when available."""
    if league_swings.empty:
        return {
            "available": False,
            "reason": "league_swings.parquet not found -- set league_id "
            "(or add data/manual/league_snapshot.json) and run fpl.league first.",
        }
    eo_col = "weighted_league_eo" if "weighted_league_eo" in league_swings.columns else "league_eo"
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
    return {"available": True, "runs": runs, "latest": runs[-1]}


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
    league_picks = load_optional("league_picks")
    league_history = load_optional("league_history")
    league_managers = load_optional("league_managers")
    league_swings = load_optional("league_swings")
    league_chips_grid = load_optional("league_chips_grid")
    league_squad_overlap = load_optional("league_squad_overlap")
    league_strategy = load_optional("league_strategy")
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

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "config": {
            "team_id": team_id,
            "league_id": config.get("league_id"),
            "strategy_mode": (
                league_strategy["mode"].iloc[0] if not league_strategy.empty else config.get("strategy")
            ),
        },
        "verdict": build_verdict(optimise_transfers, optimise_squad),
        "xi": build_xi(optimise_squad),
        "captain_board": build_captain_board(optimise_captain_board),
        "captaincy_risk": build_captaincy_risk(optimise_captaincy_risk),
        "transfer_plan": build_transfer_plan(optimise_transfers),
        "chips": build_chips(optimise_chip_plan, league_chips_grid),
        "threats": build_threats(league_squad_overlap),
        "eo_threats": build_eo_threats(league_swings),
        "differentials": build_differentials(
            league_swings, xp_by_gameweek, config, league_picks, league_history, league_managers
        ),
        "league_table": build_league_table(league_history, league_managers, config),
        "ownership_heatmap": build_ownership_heatmap(league_picks, league_managers, league_swings),
        "vs_leader": build_vs_leader(league_history, league_managers, league_squad_overlap, config),
        "fixture_ticker": build_fixture_ticker(features),
        "model_accuracy": build_model_accuracy(backtest_log),
        "top_xp": build_top_xp(xp_by_gameweek),
    }

    path = write_json(payload)
    available = [k for k, v in payload.items() if isinstance(v, dict) and v.get("available")]
    missing = [k for k, v in payload.items() if isinstance(v, dict) and v.get("available") is False]
    print(f"Checkpoint: wrote {path}")
    print(f"  panels available: {', '.join(available) or 'none'}")
    if missing:
        print(f"  panels missing (step not run / config not set): {', '.join(missing)}")


if __name__ == "__main__":
    main()
