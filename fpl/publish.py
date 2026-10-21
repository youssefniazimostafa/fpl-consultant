"""Publish -- exports ``site/data/latest.json``, the one file the static
``site/index.html`` fetches to render every panel. This is the last stage
of ``fpl.pipeline`` (fetch -> external data -> features -> model -> league
-> optimise -> **publish**), so it only *reads* tables other steps already
wrote; it never calls the FPL API or re-derives a number another module
owns.

**Panels, and which step's tables they come from:**

* ``verdict`` / ``xi`` / ``captain_board`` / ``transfer_plan`` / ``chips.
  yours`` -- Step 9's ``optimise_*.parquet``. Needs ``team_id`` (and for
  the transfer/chip panels, ``league_id`` too) set in ``config.json``.
* ``threats`` / ``differentials`` / ``chips.rivals`` -- Step 8's
  ``league_*.parquet``. Needs ``league_id`` set.
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
) -> dict[str, Any]:
    """Low league-ownership players (below ``DIFFERENTIAL_OWNERSHIP_MAX_PCT``
    of rivals) you don't currently own, ranked by next-gameweek xP -- the
    picks whose haul swings the table almost regardless of their EO."""
    if league_swings.empty or xp_by_gameweek.empty:
        return {
            "available": False,
            "reason": "league_swings.parquet or xp_by_gameweek.parquet not found -- "
            "run fpl.xp and fpl.league first.",
        }
    team_id = config.get("team_id")
    my_squad_ids: set[int] = set()
    if team_id is not None and not league_picks.empty and not league_history.empty:
        try:
            my_squad_ids, _, _ = resolve_my_squad(league_picks, league_history, int(team_id))
        except ValueError:
            pass

    weeks = sorted(int(c[2:-3]) for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp"))
    next_gw = weeks[0] if weeks else None
    xp_col = xp_by_gameweek[["player_id", f"gw{next_gw}_xp"]].rename(
        columns={f"gw{next_gw}_xp": "next_gw_xp"}
    ) if next_gw is not None else xp_by_gameweek[["player_id"]].assign(next_gw_xp=0.0)

    candidates = league_swings.merge(xp_col, on="player_id", how="left")
    candidates = candidates[
        (candidates["league_ownership_pct"] < DIFFERENTIAL_OWNERSHIP_MAX_PCT)
        & (~candidates["player_id"].isin(my_squad_ids))
    ]
    candidates = candidates.sort_values("next_gw_xp", ascending=False).head(DIFFERENTIAL_LIMIT)
    return {"available": True, "next_gameweek": next_gw, "players": records(candidates)}


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
    league_swings = load_optional("league_swings")
    league_chips_grid = load_optional("league_chips_grid")
    league_squad_overlap = load_optional("league_squad_overlap")
    league_strategy = load_optional("league_strategy")
    optimise_squad = load_optional("optimise_squad")
    optimise_transfers = load_optional("optimise_transfers")
    optimise_chip_plan = load_optional("optimise_chip_plan")
    optimise_captain_board = load_optional("optimise_captain_board")
    backtest_log = load_backtest_log()

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "config": {
            "team_id": config.get("team_id"),
            "league_id": config.get("league_id"),
            "strategy_mode": (
                league_strategy["mode"].iloc[0] if not league_strategy.empty else config.get("strategy")
            ),
        },
        "verdict": build_verdict(optimise_transfers, optimise_squad),
        "xi": build_xi(optimise_squad),
        "captain_board": build_captain_board(optimise_captain_board),
        "transfer_plan": build_transfer_plan(optimise_transfers),
        "chips": build_chips(optimise_chip_plan, league_chips_grid),
        "threats": build_threats(league_squad_overlap),
        "differentials": build_differentials(
            league_swings, xp_by_gameweek, config, league_picks, league_history
        ),
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
