"""Step 11 -- the post-gameweek scoring run and feedback loop.

Runs the day after each gameweek, once scores lock (FPL finalises bonus
points and clears "provisional" flags a few hours after the last match of
the round -- in practice always by 9am UK the next day, which is when the
scheduled workflow fires). For the gameweek that just finished it:

1. Finds the pre-deadline prediction snapshot ``fpl.xp.log_live_predictions``
   wrote for that gameweek (``data/predictions/gw{N}.parquet``, frozen once
   the deadline passes; older whole-horizon, timestamped files from before
   that scheme are still read as a fallback), and the actual results from
   the live API's ``event/{gw}/live/`` endpoint -- every rostered player,
   every gameweek, regardless of whether he played, with FPL's own
   per-category point breakdown (``explain``).
2. Joins the two on ``player_id`` and computes the residual (actual -
   predicted), split by component -- minutes, attacking returns (goals +
   assists), clean sheets/conceded, bonus, defensive contribution, saves,
   cards -- using exactly the FPL scoring categories the live endpoint
   reports, so no scoring rule is re-implemented by hand here.
3. Reads the matching ``data/manual_snapshots/<gw>.json`` retrospective
   archive (if present) to record each rival's actual GW score, captain,
   and chip used.
4. Scores any pending entries in ``data/manual/override_log.json`` (Part
   B.3: "when I go against the site's recommendation") now that the
   outcome is known.
5. Writes ``data/accuracy/gw<N>.json`` (this gameweek's full detail) and
   appends one row to the cumulative ``data/accuracy/accuracy_log.csv``
   (so the site's accuracy chart never has to re-derive history from
   every individual gameweek file).

Snapshots logged before per-component breakdown columns existed degrade
gracefully to an overall MAE/bias only -- see
``fpl.xp.PREDICTION_BREAKDOWN_COLUMNS``.

Run with `uv run python -m fpl.scoring [gw]` (``gw`` optional -- defaults
to the latest finished gameweek per bootstrap-static).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.fetch import RAW_DIR, TABLES_DIR, get
from fpl.league_snapshot import SnapshotError, match_players, validate_gw_totals

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = ROOT / "data" / "predictions"
ACCURACY_DIR = ROOT / "data" / "accuracy"
MANUAL_SNAPSHOTS_DIR = ROOT / "data" / "manual_snapshots"
OVERRIDE_LOG_PATH = ROOT / "data" / "manual" / "override_log.json"
ACCURACY_LOG_CSV = ACCURACY_DIR / "accuracy_log.csv"

# predicted-column -> live API `explain` identifier(s) it should be checked
# against. The xp model doesn't predict own_goals/penalties_missed at all
# (no corresponding column in fpl.xp.build_xp), so any actual points from
# those categories are entirely attributed to "unmodeled" below -- a
# visible, honest error source rather than silently folded into another
# bucket.
COMPONENT_BUCKETS: dict[str, list[str]] = {
    "appearance_pts": ["minutes"],
    "expected_goal_pts": ["goals_scored"],
    "expected_assist_pts": ["assists"],
    "clean_sheet_pts": ["clean_sheets"],
    "conceded_pts": ["goals_conceded"],
    "defcon_pts": ["defensive_contribution"],
    "saves_pts": ["saves"],
    "bonus_pts": ["bonus"],
    "card_pts": ["yellow_cards", "red_cards"],
}
UNMODELED_IDENTIFIERS = ["own_goals", "penalties_missed", "penalties_saved"]


def _day_dir() -> Path:
    d = RAW_DIR / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# gameweek resolution
# ---------------------------------------------------------------------------


def latest_finished_gameweek(bootstrap: dict[str, Any]) -> int:
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    if not finished:
        raise RuntimeError("no finished gameweek yet -- nothing to score")
    return max(finished)


def deadline_time(bootstrap: dict[str, Any], gw: int) -> pd.Timestamp:
    for event in bootstrap["events"]:
        if event["id"] == gw:
            return pd.Timestamp(event["deadline_time"])
    raise RuntimeError(f"gameweek {gw} not found in bootstrap-static events")


# ---------------------------------------------------------------------------
# 1. the prediction snapshot to score
# ---------------------------------------------------------------------------


_LEGACY_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{6}")


def _find_legacy_prediction_snapshot(gw: int, deadline: pd.Timestamp) -> Path | None:
    """The old scheme: the latest ``data/predictions/<timestamp>.parquet``
    logged strictly before ``gw``'s deadline that actually has rows for it
    -- each snapshot logged a whole horizon, not just the next gameweek, so
    an older file can still be the right one if a newer one's horizon
    hadn't reached this gw yet. Kept as a fallback for gameweeks scored
    from before fpl.xp switched to one file per gameweek, and reused by
    fpl.migrate_prediction_snapshots to pick the file to migrate."""
    if not PREDICTIONS_DIR.exists():
        return None
    candidates = sorted(PREDICTIONS_DIR.glob("*.parquet"))
    best: Path | None = None
    best_logged_at: pd.Timestamp | None = None
    for path in candidates:
        # Filenames are "YYYY-MM-DDTHHMMSS"; parse explicitly rather than
        # relying on pandas' guesswork on a non-standard compact time.
        stem = path.stem
        if not _LEGACY_TIMESTAMP_RE.fullmatch(stem):
            continue  # skip the new gw{N}.parquet files (and anything else)
        logged_at = pd.Timestamp(
            f"{stem[:10]}T{stem[11:13]}:{stem[13:15]}:{stem[15:17]}", tz="UTC"
        )
        if logged_at >= deadline:
            continue
        df = pd.read_parquet(path, columns=["event"])
        if gw not in set(df["event"].unique()):
            continue
        if best_logged_at is None or logged_at > best_logged_at:
            best, best_logged_at = path, logged_at
    return best


def find_prediction_snapshot(gw: int, deadline: pd.Timestamp) -> Path | None:
    """The frozen pre-deadline snapshot for ``gw``: prefers the current
    ``data/predictions/gw{N}.parquet`` scheme (one file per gameweek,
    overwritten every run until the deadline passes, then left alone),
    falling back to the older whole-horizon, timestamped-file scheme for
    gameweeks that were only ever logged before that change."""
    if not PREDICTIONS_DIR.exists():
        return None

    gw_path = PREDICTIONS_DIR / f"gw{gw}.parquet"
    if gw_path.exists():
        df = pd.read_parquet(gw_path, columns=["logged_at", "event"])
        if gw in set(df["event"].unique()):
            logged_at = pd.Timestamp(df["logged_at"].iloc[0])
            if logged_at < deadline:
                return gw_path
            # Clock skew / a retried run landed at-or-after the deadline --
            # don't trust it; fall through in case an older, genuinely
            # pre-deadline legacy file still exists for this gameweek.

    return _find_legacy_prediction_snapshot(gw, deadline)


def load_prediction_snapshot(path: Path, gw: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["event"] == gw].copy()


# ---------------------------------------------------------------------------
# 2. actual results for the gameweek, from the live API
# ---------------------------------------------------------------------------


def fetch_live_results(gw: int) -> pd.DataFrame:
    """One row per player who has a live-API ``elements`` entry for ``gw``,
    with actual total points, minutes, and actual points per scoring
    category (summed across fixtures, for players with a blank/double
    gameweek)."""
    live = get(f"event/{gw}/live/", _day_dir())
    if live is None:
        raise RuntimeError(f"event/{gw}/live/ returned nothing -- has this gameweek been played yet?")

    rows = []
    for element in live["elements"]:
        stats = element["stats"]
        actual_by_identifier: dict[str, float] = {}
        for fixture_explain in element.get("explain", []):
            for item in fixture_explain["stats"]:
                actual_by_identifier[item["identifier"]] = (
                    actual_by_identifier.get(item["identifier"], 0) + item["points"]
                )
        row = {
            "player_id": element["id"],
            "actual_total_points": stats["total_points"],
            "actual_minutes": stats["minutes"],
        }
        for bucket, identifiers in COMPONENT_BUCKETS.items():
            row[f"actual_{bucket}"] = sum(actual_by_identifier.get(i, 0) for i in identifiers)
        row["actual_unmodeled_pts"] = sum(actual_by_identifier.get(i, 0) for i in UNMODELED_IDENTIFIERS)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. residuals, overall and by component
# ---------------------------------------------------------------------------


def score_predictions(gw: int, bootstrap: dict[str, Any] | None = None) -> dict[str, Any]:
    bootstrap = bootstrap or get("bootstrap-static/", _day_dir())
    deadline = deadline_time(bootstrap, gw)
    snapshot_path = find_prediction_snapshot(gw, deadline)
    if snapshot_path is None:
        return {
            "available": False,
            "reason": f"no prediction snapshot found in data/predictions/ logged before GW{gw}'s "
            f"deadline ({deadline.isoformat()}) -- can't score a prediction that wasn't saved.",
        }

    predicted = load_prediction_snapshot(snapshot_path, gw)
    actual = fetch_live_results(gw)
    merged = predicted.merge(actual, on="player_id", how="left")
    missing_actuals = merged["actual_total_points"].isna().sum()
    merged["actual_total_points"] = merged["actual_total_points"].fillna(0)
    merged["actual_minutes"] = merged["actual_minutes"].fillna(0)
    merged["residual"] = merged["actual_total_points"] - merged["xp"]

    has_breakdown = "appearance_pts" in predicted.columns and "p_play" in predicted.columns
    by_component: list[dict[str, Any]] | None = None
    if has_breakdown:
        rows = []
        for predicted_col in COMPONENT_BUCKETS:
            actual_col = f"actual_{predicted_col}"
            merged[actual_col] = merged[actual_col].fillna(0)
            resid = merged[actual_col] - merged[predicted_col]
            rows.append(
                {
                    "component": predicted_col.replace("_pts", ""),
                    "mae": float(resid.abs().mean()),
                    "bias": float(resid.mean()),
                }
            )
        # the one actual component the model never predicts at all
        merged["actual_unmodeled_pts"] = merged["actual_unmodeled_pts"].fillna(0)
        rows.append(
            {
                "component": "unmodeled (own goals / penalty misses / saves)",
                "mae": float(merged["actual_unmodeled_pts"].abs().mean()),
                "bias": float(merged["actual_unmodeled_pts"].mean()),
            }
        )
        by_component = sorted(rows, key=lambda r: r["mae"], reverse=True)

    worst = merged.reindex(merged["residual"].abs().sort_values(ascending=False).index).head(15)
    experiment = score_ep_next_experiment(merged, gw)

    return {
        "available": True,
        "gameweek": gw,
        "snapshot_path": str(snapshot_path.relative_to(ROOT)),
        # gw{N}.parquet files carry their real time in the logged_at column;
        # legacy files only in their (timestamp) filename.
        "snapshot_logged_at": (
            pd.Timestamp(predicted["logged_at"].iloc[0]).isoformat()
            if "logged_at" in predicted.columns and not predicted.empty
            else snapshot_path.stem
        ),
        "n_players": int(len(merged)),
        "missing_actuals": int(missing_actuals),
        "has_component_breakdown": has_breakdown,
        "mae": float(merged["residual"].abs().mean()),
        "bias": float(merged["residual"].mean()),
        "by_component": by_component,
        "worst_predictions": records(
            worst[["web_name", "position", "team_short", "xp", "actual_total_points", "residual"]]
        ),
        "ep_next_experiment": experiment,
        "_merged": merged,  # kept for the CSV writer below; stripped before JSON export
    }


# ---------------------------------------------------------------------------
# 3b. live ep_next experiment (measurement only -- nothing here feeds the model)
# ---------------------------------------------------------------------------

EXPERIMENT_MODELS = {"xp": "our xP", "ep_next": "FPL ep_next", "blend": "50/50 blend"}
EXPERIMENT_METRICS = ["mae", "starter_spearman", "top10", "top20", "captain_pts"]


def score_ep_next_experiment(merged: pd.DataFrame, gw: int) -> dict[str, Any]:
    """Our xP vs FPL's pre-deadline ep_next vs a 50/50 blend, on the players
    that have both, using the same metric definitions as fpl.backtest:
    MAE, Spearman on predicted likely starters (p_start > 0.5, averaged
    across positions), top-10/20 hit rate per position, and the actual
    points of each model's single top-ranked player (captain points,
    un-doubled).

    ``ep_next`` is only present in snapshots logged after this experiment
    started (see ``fpl.xp.EXPERIMENT_COLUMNS``) -- older gameweeks report
    ``available: False`` rather than being silently scored on nothing."""
    from fpl.backtest import captain_accuracy, spearman_by_position, topn_hit_rate

    if "ep_next" not in merged.columns or merged["ep_next"].notna().sum() == 0:
        return {"available": False, "reason": "this gameweek's snapshot has no ep_next column (logged before the experiment started)."}

    if "p_start" not in merged.columns:
        merged = merged.assign(p_start=np.nan)
    # Snapshot rows are per fixture; ep_next and actual points are per
    # player for the whole gameweek -- collapse to one row per player.
    per_player = merged.groupby("player_id", as_index=False).agg(
        position=("position", "first"),
        xp=("xp", "sum"),
        ep_next=("ep_next", "first"),
        p_start=("p_start", "max"),
        total_points=("actual_total_points", "first"),
    )
    per_player = per_player.dropna(subset=["ep_next"])
    per_player["blend"] = 0.5 * per_player["xp"] + 0.5 * per_player["ep_next"]
    per_player["event"] = gw
    starters = per_player[per_player["p_start"] > 0.5]

    models = {}
    for col in EXPERIMENT_MODELS:
        starter_corr = spearman_by_position(starters, col)
        metrics = {
            "mae": (per_player[col] - per_player["total_points"]).abs().mean(),
            "starter_spearman": np.mean(list(starter_corr.values())) if starter_corr else None,
            "top10": topn_hit_rate(per_player, col, 10),
            "top20": topn_hit_rate(per_player, col, 20),
            "captain_pts": captain_accuracy(per_player, col),
        }
        # None, not NaN, for anything undefined -- the gw JSON is written
        # with allow_nan=False.
        models[col] = {k: _to_native(float(v)) if v is not None else None for k, v in metrics.items()}
    return {
        "available": True,
        "n_players": int(len(per_player)),
        "n_likely_starters": int(len(starters)),
        "models": models,
    }


# ---------------------------------------------------------------------------
# 4. rival league results for the gameweek (from the retrospective archive)
# ---------------------------------------------------------------------------


def score_league_results(gw: int) -> dict[str, Any]:
    path = MANUAL_SNAPSHOTS_DIR / f"{gw}.json"
    if not path.exists():
        return {
            "available": False,
            "reason": f"no data/manual_snapshots/{gw}.json -- log this gameweek's standings/squads "
            "there (see data/manual/README.md) to get rival GW scores, captains and chips.",
        }
    snapshot = json.loads(path.read_text())
    validate_gw_totals(snapshot, skip_me=False)  # raise loudly if the transcription is wrong

    rows = []
    for s in snapshot["standings"]:
        team = s["team"]
        squad = snapshot["squads"][team]
        captain = next((p for p in squad["xi"] if p.get("role") == "C"), None)
        if captain is None:
            # nominal captain was benched (didn't play) -- the vice took over
            captain = next((p for p in squad["xi"] if p.get("role") == "V"), None)
        rows.append(
            {
                "rank": s["rank"],
                "team": team,
                "manager": s["manager"],
                "gw_points": s["gw"],
                "total_points": s["total"],
                "is_me": bool(s.get("is_me")),
                "captain": captain["name"] if captain else None,
                "captain_pts": captain["pts"] if captain else None,
                "chip": squad.get("chip"),
                # FPL doesn't show transfer-hit cost anywhere this project can
                # see without that rival's own team_id (private league, no
                # league_id configured) -- left for a future manual field.
                "hits": None,
            }
        )
    return {"available": True, "gameweek": gw, "teams": records(pd.DataFrame(rows).sort_values("rank"))}


# ---------------------------------------------------------------------------
# 5. override log -- was going against the site's recommendation right?
# ---------------------------------------------------------------------------


def load_override_log() -> list[dict[str, Any]]:
    if not OVERRIDE_LOG_PATH.exists():
        return []
    return json.loads(OVERRIDE_LOG_PATH.read_text())


def score_override_log(gw: int, live_results: pd.DataFrame | None) -> dict[str, Any]:
    """Resolves every pending entry for ``gw`` in ``data/manual/
    override_log.json`` (``gameweek`` set, ``outcome`` still null) now that
    actual points are known, and rewrites the file with the outcome filled
    in -- so it accumulates a permanent track record rather than needing
    rescoring by hand."""
    entries = load_override_log()
    if not entries:
        return {"available": False, "reason": "data/manual/override_log.json is empty or missing."}

    by_player = (
        live_results.set_index("player_id")["actual_total_points"]
        if live_results is not None
        else pd.Series(dtype=float)
    )
    players = pd.read_parquet(TABLES_DIR / "players.parquet") if (TABLES_DIR / "players.parquet").exists() else None
    name_to_id = players.set_index("web_name")["id"] if players is not None else pd.Series(dtype=int)

    changed = False
    scored_this_gw = []
    for entry in entries:
        if entry.get("gameweek") != gw or entry.get("outcome") is not None:
            continue
        site_player = entry.get("site_recommendation")
        chosen_player = entry.get("what_you_did")

        def points_for(name: str | None) -> float | None:
            if not name or name not in name_to_id.index:
                return None
            pid = name_to_id.loc[name]
            pid = pid.iloc[0] if isinstance(pid, pd.Series) else pid
            return float(by_player.get(pid)) if pid in by_player.index else None

        site_pts = points_for(site_player)
        chosen_pts = points_for(chosen_player)
        if site_pts is None or chosen_pts is None:
            outcome = "unresolved (player name didn't match a live player -- check spelling)"
        elif chosen_pts > site_pts:
            outcome = "override was right"
        elif chosen_pts < site_pts:
            outcome = "override was wrong"
        else:
            outcome = "tied"
        entry["site_points"] = site_pts
        entry["chosen_points"] = chosen_pts
        entry["outcome"] = outcome
        changed = True
        scored_this_gw.append(entry)

    if changed:
        OVERRIDE_LOG_PATH.write_text(json.dumps(entries, indent=2))
    return {"available": True, "scored_this_gw": scored_this_gw, "all_entries": entries}


# ---------------------------------------------------------------------------
# JSON-safety (same convention as fpl.publish.records/_to_native)
# ---------------------------------------------------------------------------


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# 6. write the accuracy log
# ---------------------------------------------------------------------------


def write_accuracy_outputs(gw: int, prediction_result: dict[str, Any], league_result: dict[str, Any], override_result: dict[str, Any]) -> Path:
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "gameweek": gw,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "predictions": {k: v for k, v in prediction_result.items() if k != "_merged"},
        "league_results": league_result,
        "override_log": override_result,
    }
    gw_path = ACCURACY_DIR / f"gw{gw}.json"
    gw_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    if prediction_result.get("available"):
        row = {
            "gameweek": gw,
            "generated_at": payload["generated_at"],
            "n_players": prediction_result["n_players"],
            "mae": prediction_result["mae"],
            "bias": prediction_result["bias"],
            "has_component_breakdown": prediction_result["has_component_breakdown"],
        }
        if prediction_result.get("by_component"):
            for c in prediction_result["by_component"]:
                row[f"mae_{c['component']}"] = c["mae"]
                row[f"bias_{c['component']}"] = c["bias"]
        experiment = prediction_result.get("ep_next_experiment") or {}
        if experiment.get("available"):
            row["exp_n_players"] = experiment["n_players"]
            for model, metrics in experiment["models"].items():
                for metric in EXPERIMENT_METRICS:
                    row[f"exp_{metric}_{model}"] = metrics[metric]
        log_df = pd.DataFrame([row])
        if ACCURACY_LOG_CSV.exists():
            existing = pd.read_csv(ACCURACY_LOG_CSV)
            existing = existing[existing["gameweek"] != gw]  # idempotent re-run: replace, don't duplicate
            log_df = pd.concat([existing, log_df], ignore_index=True)
        log_df.sort_values("gameweek").to_csv(ACCURACY_LOG_CSV, index=False)

    return gw_path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    gw_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    bootstrap = get("bootstrap-static/", _day_dir())
    if bootstrap is None:
        print("bootstrap-static/ returned nothing -- check network/API status.")
        return
    gw = gw_arg or latest_finished_gameweek(bootstrap)
    print(f"Scoring gameweek {gw}...")

    prediction_result = score_predictions(gw, bootstrap)
    if prediction_result["available"]:
        print(
            f"  predictions: {prediction_result['n_players']} players, "
            f"MAE={prediction_result['mae']:.2f}, bias={prediction_result['bias']:+.2f} "
            f"(snapshot: {prediction_result['snapshot_path']})"
        )
        if prediction_result["by_component"]:
            worst = prediction_result["by_component"][0]
            print(f"  biggest error source: {worst['component']} (MAE={worst['mae']:.2f})")
        experiment = prediction_result["ep_next_experiment"]
        if experiment["available"]:
            print(f"  ep_next experiment ({experiment['n_players']} players):")
            print(f"    {'model':<13}" + "".join(f"{m:>18}" for m in EXPERIMENT_METRICS))
            for model, label in EXPERIMENT_MODELS.items():
                cells = "".join(
                    f"{'n/a' if v is None else f'{v:.3f}':>18}"
                    for v in (experiment["models"][model][m] for m in EXPERIMENT_METRICS)
                )
                print(f"    {label:<13}{cells}")
        else:
            print(f"  ep_next experiment: {experiment['reason']}")
    else:
        print(f"  predictions: {prediction_result['reason']}")

    try:
        league_result = score_league_results(gw)
    except SnapshotError as exc:
        league_result = {"available": False, "reason": str(exc)}
    if league_result["available"]:
        print(f"  league results: {len(league_result['teams'])} teams recorded")
    else:
        print(f"  league results: {league_result['reason']}")

    live_results_df = prediction_result.get("_merged") if prediction_result["available"] else None
    override_result = score_override_log(gw, live_results_df)
    if override_result["available"] and override_result["scored_this_gw"]:
        print(f"  override log: scored {len(override_result['scored_this_gw'])} entr(y/ies) for this gw")

    path = write_accuracy_outputs(gw, prediction_result, league_result, override_result)
    print(f"Checkpoint: wrote {path}")
    score_consultant(gw, bootstrap)


def score_consultant(latest_gw: int, bootstrap: dict[str, Any]) -> None:
    """"You vs the consultant" (fpl/consultant.py): score every frozen
    recommendation whose gameweek has finished -- new ones, plus the latest
    finished gameweek again (bonus points can still settle) -- against what
    you actually did, all from the FPL API. Never fails the scoring run."""
    import traceback

    from fpl.consultant import CSV_PATH, RECOMMENDATIONS_DIR, score_gameweek
    from fpl.fetch import load_config
    from fpl.league import resolve_team_id

    try:
        managers_path = TABLES_DIR / "league_managers.parquet"
        managers = pd.read_parquet(managers_path) if managers_path.exists() else pd.DataFrame()
        team_id = resolve_team_id(load_config(), managers)
        if team_id is None:
            print("  consultant vs you: no team_id -- skipped")
            return
        done = set(pd.read_csv(CSV_PATH)["gameweek"]) if CSV_PATH.exists() else set()
        frozen = sorted(int(p.stem[2:]) for p in RECOMMENDATIONS_DIR.glob("gw*.json")) if RECOMMENDATIONS_DIR.exists() else []
        todo = [g for g in frozen if g <= latest_gw and (g not in done or g == latest_gw)]
        if not todo:
            print(f"  consultant vs you: nothing to score yet (frozen: {frozen or 'none'})")
        for g in todo:
            r = score_gameweek(g, team_id, lambda path: get(path, _day_dir()), bootstrap)
            if r["available"]:
                print(f"  consultant vs you GW{g}: consultant {r['consultant_points']}, you {r['my_points']} "
                      f"(followed transfers: {r['followed_transfers']}, captain: {r['followed_captain']})")
            else:
                print(f"  consultant vs you GW{g}: {r['reason']}")
    except Exception:
        print("  consultant vs you: scoring failed -- continuing:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
