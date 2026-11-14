"""The desktop app's "Analytics" page -- one ``analytics`` section in
latest.json with the series the page charts that no other section
already carries. Everything else on the page reuses existing sections
(ownership_heatmap, consultant_vs_you, live_accuracy, season_simulation,
captain_board, decision, model_accuracy).

* league_series: your points per gameweek (after hits) vs the league
  average and the current leader, and your league position each week
  (league_history);
* team_projection: your team's frozen pre-deadline projection vs what it
  scored, for every gameweek with a frozen prediction
  (data/predictions/gw{N}.parquet x your confirmed picks);
* misses: the biggest prediction misses per scored gameweek
  (data/accuracy/gw{N}.json, written by fpl.scoring);
* fixture_difficulty: every club's next fixtures with FPL's own
  difficulty rating;
* experiment: the live ep_next comparison rows from the accuracy log,
  once any gameweek has been scored with it;
* backtest: the matched 10-gameweek and 37-gameweek backtest runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MISSES_PER_GW = 5
FIXTURE_WEEKS = 6


def _num(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)


def build_analytics(
    *,
    team_id: int | None,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    league_picks: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    next_gameweek: int | None,
    predictions_dir: Path,
    accuracy_dir: Path,
    model_accuracy: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {"available": True}

    # --- league series ---
    series: list[dict[str, Any]] = []
    if not league_history.empty and team_id is not None:
        h = league_history.copy()
        h["net"] = h["points"] - h["event_transfers_cost"].fillna(0)
        latest_gw = int(h["gameweek"].max())
        leader_id = int(h[h["gameweek"] == latest_gw].sort_values("total_points", ascending=False)["manager_id"].iloc[0])
        names = league_managers.set_index("manager_id")["entry_name"].to_dict() if not league_managers.empty else {}
        for gw, wk in h.groupby("gameweek"):
            wk = wk.sort_values("total_points", ascending=False).reset_index(drop=True)
            me = wk[wk["manager_id"] == team_id]
            if me.empty:
                continue
            my_total = int(me["total_points"].iloc[0])
            series.append({
                "gameweek": int(gw),
                "you": int(me["net"].iloc[0]),
                "league_average": round(float(wk["net"].mean()), 1),
                "leader": int(wk[wk["manager_id"] == leader_id]["net"].iloc[0]) if (wk["manager_id"] == leader_id).any() else None,
                "position": int((wk["total_points"] > my_total).sum()) + 1,
                "managers": len(wk),
            })
        out["leader_name"] = names.get(leader_id)
    out["league_series"] = series

    # --- your team: frozen projection vs actual ---
    projection: list[dict[str, Any]] = []
    if not league_picks.empty and team_id is not None and not league_history.empty:
        mine = league_picks[league_picks["manager_id"] == team_id]
        for path in sorted(predictions_dir.glob("gw*.parquet"), key=lambda p: int(p.stem[2:])):
            gw = int(path.stem[2:])
            picks = mine[mine["gameweek"] == gw]
            hist = league_history[(league_history["manager_id"] == team_id) & (league_history["gameweek"] == gw)]
            if picks.empty or hist.empty:
                continue  # not played yet
            xp = pd.read_parquet(path).groupby("player_id")["xp"].sum()
            projected = float(sum(xp.get(int(r.player_id), 0.0) * int(r.multiplier) for r in picks.itertuples()))
            projection.append({"gameweek": gw, "projected": round(projected, 1), "actual": int(hist["points"].iloc[0])})
    out["team_projection"] = projection

    # --- biggest misses per scored gameweek ---
    misses: list[dict[str, Any]] = []
    for path in sorted(accuracy_dir.glob("gw*.json"), key=lambda p: int(p.stem[2:])):
        detail = json.loads(path.read_text())
        worst = ((detail.get("predictions") or {}).get("worst_predictions")) or []
        top = sorted(worst, key=lambda w: -abs(w.get("residual") or 0))[:MISSES_PER_GW]
        if top:
            misses.append({"gameweek": int(path.stem[2:]), "players": [
                {"web_name": w["web_name"], "team_short": w.get("team_short"), "expected": round(float(w["xp"]), 1),
                 "actual": int(w["actual_total_points"]), "difference": round(float(w["residual"]), 1)} for w in top]})
    out["misses"] = misses

    # --- every club's next fixtures, FPL's difficulty ---
    difficulty: list[dict[str, Any]] = []
    if not fixtures.empty and not teams.empty and next_gameweek:
        weeks = list(range(next_gameweek, next_gameweek + FIXTURE_WEEKS))
        tm = teams.set_index("id")
        fx = fixtures[fixtures["event"].isin(weeks)]
        for tid, t in tm.sort_values("short_name").iterrows():
            cells = {}
            for g in weeks:
                games = []
                for r in fx[fx["event"] == g].itertuples():
                    if r.team_h == tid:
                        games.append({"opponent": tm.loc[r.team_a, "short_name"], "home": True, "difficulty": int(r.team_h_difficulty)})
                    elif r.team_a == tid:
                        games.append({"opponent": tm.loc[r.team_h, "short_name"], "home": False, "difficulty": int(r.team_a_difficulty)})
                cells[str(g)] = games
            avg = [g["difficulty"] for gs in cells.values() for g in gs]
            difficulty.append({"team_short": t["short_name"], "team_code": int(t["code"]), "cells": cells,
                               "average": round(sum(avg) / len(avg), 2) if avg else None})
        difficulty.sort(key=lambda r: (r["average"] is None, r["average"]))
        out["fixture_weeks"] = weeks
    out["fixture_difficulty"] = difficulty

    # --- ep_next experiment (accuracy log rows that have it) ---
    experiment: list[dict[str, Any]] = []
    log = accuracy_dir / "accuracy_log.csv"
    if log.exists():
        df = pd.read_csv(log)
        cols = [c for c in df.columns if c.startswith("exp_mae_")]
        if cols:
            for r in df.dropna(subset=cols, how="all").itertuples():
                experiment.append({"gameweek": int(r.gameweek), **{c[len("exp_mae_"):]: _num(getattr(r, c)) for c in cols}})
    out["experiment"] = experiment

    # --- matched backtests: latest 10-gameweek and 37-gameweek runs ---
    # The newest *matched pair*: one backtest run that wrote both a
    # 10-gameweek table (only the weeks where FPL's historical projection
    # exists, scope "fpl_xp_valid_gws") and a 37-gameweek one (all weeks,
    # FPL's projection left blank, scope "all_gws").
    runs = (model_accuracy or {}).get("runs") or []
    sec = lambda r: str(r["run_at"])[:19]  # noqa: E731 -- the pair's two rows differ in microseconds
    paired = [t for t in {sec(r) for r in runs}
              if {10, 37} <= {r.get("n_gameweeks") for r in runs if sec(r) == t}]
    backtest = []
    pair_at = max(paired) if paired else None
    for n in (10, 37):
        r = next((x for x in runs if sec(x) == pair_at and x.get("n_gameweeks") == n), None)
        if not r:
            continue
        rows = []
        for key, label in (("xp", "Our model"), ("fpl_xp", "FPL's own projection"), ("ppg", "Points per game so far"), ("last5", "Average of last 5")):
            rows.append({
                "model": label,
                "mae": _num(r.get(f"mae_{key}")),
                "rank_all": _num(r.get(f"spearman_{key}_overall")),
                "rank_starters": _num(r.get(f"spearman_{key}_likely_starter")),
                "top10": _num(r.get(f"top10_{key}")),
                "top20": _num(r.get(f"top20_{key}")),
                "captain": _num(r.get(f"captain_{key}")),
            })
        backtest.append({"gameweeks": n, "run_at": r["run_at"], "predictions": r.get("n_predictions"),
                         "scope": r.get("scope"), "rows": rows})
    out["backtest"] = backtest
    return out
