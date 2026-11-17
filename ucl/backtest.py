"""Backtest of the UCL expected points, with no lookahead.

For every played matchday m (2025/26 MD2-17 and 2026/27 MD2 onwards), the
model is fitted only on matches before m (plus all of 2025/26 when
predicting 2026/27) and predicts m; then the predictions are compared with
the real points. UEFA's injury news isn't stored historically, so the
backtest runs without it (the live model uses it, so it should do a bit
better than this).

Compared with a simple baseline: points per game so far this season
(players without a game yet get 0).

* Ranking: Spearman rank correlation between predicted and real points,
  over the players who played 60+ minutes ("starters") and over everyone.
* Top-10 hit rate: how many of the predicted top 10 were in the real top 10.
* Top-10 points: average real points of the predicted top 10.
* Captain points: real points of the predicted best player.

Run with ``uv run python -m ucl.backtest``; writes data/ucl/backtest.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ucl import feed, model

OUT = feed.DATA / "backtest.json"


def baseline(tour: int, md: int, pids: list[int]) -> pd.Series:
    _, _, a = model.load(tour)
    past = a[a["md"] < md]
    ppg = past.groupby("pid")["points"].mean()
    return pd.Series([float(ppg.get(p, 0.0)) for p in pids], index=pids)


def metrics(pred: pd.Series, actual: pd.Series, mins: pd.Series) -> dict:
    rng = np.random.default_rng(0)
    jitter = pd.Series(rng.random(len(pred)) * 1e-6, index=pred.index)
    p = pred + jitter
    top = p.nlargest(10).index
    real_top = actual.sort_values(ascending=False).index[:10]
    starters = mins[mins >= 60].index
    return {
        "rank_starters": float(spearmanr(p.loc[starters], actual.loc[starters]).statistic) if len(starters) > 5 else None,
        "rank_all": float(spearmanr(p, actual).statistic),
        "top10_hits": int(len(set(top) & set(real_top))),
        "top10_points": float(actual.loc[top].mean()),
        "captain_points": float(actual.loc[p.idxmax()]),
    }


def run_md(tour: int, md: int, cfg: dict | None = None) -> dict | None:
    xp, _ = model.predict(tour, as_of_md=md, mds=[md], use_status=False, cfg=cfg)
    if xp.empty:
        return None
    _, _, a = model.load(tour)
    real = a[a["md"] == md].groupby("pid").agg(points=("points", "sum"), mins=("oF", "sum"))
    pred = xp.groupby("pid")["xp"].sum()
    pids = list(pred.index)
    actual = real["points"].reindex(pids).fillna(0.0)
    mins = real["mins"].reindex(pids).fillna(0)
    base = baseline(tour, md, pids)
    return {"tour": tour, "md": md, "players": len(pids), "model": metrics(pred, actual, mins),
            "ppg": metrics(base, actual, mins), "mae_model": float((pred - actual).abs().mean())}


def run(tours: tuple[int, ...] = (feed.PREV_TOUR, feed.TOUR), verbose: bool = True) -> dict:
    rows = []
    for tour in tours:
        if not (model.HIST / f"appearances_{tour}.parquet").exists():
            continue
        m, _, a = model.load(tour)
        played = sorted(set(a["md"])) if len(a) else []
        for md in played:
            if md < 2:
                continue
            r = run_md(tour, md)
            if r:
                rows.append(r)
                if verbose:
                    print(f"  {tour} MD{md}: rank {r['model']['rank_starters']:.2f} vs {r['ppg']['rank_starters']:.2f}, "
                          f"top10 {r['model']['top10_hits']} vs {r['ppg']['top10_hits']}, "
                          f"captain {r['model']['captain_points']:.0f} vs {r['ppg']['captain_points']:.0f}", flush=True)

    def avg(group, who, k):
        vals = [r[who][k] for r in group if r[who][k] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {}
    for label, group in (("all", rows), ("2025/26", [r for r in rows if r["tour"] == feed.PREV_TOUR]),
                         ("2026/27", [r for r in rows if r["tour"] == feed.TOUR])):
        if not group:
            continue
        summary[label] = {"matchdays": len(group), **{
            f"{k}_{who}": avg(group, who, k) for k in ("rank_starters", "rank_all", "top10_hits", "top10_points",
                                                        "captain_points") for who in ("model", "ppg")}}
    out = {"rows": rows, "summary": summary,
           "method": "Each matchday predicted only from matches before it; injury news not used (not stored historically)."}
    out = feed.clean(out)
    OUT.write_text(json.dumps(out, indent=1, allow_nan=False))
    return out


def plain_words(bt: dict) -> list[str]:
    s = bt.get("summary", {}).get("all")
    if not s:
        return ["Not enough played matchdays to test the model yet."]
    lines = [
        f"Replayed {s['matchdays']} matchdays, predicting each one only from the matches before it.",
        f"Ranking the players who started: {s['rank_starters_model']:.2f} for XPert against {s['rank_starters_ppg']:.2f} "
        "for points per game so far (1 = perfect order, 0 = random).",
        f"Of the 10 players we expected to score most, {s['top10_hits_model']:.1f} on average were in the real top 10 "
        f"(points per game: {s['top10_hits_ppg']:.1f}). They scored {s['top10_points_model']:.1f} points each on average "
        f"({s['top10_points_ppg']:.1f}).",
        f"Our captain pick scored {s['captain_points_model']:.1f} points on average (points per game's pick: "
        f"{s['captain_points_ppg']:.1f}), before doubling.",
    ]
    return lines


if __name__ == "__main__":
    res = run()
    print(json.dumps(res["summary"], indent=1))
    print("\n".join(plain_words(res)))
