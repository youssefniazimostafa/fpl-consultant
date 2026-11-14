"""Step 14: test how the xP model weights recent form, on past seasons.

Each setting is scored with fpl.backtest's own replay (every gameweek
predicted only from rows strictly before it -- no lookahead) on 2025-26
(the tuning season) and, unchanged, on 2024-25 and 2023-24 (never used for
tuning). Settings tried, all in fpl.features: the recency half-life
(HALF_LIFE_MATCHES), the shrinkage toward the position average
(SHRINKAGE_MATCHES), spike damping (SPIKE_MODE) and form-change detection
(FORM_CHANGE_ON).

Metrics, per gameweek, then paired against today's settings over the
season's gameweeks (mean difference +- standard error):

* starter rank correlation: Spearman(xP, actual points) among predicted
  starters (p_start > 0.5), averaged over the four positions -- the
  backtest's own decision metric;
* top-10 hit rate: share of each position's actual top 10 that the model
  also had in its top 10 (``fpl.backtest.topn_hit_rate``);
* captain points: actual points of the model's highest-xP player;
* error: mean absolute error in points per player-week (all players).

Rule (same as the transfer calibration): a setting is switched on only if
it improves starter rank correlation on BOTH unseen seasons by more than
one paired SE, and no other metric gets worse by more than one SE on
either.

Run with ``uv run python -m fpl.form_calibration``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl import backtest as B
from fpl import features as F

SEASONS = [None, "2024-25", "2023-24"]  # None = the tuning season (2025-26)
DEFAULTS = {k: getattr(F, k) for k in ["HALF_LIFE_MATCHES", "SHRINKAGE_MATCHES", "SPIKE_MODE", "SPIKE_POINTS",
                                       "SPIKE_WEIGHT", "SPIKE_CAP_QUANTILE", "FORM_CHANGE_ON", "FORM_CHANGE_OLD_WEIGHT"]}
METRICS = ["starter_rho", "top10", "captain", "mae"]
HIGHER_IS_BETTER = {"starter_rho": True, "top10": True, "captain": True, "mae": False}


def per_gw_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, d in results.groupby("event"):
        starters = d[d["p_start"] > 0.5]
        rhos = []
        for _, s in starters.groupby("position"):
            if len(s) >= 5 and s["xp"].nunique() > 1 and s["total_points"].nunique() > 1:
                rhos.append(spearmanr(s["xp"], s["total_points"])[0])
        rows.append({
            "event": g,
            "starter_rho": float(np.mean(rhos)) if rhos else np.nan,
            "top10": B.topn_hit_rate(d, "xp", 10),
            "captain": float(d.loc[d["xp"].idxmax(), "total_points"]),
            "mae": float((d["xp"] - d["total_points"]).abs().mean()),
        })
    return pd.DataFrame(rows).set_index("event")


def run_setting(setting: dict[str, Any], season: str | None) -> pd.DataFrame:
    for k, v in {**DEFAULTS, **setting}.items():
        setattr(F, k, v)
    try:
        return per_gw_metrics(B.run_backtest(season))
    finally:
        for k, v in DEFAULTS.items():
            setattr(F, k, v)


def label(setting: dict[str, Any]) -> str:
    if not setting:
        return "today (half-life 6, shrinkage 3)"
    bits = []
    if "HALF_LIFE_MATCHES" in setting or "SHRINKAGE_MATCHES" in setting:
        bits.append(f"half-life {setting.get('HALF_LIFE_MATCHES', DEFAULTS['HALF_LIFE_MATCHES'])}, "
                    f"shrinkage {setting.get('SHRINKAGE_MATCHES', DEFAULTS['SHRINKAGE_MATCHES'])}")
    if setting.get("SPIKE_MODE") == "discount":
        bits.append(f"discount {setting['SPIKE_POINTS']}+ pt matches x{setting['SPIKE_WEIGHT']}")
    if setting.get("SPIKE_MODE") == "cap":
        bits.append(f"cap at q{setting['SPIKE_CAP_QUANTILE']}")
    if setting.get("FORM_CHANGE_ON"):
        bits.append(f"form change, old x{setting['FORM_CHANGE_OLD_WEIGHT']}")
    return "; ".join(bits)


def grid() -> list[dict[str, Any]]:
    g: list[dict[str, Any]] = [{}]
    for hl in [3, 4, 6, 8, 10, 15]:
        for k in [1, 3, 6, 10]:
            if (hl, k) != (6, 3):
                g.append({"HALF_LIFE_MATCHES": hl, "SHRINKAGE_MATCHES": k})
    for pts, w in [(10, 0.5), (13, 0.5), (13, 0.75), (15, 0.5)]:
        g.append({"SPIKE_MODE": "discount", "SPIKE_POINTS": pts, "SPIKE_WEIGHT": w})
    for q in [0.98, 0.99, 0.995]:
        g.append({"SPIKE_MODE": "cap", "SPIKE_CAP_QUANTILE": q})
    for w in [0.25, 0.5]:
        g.append({"FORM_CHANGE_ON": True, "FORM_CHANGE_OLD_WEIGHT": w})
    return g


def paired(base: pd.DataFrame, other: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for m in METRICS:
        d = (other[m] - base[m]).dropna()
        out[m] = (float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"))
    return out


def verdict(diffs: dict[str | None, dict[str, tuple[float, float]]]) -> bool:
    for season in SEASONS[1:]:
        d = diffs[season]
        mean, se = d["starter_rho"]
        if not mean > se:
            return False
        for m in METRICS[1:]:
            mm, ss = d[m]
            worse = -mm if HIGHER_IS_BETTER[m] else mm
            if worse > ss:
                return False
    return True


def main() -> None:
    settings = grid()
    results = {s: {label(x): run_setting(x, s) for x in settings} for s in SEASONS}
    base_label = label({})
    rows = []
    for x in settings:
        lab = label(x)
        diffs = {s: paired(results[s][base_label], results[s][lab]) for s in SEASONS}
        row = {"setting": lab}
        for s in SEASONS:
            name = s or "2025-26"
            for m in METRICS:
                if lab == base_label:
                    row[f"{name} {m}"] = round(float(results[s][lab][m].mean()), 3)
                else:
                    mean, se = diffs[s][m]
                    row[f"{name} {m}"] = f"{mean:+.3f}±{se:.3f}"
        row["switch on?"] = "baseline" if lab == base_label else ("YES" if verdict(diffs) else "no")
        rows.append(row)
    table = pd.DataFrame(rows).set_index("setting")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 30)
    for s in SEASONS:
        name = s or "2025-26"
        print(f"\n=== {name} ({'tuning' if s is None else 'unseen'}): baseline means, then differences vs today")
        print(table[[c for c in table.columns if c.startswith(name)] + ["switch on?"]].to_string())
    table.to_csv(B.TABLES_DIR / "form_calibration.csv")


if __name__ == "__main__":
    main()
