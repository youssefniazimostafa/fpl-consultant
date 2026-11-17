"""Backtest of the UCL model refinements, one factor at a time.

The rule (fixed before any run):
* each factor is added on its own, on top of the factors already kept,
  and is switchable (ucl.model.CFG);
* it is backtested with no lookahead on 2025/26 MD2-17 (tuning) and
  2026/27 MD1 onwards (unseen; MD1 is predicted from 2025/26 and the
  domestic results alone);
* metrics: rank of the players who started (Spearman), top-10 hits,
  real points of the predicted top 10, captain points, and the average
  miss |xP - points| (lower is better);
* KEPT when, on 2026/27, it improves at least 3 of the 5 metrics, and on
  2025/26 it doesn't clearly hurt: no drop of 0.02+ in starter ranking,
  0.3+ in top-10 points, 0.5+ in captain points, and no rise of 0.05+
  in the average miss;
* simple, round values only.

Run ``uv run python -m ucl.factors``; writes data/ucl/factor_tests.json.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ucl import backtest, feed, model

OUT = feed.DATA / "factor_tests.json"
METRICS = ("rank_starters", "top10_hits", "top10_points", "captain_points", "avg_miss")
HURT = {"rank_starters": -0.02, "top10_points": -0.3, "captain_points": -0.5, "avg_miss": 0.05}

# (name, what it changes, cfg overrides on top of what's kept so far); one at a time, in this order
PHASE1 = [
    ("Elo team ratings, 50% of expected goals", {"elo_weight": 0.5}),
    ("Elo team ratings, 100% of expected goals", {"elo_weight": 1.0}),
    ("Prior from price (x price/typical)", {"prior_price": 1.0}),
    ("Prior from team strength (x team goals/average)", {"prior_team": 1.0}),
    ("Slower fade: prior worth 6 matches", {"shrink_90s": 6.0}),
    ("Slower fade: prior worth 10 matches", {"shrink_90s": 10.0}),
    # added after the first run (all six above dropped): two milder variants
    ("Elo team ratings, 30% of expected goals", {"elo_weight": 0.3}),
    ("Prior from Elo team strength (all matches)", {"prior_team": 1.0, "prior_team_elo": True}),
]


def season_scores(cfg: dict) -> dict:
    out = {}
    for tour, first in ((feed.PREV_TOUR, 2), (feed.TOUR, 1)):
        _, _, a = model.load(tour)
        rows = [r for md in sorted(set(a["md"])) if md >= first and (r := backtest.run_md(tour, md, cfg))]
        out[tour] = {"matchdays": len(rows), **{
            k: round(float(np.mean([r["model"][k] for r in rows if r["model"][k] is not None])), 3)
            for k in ("rank_starters", "top10_hits", "top10_points", "captain_points")},
            "avg_miss": round(float(np.mean([r["mae_model"] for r in rows])), 3)}
    return out


def better(new: dict, old: dict, k: str) -> bool:
    return new[k] < old[k] if k == "avg_miss" else new[k] > old[k]


def judge(new: dict, old: dict) -> tuple[bool, str]:
    unseen = new[feed.TOUR], old[feed.TOUR]
    wins = [k for k in METRICS if better(unseen[0], unseen[1], k)]
    tune_new, tune_old = new[feed.PREV_TOUR], old[feed.PREV_TOUR]
    hurt = [k for k, lim in HURT.items()
            if (tune_new[k] - tune_old[k] >= lim if lim > 0 else tune_new[k] - tune_old[k] <= lim)]
    keep = len(wins) >= 3 and not hurt
    why = f"2026/27 better on {len(wins)}/5 ({', '.join(wins) or 'none'})"
    why += f"; 2025/26 clearly worse on {', '.join(hurt)}" if hurt else "; 2025/26 not clearly worse"
    return keep, why


def team_test(weights=(0.0, 0.5, 1.0)) -> dict:
    """Team level, for the knockout simulation (which no player metric sees):
    mean Poisson log-likelihood of each UCL score (home and away goals) when
    every match is predicted only from matches before its matchday, with
    the given share of Elo in the expected goals. Same rule: kept if better
    on 2026/27 and not worse on 2025/26."""
    out = {}
    for w in weights:
        res = {}
        for tour, first in ((feed.PREV_TOUR, 2), (feed.TOUR, 1)):
            m, _, _ = model.load(tour)
            lls = []
            for md in sorted(m.loc[m["finished"], "md"].unique()):
                if md < first:
                    continue
                _, team = model.predict(tour, as_of_md=int(md), mds=[], use_status=False,
                                        cfg={"elo_weight_knockout": w} if w else {})
                for r in m[(m["md"] == md) & m["finished"]].itertuples():
                    st = str(r.stage).lower()
                    neutral = "final" in st and "semi" not in st and "quarter" not in st
                    lh, la = team.knockout.lambdas(int(r.home_id), int(r.away_id), neutral)
                    lls.append(sum(g * math.log(lam) - lam - math.lgamma(g + 1)
                                   for lam, g in ((lh, r.home_goals), (la, r.away_goals))))
            res[tour] = {"matches": len(lls), "loglik": round(float(np.mean(lls)), 4)}
        out[str(w)] = res
        print("knockout Elo share", w, json.dumps(res), flush=True)
    base = out["0.0"]
    ok = {w: r[feed.TOUR]["loglik"] > base[feed.TOUR]["loglik"] and r[feed.PREV_TOUR]["loglik"] >= base[feed.PREV_TOUR]["loglik"]
          for w, r in out.items() if w != "0.0"}
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev["knockout_team_test"] = {"results": out, "passes": ok}
    OUT.write_text(json.dumps(feed.clean(prev), indent=1))
    return prev["knockout_team_test"]


def run(factors=PHASE1, base: dict | None = None, label: str = "phase1") -> dict:
    kept = dict(base or {})
    cur = season_scores(kept)
    print("baseline", json.dumps(cur), flush=True)
    log = [{"factor": "baseline (model before this work)" if not base else "starting point", "cfg": dict(kept),
            "scores": cur, "kept": True, "why": ""}]
    for name, over in factors:
        trial = {**kept, **over}
        sc = season_scores(trial)
        keep, why = judge(sc, cur)
        print(f"{'KEEP' if keep else 'drop'} {name}: {why}\n   {json.dumps(sc)}", flush=True)
        log.append({"factor": name, "cfg": over, "scores": sc, "kept": keep, "why": why})
        if keep:
            kept, cur = trial, sc
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[label] = {"rule": __doc__.split("The rule (fixed before any run):")[1].split("Run ``")[0].strip(),
                   "tests": log, "final_cfg": kept}
    OUT.write_text(json.dumps(feed.clean(prev), indent=1))
    return prev[label]


if __name__ == "__main__":
    import sys

    if "--team" in sys.argv:
        print(team_test())
    else:
        res = run(base={k: 0.0 if isinstance(v, float) and k.startswith(("elo", "prior")) else v
                        for k, v in model.CFG.items()} | {"prior_team_elo": False, "shrink_90s": model.SHRINK_90S})
        print("final:", res["final_cfg"])
