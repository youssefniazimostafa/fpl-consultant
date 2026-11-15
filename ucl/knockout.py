"""Knockout awareness: simulate the rest of the league phase and the knockouts.

From the current 36-club table and the team model's goal expectations,
the remaining league-phase matches are simulated N_SIMS times. 1st-8th go
straight to the round of 16 (so their players don't play in the knockout
play-offs, MD9-10), 9th-24th play the play-offs, 25th-36th are out. The
knockouts are then simulated as two-legged ties (seeded: 9-16 v 17-24 in
the play-offs, top 8 v play-off winners in the round of 16, then an open
draw) and a one-off final on neutral ground; level ties are decided by a
coin flip (extra time and penalties).

Output per club: P(top 8), P(9th-24th), P(out after the league phase),
P(reaching each round), and the expected number of UCL matches left, which
is also the expected number of matchdays its players can still score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl import rules
from ucl.model import TeamModel

N_SIMS = 4000
SEED = 7


def table(matches: pd.DataFrame) -> pd.DataFrame:
    lg = matches[(matches["md"] <= 8)]
    teams = sorted(set(lg["home_id"]) | set(lg["away_id"]))
    t = pd.DataFrame(0, index=teams, columns=["played", "pts", "gf", "ga"])
    for r in lg[lg["finished"]].itertuples():
        h, a, hg, ag = int(r.home_id), int(r.away_id), int(r.home_goals), int(r.away_goals)
        t.loc[[h, a], "played"] += 1
        t.loc[h, ["gf", "ga"]] += [hg, ag]
        t.loc[a, ["gf", "ga"]] += [ag, hg]
        t.loc[h, "pts"] += 3 if hg > ag else 1 if hg == ag else 0
        t.loc[a, "pts"] += 3 if ag > hg else 1 if hg == ag else 0
    t["gd"] = t["gf"] - t["ga"]
    names = {}
    for r in matches.itertuples():
        names[int(r.home_id)] = (r.home, r.home_code)
        names[int(r.away_id)] = (r.away, r.away_code)
    t["team"] = [names[i][0] for i in t.index]
    t["code"] = [names[i][1] for i in t.index]
    return t.sort_values(["pts", "gd", "gf"], ascending=False)


def _tie(rng, model: TeamModel, a: int, b: int, legs: int = 2) -> int:
    if legs == 1:
        lh, la = model.lambdas(a, b, neutral=True)
        ga, gb = rng.poisson(lh), rng.poisson(la)
    else:
        l1h, l1a = model.lambdas(a, b)
        l2h, l2a = model.lambdas(b, a)
        ga = rng.poisson(l1h) + rng.poisson(l2a)
        gb = rng.poisson(l1a) + rng.poisson(l2h)
    if ga == gb:
        return a if rng.random() < 0.5 else b
    return a if ga > gb else b


def simulate(matches: pd.DataFrame, model: TeamModel, n: int = N_SIMS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = table(matches)
    teams = list(base.index)
    ix = {t: i for i, t in enumerate(teams)}
    rest = matches[(matches["md"] <= 8) & (~matches["finished"])]
    lam = np.array([model.lambdas(int(r.home_id), int(r.away_id)) for r in rest.itertuples()]).reshape(-1, 2)
    hi = np.array([ix[int(h)] for h in rest["home_id"]], dtype=int)
    ai = np.array([ix[int(a)] for a in rest["away_id"]], dtype=int)
    K = len(teams)
    cnt = {k: np.zeros(K) for k in ("top8", "po", "out", "r16", "qf", "sf", "final", "win")}
    for _ in range(n):
        pts = base["pts"].to_numpy(float).copy()
        gd = base["gd"].to_numpy(float).copy()
        gf = base["gf"].to_numpy(float).copy()
        if len(rest):
            hg = rng.poisson(lam[:, 0])
            ag = rng.poisson(lam[:, 1])
            np.add.at(pts, hi, np.where(hg > ag, 3, np.where(hg == ag, 1, 0)))
            np.add.at(pts, ai, np.where(ag > hg, 3, np.where(hg == ag, 1, 0)))
            np.add.at(gd, hi, hg - ag)
            np.add.at(gd, ai, ag - hg)
            np.add.at(gf, hi, hg)
            np.add.at(gf, ai, ag)
        key = pts * 1e6 + gd * 1e3 + gf + rng.random(K) * 0.5
        order = list(np.argsort(-key))
        top8, po, out = order[:8], order[8:24], order[24:]
        for g, k in ((top8, "top8"), (po, "po"), (out, "out")):
            cnt[k][g] += 1
        # play-offs: 9-16 v 17-24 (seeded, random within)
        seeded, unseeded = po[:8], list(rng.permutation(po[8:]))
        winners = [ix[_tie(rng, model, teams[s], teams[u])] for s, u in zip(seeded, unseeded)]
        r16 = top8 + winners
        cnt["r16"][r16] += 1
        pairs = list(zip(top8, rng.permutation(winners)))
        qf = [ix[_tie(rng, model, teams[a], teams[b])] for a, b in pairs]
        cnt["qf"][qf] += 1
        q = list(rng.permutation(qf))
        sf = [ix[_tie(rng, model, teams[q[i]], teams[q[i + 1]])] for i in range(0, 8, 2)]
        cnt["sf"][sf] += 1
        s = list(rng.permutation(sf))
        fin = [ix[_tie(rng, model, teams[s[i]], teams[s[i + 1]])] for i in range(0, 4, 2)]
        cnt["final"][fin] += 1
        cnt["win"][ix[_tie(rng, model, teams[fin[0]], teams[fin[1]], legs=1)]] += 1
    out = base.copy()
    for k, v in cnt.items():
        out["p_" + k] = v / n
    left_league = 8 - out["played"]
    # matchdays still to come: league games, then 2 per two-legged round, 1 final
    out["exp_matches"] = left_league + 2 * out["p_po"] + 2 * out["p_r16"] + 2 * out["p_qf"] + 2 * out["p_sf"] + out["p_final"]
    out["rank"] = range(1, len(out) + 1)
    return out


def plays_md(ko: pd.DataFrame, team_id: int, md: int) -> float:
    """Chance the club plays on knockout matchday `md`."""
    if md <= 8 or team_id not in ko.index:
        return 1.0 if md <= 8 else 0.0
    r = ko.loc[team_id]
    stage = rules.STAGE[md]
    return float({"playoff": r.p_po, "r16": r.p_r16, "qf": r.p_qf, "sf": r.p_sf, "final": r.p_final}[stage])
