"""Between-days advice: changes to make after the first match day of a matchday.

UCL Fantasy lets you change your team between the days of one matchday: up
to 4 substitutions (a starter out, a bench player whose match hasn't started
in) and a captain switch (the new captain's points are doubled instead;
the old captain's points still count once). Making any manual change turns
off the automatic substitutions for that matchday. Not allowed on MD8 (one
day) or the final.

The advice compares two plans by simulation (ucl.sim), using real points
for players who have played and the model for the rest:

* keep: change nothing, automatic substitutions stay on;
* change: the best substitutions (each must gain at least MIN_SWAP_GAIN on
  its own) and captain switch, automatic substitutions off.

It recommends the changes only when they gain at least MIN_GAIN points on
average after the lost automatic substitutions; otherwise "keep your team".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl import rules, sim

MIN_SWAP_GAIN = 0.5
MIN_GAIN = 1.0
N = 3000


def _expected(xi: list[int], bench: list[int], captain: int, draws: dict[int, np.ndarray],
              played: dict[int, bool], pos: dict[int, int], autosubs: bool) -> float:
    n = len(next(iter(draws.values())))
    total = np.zeros(n)
    xi_d = np.stack([draws[p] for p in xi], axis=1)
    total += xi_d.sum(axis=1) + draws[captain]
    if autosubs:
        # a starter who didn't play is replaced by the first fitting bench player who did
        for i in range(n):
            did = {p: draws[p][i] != 0 for p in xi + bench}
            if all(did[p] for p in xi):
                continue
            final = rules.auto_subs([(p, pos[p], bool(did[p])) for p in xi], [(b, pos[b], bool(did[b])) for b in bench])
            total[i] += sum(draws[p][i] for p in final) - sum(draws[p][i] for p in xi)
    return float(total.mean())


def advise(team: dict, md: int, xp_md: pd.DataFrame, players: pd.DataFrame, matches: pd.DataFrame,
           actual: dict[int, float], now: pd.Timestamp) -> dict:
    """team: {"xi": [...], "bench": [...ordered], "captain": pid, "vice": pid}."""
    if not rules.between_days_allowed(md):
        return {"status": "not_allowed", "md": md,
                "note": "No changes between days on this matchday: MD8 is played on one day, and so is the final."}
    mm = matches[matches["md"] == md]
    days = sorted(mm["gd"].unique())
    if len(days) < 2:
        return {"status": "not_allowed", "md": md, "note": "This matchday is played on one day."}
    started = mm[mm["kickoff"] <= now]
    if started.empty:
        return {"status": "before", "md": md, "note": "Between-days advice appears after the first match day."}
    if len(started) == len(mm):
        return {"status": "over", "md": md, "note": "Every match of this matchday has started."}
    club_started = set(started["home_id"]) | set(started["away_id"])
    pl = players.set_index("pid")
    pos = {int(p): int(pl.loc[p, "pos"]) for p in team["xi"] + team["bench"]}
    played = {p: int(pl.loc[p, "team_id"]) in club_started for p in pos}
    rows = xp_md.set_index("pid")
    draws_src = rows.reindex(list(pos)).fillna(0)
    draws_src["pos"] = [pos[p] for p in draws_src.index]
    raw = sim.sample(draws_src.reset_index(), n=N)
    draws = {}
    for j, p in enumerate(draws_src.index):
        draws[int(p)] = np.full(N, float(actual.get(int(p), 0.0))) if played[int(p)] else raw[:, j]
    val = {p: (float(actual.get(p, 0.0)) if played[p] else float(rows["xp"].get(p, 0.0))) for p in pos}

    xi, bench, cap = list(team["xi"]), list(team["bench"]), team["captain"]
    keep = _expected(xi, bench, cap, draws, played, pos, autosubs=True)

    subs = []
    new_xi, new_bench = list(xi), list(bench)
    for _ in range(rules.BETWEEN_DAYS_MAX_SUBS):
        best = None
        for o in new_xi:
            for i in new_bench:
                if played[i]:
                    continue
                trial = [p for p in new_xi if p != o] + [i]
                if not rules.valid_xi([pos[p] for p in trial]):
                    continue
                g = val[i] - val[o]
                if g >= MIN_SWAP_GAIN and (best is None or g > best[0]):
                    best = (g, o, i)
        if not best:
            break
        g, o, i = best
        new_xi = [p for p in new_xi if p != o] + [i]
        new_bench = [p for p in new_bench if p != i] + [o]
        subs.append({"out": o, "in": i, "gain": round(g, 2), "out_played": played[o],
                     "out_value": round(val[o], 2), "in_value": round(val[i], 2)})
    new_cap = cap if cap in new_xi else max(new_xi, key=lambda p: val[p])
    switch = None
    if played.get(new_cap):
        unplayed = [p for p in new_xi if not played[p]]
        if unplayed:
            cand = max(unplayed, key=lambda p: val[p])
            if val[cand] - val[new_cap] >= MIN_SWAP_GAIN:
                switch = {"from": new_cap, "to": cand, "gain": round(val[cand] - val[new_cap], 2),
                          "from_points": val[new_cap], "to_xp": round(val[cand], 2)}
                new_cap = cand
    changed = bool(subs) or new_cap != cap
    change = _expected(new_xi, new_bench, new_cap, draws, played, pos, autosubs=False) if changed else keep
    # value of automatic substitutions kept by not touching the team
    no_auto = _expected(xi, bench, cap, draws, played, pos, autosubs=False)
    gain = change - keep
    recommend = changed and gain >= MIN_GAIN
    return {
        "status": "advice", "md": md, "recommend": recommend,
        "subs": subs if recommend else [], "captain_switch": switch if recommend else None,
        "considered": {"subs": subs, "captain_switch": switch},
        "expected_keep": round(keep, 2), "expected_change": round(change, 2), "gain": round(gain, 2),
        "autosub_value": round(keep - no_auto, 2),
        "played": [p for p in pos if played[p]],
        "note": ("Make these changes: they should gain about {:.1f} points even after losing the automatic "
                 "substitutions.".format(gain)) if recommend else
                ("Keep your team: no change gains at least {:.0f} point on average once the automatic substitutions "
                 "(worth about {:.1f} here) are lost.".format(MIN_GAIN, keep - no_auto)),
    }


def flexibility_value(team: dict, xp_md: pd.DataFrame, pos: dict[int, int], n: int = 1500) -> dict:
    """Expected points from the between-days option for a squad, before the
    matchday: simulate the first day, apply the same advice rule with the
    model's values for the later days, and compare with keeping the team."""
    xp = xp_md.set_index("pid")
    ids = team["xi"] + team["bench"]
    src = xp.reindex(ids).fillna(0)
    src["pos"] = [pos[p] for p in ids]
    first = int(src["gd"][src["gd"] > 0].min()) if (src["gd"] > 0).any() else 0
    if not first or src["gd"].nunique() < 2:
        return {"value": 0.0, "captain_switch_value": 0.0, "note": "All your players play on the same day."}
    d = sim.sample(src.reset_index(), n=n, seed=5)
    col = {p: j for j, p in enumerate(ids)}
    day1 = {p for p in ids if int(src.loc[p, "gd"]) == first}
    gain_total = np.zeros(n)
    cap_gain = np.zeros(n)
    for i in range(n):
        val = {p: (d[i, col[p]] if p in day1 else float(src.loc[p, "xp"])) for p in ids}
        xi, bench, cap = list(team["xi"]), list(team["bench"]), team["captain"]
        g = 0.0
        swaps = 0
        for _ in range(rules.BETWEEN_DAYS_MAX_SUBS):
            best = None
            for o in xi:
                for b in bench:
                    if b in day1:
                        continue
                    trial = [p for p in xi if p != o] + [b]
                    if not rules.valid_xi([pos[p] for p in trial]):
                        continue
                    gg = val[b] - val[o]
                    if gg >= MIN_SWAP_GAIN and (best is None or gg > best[0]):
                        best = (gg, o, b)
            if not best:
                break
            gg, o, b = best
            xi = [p for p in xi if p != o] + [b]
            bench = [p for p in bench if p != b] + [o]
            g += gg
            swaps += 1
        cg = 0.0
        if cap in day1:
            later = [p for p in xi if p not in day1]
            if later:
                c2 = max(later, key=lambda p: val[p])
                if val[c2] - val[cap] >= MIN_SWAP_GAIN:
                    cg = val[c2] - val[cap]
        if g + cg >= MIN_GAIN:
            gain_total[i] = g + cg
            cap_gain[i] = cg
    return {"value": round(float(gain_total.mean()), 2), "captain_switch_value": round(float(cap_gain.mean()), 2),
            "share_of_weeks": round(float((gain_total > 0).mean()), 2)}
