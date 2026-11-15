"""Squad, XI, captain and transfers over a horizon of matchdays (MILP, HiGHS).

The horizon runs from the current matchday to the matchday before the next
unlimited-transfer window (e.g. MD5-8 now, since transfers are unlimited
before the play-offs), so nothing after it depends on today's squad.

Per matchday: 15 players (2/5/5/3) within the budget and the club limit for
that stage, a valid XI, a captain (doubled), transfers with the free-transfer
schedule from ucl.rules (league phase: 2 per matchday, at most 1 carried)
and -4 per extra transfer. Later matchdays count a little less (DISCOUNT),
each transfer costs TRANSFER_FRICTION points so the plan doesn't churn on
small differences, and bench players count at BENCH_WEIGHT (more if they
play on a later match day than the first, because then they can be brought
in by hand between days, see ucl.between_days).

Chips: `solve(..., wildcard_md=m)` makes transfers before m free and
permanent; `limitless_md=m` picks a one-off squad for m with no budget
(club limits still apply) and the normal squad returns afterwards.
"""

from __future__ import annotations

import pandas as pd
import pulp

from ucl import rules

DISCOUNT = 0.92
BENCH_WEIGHT = 0.08
LATER_DAY_BENCH_WEIGHT = 0.2
TRANSFER_FRICTION = 0.5
CANDIDATES_PER_POS = 45
TIME_LIMIT = 60


def horizon(md: int) -> list[int]:
    out = [md]
    m = md + 1
    while m <= rules.N_MATCHDAYS and not rules.is_unlimited(m):
        out.append(m)
        m += 1
    return out


def candidates(xp: pd.DataFrame, players: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
    tot = xp.groupby("pid")["xp"].sum()
    p = players.set_index("pid").join(tot.rename("hxp")).fillna({"hxp": 0.0})
    p = p[(p["active"] == 1) | p.index.isin(keep)]
    picks = set(keep)
    for pos, sub in p.groupby("pos"):
        picks |= set(sub.nlargest(CANDIDATES_PER_POS, "hxp").index)
        picks |= set(sub.nsmallest(6, "price").index)  # cheap bench options
        sub = sub.assign(vpm=sub["hxp"] / sub["price"].clip(lower=3.5))
        picks |= set(sub.nlargest(15, "vpm").index)
    return p.loc[sorted(picks)].reset_index()


def solve(xp: pd.DataFrame, players: pd.DataFrame, md: int, squad: list[int] | None = None,
          free_transfers: int | None = None, bank: float | None = None, wildcard_md: int | None = None,
          limitless_md: int | None = None, mds: list[int] | None = None) -> dict:
    """xp: rows (pid, md, xp, gd) from ucl.model.predict. With no `squad`, the
    first matchday builds a new team (free, within the budget)."""
    mds = mds or horizon(md)
    keep = set(squad or [])
    cand = candidates(xp[xp["md"].isin(mds)], players, keep)
    P = list(cand["pid"].astype(int))
    pos = dict(zip(P, cand["pos"].astype(int)))
    club = dict(zip(P, cand["team_id"].astype(int)))
    price = dict(zip(P, cand["price"].astype(float)))
    agg = xp[xp["md"].isin(mds)].groupby(["pid", "md"]).agg(xp=("xp", "sum"), gd=("gd", "min")).reset_index()
    X = {(int(r.pid), int(r.md)): float(r.xp) for r in agg.itertuples()}
    GD = {(int(r.pid), int(r.md)): int(r.gd) for r in agg.itertuples()}
    first_gd = {m: min([g for (p_, mm), g in GD.items() if mm == m] or [0]) for m in mds}
    clubs = sorted(set(club.values()))

    prob = pulp.LpProblem("ucl", pulp.LpMaximize)
    x = {(p, m): prob.add_variable(f"x_{p}_{m}", cat="Binary") for p in P for m in mds}
    s = {(p, m): prob.add_variable(f"s_{p}_{m}", cat="Binary") for p in P for m in mds}
    c = {(p, m): prob.add_variable(f"c_{p}_{m}", cat="Binary") for p in P for m in mds}
    tin = {(p, m): prob.add_variable(f"i_{p}_{m}", cat="Binary") for p in P for m in mds}
    tout = {(p, m): prob.add_variable(f"o_{p}_{m}", cat="Binary") for p in P for m in mds}
    hits = {m: prob.add_variable(f"h_{m}", lowBound=0, cat="Integer") for m in mds}
    ft = {m: prob.add_variable(f"f_{m}", lowBound=0, upBound=10, cat="Integer") for m in mds}
    carry = {m: prob.add_variable(f"cy_{m}", cat="Binary") for m in mds}
    L = limitless_md
    if L is not None:
        z = {p: prob.add_variable(f"z_{p}", cat="Binary") for p in P}
        zs = {p: prob.add_variable(f"zs_{p}", cat="Binary") for p in P}
        zc = {p: prob.add_variable(f"zc_{p}", cat="Binary") for p in P}

    def team_constraints(xv, sv, cv, m, with_budget: bool, budget: float):
        nonlocal prob
        for q, n in rules.SQUAD_SHAPE.items():
            prob += pulp.lpSum(xv[p] for p in P if pos[p] == q) == n
            prob += pulp.lpSum(sv[p] for p in P if pos[p] == q) >= rules.XI_MIN[q]
            prob += pulp.lpSum(sv[p] for p in P if pos[p] == q) <= rules.XI_MAX[q]
        prob += pulp.lpSum(sv.values()) == rules.XI_SIZE
        prob += pulp.lpSum(cv.values()) == 1
        for p in P:
            prob += sv[p] <= xv[p]
            prob += cv[p] <= sv[p]
        lim = rules.club_limit(m)
        for t in clubs:
            prob += pulp.lpSum(xv[p] for p in P if club[p] == t) <= lim
        if with_budget:
            prob += pulp.lpSum(price[p] * xv[p] for p in P) <= budget

    start_value = sum(price.get(p, 0) for p in keep)
    budget = (start_value + bank) if (squad and bank is not None) else rules.budget(md)
    for m in mds:
        bud = budget + (rules.budget(m) - rules.budget(md)) if squad else rules.budget(m)
        team_constraints({p: x[p, m] for p in P}, {p: s[p, m] for p in P}, {p: c[p, m] for p in P}, m, True, bud)
    if L is not None:
        team_constraints(z, zs, zc, L, False, 0)

    obj = []
    for i, m in enumerate(mds):
        w = DISCOUNT ** i
        if m == L:
            obj += [w * X.get((p, m), 0) * (zs[p] + zc[p]) for p in P]
        else:
            for p in P:
                v = X.get((p, m), 0)
                bw = LATER_DAY_BENCH_WEIGHT if GD.get((p, m), 0) > first_gd[m] and rules.between_days_allowed(m) else BENCH_WEIGHT
                obj.append(w * v * (s[p, m] + c[p, m]))
                obj.append(w * bw * v * (x[p, m] - s[p, m]))
        # transfers into matchday m
        prev = None if i == 0 else mds[i - 1]
        for p in P:
            before = (1 if p in keep else 0) if prev is None else x[p, prev]
            if prev is None and not squad:
                prob += tin[p, m] - x[p, m] == 0
                prob += tout[p, m] == 0
            else:
                prob += x[p, m] - before == tin[p, m] - tout[p, m]
        n_in = pulp.lpSum(tin[p, m] for p in P)
        free_window = (prev is None and not squad) or rules.is_unlimited(m) or m == wildcard_md
        if m == L:
            prob += n_in == 0  # the normal squad waits behind the one-off Limitless team
        if free_window:
            prob += hits[m] == 0
            prob += carry[m] == 0
            if i + 1 < len(mds):
                prob += ft[mds[i + 1]] == (rules.FREE_TRANSFERS[mds[i + 1]] or 15)
        else:
            if i == 0:
                prob += ft[m] == (free_transfers if free_transfers is not None else rules.FREE_TRANSFERS[m])
            prob += hits[m] >= n_in - ft[m]
            prob += carry[m] <= ft[m] - n_in + hits[m]
            if rules.STAGE[m] != "league":
                prob += carry[m] == 0
            if i + 1 < len(mds):
                nxt = mds[i + 1]
                base = rules.FREE_TRANSFERS[nxt] or 15
                prob += ft[nxt] == base + (carry[m] if rules.STAGE[nxt] == "league" else 0)
            obj.append(-rules.HIT_COST * hits[m] - TRANSFER_FRICTION * w * n_in)
    prob += pulp.lpSum(obj)
    stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=TIME_LIMIT))
    if not stats.has_solution:
        raise RuntimeError(f"UCL optimiser: {stats.status}")

    def on(v):
        return (v.varValue or 0) > 0.5

    plan = []
    for i, m in enumerate(mds):
        if m == L:
            sq = [p for p in P if on(z[p])]
            xi = [p for p in P if on(zs[p])]
            cap = next(p for p in P if on(zc[p]))
        else:
            sq = [p for p in P if on(x[p, m])]
            xi = [p for p in P if on(s[p, m])]
            cap = next(p for p in P if on(c[p, m]))
        ins = [p for p in P if on(tin[p, m])] if not (i == 0 and not squad) else []
        outs = [p for p in P if on(tout[p, m])]
        chip = "limitless" if m == L else "wildcard" if m == wildcard_md else None
        plan.append({"md": m, "squad": sq, "xi": xi, "captain": cap, "in": ins, "out": outs,
                     "hits": int(round(hits[m].varValue or 0)), "ft": None if (i == 0 and not squad) else int(round(ft[m].varValue or 0)),
                     "chip": chip, "cost": round(sum(price[p] for p in sq), 1),
                     "xp": round(sum(X.get((p, m), 0) for p in xi) + X.get((cap, m), 0), 2)})
    return {"objective": float(stats.objective), "plan": plan, "mds": mds,
            "price": price}


def bench_order(squad: list[int], xi: list[int], xpm: dict[int, float], pos: dict[int, int]) -> list[int]:
    """Bench: the reserve keeper first (he can only replace the keeper), then
    outfield players by expected points."""
    bench = [p for p in squad if p not in xi]
    gk = [p for p in bench if pos[p] == rules.GK]
    out = sorted([p for p in bench if pos[p] != rules.GK], key=lambda p: -xpm.get(p, 0))
    return gk + out
