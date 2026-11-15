"""Step 16: test the risk dial (fpl.optimise.risk_dial) in simulated
10-manager mini-leagues on past seasons.

For each season (2025-26 to tune; 2024-25 and 2023-24 unseen), start point
(a gameweek mid-season and late in the season) and starting gap (you lead
by 10, or trail the leader by 10 / 30 / 60), build several leagues:

* nine rivals about as strong as a decent manager, whose squads follow real
  ownership: each is the optimiser's best squad on his own noisy view of
  expected points (x lognormal(0, 0.3) per player), tilted up to +30% toward
  the players FPL's "selected" counts say are most owned that gameweek
  (vaastav's merged_gw). Each week a rival
  usually makes one transfer (his weakest player by expected points out, a
  popular, decent player in), plays his best XI and most often captains
  his most-owned starter;
* rival starting totals: one rival at the gap, the others spread between
  40 behind you and the gap;
* your team: the optimiser's best squad at the start, then every week the
  same replay as the transfer calibration (leakage-free forecasts, 1 free
  transfer, banking to 5, no hits, the per-week bar and the saved-transfer
  value) with the rivals' rank-weighted ownership feeding the value, under
  each policy: today's "balanced" (lambda 0.02) or the risk dial.

Everyone scores actual points (FPL autosubs, vice-captain). The same
leagues are replayed under every policy, so the comparison is paired.
Metrics: chance of finishing first, average final position, average points.

Run with ``uv run python -m fpl.strategy_calibration --stage tune`` and
``--stage check --lam .. --cap ..``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from fpl import backtest as B
from fpl import optimise as O
from fpl import transfer_calibration as C

STARTS = (12, 26)          # 27 and 13 gameweeks to go (to GW38)
GAPS = (-10, 10, 30, 60)   # leader minus you (negative = you lead by 10)
N_LEAGUES = 8
RIVALS = 9
RANK_WEIGHT_SCALE = 30     # same rank weighting as fpl.league's live EO
TRANSFER_PROB = 0.7
CAPTAIN_MOST_OWNED_PROB = 0.6

_S: dict[str, Any] = {}


SOLVE_TIME_LIMIT = 30  # seconds per MILP in the simulation only, so one hard solve can't stall a worker


def _init(season: str | None) -> None:
    import functools

    O.pulp.HiGHS = functools.partial(O.pulp.HiGHS, timeLimit=SOLVE_TIME_LIMIT, gapRel=0.001)
    f = C.load_forecasts(season=season)
    _S["season"] = C.Season(f, season)
    _, gw = B.load_backtest_season(season)
    _S["selected"] = gw.groupby(["player_id", "GW"])["selected"].max().to_dict()


def _xi_and_captain(squad: set[int], raw: dict, g: int, info: dict, rng: np.random.Generator, selected: dict) -> dict[str, Any]:
    xp = lambda p: raw.get((p, g), 0.0)  # noqa: E731
    pos = lambda p: info[p]["position"]  # noqa: E731
    gk = max((p for p in squad if pos(p) == "GKP"), key=xp)
    out = [p for p in squad if pos(p) != "GKP"]
    xi = {gk}
    for position, n in (("DEF", 3), ("MID", 2), ("FWD", 1)):
        xi |= set(sorted((p for p in out if pos(p) == position), key=xp, reverse=True)[:n])
    rest = sorted((p for p in out if p not in xi), key=xp, reverse=True)
    for p in rest:
        if len(xi) == 11:
            break
        counts = {k: sum(1 for q in xi | {p} if pos(q) == k) for k in ("DEF", "MID", "FWD")}
        if counts["DEF"] <= 5 and counts["MID"] <= 5 and counts["FWD"] <= 3:
            xi.add(p)
    if rng.random() < CAPTAIN_MOST_OWNED_PROB:
        cap = max(xi, key=lambda p: selected.get((p, g), 0))
    else:
        cap = max(xi, key=xp)
    return {"squad_ids": set(squad), "xi_ids": xi, "captain_id": cap}


def _sample_squad(info: dict, t: int, rng: np.random.Generator, selected: dict, raw: dict) -> set[int]:
    for _ in range(200):
        squad: set[int] = set()
        clubs: dict[Any, int] = {}
        budget = C.BUDGET
        ok = True
        slots = [pos for pos, n in O.POSITION_QUOTAS.items() for _ in range(n)]
        rng.shuffle(slots)
        for i, pos in enumerate(slots):
            left = len(slots) - i - 1
            cands = [p for p, r in info.items() if r["position"] == pos and p not in squad
                     and clubs.get(r["team"], 0) < O.MAX_PER_CLUB and (p, t) in raw
                     and r["now_cost"] <= budget - 40 * left]
            if not cands:
                ok = False
                break
            w = np.array([max(selected.get((p, t), 0), 1) ** 0.8 for p in cands], dtype=float)
            p = cands[rng.choice(len(cands), p=w / w.sum())]
            squad.add(p)
            clubs[info[p]["team"]] = clubs.get(info[p]["team"], 0) + 1
            budget -= info[p]["now_cost"]
        if ok and len(squad) == 15:
            return squad
    raise RuntimeError("couldn't sample a rival squad")


RIVAL_NOISE = 0.3      # each rival's "opinion": xP x lognormal(0, RIVAL_NOISE) per player
RIVAL_OWNERSHIP_TILT = 0.3  # plus up to +30% for the most-owned players


def _rival_squad(ctx: dict, t: int, rng: np.random.Generator, selected: dict) -> set[int]:
    """A rival as good as a decent manager: the optimiser's best squad on his
    own noisy view of expected points, tilted toward popular players, so
    squads differ but follow real ownership."""
    info, raw, weeks = ctx["info"], ctx["raw"], ctx["weeks"]
    top_owned: set[int] = set()
    for pos in O.POSITION_QUOTAS:
        ids = [p for p, r in info.items() if r["position"] == pos and (p, t) in raw]
        top_owned |= set(sorted(ids, key=lambda p: -selected.get((p, t), 0))[:15])
    pool = sorted(ctx["top"] | top_owned)
    max_sel = max((selected.get((p, t), 0) for p in pool), default=1) or 1
    noise = {p: float(rng.lognormal(0.0, RIVAL_NOISE)) for p in pool}
    value = {(p, g): raw.get((p, g), 0.0) * noise[p] * (1 + RIVAL_OWNERSHIP_TILT * selected.get((p, t), 0) / max_sel)
             for p in pool for g in weeks}
    prices = {p: info[p]["now_cost"] for p in pool}
    plan = O.solve_horizon(pool, value, info, set(), prices, C.BUDGET, O.UNLIMITED_FREE_TRANSFERS, weeks,
                           {g: 0.9**i for i, g in enumerate(weeks)}, held_weeks=frozenset(weeks[1:]),
                           no_hit_weeks=frozenset(weeks))
    return set(plan["weeks"][0]["squad_ids"])


def _rival_transfer(squad: set[int], bank: int, info: dict, raw: dict, t: int, rng: np.random.Generator, selected: dict) -> tuple[set[int], int]:
    xp = lambda p: raw.get((p, t), 0.0)  # noqa: E731
    out = min(squad, key=xp)
    clubs: dict[Any, int] = {}
    for p in squad - {out}:
        clubs[info[p]["team"]] = clubs.get(info[p]["team"], 0) + 1
    budget = info[out]["now_cost"] + bank
    cands = [p for p, r in info.items() if p not in squad and r["position"] == info[out]["position"]
             and r["now_cost"] <= budget and clubs.get(r["team"], 0) < O.MAX_PER_CLUB and xp(p) > xp(out)]
    if not cands:
        return squad, bank
    w = np.array([max(selected.get((p, t), 0), 1) ** 0.5 * max(xp(p), 0.1) ** 2 for p in cands], dtype=float)
    inn = cands[rng.choice(len(cands), p=w / w.sum())]
    return (squad - {out}) | {inn}, budget - info[inn]["now_cost"]


def _league_eo(rival_weeks: list[dict], rival_totals: list[float], my_total: float) -> dict[int, float]:
    ws = [math.exp(-abs(tot - my_total) / RANK_WEIGHT_SCALE) for tot in rival_totals]
    eo: dict[int, float] = {}
    for w, wk in zip(ws, rival_weeks):
        for p in wk["xi_ids"]:
            eo[p] = eo.get(p, 0.0) + w * (2 if p == wk["captain_id"] else 1)
    total = sum(ws) or 1.0
    return {p: v / total for p, v in eo.items()}


def play(policy: dict[str, Any], start: int, gap: int, league: int) -> dict[str, Any]:
    season: C.Season = _S["season"]
    selected = _S["selected"]
    rng = np.random.default_rng(1000 * start + 10 * (gap + 100) + league)
    ctx0 = season.context(start)
    info0, raw0 = ctx0["info"], ctx0["raw"]
    rivals = []
    offsets = [gap] + list(rng.uniform(-40, max(gap, -10), RIVALS - 1))
    if gap < 0:  # you lead by -gap: everyone else at or below
        offsets = [gap] + list(rng.uniform(-40, gap, RIVALS - 1))
    for off in offsets:
        sq = _rival_squad(ctx0, start, rng, selected)
        rivals.append({"squad": sq, "bank": C.BUDGET - sum(info0[p]["now_cost"] for p in sq), "total": float(off)})

    # your squad: the optimiser's best 15 at the start (same for every policy)
    pool = sorted(ctx0["top"])
    weeks0 = ctx0["weeks"]
    value0 = {(p, g): raw0.get((p, g), 0.0) for p in pool for g in weeks0}
    prices = {p: info0[p]["now_cost"] for p in pool}
    w0 = O.solve_horizon(pool, value0, info0, set(), prices, C.BUDGET, O.UNLIMITED_FREE_TRANSFERS, weeks0,
                         {g: 0.9**i for i, g in enumerate(weeks0)}, held_weeks=frozenset(weeks0[1:]),
                         no_hit_weeks=frozenset(weeks0))
    squad = set(w0["weeks"][0]["squad_ids"])
    purchase = {p: info0[p]["now_cost"] for p in squad}
    bank = C.BUDGET - sum(purchase.values())
    ft, my_total, my_points = 1, 0.0, 0.0
    dials = []
    O.TRANSFER_BAR_MODE = "per_week"
    for t in range(start, C.LAST_GW + 1):
        ctx = season.context(t)
        info, raw = ctx["info"], ctx["raw"]
        # rivals move first (their deadline picks are hidden from you; their
        # last week's teams set your league EO, like the live app)
        rival_weeks = []
        for r in rivals:
            if t > start and rng.random() < TRANSFER_PROB:
                r["squad"], r["bank"] = _rival_transfer(r["squad"], r["bank"], info, raw, t, rng, selected)
            for p in r["squad"]:
                info.setdefault(p, info0.get(p))
            rival_weeks.append(_xi_and_captain(r["squad"], raw, t, info, rng, selected))
        eo = _league_eo(rival_weeks, [r["total"] for r in rivals], my_total)
        # the dial from the standings before this deadline
        totals = sorted([r["total"] for r in rivals], reverse=True)
        leader = totals[0]
        gap_now = leader - my_total if leader > my_total else -(my_total - leader)
        between = max(sum(1 for x in totals if x > my_total) - 1, 0)
        weeks_left = C.LAST_GW - t + 1
        if policy["kind"] == "dial":
            dial, _ = O.risk_dial(gap_now, weeks_left, between)
            value_fn = lambda rw: O.dial_adjusted(rw, eo, dial, policy["lam"], policy["cap"])  # noqa: E731
        else:
            dial = 0.0
            value_fn = lambda rw: O.strategy_adjusted(rw, eo, "balanced")  # noqa: E731
        dials.append(dial)
        weeks = ctx["weeks"]
        disc = {g: 0.9**i for i, g in enumerate(weeks)}
        for p in squad:
            info.setdefault(p, {"web_name": str(p), "position": season.position[p], "team": "?",
                                "now_cost": purchase[p], "status": "a", "chance_of_playing_next_round": None})
        pool = sorted(ctx["top"] | squad)
        raw_pool = {(p, g): raw.get((p, g), 0.0) for p in pool for g in weeks}
        value = value_fn(raw_pool)
        sell = {p: C.sell_value(purchase[p], info[p]["now_cost"]) if p in squad else info[p]["now_cost"] for p in pool}
        args = (pool, value, info, squad, sell, bank, ft, weeks, disc)
        no_hits = frozenset(weeks)
        kw = {"ft_value": 1.5}
        base = O.solve_horizon(*args, no_hit_weeks=no_hits, **kw)
        plan, _, _, _ = O.recommend_transfers(base, *args, raw_weekly_xp=raw_pool, uncertainty_k=None,
                                              no_hit_weeks=no_hits, gate_later_weeks=False, **kw)
        wk = plan["weeks"][0]
        for p in wk["transfers_out"]:
            bank += sell[p]
            purchase.pop(p, None)
        for p in wk["transfers_in"]:
            bank -= info[p]["now_cost"]
            purchase[p] = info[p]["now_cost"]
        squad = (squad - wk["transfers_out"]) | wk["transfers_in"]
        ft = min(O.MAX_FREE_TRANSFERS, ft - len(wk["transfers_in"]) + 1)
        pts = C.score_week(wk, raw, season, t, info)
        my_total += pts
        my_points += pts
        for r, rw in zip(rivals, rival_weeks):
            r["total"] += C.score_week(rw, raw, season, t, info)
    finals = sorted([r["total"] for r in rivals] + [my_total], reverse=True)
    rank = 1 + sum(1 for x in [r["total"] for r in rivals] if x > my_total)
    tied_first = my_total >= finals[0]
    return {"rank": rank, "first": 1.0 if tied_first else 0.0, "points": my_points, "mean_dial": float(np.mean(dials))}


def _job(job: tuple) -> tuple:
    label, policy, start, gap, league = job
    return label, start, gap, league, play(policy, start, gap, league)


def label_of(p: dict[str, Any]) -> str:
    return "balanced (today)" if p["kind"] == "balanced" else f"dial lam={p['lam']:g} cap={p['cap']:g}"


def run(policies: list[dict[str, Any]], season: str | None, workers: int) -> pd.DataFrame:
    jobs = [(label_of(p), p, s, g, lg) for p in policies for s in STARTS for g in GAPS for lg in range(N_LEAGUES)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(season,)) as ex:
        t0 = time.time()
        for i, (label, start, gap, league, r) in enumerate(ex.map(_job, jobs), start=1):
            rows.append({"policy": label, "start": start, "gap": gap, "league": league, **r})
            if i % 16 == 0:
                print(f"  progress {i}/{len(jobs)} leagues, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    base = df[df["policy"] == "balanced (today)"].set_index(["start", "gap", "league"])
    out = []
    for pol, d in df.groupby("policy"):
        d = d.set_index(["start", "gap", "league"])
        row = {"policy": pol, "P(first)": d["first"].mean(), "avg position": d["rank"].mean(),
               "avg points": d["points"].mean(), "avg dial": d["mean_dial"].mean()}
        if pol != "balanced (today)":
            for m, col in (("first", "P(first) diff"), ("rank", "position diff"), ("points", "points diff")):
                diff = (d[m] - base.loc[d.index, m]).to_numpy()
                row[col] = f"{diff.mean():+.3f}±{diff.std(ddof=1) / math.sqrt(len(diff)):.3f}"
        out.append(row)
    table = pd.DataFrame(out).set_index("policy")
    print(f"\n=== {name}: {len(df) // max(df['policy'].nunique(), 1)} leagues per policy "
          f"(starts {STARTS}, gaps {GAPS}, {N_LEAGUES} leagues each)")
    print(table.round(3).to_string())
    by_gap = df.pivot_table(index="gap", columns="policy", values="first", aggfunc="mean").round(3)
    print("P(first) by starting gap (leader minus you):")
    print(by_gap.to_string())
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["tune", "check"], default="tune")
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    base = {"kind": "balanced"}
    if args.stage == "tune":
        policies = [base] + [{"kind": "dial", "lam": lam, "cap": cap} for lam in (0.15, 0.3) for cap in (0.5, 1.0)]
        seasons = [None]
    else:
        policies = [base, {"kind": "dial", "lam": args.lam, "cap": args.cap}]
        seasons = ["2024-25", "2023-24"]
    for season in seasons:
        t0 = time.time()
        df = run(policies, season, args.workers)
        name = season or "2025-26"
        summarise(df, f"{name} ({time.time() - t0:.0f}s)")
        df.to_csv(B.TABLES_DIR / f"strategy_calibration_{args.stage}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
