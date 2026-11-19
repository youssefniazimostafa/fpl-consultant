"""Step 17: test the goal-aware close-call rule (fpl/goals.py) against
today's balanced mode (ownership tie-breaker lambda = 0.02) in the simulated
10-manager leagues of the risk-dial test (fpl/strategy_calibration.py).

Same leagues, same rivals and same seeds as that test: seasons 2025-26,
2024-25 and 2023-24; start at GW12 or GW26; you lead by 10, or trail the
leader by 10, 30 or 60. N_LEAGUES leagues per (start, gap) cell, so
2 x 4 x 25 = 200 leagues per season. Each league is played once by the
rivals (their moves never depend on yours) and replayed for you under both
policies, so every comparison is paired.

* balanced (today): xP x (1 + 0.02 x (0.5 - league EO)), the optimiser's
  own transfer and captain.
* goal-aware: plain xP (no ownership term). Transfer options as in the
  app (the optimiser's move or hold, the next-best moves with its incoming
  players left out, and hold; 3 in all) and every XI captain go through the
  1-SD close-call test; close ones are simulated against the rivals' last
  confirmed XI and captain (your view before the deadline), and the best
  weighted goal chance wins only if it beats the expected-points pick by more
  than 2 simulation SEs.

Goals and thresholds are the Settings defaults, fixed before running:
finish 1st (weight 2) and top 2 (weight 1), by the end of the season; close =
1 SD. Nothing is tuned. Pass rule: see fpl.goals (USE_GOAL_AWARE).

Run with ``uv run python -m fpl.goal_calibration [--season 2024-25]``.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from fpl import backtest as B
from fpl import goals as G
from fpl import optimise as O
from fpl import strategy_calibration as S
from fpl import transfer_calibration as C

N_LEAGUES = 25
N_OPTIONS = 3
FT_VALUE = O.FT_VALUE_XP
POOLED_WEEK_SD = 16.5   # mean per-manager weekly-score SD in your real league (league_history, 10 managers)
MIN_WEEKS_MEAN = 3
MIN_WEEKS_SD = 5
GOALS = G.DEFAULT_GOALS["goals"]
TARGET = C.LAST_GW


def _team_xp(team: dict[str, Any], raw: dict, t: int) -> float:
    pts = sum(raw.get((p, t), 0.0) for p in team["xi_ids"])
    return pts + (raw.get((team["captain_id"], t), 0.0) if team["captain_id"] is not None else 0.0)


def _rest(history: list[float], fallback_mean: float) -> tuple[float, float]:
    mean = float(np.mean(history)) if len(history) >= MIN_WEEKS_MEAN else fallback_mean
    sd = max(float(np.std(history, ddof=1)), G.REST_MIN_SD) if len(history) >= MIN_WEEKS_SD else POOLED_WEEK_SD
    return mean, sd


def _rivals(start: int, gap: int, league: int) -> dict[str, Any]:
    """The rivals' whole season (identical to strategy_calibration.play's
    rng sequence) and your starting squad."""
    season: C.Season = S._S["season"]
    selected = S._S["selected"]
    rng = np.random.default_rng(1000 * start + 10 * (gap + 100) + league)
    ctx0 = season.context(start)
    info0, raw0 = ctx0["info"], ctx0["raw"]
    offsets = [gap] + list(rng.uniform(-40, max(gap, -10), S.RIVALS - 1))
    if gap < 0:
        offsets = [gap] + list(rng.uniform(-40, gap, S.RIVALS - 1))
    rivals = []
    for off in offsets:
        sq = S._rival_squad(ctx0, start, rng, selected)
        rivals.append({"squad": sq, "bank": C.BUDGET - sum(info0[p]["now_cost"] for p in sq), "total": float(off)})
    weeks: dict[int, dict[str, Any]] = {}
    for t in range(start, C.LAST_GW + 1):
        ctx = season.context(t)
        info, raw = ctx["info"], ctx["raw"]
        rw = []
        for r in rivals:
            if t > start and rng.random() < S.TRANSFER_PROB:
                r["squad"], r["bank"] = S._rival_transfer(r["squad"], r["bank"], info, raw, t, rng, selected)
            for p in r["squad"]:
                info.setdefault(p, info0.get(p))
            rw.append(S._xi_and_captain(r["squad"], raw, t, info, rng, selected))
        before = [r["total"] for r in rivals]
        pts = [C.score_week(w, raw, season, t, info) for w in rw]
        for r, x in zip(rivals, pts):
            r["total"] += x
        weeks[t] = {"teams": rw, "totals_before": before, "points": pts}

    pool = sorted(ctx0["top"])
    weeks0 = ctx0["weeks"]
    value0 = {(p, g): raw0.get((p, g), 0.0) for p in pool for g in weeks0}
    prices = {p: info0[p]["now_cost"] for p in pool}
    w0 = O.solve_horizon(pool, value0, info0, set(), prices, C.BUDGET, O.UNLIMITED_FREE_TRANSFERS, weeks0,
                         {g: 0.9**i for i, g in enumerate(weeks0)}, held_weeks=frozenset(weeks0[1:]),
                         no_hit_weeks=frozenset(weeks0))
    return {"weeks": weeks, "squad": set(w0["weeks"][0]["squad_ids"]), "info0": info0}


def _goal_week(
    plan: dict[str, Any], args: tuple, kw: dict, raw_pool: dict, info: dict, ctx: dict, t: int,
    rival_view: list[dict[str, Any]], me: dict[str, Any], stats: dict[str, int],
) -> dict[str, Any]:
    """The goal-aware transfer and captain choice for one week."""
    pool, value, _, squad, sell, bank, ft, weeks, disc = args
    position = {p: info[p]["position"] for p in pool}
    table = ctx["residual_sd"]
    rest_weeks = TARGET - t

    # --- transfer options: the app's three ---
    rec_moves = bool(plan["weeks"][0]["transfers_in"])
    options = [plan]
    banned = set(plan["weeks"][0]["transfers_in"])
    for _ in range(N_OPTIONS - 1 - (1 if rec_moves else 0)):
        ids = [p for p in pool if p not in banned or p in squad]
        alt = O.solve_horizon(ids, value, info, squad, sell, bank, ft, weeks, disc,
                              no_hit_weeks=frozenset(weeks), **kw)
        if alt["status"] != "Optimal" or not alt["weeks"][0]["transfers_in"]:
            break
        options.append(alt)
        banned |= set(alt["weeks"][0]["transfers_in"])
    if rec_moves:
        options.append(O.solve_horizon(*args, held_weeks=frozenset({t}), no_hit_weeks=frozenset(weeks), **kw))
    # an option whose solve returned no solution (time limit) can't be compared
    options = [plan] + [o for o in options[1:] if o["status"] == "Optimal" and len(o["weeks"][0]["xi_ids"]) == 11]
    vals = [G.plan_value(o, raw_pool, disc, ft, FT_VALUE) for o in options]
    close = [0] + [i for i in range(1, len(options))
                   if vals[0] - vals[i] <= G.CLOSE_SD * G.plan_diff_sd(options[0], options[i], raw_pool, disc, position, table)]
    chosen = plan
    if len(close) > 1:
        stats["transfer_close"] += 1
        ref_future = G.future_points(plan, raw_pool, TARGET) + FT_VALUE * G.ft_end(plan, ft)
        opts = []
        for i in close:
            w0 = options[i]["weeks"][0]
            fut = G.future_points(options[i], raw_pool, TARGET) + FT_VALUE * G.ft_end(options[i], ft)
            opts.append({"xi_ids": sorted(w0["xi_ids"]), "captain_id": w0["captain_id"], "hits": 0,
                         "future_delta": fut - ref_future})
        xp_map = {p: raw_pool.get((p, t), 0.0) for o in opts for p in o["xi_ids"]}
        xp_map.update({p: ctx["raw"].get((p, t), 0.0) for r in rival_view for p in r["xi_ids"]})
        sd_map = {p: G.error_sd(S._S["season"].position[p], xp_map[p], table) for p in xp_map}
        sim = G.simulate(opts, rival_view, me, xp_map, sd_map, GOALS, rest_weeks, ref=0, seed=t)
        pick, verdict = G.choose(sim, 0)
        stats["draws"] += sim["draws"]
        stats[f"transfer_{verdict.replace(' ', '_')}"] += 1
        chosen = options[close[pick]]

    # --- captain within the chosen XI ---
    wk = dict(chosen["weeks"][0])
    xi = sorted(wk["xi_ids"])
    if len(xi) != 11:
        return wk
    xp_t = {p: raw_pool.get((p, t), 0.0) for p in xi}
    best, rows = G.captain_close_calls(xi, xp_t, position, table)
    cands = [r["player_id"] for r in rows if r["close"]]
    wk["captain_id"] = best
    if len(cands) > 1:
        stats["captain_close"] += 1
        opts = [{"xi_ids": xi, "captain_id": c, "hits": 0, "future_delta": 0.0} for c in cands]
        xp_map = dict(xp_t)
        xp_map.update({p: ctx["raw"].get((p, t), 0.0) for r in rival_view for p in r["xi_ids"]})
        sd_map = {p: G.error_sd(S._S["season"].position[p], xp_map[p], table) for p in xp_map}
        sim = G.simulate(opts, rival_view, me, xp_map, sd_map, GOALS, rest_weeks, ref=0, seed=1000 + t)
        pick, verdict = G.choose(sim, 0)
        stats["draws"] += sim["draws"]
        stats[f"captain_{verdict.replace(' ', '_')}"] += 1
        wk["captain_id"] = cands[pick]
    return wk


def play(policy: str, league_data: dict[str, Any], start: int) -> dict[str, Any]:
    season: C.Season = S._S["season"]
    weeks_data = league_data["weeks"]
    info0 = league_data["info0"]
    squad = set(league_data["squad"])
    purchase = {p: info0[p]["now_cost"] for p in squad}
    bank = C.BUDGET - sum(purchase.values())
    ft, my_total, my_points = 1, 0.0, 0.0
    my_hist: list[float] = []
    rival_hist: list[list[float]] = [[] for _ in range(S.RIVALS)]
    stats = {k: 0 for k in ("transfer_close", "captain_close", "draws", "transfer_goals", "transfer_agree",
                            "transfer_too_close", "captain_goals", "captain_agree", "captain_too_close")}
    O.TRANSFER_BAR_MODE = "per_week"
    for t in range(start, C.LAST_GW + 1):
        ctx = season.context(t)
        info, raw = ctx["info"], ctx["raw"]
        wd = weeks_data[t]
        weeks = ctx["weeks"]
        disc = {g: 0.9**i for i, g in enumerate(weeks)}
        for p in squad:
            info.setdefault(p, {"web_name": str(p), "position": season.position[p], "team": "?",
                                "now_cost": purchase[p], "status": "a", "chance_of_playing_next_round": None})
        pool = sorted(ctx["top"] | squad)
        raw_pool = {(p, g): raw.get((p, g), 0.0) for p in pool for g in weeks}
        if policy == "balanced":
            eo = S._league_eo(wd["teams"], wd["totals_before"], my_total)
            value = O.strategy_adjusted(raw_pool, eo, "balanced")
        else:
            value = raw_pool
        sell = {p: C.sell_value(purchase[p], info[p]["now_cost"]) if p in squad else info[p]["now_cost"] for p in pool}
        args = (pool, value, info, squad, sell, bank, ft, weeks, disc)
        no_hits = frozenset(weeks)
        kw = {"ft_value": FT_VALUE}
        base = O.solve_horizon(*args, no_hit_weeks=no_hits, **kw)
        plan, _, _, _ = O.recommend_transfers(base, *args, raw_weekly_xp=raw_pool, uncertainty_k=None,
                                              no_hit_weeks=no_hits, gate_later_weeks=False, **kw)
        if policy == "balanced":
            wk = plan["weeks"][0]
        else:
            # what you can see before the deadline: rivals' last confirmed teams
            seen = weeks_data[t - 1]["teams"] if t - 1 in weeks_data else wd["teams"]
            rival_view = []
            for i, team in enumerate(seen):
                mean, sd = _rest(rival_hist[i], _team_xp(team, raw, t))
                rival_view.append({"xi_ids": sorted(team["xi_ids"]), "captain_id": team["captain_id"],
                                   "total": wd["totals_before"][i], "rest_mean": mean, "rest_sd": sd})
            my_xp = _team_xp(plan["weeks"][0], raw, t)
            mean, sd = _rest(my_hist, my_xp)
            me = {"total": my_total, "rest_mean": mean, "rest_sd": sd}
            wk = _goal_week(plan, args, kw, raw_pool, info, ctx, t, rival_view, me, stats)
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
        my_hist.append(pts)
        for i, x in enumerate(wd["points"]):
            rival_hist[i].append(x)
    rivals_final = [wd_tot + 0 for wd_tot in (weeks_data[C.LAST_GW]["totals_before"][i] + weeks_data[C.LAST_GW]["points"][i]
                                                for i in range(S.RIVALS))]
    above = sum(1 for x in rivals_final if x > my_total)
    return {"rank": 1 + above, "first": float(above == 0), "top2": float(above < 2), "points": my_points, **stats}


def _job(job: tuple) -> list[dict[str, Any]]:
    start, gap, league = job
    data = _rivals(start, gap, league)
    out = []
    for policy in ("balanced", "goal"):
        out.append({"policy": policy, "start": start, "gap": gap, "league": league, **play(policy, data, start)})
    return out


def run(season: str | None, workers: int, n_leagues: int) -> pd.DataFrame:
    jobs = [(s, g, lg) for s in S.STARTS for g in S.GAPS for lg in range(n_leagues)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=S._init, initargs=(season,)) as ex:
        t0 = time.time()
        for i, r in enumerate(ex.map(_job, jobs), start=1):
            rows.extend(r)
            if i % 10 == 0:
                print(f"  progress {i}/{len(jobs)} leagues, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, name: str) -> dict[str, Any]:
    w = np.array([g["weight"] for g in GOALS], dtype=float)
    w = w / w.sum()
    df = df.assign(goal=w[0] * df["first"] + w[1] * df["top2"])
    key = ["start", "gap", "league"]
    b = df[df["policy"] == "balanced"].set_index(key).sort_index()
    g = df[df["policy"] == "goal"].set_index(key).sort_index()
    n = len(b)
    res: dict[str, Any] = {"season": name, "leagues": n}
    for m in ("first", "top2", "goal", "points", "rank"):
        d = (g[m] - b[m]).to_numpy()
        res[m] = {"balanced": float(b[m].mean()), "goal_aware": float(g[m].mean()),
                  "diff": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(n))}
    res["pass"] = bool(
        res["goal"]["diff"] > res["goal"]["se"]
        and res["first"]["diff"] > -res["first"]["se"]
        and res["top2"]["diff"] > -res["top2"]["se"]
        and res["points"]["diff"] > -res["points"]["se"]
    )
    weeks = (C.LAST_GW - g.reset_index()["start"] + 1).sum()
    res["usage"] = {k: int(g[k].sum()) for k in ("captain_close", "captain_goals", "captain_agree", "captain_too_close",
                                                 "transfer_close", "transfer_goals", "transfer_agree", "transfer_too_close")}
    res["usage"]["weeks"] = int(weeks)
    res["usage"]["mean_draws_per_sim"] = float(g["draws"].sum() / max(g["captain_close"].sum() + g["transfer_close"].sum(), 1))
    by_gap = []
    for gap in S.GAPS:
        bb, gg = b.xs(gap, level="gap"), g.xs(gap, level="gap")
        row = {"gap": gap}
        for m in ("first", "top2", "points"):
            d = (gg[m] - bb[m]).to_numpy()
            row[m] = f"{bb[m].mean():.3f} -> {gg[m].mean():.3f} ({d.mean():+.3f}±{d.std(ddof=1) / math.sqrt(len(d)):.3f})"
        by_gap.append(row)
    res["by_gap"] = by_gap
    print(f"\n=== {name}: {n} leagues, paired ===")
    for m, label in (("first", "P(1st)"), ("top2", "P(top 2)"), ("goal", "weighted goal"), ("points", "avg points"), ("rank", "avg position")):
        r = res[m]
        print(f"  {label:14s} balanced {r['balanced']:.3f}  goal-aware {r['goal_aware']:.3f}  diff {r['diff']:+.3f} ± {r['se']:.3f}")
    print(f"  usage: {res['usage']}")
    print("  by starting gap (leader minus you):")
    print(pd.DataFrame(by_gap).to_string(index=False))
    print(f"  PASS (fixed rule): {res['pass']}")
    return res


def main() -> None:
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", action="append", help="2025-26 (default), 2024-25, 2023-24; repeatable")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--leagues", type=int, default=N_LEAGUES, help="leagues per (start, gap) cell")
    args = ap.parse_args()
    seasons = args.season or ["2025-26", "2024-25", "2023-24"]
    out = []
    for s in seasons:
        t0 = time.time()
        df = run(None if s == "2025-26" else s, args.workers, args.leagues)
        df.to_csv(B.TABLES_DIR / f"goal_calibration_{s}.csv", index=False)
        res = summarise(df, f"{s} ({time.time() - t0:.0f}s)")
        res["season"] = s
        out.append(res)
    path = B.TABLES_DIR / "goal_calibration_results.json"
    if args.leagues == N_LEAGUES:
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
