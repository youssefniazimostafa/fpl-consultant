"""Step 13: calibrate fpl.optimise's free-transfer rules by replaying a whole
season of transfer decisions.

**Forecasts (no leakage).** For every deadline ``t`` of the 2025-26
backtest season (``fpl.backtest.load_backtest_season``) and each week of
the horizon ``t .. t+H-1``, xP is rebuilt with ``fpl.backtest``'s
machinery from rows with ``GW < t`` only (``build_features_asof(...,
history_before=t)``) -- the forecast a manager would have had at that
deadline, never a later result. Cached in
``data/tables/transfer_calibration_forecasts.parquet``.

**Replay.** Start from a sensible squad (the optimiser's own best 15 at
``START_GW`` on a GBP 100m budget, plus two alternative starting squads so
one lucky squad can't decide the result), then every later gameweek: 1
free transfer, banking up to 5, no hits (forbidden in every planned week),
no chips, the optimiser's XI and captain, scored on *actual* points with
FPL-style autosubs and vice-captain. Prices are each player's real value
at that deadline; selling price follows the FPL half-the-rise rule.

**Strategies.** ``current`` (today's optimiser: 0.9 discount, flat 1.75 xP
threshold), ``never`` (never transfer), ``flagged`` (only sell players
who missed their team's last two matches or left the league -- the
backtest has no injury-news feed, so that is the stand-in for "injured or
flagged"), and the Step 13 rules over a small grid.

**Residual SDs** (rule 2) come from the one-week-ahead residuals: in the
replay only gameweeks before each deadline (expanding window); the live
table (``residual_sd_table``) pools the full backtest season with this
season's live accuracy residuals (data/predictions/gw{N}.parquet vs
player_matches).

Run with ``uv run python -m fpl.transfer_calibration`` (``--quick`` for a
smaller grid).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl import backtest as B
from fpl import optimise as O
from fpl.xp import build_prior_season_rates, build_xp, find_prior_season_dir

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
FORECAST_CACHE = TABLES_DIR / "transfer_calibration_forecasts.parquet"
RESULTS_PATH = TABLES_DIR / "transfer_calibration_results.json"
WEEKLY_PATH = TABLES_DIR / "transfer_calibration_weekly.json"

HORIZON = 5                 # same as config.json's horizon
START_GW = 5                # GW2-4 give the first residuals for rule 2's expanding window
LAST_GW = 38
BUDGET = 1000               # GBP 100.0m in tenths
POOL_PER_POSITION = 20      # replay pool: squad + top 20 per position (live uses 40)
MIN_CELL_ROWS = 40          # residual cell smaller than this falls back to the position's pooled SD
FLAG_MISSED_MATCHES = 2     # "flagged": 0 minutes in each of his team's last 2 matches


# ---------------------------------------------------------------------------
# season data and leakage-free forecasts
# ---------------------------------------------------------------------------


def forecast_cache(season: str | None) -> Path:
    return FORECAST_CACHE if season is None else FORECAST_CACHE.with_name(f"transfer_calibration_forecasts_{season}.parquet")


def build_forecasts(horizon: int = HORIZON, season: str | None = None) -> pd.DataFrame:
    season_dir, gw = B.load_backtest_season(season)
    players_like = B.build_players_like(gw)
    pairs = B.build_team_match_pairs(gw)
    scoring = pd.read_parquet(TABLES_DIR / "scoring_rules.parquet")
    prior = build_prior_season_rates(
        gw.groupby("player_id")["name"].first(),
        season_dir=find_prior_season_dir(before=season_dir.name),
        cache_path=B.PRIOR_CACHE_PATH if season is None else TABLES_DIR / f"transfer_calibration_prior_map_{season}.csv",
    )
    out = []
    for t in range(B.MIN_BACKTEST_GW, LAST_GW + 1):
        for h in range(horizon):
            target = t + h
            if target > LAST_GW:
                break
            features = B.build_features_asof(gw, players_like, pairs, target, history_before=t)
            if features is None:
                continue
            xp = build_xp(features, scoring, players_like, gw_so_far=t - 1, prior=prior)
            xp["decision_gw"] = t
            out.append(xp)
    cols = ["decision_gw", "event", "fixture_id", "player_id", "position", "team_short", "xp", "p_play", "p_60",
            "appearance_pts", "expected_goal_pts", "expected_assist_pts", "clean_sheet_pts", "conceded_pts",
            "defcon_pts", "saves_pts", "bonus_pts", "card_pts"]
    return pd.concat(out, ignore_index=True)[cols]


def load_forecasts(rebuild: bool = False, season: str | None = None) -> pd.DataFrame:
    path = forecast_cache(season)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    t0 = time.time()
    df = build_forecasts(season=season)
    df.to_parquet(path, index=False)
    print(f"  built {len(df)} forecast rows in {time.time() - t0:.0f}s -> {path.name}")
    return df


class Season:
    """Everything the replay needs, indexed for fast per-deadline lookups."""

    def __init__(self, forecasts: pd.DataFrame, season: str | None = None):
        _, gw = B.load_backtest_season(season)
        self.scoring = pd.read_parquet(TABLES_DIR / "scoring_rules.parquet")
        self.position = gw.groupby("player_id")["position"].first().to_dict()
        self.name = gw.groupby("player_id")["name"].first().to_dict()
        per_gw = gw.groupby(["player_id", "GW"]).agg(
            points=("total_points", "sum"), minutes=("minutes", "sum"),
            value=("value", "last"), team=("team_name", "last"),
        ).reset_index()
        self.points = {(p, g): v for p, g, v in zip(per_gw.player_id, per_gw.GW, per_gw.points)}
        self.minutes = {(p, g): v for p, g, v in zip(per_gw.player_id, per_gw.GW, per_gw.minutes)}
        self.per_gw = per_gw.sort_values(["player_id", "GW"])
        # per-player match minutes (for the "flagged" stand-in): rows are team matches
        self.match_minutes = gw.sort_values("kickoff_time").groupby("player_id")[["GW", "minutes"]].apply(
            lambda d: list(zip(d["GW"], d["minutes"]))
        ).to_dict()
        f = forecasts
        self.forecasts = {t: d for t, d in f.groupby("decision_gw")}
        one_week = f[f["decision_gw"] == f["event"]]
        res = one_week.groupby(["player_id", "event"]).agg(xp=("xp", "sum"), position=("position", "first")).reset_index()
        res["actual"] = [self.points.get((p, g), 0) for p, g in zip(res.player_id, res.event)]
        self.residuals = res
        self._ctx: dict[int, dict[str, Any]] = {}

    def info_at(self, t: int) -> dict[int, dict[str, Any]]:
        known = self.per_gw[self.per_gw["GW"] <= t].groupby("player_id").last()
        return {
            int(p): {"web_name": self.name.get(p, str(p)), "position": self.position[p], "team": r.team,
                     "now_cost": int(r.value), "status": "a", "chance_of_playing_next_round": None}
            for p, r in known.iterrows()
        }

    def context(self, t: int) -> dict[str, Any]:
        if t in self._ctx:
            return self._ctx[t]
        weeks = list(range(t, min(t + HORIZON - 1, LAST_GW) + 1))
        f = self.forecasts[t]
        weekly = f.groupby(["player_id", "event"])["xp"].sum()
        raw = {(int(p), int(g)): float(v) for (p, g), v in weekly.items()}
        horizon_xp = weekly.groupby(level=0).sum()
        info = self.info_at(t)
        top: set[int] = set()
        for pos in O.POSITION_QUOTAS:
            ids = [p for p in horizon_xp.index if self.position.get(p) == pos and p in info]
            top |= set(horizon_xp.loc[ids].sort_values(ascending=False).head(POOL_PER_POSITION).index.astype(int))
        ctx = {
            "weeks": weeks, "raw": raw, "info": info, "top": top,
            "risk_sd": O.outcome_sd_by_week(f.rename(columns={}), self.scoring),
            "residual_sd": self.residual_table(before=t),
        }
        self._ctx[t] = ctx
        return ctx

    def residual_table(self, before: int | None = None) -> dict[str, tuple[float, ...]]:
        res = self.residuals if before is None else self.residuals[self.residuals["event"] < before]
        return residual_sd_from(res)

    def flagged(self, p: int, t: int) -> bool:
        recent = [(g, m) for g, m in self.match_minutes.get(p, []) if g < t]
        if not recent or recent[-1][0] < t - 3:
            return True  # left the league (no team match for 3 gameweeks)
        last = recent[-FLAG_MISSED_MATCHES:]
        return len(last) == FLAG_MISSED_MATCHES and all(m == 0 for _, m in last)


def residual_sd_from(res: pd.DataFrame) -> dict[str, tuple[float, ...]]:
    """SD of (actual - xp) per position and XP_BANDS band; a thin cell
    falls back to the band below it (the top band is the thin one), the
    lowest band to the position's pooled SD."""
    res = res.assign(err=res["actual"] - res["xp"])
    band = np.searchsorted(np.array(O.XP_BANDS), res["xp"].to_numpy(), side="right")
    res = res.assign(band=band)
    table = {}
    for pos in O.POSITION_QUOTAS:
        r = res[res["position"] == pos]
        pooled = float(r["err"].std()) if len(r) > 1 else 2.0
        cells = []
        for b in range(len(O.XP_BANDS) + 1):
            c = r[r["band"] == b]["err"]
            fallback = cells[-1] if cells else round(pooled, 2)
            cells.append(round(float(c.std()), 2) if len(c) >= MIN_CELL_ROWS else fallback)
        table[pos] = tuple(cells)
    return table


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def sell_value(purchase: int, now: int) -> int:
    return now if now < purchase else purchase + (now - purchase) // 2


def score_week(plan_week: dict[str, Any], raw: dict, season: Season, g: int, info: dict) -> float:
    """Actual points for one gameweek: XI with FPL autosubs (bench in xP
    order, keeper for keeper, formation kept valid), captain x2 or the
    vice if the captain didn't play."""
    xi = set(plan_week["xi_ids"])
    bench = sorted(plan_week["squad_ids"] - xi, key=lambda p: (info[p]["position"] != "GKP", -raw.get((p, g), 0.0)))
    played = lambda p: season.minutes.get((p, g), 0) > 0  # noqa: E731
    for b in bench:
        if not played(b):
            continue
        for m in sorted(p for p in xi if not played(p)):
            if (info[m]["position"] == "GKP") != (info[b]["position"] == "GKP"):
                continue
            trial = (xi - {m}) | {b}
            counts = {pos: sum(1 for p in trial if info[p]["position"] == pos) for pos in O.FORMATION_BOUNDS}
            if all(lo <= counts[pos] <= hi for pos, (lo, hi) in O.FORMATION_BOUNDS.items()):
                xi = trial
                break
    total = sum(season.points.get((p, g), 0) for p in xi)
    cap = plan_week["captain_id"]
    others = [p for p in plan_week["xi_ids"] if p != cap]
    vice = max(others, key=lambda p: raw.get((p, g), 0.0)) if others else None
    if cap is not None and played(cap):
        total += season.points.get((cap, g), 0)
    elif vice is not None and played(vice) and vice in xi:
        total += season.points.get((vice, g), 0)
    return float(total)


def initial_squad(season: Season, variant: int) -> set[int]:
    """Variant 0: the optimiser's best 15 for START_GW's 5-week horizon.
    1: the same with its 3 highest-xP picks banned. 2: best 15 for START_GW
    alone (a short-term squad)."""
    ctx = season.context(START_GW)
    pool = sorted(ctx["top"])
    weeks = ctx["weeks"] if variant != 2 else [START_GW]
    value = {k: v for k, v in ctx["raw"].items()}
    value = {(p, g): value.get((p, g), 0.0) for p in pool for g in weeks}
    disc = {g: 0.9**i for i, g in enumerate(weeks)}

    def solve(banned: set[int]) -> set[int]:
        ids = [p for p in pool if p not in banned]
        prices = {p: ctx["info"][p]["now_cost"] for p in ids}
        plan = O.solve_horizon(ids, value, ctx["info"], set(), prices, BUDGET, O.UNLIMITED_FREE_TRANSFERS, weeks, disc,
                               held_weeks=frozenset(weeks[1:]), no_hit_weeks=frozenset(weeks))
        return plan["weeks"][0]["squad_ids"]

    squad = solve(set())
    if variant == 1:
        best = sorted(squad, key=lambda p: -sum(value.get((p, g), 0.0) for g in weeks))[:3]
        squad = solve(set(best))
    return squad


def replay(strategy: dict[str, Any], season: Season, variant: int) -> dict[str, Any]:
    kind = strategy["kind"]
    squad = set(initial_squad(season, variant))
    info0 = season.info_at(START_GW)
    purchase = {p: info0[p]["now_cost"] for p in squad}
    bank = BUDGET - sum(purchase.values())
    ft = 1
    weekly_points, n_transfers = {}, 0
    for t in range(START_GW, LAST_GW + 1):
        ctx = season.context(t)
        info, raw = ctx["info"], ctx["raw"]
        weeks = ctx["weeks"] if kind not in ("never",) else [t]
        delta = strategy.get("discount", 0.9)
        disc = {g: delta**i for i, g in enumerate(weeks)}
        for p in squad:  # a player who left still has a row in info (last known)
            info.setdefault(p, {"web_name": str(p), "position": season.position[p], "team": "?", "now_cost": purchase[p],
                                "status": "a", "chance_of_playing_next_round": None})
        pool = sorted(ctx["top"] | squad)
        value = {(p, g): raw.get((p, g), 0.0) for p in pool for g in weeks}
        sell = {p: sell_value(purchase[p], info[p]["now_cost"]) if p in squad else info[p]["now_cost"] for p in pool}
        no_hits = frozenset(weeks)
        args = (pool, value, info, squad, sell, bank, ft, weeks, disc)

        if t == START_GW or kind == "never":
            plan = O.solve_horizon(*args, held_weeks=frozenset(weeks))
        elif kind == "flagged":
            flagged = {p for p in squad if season.flagged(p, t)}
            locked = {p: (sell[p] if p in flagged else -10**6) for p in pool}  # unflagged can't be sold
            largs = (pool, value, info, squad, locked, bank, ft, weeks, disc)
            base = O.solve_horizon(*largs, held_weeks=frozenset(weeks[1:]), no_hit_weeks=no_hits)
            # same gate as today's optimiser, so a benched backup keeper
            # (always "missing" matches) isn't swapped for nothing
            plan, _, _, _ = O.recommend_transfers(base, *largs, uncertainty_k=None, no_hit_weeks=no_hits)
        else:
            rho = strategy.get("risk", 0.0)
            kw = {"ft_value": strategy.get("ft_value", 0.0), "risk_sd": ctx["risk_sd"] if rho else None,
                  "risk_weight": rho}
            O.RESIDUAL_SD = ctx["residual_sd"]  # expanding window: residuals before t only
            base = O.solve_horizon(*args, no_hit_weeks=no_hits, **kw)
            plan, _, _, _ = O.recommend_transfers(
                base, *args, raw_weekly_xp=raw, uncertainty_k=strategy.get("k"), no_hit_weeks=no_hits, **kw
            )

        wk = plan["weeks"][0]
        ins, outs = wk["transfers_in"], wk["transfers_out"]
        for p in outs:
            bank += sell[p]
            purchase.pop(p, None)
        for p in ins:
            bank -= info[p]["now_cost"]
            purchase[p] = info[p]["now_cost"]
        squad = (squad - outs) | ins
        assert len(squad) == 15 and bank >= 0, (t, len(squad), bank)
        if t > START_GW:
            n_transfers += len(ins)
            ft = min(O.MAX_FREE_TRANSFERS, ft - len(ins) + 1)
        weekly_points[t] = score_week(wk, raw, season, t, info)
    scored = {g: v for g, v in weekly_points.items() if g > START_GW}
    return {"points": sum(scored.values()), "transfers": n_transfers, "weekly": weekly_points}


# ---------------------------------------------------------------------------
# grid search (parallel)
# ---------------------------------------------------------------------------

_SEASON: Season | None = None


def _init_worker(season: str | None = None) -> None:
    global _SEASON
    _SEASON = Season(load_forecasts(season=season), season)


def _run(job: tuple[str, dict[str, Any], int]) -> tuple[str, int, dict[str, Any]]:
    label, strategy, variant = job
    assert _SEASON is not None
    return label, variant, replay(strategy, _SEASON, variant)


def label_of(s: dict[str, Any]) -> str:
    if s["kind"] != "rules":
        return s["kind"]
    k = "legacy1.75" if s.get("k") is None else f"k={s['k']:g}"
    return f"V={s.get('ft_value', 0):g} {k} rho={s.get('risk', 0):g} d={s.get('discount', 0.9):g}"


def run_grid(
    strategies: list[dict[str, Any]], variants: tuple[int, ...], workers: int, season: str | None = None
) -> pd.DataFrame:
    jobs = [(label_of(s), s, v) for s in strategies for v in variants]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(season,)) as ex:
        for label, variant, r in ex.map(_run, jobs):
            rows.append({"strategy": label, "variant": variant, "points": r["points"], "transfers": r["transfers"],
                         "weekly": r["weekly"]})
    df = pd.DataFrame(rows)
    weekly = {(r["strategy"], r["variant"]): r["weekly"] for r in rows}
    WEEKLY_PATH.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in weekly.items()}))
    wide = df.pivot_table(index="strategy", columns="variant", values="points")
    wide.columns = [f"squad{c}" for c in wide.columns]
    wide["mean_points"] = wide.mean(axis=1)
    wide["mean_transfers"] = df.groupby("strategy")["transfers"].mean()
    return wide.sort_values("mean_points", ascending=False)


def print_paired_se(table: pd.DataFrame, squads: list[str]) -> None:
    """Each strategy minus ``current``, per starting squad: mean difference
    and its paired standard error (SD of the per-squad differences /
    sqrt(n squads))."""
    if "current" not in table.index or len(squads) < 1:
        return
    print("\nvs current (paired over starting squads):")
    for label, row in table.iterrows():
        if label == "current":
            continue
        d = np.array([row[c] - table.loc["current", c] for c in squads])
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  {label:32s} diff {d.mean():+7.1f}  SE {se:6.1f}  per squad {np.round(d, 0).tolist()}")


def residual_sd_table() -> dict[str, tuple[float, ...]]:
    """The live RESIDUAL_SD: full-season one-week-ahead backtest residuals
    pooled with this season's live residuals."""
    season = Season(load_forecasts())
    res = [season.residuals[["position", "xp", "actual"]]]
    pm_path = TABLES_DIR / "player_matches.parquet"
    if pm_path.exists():
        pm = pd.read_parquet(pm_path)
        actual = pm.groupby(["player_id", "gameweek"])["total_points"].sum()
        for f in sorted((ROOT / "data" / "predictions").glob("gw*.parquet")):
            pred = pd.read_parquet(f)
            g = int(f.stem[2:])
            pred = pred[pred["event"] == g]
            if pred.empty or g not in set(pm["gameweek"]):
                continue
            pp = pred.groupby("player_id").agg(xp=("xp", "sum"), position=("position", "first")).reset_index()
            pp["actual"] = [actual.get((p, g), 0) for p in pp.player_id]
            res.append(pp[["position", "xp", "actual"]])
    return residual_sd_from(pd.concat(res, ignore_index=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached forecasts")
    ap.add_argument("--stage", choices=["1", "2", "3", "baselines", "residuals", "oos", "sens"], default="1")
    ap.add_argument("--season", default=None, help="replay this season instead (e.g. 2024-25)")
    ap.add_argument("--variants", default="0,1,2", help="starting squads to replay")
    ap.add_argument("--sweep", choices=["k", "V"], default="k", help="stage sens: which rule to sweep")
    ap.add_argument("--V", type=float, default=0.0)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--discount", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()

    load_forecasts(rebuild=args.rebuild, season=args.season)
    variants = tuple(int(v) for v in args.variants.split(","))
    base = [{"kind": "current"}, {"kind": "never"}, {"kind": "flagged"}]
    if args.stage == "residuals":
        print(json.dumps(residual_sd_table()))
        return
    if args.stage == "baselines":
        grid = base
    elif args.stage == "1":   # saved-FT value x margin, today's discount, no risk
        grid = base + [
            {"kind": "rules", "ft_value": V, "k": k, "discount": 0.9}
            for V, k in itertools.product([0, 0.5, 1, 1.5, 2, 3], [None, 0.0, 0.1, 0.2, 0.3, 0.5])
            if not (V == 0 and k is None)  # = current
        ]
    elif args.stage == "oos":  # the chosen settings, unchanged, each rule alone, neighbours, baselines
        tuned = {"kind": "rules", "ft_value": args.V, "k": args.k, "discount": args.discount, "risk": args.rho}
        grid = base + [
            tuned,
            {**tuned, "k": None},                      # rule 1 only (legacy threshold)
            {**tuned, "ft_value": 0.0},                # rule 2 only
            {**tuned, "ft_value": args.V - 0.5},       # neighbours
            {**tuned, "k": round((args.k or 0) - 0.1, 2)},
            {**tuned, "k": round((args.k or 0) + 0.2, 2)},
        ]
    elif args.stage == "sens":  # one rule alone: k sweep (V fixed) or, with --k, V sweep
        if args.sweep == "k":
            grid = [{"kind": "current"}] + [
                {"kind": "rules", "ft_value": args.V, "k": k, "discount": args.discount} for k in [0.0, 0.1, 0.2, 0.3, 0.5]
            ]
        else:
            grid = [{"kind": "current"}] + [
                {"kind": "rules", "ft_value": V, "k": args.k, "discount": args.discount} for V in [0.5, 1.0, 1.5, 2.0, 3.0]
            ]
    elif args.stage == "2":   # discount x reliability at the stage-1 choice
        grid = [
            {"kind": "rules", "ft_value": args.V, "k": args.k, "discount": d, "risk": r}
            for d, r in itertools.product([0.7, 0.8, 0.85, 0.9, 0.95, 1.0], [0, 0.05, 0.1, 0.2])
        ]
    else:                     # stage 3: re-check V and k at the stage-2 discount/risk
        grid = [
            {"kind": "rules", "ft_value": V, "k": k, "discount": args.discount, "risk": args.rho}
            for V, k in itertools.product([0, 0.5, 1, 1.5, 2, 3], [None, 0.0, 0.1, 0.2, 0.3])
        ]
    t0 = time.time()
    global WEEKLY_PATH
    WEEKLY_PATH = WEEKLY_PATH.with_name(f"transfer_calibration_weekly_stage{args.stage}.json")
    table = run_grid(grid, variants, args.workers, args.season)
    print(table.round(1).to_string())
    print_paired_se(table, [c for c in table.columns if c.startswith("squad")])
    print(f"({len(grid)} strategies x {len(variants)} starting squads in {time.time() - t0:.0f}s)")
    tag = args.stage + (f"_{args.season}" if args.season else "")
    RESULTS_PATH.with_name(f"transfer_calibration_stage{tag}.csv").write_text(table.round(2).to_csv())


if __name__ == "__main__":
    main()
