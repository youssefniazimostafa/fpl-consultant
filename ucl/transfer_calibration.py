"""Calibrate the UCL transfer bar (ucl.optimise.TRANSFER_FRICTION) and a
saved-transfer value (ucl.optimise.FT_VALUE), the same way as the PL side
(fpl/transfer_calibration.py): a season replay.

Replay: 2025/26 MD1-17, then 2026/27 MD1-4 as the unseen check. Each
matchday the model is fitted only on matches before it (ucl.model.predict
with as_of_md), the optimiser picks the team with the config being tested,
and the team scores its real points (best of the XI with automatic
substitutions, captain doubled, vice-captain if the captain didn't play,
minus hits). UCL rules throughout (ucl.rules): new team free at MD1,
unlimited transfers before MD9 and MD11 (team rebuilt within the budget),
2 free transfers in the league phase with 1 carried over, 2/3/5/3/5/3/5 in
the knockouts with no carry-over, -4 per extra transfer, club limits and
budget by stage. No chips, no injury news (not stored historically).

No lookahead in the fixtures either: in the league phase every league
matchday's fixtures were known from the draw; in the knockouts the plan only
looks at the current round (the next round's opponents weren't known).
Prices: each player's UEFA price at that matchday where stored, else the
nearest known one.

Decision rule, fixed before running: pick the config with the most points on
2025/26; replace today's 0.5 bar (FT value 0) only if that config also scores
more points than 0.5 on 2026/27 MD1-4. (That unseen check is small: four
matchdays. The bar also shapes the first squad, since the optimiser plans
MD1 with the later matchdays' transfer costs in view.)

Result (2026-11-15, season points, no chips):
                2025/26 (FT value 0 / 1 / 2)    2026/27 MD1-4 (0 / 1 / 2)
  bar 0.5        1109 / 1091 / 1114               254 / 244 / 241
  bar 1.0        1100 / 1071 / 1081               256 / 266 / 241
  bar 1.5        1072 / 1070 / 1079               240 / 266 / 241
  bar 2.0        1065 / 1062 / 1074               268 / 297 / 260
  bar 3.0        1048 / 1076 / 1062               291 / 297 / 279
  bar 4.0        1071 / 1053 / 1042               291 / 297 / 287
Best on 2025/26: bar 0.5 + FT value 2 (1114, +5 on 0.5/0), but 241 vs 254
on 2026/27 -> rule says keep bar 0.5, FT value 0. The two seasons point
opposite ways (high bars best on the 4 unseen matchdays, worst on 2025/26),
so neither direction is established.

Re-run 2026-11-17 with the refined model (ucl.model.CFG: Elo team-strength
prior; predictions are cached per model setting in data/ucl/raw/calib/):
                2025/26 (FT value 0 / 1 / 2)    2026/27 MD1-4 (0 / 1 / 2)
  bar 0.5        1082 / 1116 / 1114               265 / 229 / 224
  bar 1.0        1089 / 1117 / 1112               262 / 268 / 254
  bar 1.5        1084 / 1103 / 1103               262 / 268 / 256
  bar 2.0        1068 / 1069 / 1072               268 / 256 / 256
  bar 3.0        1061 / 1069 / 1054               258 / 249 / 256
  bar 4.0        1030 / 1046 / 1028               246 / 246 / 241
Best on 2025/26: bar 1.0 + FT value 1 (1117, +35 on 0.5/0), and 268 vs 265
on 2026/27 -> rule says adopt bar 1.0, FT value 1 (now in ucl.optimise).
The unseen margin is only 3 points over four matchdays.

Run with ``uv run python -m ucl.transfer_calibration``; writes
data/ucl/transfer_calibration.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ucl import feed, model, optimise, rules

OUT = feed.DATA / "transfer_calibration.json"
CACHE = feed.RAW / "calib"
BARS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
FT_VALUES = [0.0, 1.0, 2.0]
BASELINE = (0.5, 0.0)


def known_mds(tour: int, md: int, matches: pd.DataFrame) -> list[int]:
    """The planning horizon from md with only fixtures known at the time."""
    h = optimise.horizon(md)
    if rules.STAGE[md] != "league":
        h = [m for m in h if rules.STAGE[m] == rules.STAGE[md]]
    have = set(matches["md"])
    return [m for m in h if m in have]


def predictions(tour: int, md: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    import hashlib  # the cache is per model setting, so a model change can't reuse old predictions

    tag = hashlib.sha1(json.dumps(model.CFG, sort_keys=True).encode()).hexdigest()[:8]
    f = CACHE / f"xp_{tour}_{md}_{tag}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    m, _, _ = model.load(tour)
    xp, _ = model.predict(tour, as_of_md=md, mds=known_mds(tour, md, m), use_status=False)
    xp.to_parquet(f, index=False)
    return xp


def prices_at(tour: int) -> dict[int, dict[int, float]]:
    _, players, apps = model.load(tour)
    a = pd.read_parquet(model.HIST / f"appearances_{tour}.parquet")  # includes listed-but-unplayed rows
    out: dict[int, dict[int, float]] = {}
    for pid, g in a.dropna(subset=["price"]).groupby("pid"):
        out[int(pid)] = dict(zip(g["md"].astype(int), g["price"].astype(float)))
    return out


def price_for(pmap: dict[int, float] | None, md: int, fallback: float) -> float:
    if not pmap:
        return fallback
    before = [m for m in pmap if m <= md]
    m = max(before) if before else min(pmap)
    return pmap[m]


def actual_points(tour: int) -> pd.DataFrame:
    _, _, apps = model.load(tour)
    return apps


def score(team: dict, md: int, apps: pd.DataFrame, pos: dict[int, int]) -> int:
    a = apps[apps["md"] == md].groupby("pid").agg(points=("points", "sum"), mins=("oF", "sum"))
    pts = a["points"].to_dict()
    played = {p: a["mins"].get(p, 0) > 0 for p in team["xi"] + team["bench"]}
    final = rules.auto_subs([(p, pos[p], played[p]) for p in team["xi"]], [(b, pos[b], played[b]) for b in team["bench"]])
    cap = team["captain"] if played.get(team["captain"]) else team["vice"]
    return int(sum(pts.get(p, 0) for p in final) + (pts.get(cap, 0) if cap in final else 0))


def replay(tour: int, bar: float, ftv: float, squad_in=None, verbose=False) -> dict:
    matches, players, _ = model.load(tour)
    apps = actual_points(tour)
    pmaps = prices_at(tour)
    pos = dict(zip(players["pid"].astype(int), players["pos"].astype(int)))
    mds = sorted(set(apps["md"]))
    squad, ft, total, per_md, moves, bank = None, None, 0, {}, 0, 0.0
    for md in mds:
        xp = predictions(tour, md)
        if xp.empty:
            continue
        xp = xp.assign(pos=xp["pid"].map(pos))
        pl = players.copy()
        pl["price"] = [price_for(pmaps.get(int(p)), md, float(pr)) for p, pr in zip(pl["pid"], pl["price"])]
        pl["active"] = 1
        H = known_mds(tour, md, matches)
        price = dict(zip(pl["pid"].astype(int), pl["price"]))
        if squad is None or rules.is_unlimited(md):
            # new team / unlimited window: rebuilt within the stage budget
            res = optimise.solve(xp, pl, md, mds=H, friction=bar, ft_value=ftv)
            used, hits = 0, 0
            funds = rules.budget(md)
        else:
            # players sell at today's price: squad value + bank is the money available
            funds = sum(price[p] for p in squad) + bank
            res = optimise.solve(xp, pl, md, squad=squad, free_transfers=ft, bank=bank, mds=H, friction=bar, ft_value=ftv)
            used = len(res["plan"][0]["in"])
            hits = res["plan"][0]["hits"]
        st = res["plan"][0]
        xpm = xp[xp["md"] == md].groupby("pid")["xp"].sum().to_dict()
        xi = sorted(st["xi"], key=lambda p: -xpm.get(p, 0))
        bench = optimise.bench_order(st["squad"], st["xi"], xpm, pos)
        team = {"xi": st["xi"], "bench": bench, "captain": st["captain"],
                "vice": next(p for p in xi if p != st["captain"])}
        pts = score(team, md, apps, pos) - rules.HIT_COST * hits
        total += pts
        per_md[md] = pts
        moves += used
        squad = st["squad"]
        bank = max(0.0, funds - sum(price[p] for p in squad))
        ft = rules.free_transfers_next(md, None if rules.is_unlimited(md) or md == mds[0] else ft, used)
        if verbose:
            print(f"  {tour} MD{md}: {pts} pts, {used} transfer(s), {hits} hit(s)", flush=True)
    return {"tour": tour, "bar": bar, "ft_value": ftv, "points": total, "per_md": per_md, "transfers": moves}


def _job(args):
    tour, bar, ftv = args
    return replay(tour, bar, ftv)


def main() -> None:
    for tour in (feed.PREV_TOUR, feed.TOUR):  # warm the prediction cache once (shared by every config)
        m, _, a = model.load(tour)
        for md in sorted(set(a["md"])):
            predictions(tour, md)
        print(f"predictions cached for tour {tour}", flush=True)
    jobs = [(t, b, v) for t in (feed.PREV_TOUR, feed.TOUR) for b in BARS for v in FT_VALUES]
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_job, jobs))
    df = pd.DataFrame([{k: r[k] for k in ("tour", "bar", "ft_value", "points", "transfers")} for r in results])
    print(df.pivot_table(index="bar", columns=["tour", "ft_value"], values="points").to_string())
    seen = df[df["tour"] == feed.PREV_TOUR].sort_values("points", ascending=False)
    best = seen.iloc[0]
    unseen = df[df["tour"] == feed.TOUR].set_index(["bar", "ft_value"])["points"]
    base_unseen = float(unseen.loc[BASELINE])
    best_unseen = float(unseen.loc[(best["bar"], best["ft_value"])])
    adopt = (best["bar"], best["ft_value"]) != BASELINE and best_unseen > base_unseen
    verdict = {
        "best_2025_26": {"bar": float(best["bar"]), "ft_value": float(best["ft_value"]), "points": int(best["points"])},
        "baseline_2025_26": int(df[(df["tour"] == feed.PREV_TOUR) & (df["bar"] == BASELINE[0]) & (df["ft_value"] == BASELINE[1])]["points"].iloc[0]),
        "best_unseen_2026_27": best_unseen, "baseline_unseen_2026_27": base_unseen, "adopt": bool(adopt),
    }
    print(json.dumps(verdict, indent=1))
    OUT.write_text(json.dumps({"results": results, "verdict": verdict}, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(replay(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), verbose=True))
    else:
        main()
