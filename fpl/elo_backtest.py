"""Validates fpl.elo's derived fallback against ClubElo's real history.

For a few seed points (the day before gameweek G's first kickoff), it
seeds from ClubElo's ratings as of that day, applies every finished FPL
result from then up to the last date in the ClubElo history, and compares
each team's derived rating with ClubElo's actual rating on that last date.
A "no update" row (seed ratings left unchanged) is printed alongside, so
it's clear whether applying FPL results actually gets closer to ClubElo.

Uses data/last_good/elo.parquet (the last real ClubElo fetch) and
data/tables/fixtures.parquet. Read-only: writes nothing.

Run with `uv run python -m fpl.elo_backtest` (add --grid to also print a
small K-factor x home-advantage grid).
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd
from scipy.stats import spearmanr

import fpl.elo as elo
from fpl.source_health import current_pl_team_ids, load_last_good

TABLES_DIR = elo.TABLES_DIR


def evaluate(history: pd.DataFrame, fixtures: pd.DataFrame, team_ids: set[int], seed_date, truth_date) -> dict:
    seed = elo.latest_ratings(history[history["team_id"].isin(team_ids)], as_of=seed_date)
    truth = elo.latest_ratings(history[history["team_id"].isin(team_ids)], as_of=truth_date)
    ratings = {int(t): float(r) for t, r in seed.items()}
    results = elo.finished_results(fixtures, after=seed_date, until=truth_date)
    elo.apply_results(ratings, results)
    derived = pd.Series(ratings).reindex(truth.index)
    err = (derived - truth).abs()
    base = (seed.reindex(truth.index) - truth).abs()
    return {
        "seed_date": seed_date,
        "matches": len(results),
        "mean_abs": err.mean(),
        "max_abs": err.max(),
        "max_team": err.idxmax(),
        "spearman": spearmanr(derived, truth).statistic,
        "noupd_mean_abs": base.mean(),
        "noupd_max_abs": base.max(),
        "noupd_spearman": spearmanr(seed.reindex(truth.index), truth).statistic,
        "per_team": pd.DataFrame({"truth": truth, "derived": derived, "abs_err": err}),
    }


def seed_points(fixtures: pd.DataFrame, truth_date, gameweeks_back=(1, 2, 3, 4, 6)) -> list:
    """Day before the first kickoff of each gameweek that's N gameweeks
    before the last one fully played by ``truth_date``."""
    f = fixtures.copy()
    f["kickoff_date"] = pd.to_datetime(f["kickoff_time"], utc=True).dt.date
    first = f.groupby("event")["kickoff_date"].min()
    last = f.groupby("event")["kickoff_date"].max()
    complete = last[last <= truth_date].index.max()
    return [(n, first[complete - n + 1] - timedelta(days=1)) for n in gameweeks_back if complete - n + 1 in first.index]


def main() -> None:
    history, _ = load_last_good("elo")
    fixtures = pd.read_parquet(TABLES_DIR / "fixtures.parquet")
    names = elo._team_names()
    team_ids = current_pl_team_ids()
    truth_date = max(history["date"])
    print(f"ClubElo history ends {truth_date}; constants K={elo.K_FACTOR}, home advantage={elo.HOME_ADVANTAGE}\n")

    print(f"{'GWs':>4} {'seed date':>11} {'matches':>8} | {'mean|d|':>8} {'max|d|':>8} {'team':>5} {'rho':>6} | "
          f"{'no-update: mean':>15} {'max':>6} {'rho':>6}")
    worst = None
    for n, seed_date in seed_points(fixtures, truth_date):
        r = evaluate(history, fixtures, team_ids, seed_date, truth_date)
        print(f"{n:>4} {seed_date!s:>11} {r['matches']:>8} | {r['mean_abs']:8.1f} {r['max_abs']:8.1f} "
              f"{names[r['max_team']]:>5} {r['spearman']:6.3f} | {r['noupd_mean_abs']:15.1f} "
              f"{r['noupd_max_abs']:6.1f} {r['noupd_spearman']:6.3f}")
        if n == 4:
            worst = r

    if worst is not None:
        per_team = worst["per_team"].rename(index=names).sort_values("abs_err", ascending=False)
        print(f"\nPer team, seeded 4 gameweeks back ({worst['seed_date']}):")
        print(per_team.round(1).to_string())

    if "--grid" in sys.argv:
        print("\nGrid (seeded 4 gameweeks back): mean|d| / max|d| / rho")
        _, seed_date = [p for p in seed_points(fixtures, truth_date) if p[0] == 4][0]
        k0, h0 = elo.K_FACTOR, elo.HOME_ADVANTAGE
        for k in (10, 15, 20, 25, 30):
            cells = []
            for h in (0, 30, 60, 100):
                elo.K_FACTOR, elo.HOME_ADVANTAGE = float(k), float(h)
                r = evaluate(history, fixtures, team_ids, seed_date, truth_date)
                cells.append(f"H={h:<3} {r['mean_abs']:5.1f}/{r['max_abs']:5.1f}/{r['spearman']:.3f}")
            print(f"  K={k:<3} " + "   ".join(cells))
        elo.K_FACTOR, elo.HOME_ADVANTAGE = k0, h0


if __name__ == "__main__":
    main()
