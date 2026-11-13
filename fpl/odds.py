"""Betting odds -> expected goals, via The Odds API — builds
``data/tables/odds_goals.parquet`` (one row per upcoming fixture: bookmaker
win/draw/over-2.5 probabilities, the Poisson lambda per team that reproduces
them, and each team's resulting clean-sheet probability).

Bookmaker odds are the sharpest public forecast of each match. The steps:

1. Convert each price to a probability (1 / decimal odds), then divide by
   the sum of a market's probabilities so the bookmaker's margin ("overround")
   is removed.
2. Average the de-vigged probabilities across bookmakers into one consensus
   view per match.
3. Find the home/away goal rates (lambda) that make a Poisson scoreline model
   reproduce those win/draw/loss and over/under 2.5 probabilities, via
   `scipy.optimize.minimize`.
4. From lambda: expected goals = lambda; clean-sheet probability =
   e ** (-opponent's lambda).

Needs a free key from https://the-odds-api.com (sport key `soccer_epl`),
read from the `ODDS_API_KEY` environment variable -- store it as a GitHub
secret, never in code. Free plans only cover current odds and have a modest
monthly request budget, so this fetches at most once a day (the cache keys
on today's date) and should be run at most once a day in any scheduled job.

Odds can go missing for a run (no key, quota exhausted, an off day with no
markets yet) -- this is run through ``fpl.source_health`` the same as Elo
and Understat, so a missing/invalid fetch falls back to the last good
table instead of silently dropping the input, and the site shows a warning.

Run with `uv run python -m fpl.odds`.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

from fpl.net import cached_get
from fpl.source_health import check_value_range, combine_checks, record, write_health_record

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"

SPORT_KEY = "soccer_epl"
BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
MAX_GOALS = 10  # grid size for the Poisson scoreline model; P(either team scores 10+) is negligible

# Mirrors MAX_GOALS: a fitted lambda outside this range is a solver/data bug,
# not a real match.
PLAUSIBLE_LAMBDA_RANGE = (0.0, float(MAX_GOALS))


def _sanity_check(odds_goals: pd.DataFrame) -> str | None:
    return combine_checks(
        check_value_range(odds_goals, "lambda_home", *PLAUSIBLE_LAMBDA_RANGE),
        check_value_range(odds_goals, "lambda_away", *PLAUSIBLE_LAMBDA_RANGE),
    )

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fpl-consultant (ymostafa1@babson.edu)"})


def load_aliases() -> pd.DataFrame:
    return pd.read_csv(ALIASES_PATH)


def fetch_odds(day_dir: Path, api_key: str) -> list[dict]:
    cache_file = day_dir / "odds_soccer_epl.json"
    url = f"{BASE}?regions=uk&markets=h2h,totals&oddsFormat=decimal&apiKey={api_key}"
    text = cached_get(SESSION, url, cache_file)
    return json.loads(text) if text else []


# ---------------------------------------------------------------------------
# de-vig: decimal odds -> a single consensus probability vector per match
# ---------------------------------------------------------------------------


def devig(prices: dict) -> dict:
    """Convert a market's decimal odds to probabilities that sum to 1,
    removing the bookmaker's overround."""
    implied = {outcome: 1.0 / price for outcome, price in prices.items()}
    total = sum(implied.values())
    return {outcome: p / total for outcome, p in implied.items()}


def consensus_probabilities(event: dict) -> dict | None:
    """Average de-vigged h2h and totals-2.5 probabilities across every
    bookmaker that offers both markets for this event."""
    home, away = event["home_team"], event["away_team"]
    h2h_rows, totals_rows = [], []

    for bookmaker in event.get("bookmakers", []):
        markets = {m["key"]: m for m in bookmaker.get("markets", [])}

        h2h = markets.get("h2h")
        if h2h:
            prices = {o["name"]: o["price"] for o in h2h["outcomes"]}
            if home in prices and away in prices and "Draw" in prices:
                probs = devig({home: prices[home], away: prices[away], "Draw": prices["Draw"]})
                h2h_rows.append(probs)

        totals = markets.get("totals")
        if totals:
            prices = {(o["name"], o["point"]): o["price"] for o in totals["outcomes"]}
            over = next((p for (name, point), p in prices.items() if name == "Over" and point == 2.5), None)
            under = next((p for (name, point), p in prices.items() if name == "Under" and point == 2.5), None)
            if over and under:
                probs = devig({"Over": over, "Under": under})
                totals_rows.append(probs)

    if not h2h_rows or not totals_rows:
        return None

    h2h_df = pd.DataFrame(h2h_rows)
    totals_df = pd.DataFrame(totals_rows)
    return {
        "p_home_win": h2h_df[home].mean(),
        "p_draw": h2h_df["Draw"].mean(),
        "p_away_win": h2h_df[away].mean(),
        "p_over_2_5": totals_df["Over"].mean(),
        "p_under_2_5": totals_df["Under"].mean(),
        "n_bookmakers": len(h2h_rows),
    }


# ---------------------------------------------------------------------------
# Poisson fit: consensus probabilities -> (lambda_home, lambda_away)
# ---------------------------------------------------------------------------


def _scoreline_grid(lam_home: float, lam_away: float) -> np.ndarray:
    goals = np.arange(MAX_GOALS + 1)
    return np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))


def _model_probabilities(lam_home: float, lam_away: float) -> dict:
    grid = _scoreline_grid(lam_home, lam_away)
    goals = np.arange(MAX_GOALS + 1)
    over_mask = (goals[:, None] + goals[None, :]) > 2.5
    return {
        "p_home_win": float(np.tril(grid, -1).sum()),
        "p_draw": float(np.trace(grid)),
        "p_away_win": float(np.triu(grid, 1).sum()),
        "p_over_2_5": float(grid[over_mask].sum()),
        "p_under_2_5": float(grid[~over_mask].sum()),
    }


def fit_lambdas(target: dict) -> tuple[float, float]:
    """Find (lambda_home, lambda_away) whose Poisson scoreline model best
    reproduces the bookmaker-implied win/draw/loss and over/under 2.5
    probabilities."""

    def objective(x: np.ndarray) -> float:
        lam_home, lam_away = np.exp(x)  # optimize in log-space to keep lambdas positive
        model = _model_probabilities(lam_home, lam_away)
        return sum((model[key] - target[key]) ** 2 for key in model)

    x0 = np.log([1.3, 1.1])  # a typical home/away goal rate as a starting guess
    result = minimize(objective, x0, method="Nelder-Mead")
    lam_home, lam_away = np.exp(result.x)
    return float(lam_home), float(lam_away)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_odds_table(events: list[dict], aliases: pd.DataFrame) -> pd.DataFrame:
    team_lookup = aliases.set_index("odds_api_name")[["fpl_team_id", "fpl_short_name"]]
    rows = []
    unmapped = set()

    for event in events:
        home, away = event["home_team"], event["away_team"]
        if home not in team_lookup.index or away not in team_lookup.index:
            unmapped.update({t for t in (home, away) if t not in team_lookup.index})
            continue

        target = consensus_probabilities(event)
        if target is None:
            continue  # no bookmaker offered both h2h and totals-2.5 for this match yet

        lam_home, lam_away = fit_lambdas(target)
        rows.append(
            {
                "event_id": event["id"],
                "commence_time": event["commence_time"],
                "home_team_id": int(team_lookup.loc[home, "fpl_team_id"]),
                "home_team": team_lookup.loc[home, "fpl_short_name"],
                "away_team_id": int(team_lookup.loc[away, "fpl_team_id"]),
                "away_team": team_lookup.loc[away, "fpl_short_name"],
                "p_home_win": target["p_home_win"],
                "p_draw": target["p_draw"],
                "p_away_win": target["p_away_win"],
                "p_over_2_5": target["p_over_2_5"],
                "p_under_2_5": target["p_under_2_5"],
                "n_bookmakers": target["n_bookmakers"],
                "lambda_home": lam_home,
                "lambda_away": lam_away,
                "home_clean_sheet_prob": float(np.exp(-lam_away)),
                "away_clean_sheet_prob": float(np.exp(-lam_home)),
            }
        )

    if unmapped:
        print(
            f"  warning: {len(unmapped)} team name(s) from The Odds API aren't in "
            f"team_aliases.csv's odds_api_name column (add them by hand): {sorted(unmapped)}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    odds_goals = pd.DataFrame()
    empty_message = None

    if not api_key:
        print(
            "ODDS_API_KEY is not set; falling back to the last good odds table (if any).\n"
            "  Get a free key at https://the-odds-api.com, then either:\n"
            "    export ODDS_API_KEY=... (locally), or\n"
            "    add it as a GitHub secret named ODDS_API_KEY (CI) -- never hardcode it."
        )
        empty_message = "ODDS_API_KEY not set"
    else:
        day_dir = RAW_DIR / date.today().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        print("Fetching The Odds API h2h/totals odds for soccer_epl...")
        events = fetch_odds(day_dir, api_key)
        if not events:
            empty_message = "no odds returned (bad key, no upcoming events, or quota exhausted)"
        else:
            aliases = load_aliases()
            odds_goals = build_odds_table(events, aliases)
            if odds_goals.empty:
                empty_message = "no fixtures had usable odds"

    used, health = record("odds", odds_goals, _sanity_check, empty_message=empty_message)
    write_health_record(health)
    if health["status"] == "ok":
        print(f"  health: ok ({health['row_count']} rows)")
    elif health["status"] == "stale":
        print(f"  health: STALE -- {health['error']}; falling back to the last good odds table "
              f"(saved {health['last_success_at']})")
    else:
        print(f"  health: FAILED -- {health['error']}; no last-good odds table to fall back to either")

    if used.empty:
        print("No odds data available (fetch + fallback both empty); odds_goals.parquet not written.")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "odds_goals.parquet"
    used.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(used)} fixtures)")


if __name__ == "__main__":
    main()
