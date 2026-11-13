"""Team strength from ClubElo — builds ``data/tables/elo.parquet``
(team, date, Elo). Elo moves after every match, so it reacts faster than
FPL's fixture difficulty ratings, which barely change all season.

ClubElo's documented CSV API (``api.clubelo.com/<date>``, ``.../<Club>``) is
currently dead: every ratings request answers HTTP 502, and
``api.clubelo.com/Fixtures`` answers "Fixtures API deactivated". This looks
like a permanent change (ClubElo moved behind a login), not a transient
outage. The ``clubelo.com`` *website* still publishes ratings, though: each
club's page (e.g. https://clubelo.com/Arsenal) embeds its full Elo history
as a Vega-Lite dataset inside a ``<script>`` tag. This module reads that
embedded JSON instead of the dead CSV endpoints.

This has broken silently once already, so the result is run through
``fpl.source_health``: it must cover all 20 current PL teams and carry
plausible Elo values, or the last known-good table is used instead and
the site shows a visible warning (see ``data/tables/source_health.json``).

Run with `uv run python -m fpl.elo`.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from fpl.net import cached_get
from fpl.source_health import check_team_coverage, check_value_range, combine_checks, record, write_health_record

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"

BASE = "https://clubelo.com"

# Elo systems like ClubElo's have historically kept Premier League clubs
# somewhere around 1300-2100; padded generously to 1000-2300 so this only
# flags genuinely broken data (e.g. a parsing bug returning zeros or a
# constant placeholder), not a real top/bottom-of-division swing.
PLAUSIBLE_ELO_RANGE = (1000.0, 2300.0)


def _sanity_check(elo: pd.DataFrame) -> str | None:
    """Elo must cover every current PL team and every value must be in a
    plausible range -- a failed check is reported exactly like a failed
    fetch, so the site shows a warning either way."""
    return combine_checks(
        check_team_coverage(elo, "team_id"),
        check_value_range(elo, "elo", *PLAUSIBLE_ELO_RANGE),
    )

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fpl-consultant (ymostafa1@babson.edu)"})


def load_aliases() -> pd.DataFrame:
    return pd.read_csv(ALIASES_PATH)


def extract_vega_dataset(html: str) -> list[dict]:
    """Pull the Elo history out of a club page's embedded `var vegaJson = {...}`."""
    marker = "var vegaJson = "
    start = html.find(marker)
    if start == -1:
        raise ValueError("no embedded vegaJson block found (clubelo.com's layout may have changed)")
    start += len(marker)
    obj, _ = json.JSONDecoder().raw_decode(html, start)
    datasets = obj.get("datasets", {})
    if not datasets:
        raise ValueError("vegaJson has no datasets")
    # The club-history chart is the dataset with Date/Elo rows (the ranking
    # chart's dataset, if present, carries Name/Federation instead).
    for rows in datasets.values():
        if rows and "Date" in rows[0] and "Elo" in rows[0]:
            return rows
    raise ValueError("no Date/Elo dataset found among vegaJson's datasets")


def fetch_club_elo(day_dir: Path, slug: str) -> pd.DataFrame:
    cache_file = day_dir / f"clubelo_{slug}.html"
    html = cached_get(SESSION, f"{BASE}/{slug}", cache_file)
    if html is None:
        return pd.DataFrame()
    rows = extract_vega_dataset(html)
    history = pd.DataFrame(rows)
    history["date"] = pd.to_datetime(history["Date"]).dt.date
    return history[["date", "Elo", "Golo"]].rename(columns={"Elo": "elo", "Golo": "golo"})


def build_elo_table(day_dir: Path) -> pd.DataFrame:
    aliases = load_aliases()
    tables = []
    for _, row in aliases.iterrows():
        history = fetch_club_elo(day_dir, row["clubelo_slug"])
        if history.empty:
            print(f"  no ClubElo data for {row['fpl_short_name']} ({row['clubelo_slug']}); skipping")
            continue
        history["team_id"] = row["fpl_team_id"]
        history["team"] = row["fpl_short_name"]
        tables.append(history)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)[["team_id", "team", "date", "elo", "golo"]]


def main() -> None:
    day_dir = RAW_DIR / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ClubElo ratings (per-club page scrape; the CSV API is dead)...")
    elo = build_elo_table(day_dir)

    used, health = record("elo", elo, _sanity_check)
    write_health_record(health)
    if health["status"] == "ok":
        print(f"  health: ok ({health['row_count']} rows)")
    elif health["status"] == "stale":
        print(f"  health: STALE -- {health['error']}; falling back to the last good table "
              f"(saved {health['last_success_at']})")
    else:
        print(f"  health: FAILED -- {health['error']}; no last-good table to fall back to either")

    if used.empty:
        print("ClubElo was unavailable for every club, with no usable fallback; elo.parquet was not written.")
        return

    path = TABLES_DIR / "elo.parquet"
    used.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(used)} rows, {used['team'].nunique()} teams)")


if __name__ == "__main__":
    main()
