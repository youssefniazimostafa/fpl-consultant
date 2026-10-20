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

Run with `uv run python -m fpl.elo`.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from fpl.net import cached_get

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"

BASE = "https://clubelo.com"

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
    if elo.empty:
        print("ClubElo was unavailable for every club; elo.parquet was not written.")
        return

    path = TABLES_DIR / "elo.parquet"
    elo.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(elo)} rows, {elo['team'].nunique()} teams)")


if __name__ == "__main__":
    main()
