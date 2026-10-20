"""Download everything from the FPL API, cache the raw responses, and build
tidy Parquet tables under ``data/tables/``.

Run with:

    uv run python -m fpl.fetch

Every later step in this project should read only from ``data/tables/`` and
never call the API directly. If a parsing step below turns out to be wrong,
fix it and re-run this module: cached JSON under ``data/raw/<date>/`` means
the fix is applied without hitting the API again.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
CONFIG_PATH = ROOT / "config.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fpl-consultant (ymostafa1@babson.edu)"})


def get(path: str, day_dir: Path) -> dict[str, Any] | None:
    """Fetch ``path`` from the FPL API, caching the raw JSON under ``day_dir``.

    A cache hit never touches the network. A cache miss is retried up to 3
    times with exponential backoff and pauses 0.2s after every real request.
    A 404 (e.g. a manager with no picks yet for a given gameweek) is cached
    as a miss and returns ``None`` instead of being retried, since retrying
    a permanent "not found" wastes time without changing the outcome.
    """
    cache = day_dir / (path.strip("/").replace("/", "_") + ".json")
    if cache.exists():
        text = cache.read_text()
        return None if text == "null" else json.loads(text)

    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            r = SESSION.get(f"{BASE}/{path}", timeout=30)
            if r.status_code == 404:
                cache.write_text("null")
                time.sleep(0.2)
                return None
            r.raise_for_status()
            cache.write_text(r.text)
            time.sleep(0.2)
            return r.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed: {path}") from last_error


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text())


def write_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / name
    df.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# players.parquet / teams.parquet — from bootstrap-static
# ---------------------------------------------------------------------------


def fetch_bootstrap(day_dir: Path) -> dict[str, Any]:
    return get("bootstrap-static/", day_dir)


def build_players_table(bootstrap: dict[str, Any]) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(
        columns={"id": "team", "name": "team_name", "short_name": "team_short_name"}
    )
    positions = pd.DataFrame(bootstrap["element_types"])[
        ["id", "singular_name_short"]
    ].rename(columns={"id": "element_type", "singular_name_short": "position"})
    return players.merge(teams, on="team", how="left").merge(
        positions, on="element_type", how="left"
    )


def build_teams_table(bootstrap: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(bootstrap["teams"])


def completed_gameweeks(bootstrap: dict[str, Any]) -> list[int]:
    events = bootstrap["events"]
    return [e["id"] for e in events if e.get("finished") or e.get("is_current")]


# ---------------------------------------------------------------------------
# fixtures.parquet — from fixtures/
# ---------------------------------------------------------------------------


def build_fixtures_table(day_dir: Path) -> pd.DataFrame:
    fixtures = get("fixtures/", day_dir)
    return pd.DataFrame(fixtures)


# ---------------------------------------------------------------------------
# player_matches.parquet — from element-summary/{id}/ history, one row per
# player x match played, only for players with minutes > 0 this season
# ---------------------------------------------------------------------------


def build_player_matches_table(day_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    played_ids = players.loc[players["minutes"] > 0, "id"].tolist()
    histories = []
    for player_id in played_ids:
        summary = get(f"element-summary/{player_id}/", day_dir)
        if summary is None:
            continue
        history = pd.DataFrame(summary["history"])
        if history.empty:
            continue
        histories.append(history[history["minutes"] > 0])
    if not histories:
        return pd.DataFrame()
    matches = pd.concat(histories, ignore_index=True)
    return matches.rename(columns={"element": "player_id", "round": "gameweek"})


# ---------------------------------------------------------------------------
# league_history.parquet / league_chips.parquet / league_picks.parquet —
# from leagues-classic standings, entry/{id}/history/, and picks/
# ---------------------------------------------------------------------------


def fetch_league_managers(
    day_dir: Path, league_id: int, team_id: int | None
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = get(
            f"leagues-classic/{league_id}/standings/?page_standings={page}", day_dir
        )
        standings = payload["standings"]
        results.extend(standings["results"])
        if not standings.get("has_next"):
            break
        page += 1

    managers = pd.DataFrame(results)[["entry", "entry_name", "player_name"]].rename(
        columns={"entry": "manager_id"}
    )
    if team_id is not None and team_id not in managers["manager_id"].values:
        managers = pd.concat(
            [
                managers,
                pd.DataFrame(
                    [{"manager_id": team_id, "entry_name": None, "player_name": None}]
                ),
            ],
            ignore_index=True,
        )
    return managers.drop_duplicates(subset="manager_id").reset_index(drop=True)


def build_league_history_and_chips_tables(
    day_dir: Path, managers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    chip_rows = []
    for manager_id in managers["manager_id"]:
        history = get(f"entry/{manager_id}/history/", day_dir)
        if history is None:
            print(f"  skipping manager {manager_id}: no history found")
            continue

        current = pd.DataFrame(history["current"])
        if not current.empty:
            current["manager_id"] = manager_id
            history_rows.append(current)

        chips = pd.DataFrame(history["chips"])
        if not chips.empty:
            chips["manager_id"] = manager_id
            chip_rows.append(chips)

    history_df = (
        pd.concat(history_rows, ignore_index=True).rename(
            columns={"event": "gameweek"}
        )
        if history_rows
        else pd.DataFrame()
    )
    chips_df = pd.concat(chip_rows, ignore_index=True) if chip_rows else pd.DataFrame()
    return history_df, chips_df


def build_league_picks_table(
    day_dir: Path, managers: pd.DataFrame, gameweeks: list[int]
) -> pd.DataFrame:
    rows = []
    for manager_id in managers["manager_id"]:
        for gameweek in gameweeks:
            payload = get(f"entry/{manager_id}/event/{gameweek}/picks/", day_dir)
            if payload is None:
                continue
            picks = pd.DataFrame(payload["picks"])
            if picks.empty:
                continue
            picks["manager_id"] = manager_id
            picks["gameweek"] = gameweek
            picks["active_chip"] = payload.get("active_chip")
            rows.append(picks)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns={"element": "player_id"})


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id", type=int, default=None, help="override config.json league_id"
    )
    parser.add_argument(
        "--team-id", type=int, default=None, help="override config.json team_id"
    )
    args = parser.parse_args()

    config = load_config()
    league_id = args.league_id or config.get("league_id")
    team_id = args.team_id or config.get("team_id")

    day_dir = RAW_DIR / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching bootstrap-static...")
    bootstrap = fetch_bootstrap(day_dir)
    players = build_players_table(bootstrap)
    write_table(players, "players.parquet")
    write_table(build_teams_table(bootstrap), "teams.parquet")

    print("Fetching fixtures...")
    write_table(build_fixtures_table(day_dir), "fixtures.parquet")

    played = int((players["minutes"] > 0).sum())
    print(f"Fetching element-summary for {played} players with minutes > 0...")
    write_table(build_player_matches_table(day_dir, players), "player_matches.parquet")

    if league_id is None:
        print("No league_id set in config.json (or --league-id); skipping league tables.")
        return

    print(f"Fetching standings for league {league_id}...")
    managers = fetch_league_managers(day_dir, league_id, team_id)
    print(f"Found {len(managers)} managers (rivals + you).")

    print("Fetching manager histories and chips...")
    league_history, league_chips = build_league_history_and_chips_tables(
        day_dir, managers
    )
    write_table(league_history, "league_history.parquet")
    write_table(league_chips, "league_chips.parquet")

    gameweeks = completed_gameweeks(bootstrap)
    print(f"Fetching picks for {len(managers)} managers across {len(gameweeks)} gameweeks...")
    league_picks = build_league_picks_table(day_dir, managers, gameweeks)
    write_table(league_picks, "league_picks.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
