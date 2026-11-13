"""Team and player xG from Understat — builds ``data/tables/team_xg.parquet``
(team, match, xG for, xG against, home/away, plus rolling attack/defence
strength over the last 6 and 12 matches) and ``data/tables/player_xg.parquet``
(player season totals: xG, xA, shots, key passes), and maintains
``fpl/player_map.csv``, a name+team fuzzy match from Understat players to
FPL player ids.

The ``understatapi`` package scrapes JSON blobs (``datesData``,
``teamsData``, ``playersData``) that understat.com used to embed directly
in each league/team/player page's HTML. understat.com has since moved that
data behind an XHR call its own frontend makes
(``GET /getLeagueData/<league>/<season>``, which needs an
``X-Requested-With: XMLHttpRequest`` header or the site answers 404), so the
old embedded-HTML scrape no longer finds anything to parse. This module
calls that JSON endpoint directly instead — understat is scraped either way,
so this keeps the "once a day, cache everything" politeness the task asks
for without depending on a page layout understat.com no longer serves.

This has broken silently once already, so team_xg (the table that feeds
fixture difficulty) is run through ``fpl.source_health``: it must cover
all 20 current PL teams with plausible xG values, or the last known-good
table is used instead and the site shows a visible warning (see
``data/tables/source_health.json``). player_xg falls back the same way.

Run with `uv run python -m fpl.understat`.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from fpl.net import cached_get
from fpl.source_health import (
    check_team_coverage,
    check_value_range,
    combine_checks,
    load_last_good,
    record,
    save_last_good,
    write_health_record,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"
PLAYER_MAP_PATH = Path(__file__).resolve().parent / "player_map.csv"

# Per-match xG; mirrors fpl.odds' MAX_GOALS=10 grid -- a value past this is
# almost certainly a parsing bug, not a real match.
PLAUSIBLE_TEAM_XG_RANGE = (0.0, 10.0)


def _team_xg_sanity_check(team_xg: pd.DataFrame) -> str | None:
    """team_xg feeds fixture difficulty directly (fpl.features' Elo/xG
    fallback lambda), so its health record is what Fix 3 cares about --
    player_xg degrades the same way underneath it, without its own entry."""
    return combine_checks(
        check_team_coverage(team_xg, "team_id"),
        check_value_range(team_xg, "xg_for", *PLAUSIBLE_TEAM_XG_RANGE),
    )

LEAGUE = "EPL"
BASE = f"https://understat.com/getLeagueData/{LEAGUE}"
ROLLING_WINDOWS = (6, 12)
MATCH_SCORE_THRESHOLD = 85  # below this, a row needs a manual look in player_map.csv

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "fpl-consultant (ymostafa1@babson.edu)",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{LEAGUE}",
    }
)


def load_aliases() -> pd.DataFrame:
    return pd.read_csv(ALIASES_PATH)


def current_season() -> str:
    """Understat labels a season by its starting year, e.g. 2026 for 2026/27."""
    today = date.today()
    year = today.year if today.month >= 7 else today.year - 1
    return str(year)


def fetch_league_data(day_dir: Path, season: str) -> dict:
    cache_file = day_dir / f"understat_{LEAGUE}_{season}.json"
    text = cached_get(SESSION, f"{BASE}/{season}", cache_file)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # understat has answered with an HTML error/rate-limit page instead
        # of JSON before; treat it the same as "no data" so main() falls
        # back to the last good table instead of crashing the pipeline.
        print(f"  understat returned a non-JSON response ({len(text)} chars); treating as unavailable")
        return {}


# ---------------------------------------------------------------------------
# team_xg.parquet — one row per team x match played, from the `dates` list
# ---------------------------------------------------------------------------


def build_team_xg_table(payload: dict, aliases: pd.DataFrame) -> pd.DataFrame:
    team_lookup = aliases.set_index("understat_title")[["fpl_team_id", "fpl_short_name"]]

    rows = []
    for match in payload.get("dates", []):
        if not match.get("isResult"):
            continue  # fixture hasn't been played yet
        for side, opp_side in (("h", "a"), ("a", "h")):
            rows.append(
                {
                    "match_id": int(match["id"]),
                    "date": match["datetime"],
                    "team": match[side]["title"],
                    "opponent": match[opp_side]["title"],
                    "is_home": side == "h",
                    "xg_for": float(match["xG"][side]),
                    "xg_against": float(match["xG"][opp_side]),
                    "goals_for": int(match["goals"][side]),
                    "goals_against": int(match["goals"][opp_side]),
                }
            )
    team_xg = pd.DataFrame(rows)
    if team_xg.empty:
        return team_xg

    team_xg = team_xg.merge(team_lookup, left_on="team", right_index=True, how="left")
    team_xg = team_xg.rename(columns={"fpl_team_id": "team_id", "fpl_short_name": "team_short"})
    team_xg["date"] = pd.to_datetime(team_xg["date"])
    return team_xg.sort_values(["team_id", "date"]).reset_index(drop=True)


def add_rolling_strength(team_xg: pd.DataFrame) -> pd.DataFrame:
    """Each team's trailing attack/defence xG average over its last N matches."""
    team_xg = team_xg.sort_values(["team_id", "date"]).copy()
    grouped = team_xg.groupby("team_id")
    for window in ROLLING_WINDOWS:
        team_xg[f"attack_xg_{window}"] = grouped["xg_for"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        team_xg[f"defence_xg_{window}"] = grouped["xg_against"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return team_xg


# ---------------------------------------------------------------------------
# player_xg.parquet — one row per player, season totals
# ---------------------------------------------------------------------------

_NUMERIC_PLAYER_COLS = [
    "games",
    "time",
    "goals",
    "xG",
    "assists",
    "xA",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]


def build_player_xg_table(payload: dict) -> pd.DataFrame:
    players = pd.DataFrame(payload.get("players", []))
    if players.empty:
        return players
    players["id"] = players["id"].astype(int)
    for col in _NUMERIC_PLAYER_COLS:
        players[col] = pd.to_numeric(players[col])
    return players.rename(columns={"id": "understat_id", "player_name": "understat_name"})


# ---------------------------------------------------------------------------
# fpl/player_map.csv — Understat player -> FPL player id, by name + team
# ---------------------------------------------------------------------------


def fuzzy_match_players(
    player_xg: pd.DataFrame, players_fpl: pd.DataFrame, aliases: pd.DataFrame
) -> pd.DataFrame:
    team_lookup = aliases.set_index("understat_title")["fpl_team_id"].to_dict()
    fpl_named = players_fpl.assign(full_name=players_fpl["first_name"] + " " + players_fpl["second_name"])
    fpl_by_team = {team_id: group for team_id, group in fpl_named.groupby("team")}

    rows = []
    for _, u in player_xg.iterrows():
        # Mid-season transfers come back from Understat as a comma-joined list
        # of every club the player appeared for (e.g. "Everton,Manchester
        # City"), in no guaranteed order, so pool candidates from all of them.
        team_ids = [team_lookup[name] for name in u["team_title"].split(",") if name in team_lookup]
        candidates = pd.concat([fpl_by_team[t] for t in team_ids if t in fpl_by_team], axis=0) if team_ids else None
        fpl_id = fpl_web_name = fpl_full_name = None
        score = 0.0
        if candidates is not None and not candidates.empty:
            best = process.extractOne(u["understat_name"], candidates["full_name"], scorer=fuzz.token_sort_ratio)
            if best is not None:
                _, score, idx = best
                match = candidates.loc[idx]  # rapidfuzz returns the Series' pandas index label
                fpl_id = int(match["id"])
                fpl_web_name = match["web_name"]
                fpl_full_name = match["full_name"]
        rows.append(
            {
                "understat_id": int(u["understat_id"]),
                "understat_name": u["understat_name"],
                "understat_team": u["team_title"],
                "fpl_id": fpl_id,
                "fpl_web_name": fpl_web_name,
                "fpl_full_name": fpl_full_name,
                "match_score": round(score, 1),
                "needs_review": score < MATCH_SCORE_THRESHOLD or fpl_id is None,
            }
        )
    return pd.DataFrame(rows)


def update_player_map(player_xg: pd.DataFrame, players_fpl: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-match new Understat players against FPL and merge into
    `fpl/player_map.csv`, preserving any hand-fixed rows already there."""
    existing = pd.read_csv(PLAYER_MAP_PATH) if PLAYER_MAP_PATH.exists() else pd.DataFrame()
    known_ids = set(existing["understat_id"]) if not existing.empty else set()

    new_understat = player_xg[~player_xg["understat_id"].astype(int).isin(known_ids)]
    if new_understat.empty:
        combined = existing
    else:
        new_rows = fuzzy_match_players(new_understat, players_fpl, aliases)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(PLAYER_MAP_PATH, index=False)

    review_count = int(combined["needs_review"].sum()) if not combined.empty else 0
    if review_count:
        print(f"  {review_count} row(s) in {PLAYER_MAP_PATH.name} need a manual look (low score or no FPL match).")
    return combined


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    day_dir = RAW_DIR / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    season = current_season()
    print(f"Fetching Understat {LEAGUE} {season} league data...")
    payload = fetch_league_data(day_dir, season)

    aliases = load_aliases()
    team_xg_raw = build_team_xg_table(payload, aliases) if payload else pd.DataFrame()
    if not team_xg_raw.empty:
        team_xg_raw = add_rolling_strength(team_xg_raw)

    team_xg, health = record("understat", team_xg_raw, _team_xg_sanity_check)
    write_health_record(health)
    if health["status"] == "ok":
        print(f"  health: ok ({health['row_count']} rows)")
    elif health["status"] == "stale":
        print(f"  health: STALE -- {health['error']}; falling back to the last good team_xg table "
              f"(saved {health['last_success_at']})")
    else:
        print(f"  health: FAILED -- {health['error']}; no last-good team_xg table to fall back to either")

    if team_xg.empty:
        print("No team_xg data available (fetch + fallback both empty); team_xg.parquet not written.")
    else:
        path = TABLES_DIR / "team_xg.parquet"
        team_xg.to_parquet(path, index=False)
        print(f"  wrote {path} ({len(team_xg)} rows, {team_xg['team_id'].nunique()} teams)")

    player_xg_raw = build_player_xg_table(payload) if payload else pd.DataFrame()
    if player_xg_raw.empty:
        # Falls back the same way as team_xg, keyed separately since it's a
        # different shape -- but doesn't get its own source_health entry
        # (team_xg's record above already represents "understat" health).
        player_xg, saved_at = load_last_good("understat_player_xg")
        if player_xg.empty:
            print("No Understat player data available (fetch + fallback both empty); player_xg.parquet not written.")
            return
        print(f"  falling back to the last good player_xg table (saved {saved_at})")
    else:
        player_xg = player_xg_raw
        save_last_good("understat_player_xg", player_xg)

    players_path = TABLES_DIR / "players.parquet"
    if not players_path.exists():
        print("data/tables/players.parquet not found (run `uv run python -m fpl.fetch` first); "
              "skipping player_xg.parquet and player_map.csv.")
        return
    players_fpl = pd.read_parquet(players_path)

    print("Matching Understat players to FPL ids by name + team...")
    player_map = update_player_map(player_xg, players_fpl, aliases)

    player_xg = player_xg.merge(
        player_map[["understat_id", "fpl_id", "needs_review"]], on="understat_id", how="left"
    )
    path = TABLES_DIR / "player_xg.parquet"
    player_xg.to_parquet(path, index=False)
    matched = int(player_xg["fpl_id"].notna().sum())
    shaky = int(player_xg["needs_review"].sum())
    print(f"  wrote {path} ({len(player_xg)} rows, {matched} matched to an FPL id, {shaky} flagged needs_review)")


if __name__ == "__main__":
    main()
