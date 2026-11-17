"""Domestic league results for the UCL clubs' leagues (football-data.co.uk).

Free CSVs, no key: https://www.football-data.co.uk/
* main leagues, one file per season: mmz4281/{2425,2526,2627}/{E0,SP1,D1,I1,F1,P1,N1,B1,T1,G1}.csv
* "extra" leagues, all seasons in one file: new/{AUT,DNK,NOR}.csv

No free, key-less source was found for Czechia, Slovakia, Ukraine,
Azerbaijan, Kazakhstan or Cyprus (openfootball has no files for them), so
clubs from those leagues are rated on their UCL results only (ucl.elo).

2024/25 is fetched only to warm the ratings up before 2025/26 starts; it
is never predicted or tested.

Stored compactly in data/ucl/history/domestic.parquet (committed, ~1 MB),
so a run only re-downloads the current season's files (at most every
REFRESH_HOURS), politely (one request at a time, REQUEST_GAP between them,
timeouts and retries). A league whose download fails or fails the sanity
check keeps its stored rows, and source health says so ("domestic").

Run with ``uv run python -m ucl.domestic`` (``--all`` re-downloads every season).
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from ucl import feed

OUT = feed.DATA / "history" / "domestic.parquet"
META = feed.DATA / "history" / "domestic_meta.json"  # when it was last downloaded (file times don't survive a checkout)
BASE = "https://www.football-data.co.uk/"
MAIN = {"ENG": "E0", "ESP": "SP1", "GER": "D1", "ITA": "I1", "FRA": "F1", "POR": "P1", "NED": "N1",
        "BEL": "B1", "TUR": "T1", "GRE": "G1"}
EXTRA = {"AUT": "AUT", "DEN": "DNK", "NOR": "NOR"}
SEASONS = ("2425", "2526", "2627")  # 2425 = warm-up only
CURRENT = "2627"
REFRESH_HOURS = 12
REQUEST_GAP = 1.5
TIMEOUT = 30

_session = requests.Session()
_session.headers["User-Agent"] = feed.USER_AGENT
_last = 0.0


def _get(path: str) -> str:
    global _last
    err = None
    for attempt in range(3):
        gap = time.monotonic() - _last
        if gap < REQUEST_GAP:
            time.sleep(REQUEST_GAP - gap)
        _last = time.monotonic()
        try:
            r = _session.get(BASE + path, timeout=TIMEOUT)
            r.raise_for_status()
            return r.content.decode("utf-8-sig", errors="replace")
        except requests.RequestException as e:
            err = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"{path}: {err}")


def _dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def _main(league: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_get(f"mmz4281/{season}/{MAIN[league]}.csv")), on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return pd.DataFrame({"league": league, "season": season, "date": _dates(df["Date"]),
                         "home": df["HomeTeam"].str.strip(), "away": df["AwayTeam"].str.strip(),
                         "hg": df["FTHG"].astype(int), "ag": df["FTAG"].astype(int)})


def _season_label(league: str, s: str) -> str:
    """Our season code for an "extra" file's Season value."""
    if "/" in s:  # 2025/2026
        return s[2:4] + s[7:9]
    y = int(s)  # Norway plays the calendar year, alongside the UCL season that starts that autumn: 2026 -> 2627
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _extra(league: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_get(f"new/{EXTRA[league]}.csv")), on_bad_lines="skip")
    df = df.dropna(subset=["Home", "Away", "HG", "AG"])
    df["season"] = df["Season"].astype(str).map(lambda s: _season_label(league, s))
    df = df[df["season"].isin(SEASONS)]
    return pd.DataFrame({"league": league, "season": df["season"], "date": _dates(df["Date"]),
                         "home": df["Home"].str.strip(), "away": df["Away"].str.strip(),
                         "hg": df["HG"].astype(int), "ag": df["AG"].astype(int)})


def _check(df: pd.DataFrame, league: str) -> str | None:
    if df.empty:
        return f"{league}: no rows"
    if df["date"].isna().mean() > 0.05:
        return f"{league}: unreadable dates"
    if ((df["hg"] > 12) | (df["ag"] > 12) | (df["hg"] < 0)).any():
        return f"{league}: implausible score"
    if df[["home", "away"]].stack().nunique() < 8:
        return f"{league}: too few clubs"
    return None


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(
        columns=["league", "season", "date", "home", "away", "hg", "ag"])


def update(all_seasons: bool = False, force: bool = False) -> pd.DataFrame:
    """Refresh the stored results (current season only unless `all_seasons`)."""
    old = load()
    if not force and not all_seasons and OUT.exists() and META.exists():
        fetched = datetime.fromisoformat(json.loads(META.read_text())["fetched_at"])
        age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        if age_h < REFRESH_HOURS:
            feed.note_health("domestic", "ok", None, rows=len(old), note=f"stored copy, {age_h:.0f} h old")
            return old
    seasons = SEASONS if all_seasons or old.empty else (CURRENT,)
    parts, problems = [], []
    for league in MAIN:
        for s in seasons:
            try:
                df = _main(league, s)
                p = _check(df, league)
                if p and s == CURRENT and df.empty and not old[(old.league == league)].empty:
                    p = None  # a new season's file can be empty in August
                if p:
                    raise RuntimeError(p)
                parts.append(df)
            except Exception as e:  # noqa: BLE001 - keep the stored rows for this league/season
                problems.append(f"{league} {s}: {str(e)[:80]}")
                parts.append(old[(old.league == league) & (old.season == s)])
    for league in EXTRA:
        try:
            df = _extra(league)
            df = df[df["season"].isin(seasons)]
            p = _check(df, league)
            if p:
                raise RuntimeError(p)
            parts.append(df)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{league}: {str(e)[:80]}")
            parts.append(old[(old.league == league) & (old.season.isin(seasons))])
    keep = old[~old["season"].isin(seasons)] if len(old) else old
    new = pd.concat([keep, *parts], ignore_index=True).drop_duplicates(["league", "date", "home", "away"])
    new = new.sort_values(["date", "league", "home"]).reset_index(drop=True)
    new["hg"] = new["hg"].astype(int)
    new["ag"] = new["ag"].astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(OUT, index=False)
    META.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "seasons": list(seasons), "problems": problems}, indent=1))
    status = "ok" if not problems else ("stale" if len(new) else "failed")
    feed.note_health("domestic", status, "; ".join(problems)[:300] or None, rows=len(new),
                     latest=str(new["date"].max().date()) if len(new) else None)
    return new


if __name__ == "__main__":
    df = update(all_seasons="--all" in sys.argv, force=True)
    print(df.groupby(["league", "season"]).size().unstack(fill_value=0))
    print(feed.health().get("domestic"))
    print("as of", datetime.now(timezone.utc).isoformat(timespec="seconds"))
