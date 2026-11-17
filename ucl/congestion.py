"""Each UCL club's match calendar around a UCL match, for the rotation factor.

For a club's UCL match: days since its previous match (domestic or UCL),
days until its next domestic match, and whether that next domestic
opponent is in its league's top 4 (points per game before the UCL match).

Past matches come from the stored domestic results (ucl.domestic), which
carry the real dates. Upcoming domestic fixtures aren't in those files, so
they come from openfootball's season schedules (free, no key; England,
Spain, Germany, Italy, France, Portugal, Netherlands), cached once a day
in data/ucl/raw/fixtures/; club names are matched to football-data's
with rapidfuzz. Other leagues get no rotation adjustment (neutral).
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from ucl import domestic, elo, feed

OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/{}.json"
OF_CODE = {"ENG": "en.1", "ESP": "es.1", "GER": "de.1", "ITA": "it.1", "FRA": "fr.1", "POR": "pt.1", "NED": "nl.1"}
CACHE = feed.RAW / "fixtures"


def _fixtures(league: str) -> pd.DataFrame:
    """Upcoming domestic fixtures (date, home, away) in football-data names; empty if unavailable."""
    code = OF_CODE.get(league)
    if not code:
        return pd.DataFrame(columns=["date", "home", "away"])
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{code}.json"
    try:
        if not f.exists() or time.time() - f.stat().st_mtime > 86400:
            r = requests.get(OPENFOOTBALL.format(code), headers={"User-Agent": feed.USER_AGENT}, timeout=30)
            r.raise_for_status()
            f.write_text(r.text)
            time.sleep(feed.REQUEST_GAP)
        ms = json.loads(f.read_text())["matches"]
    except Exception as e:  # noqa: BLE001 - neutral without fixtures
        feed.note_health("fixtures", "stale", f"{league}: {str(e)[:120]}")
        return pd.DataFrame(columns=["date", "home", "away"])
    names = _name_map(league, ms)
    rows = [(pd.Timestamp(m["date"]), names.get(m["team1"]), names.get(m["team2"])) for m in ms if m.get("date")]
    return pd.DataFrame([r for r in rows if r[1] and r[2]], columns=["date", "home", "away"])


def _name_map(league: str, ms: list[dict]) -> dict[str, str]:
    """openfootball name -> football-data name, from matches both list on the same day:
    within one day's few matches, fuzzy matching is reliable; each name takes its majority."""
    from collections import Counter

    d = _results()
    d = d[(d["league"] == league) & (d["season"] == domestic.CURRENT)]
    votes: dict[str, Counter] = {}
    by_day = {k: g for k, g in d.groupby(d["date"].dt.normalize())}
    for m in ms:
        if not m.get("date"):
            continue
        g = by_day.get(pd.Timestamp(m["date"]))
        if g is None:
            continue
        pairs = [f"{h} | {a}" for h, a in zip(g["home"], g["away"])]
        hit = process.extractOne(f"{m['team1']} | {m['team2']}", pairs, scorer=fuzz.token_set_ratio)
        if hit and hit[1] >= 50:
            h, a = hit[0].split(" | ")
            votes.setdefault(m["team1"], Counter())[h] += 1
            votes.setdefault(m["team2"], Counter())[a] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items() if c.most_common(1)[0][1] >= 2}


@lru_cache(maxsize=4)
def _results() -> pd.DataFrame:
    d = domestic.load().copy()
    d["date"] = pd.to_datetime(d["date"])
    return d


def _table(league: str, season: str, before: pd.Timestamp) -> list[str]:
    d = _results()
    s = d[(d["league"] == league) & (d["season"] == season) & (d["date"] < before)]
    ppg = {}
    for t in set(s["home"]) | set(s["away"]):
        h, a = s[s["home"] == t], s[s["away"] == t]
        n = len(h) + len(a)
        if n:
            pts = 3 * ((h.hg > h.ag).sum() + (a.ag > a.hg).sum()) + (h.hg == h.ag).sum() + (a.ag == a.hg).sum()
            ppg[t] = pts / n
    return sorted(ppg, key=ppg.get, reverse=True) if len(ppg) >= 8 else []


def features(matches: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """(mid, team_id) -> {rest_before, rest_after, big_next} for each UCL match in `matches`.

    No lookahead: only fixture dates and opponents after the UCL match are
    used (known in advance), and the table only counts results before it."""
    cm = elo.club_map()
    key = {r.team_code: (r.league, r.source_name) for r in cm.itertuples() if r.source_name}
    d = _results()
    out: dict[tuple[int, int], dict] = {}
    allm = matches.copy()
    allm["day"] = pd.to_datetime(allm["kickoff"]).dt.tz_convert(None).dt.normalize()
    fx_cache: dict[str, pd.DataFrame] = {}
    for r in allm.itertuples():
        for tid, code in ((r.home_id, r.home_code), (r.away_id, r.away_code)):
            if code not in key:
                continue
            lg, club = key[code]
            dom = d[(d["league"] == lg) & ((d["home"] == club) | (d["away"] == club))][["date", "home", "away", "season"]]
            if dom["date"].max() < r.day + pd.Timedelta(days=1):  # season still running: add the schedule
                if lg not in fx_cache:
                    fx_cache[lg] = _fixtures(lg)
                fx = fx_cache[lg]
                fx = fx[((fx["home"] == club) | (fx["away"] == club)) & (fx["date"] > dom["date"].max())]
                dom = pd.concat([dom, fx.assign(season=domestic.CURRENT)], ignore_index=True)
            ucl_days = allm.loc[(allm["home_code"] == code) | (allm["away_code"] == code), "day"]
            before = pd.concat([dom["date"], ucl_days])
            before = before[before < r.day]
            after = dom[dom["date"] > r.day].sort_values("date")
            f = {"rest_before": int((r.day - before.max()).days) if len(before) else None,
                 "rest_after": None, "big_next": False}
            if len(after):
                nxt = after.iloc[0]
                opp = nxt["away"] if nxt["home"] == club else nxt["home"]
                f["rest_after"] = int((nxt["date"] - r.day).days)
                f["big_next"] = opp in _table(lg, nxt["season"], r.day)[:4]
            out[(int(r.mid), int(tid))] = f
    return out


def rotation_factor(f: dict | None, short: float, big: float) -> float:
    """Multiplier on the chance of starting / 60+ minutes."""
    if not f:
        return 1.0
    m = 1.0
    if f.get("rest_after") is not None and f["rest_after"] <= 3:
        m *= short
    if f.get("big_next"):
        m *= big
    return m
