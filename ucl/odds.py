"""Bookmaker odds for upcoming UCL matches (The Odds API, sport key
`soccer_uefa_champs_league`), turned into expected goals per side with the
PL model's own de-vig and Poisson fit (fpl.odds).

The key comes from the ODDS_API_KEY environment variable only (the Mac app
passes it from the Keychain, CI from a GitHub secret); it is never written
anywhere or printed. One request (UK region, h2h + totals = 2 credits) at
most every CACHE_HOURS, and only when a UCL match is less than LOOKAHEAD_DAYS
away. Events are matched to UEFA fixtures by kick-off time (within 3 hours)
and the closest club names.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from rapidfuzz import fuzz

from fpl.net import redact
from fpl.odds import consensus_probabilities, fit_lambdas, odds_method
from ucl import feed

URL = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds"
CACHE_HOURS = 6
LOOKAHEAD_DAYS = 6


def fetch(now: datetime) -> list[dict] | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        feed.note_health("odds", "skipped", "ODDS_API_KEY not set")
        return None
    bucket = now.strftime("%Y%m%d") + f"{now.hour // CACHE_HOURS:02d}"
    f = feed.RAW / f"odds_{bucket}.json"
    if f.exists():
        feed.note_health("odds", "ok", None, cached=True)
        return json.loads(f.read_text())
    url = f"{URL}?regions=uk&markets=h2h,totals&oddsFormat=decimal&apiKey={key}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            feed.note_health("odds", "failed", f"{redact(url)} -> HTTP {r.status_code}")
            return None
        data = r.json()
    except requests.RequestException as e:
        feed.note_health("odds", "failed", type(e).__name__)
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data))
    feed.note_health("odds", "ok", None, events=len(data), remaining=r.headers.get("x-requests-remaining"))
    return data


def lambdas(matches: pd.DataFrame, now: datetime | None = None) -> dict[int, tuple[float, float]]:
    now = now or datetime.now(timezone.utc)
    soon = matches[(~matches["finished"]) & (matches["kickoff"] > now) &
                   (matches["kickoff"] < now + timedelta(days=LOOKAHEAD_DAYS))]
    if soon.empty:
        feed.note_health("odds", "skipped", "no UCL match in the next days")
        return {}
    events = fetch(now)
    if not events:
        return {}
    out = {}
    for ev in events:
        t = pd.Timestamp(ev["commence_time"])
        near = soon[(soon["kickoff"] - t).abs() < pd.Timedelta(hours=3)]
        if near.empty:
            continue
        score = [(fuzz.token_set_ratio(ev["home_team"], r.home) + fuzz.token_set_ratio(ev["away_team"], r.away), r)
                 for r in near.itertuples()]
        best, r = max(score, key=lambda x: x[0])
        if best < 110:
            continue
        probs = consensus_probabilities(ev, odds_method())
        if probs:
            out[int(r.mid)] = fit_lambdas(probs)
    if out:
        feed.note_health("odds", "ok", None, matched=len(out))
    return out
