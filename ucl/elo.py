"""Team strength from every match: our own Elo-style goal ratings.

Each club has an attack and a defence rating (log goals). Before a match
the expected goals are

    domestic: exp(mu[league] + HOME + att[home] - def[away])          (and the mirror for the away side)
    UCL:      exp(mu_ucl + HOME + att[home] + L[league home] - def[away] - L[league away])

and after it every rating moves by K x (goals - expected goals), the
Elo idea on the goals scale. Domestic matches (ucl.domestic: 13 leagues,
2024/25 as warm-up, then 2025/26 and 2026/27) set each club's level
within its league; UCL matches between leagues move the league offsets L
(K_LEAGUE), which start from round priors ordered like UEFA's country
ranking (LEAGUE_PRIOR). Clubs from leagues with no free results
(Czechia, Slovakia, Ukraine, Azerbaijan, Kazakhstan, Cyprus) only have
their UCL matches. At each new season, ratings are pulled a third of the
way back to their league's average.

`ratings_at(cutoff)` replays everything strictly before `cutoff`, so the
backtest has no lookahead. EloTeams has the same interface as
ucl.model.TeamModel (lambdas, rating, avg_for, pot_of), so the xP model
and the knockout simulation can use it directly or blended.

Run ``uv run python -m ucl.elo`` to print today's UCL clubs, strongest first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ucl import domestic, feed

CLUB_MAP = Path(__file__).resolve().parent / "club_map.csv"
HOME = 0.2  # home advantage, log goals (about +22%)
K = 0.05  # rating step per goal of surprise, domestic and UCL
K_LEAGUE = 0.02  # league offset step per goal of surprise, UCL matches between leagues
K_MU = 0.01
SEASON_REGRESS = 1 / 3
NEW_CLUB = -0.15  # a club new to its league (promoted) starts below the league average
# league strength priors (log goals, added to attack and to defence): round values, UEFA ranking order
LEAGUE_PRIOR = {"ENG": 0.2, "ESP": 0.15, "GER": 0.1, "ITA": 0.1, "FRA": 0.05, "POR": 0.0, "NED": 0.0,
                "BEL": -0.05, "TUR": -0.05, "CZE": -0.1, "GRE": -0.1, "AUT": -0.1, "DEN": -0.1, "UKR": -0.1,
                "NOR": -0.15, "SVK": -0.2, "AZE": -0.2, "CYP": -0.2, "KAZ": -0.25}
# a club from a league with no domestic data: one of its league's better clubs
NO_DATA_CLUB = 0.25
# how much of a club's rating within its league carries over to UCL matches
# (ratings earned against weak domestic opponents overstate it); tuned on 2025/26
UCL_SCALE = 1.0


def club_map() -> pd.DataFrame:
    return pd.read_csv(CLUB_MAP, dtype=str).fillna("")


def _season_of(d: pd.Timestamp) -> int:
    return d.year if d.month >= 7 else d.year - 1


@dataclass
class State:
    att: dict = field(default_factory=dict)
    dfn: dict = field(default_factory=dict)
    lg_att: dict = field(default_factory=lambda: dict(LEAGUE_PRIOR))
    lg_def: dict = field(default_factory=lambda: dict(LEAGUE_PRIOR))
    mu: dict = field(default_factory=dict)
    mu_ucl: float = math.log(1.45)
    league_of: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)  # matches seen per club
    seen_season: dict = field(default_factory=dict)


def _ucl_matches() -> pd.DataFrame:
    rows = []
    for tour in (feed.PREV_TOUR, feed.TOUR):
        f = feed.DATA / "history" / f"matches_{tour}.parquet"
        if f.exists():
            m = pd.read_parquet(f)
            rows.append(m[m["finished"]].dropna(subset=["home_goals", "away_goals"]))
    m = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return m


def _events(cutoff: pd.Timestamp) -> list[tuple]:
    """Every match strictly before `cutoff`, oldest first: (date, kind, league/None, home key, away key, hg, ag, neutral)."""
    cm = club_map()
    key_of_code = {r.team_code: (f"{r.league}:{r.source_name}" if r.source_name else f"{r.league}:#{r.team_code}")
                   for r in cm.itertuples()}
    ev = []
    d = domestic.load()
    if len(d):
        d = d[pd.to_datetime(d["date"]) < cutoff.tz_convert(None).normalize()]
        for r in d.itertuples():
            ev.append((pd.Timestamp(r.date), 0, r.league, f"{r.league}:{r.home}", f"{r.league}:{r.away}",
                       int(r.hg), int(r.ag), False))
    m = _ucl_matches()
    if len(m):
        m = m[m["kickoff"] < cutoff]
        for r in m.itertuples():
            final = "final" in str(r.stage).lower() and not any(s in str(r.stage).lower() for s in ("semi", "quarter"))
            hk, ak = key_of_code.get(r.home_code), key_of_code.get(r.away_code)
            if hk and ak:
                ev.append((pd.Timestamp(r.kickoff).tz_convert(None), 1, None, hk, ak, int(r.home_goals),
                           int(r.away_goals), final))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def _init(st: State, key: str, season: int) -> None:
    if key in st.att:
        return
    lg = key.split(":")[0]
    st.league_of[key] = lg
    no_data = ":#" in key
    first_season = not st.seen_season.get(lg)
    v = NO_DATA_CLUB if no_data else (0.0 if first_season else NEW_CLUB)
    st.att[key] = v
    st.dfn[key] = v
    st.n[key] = 0


def run(cutoff: pd.Timestamp, k: float = K, k_league: float = K_LEAGUE) -> State:
    st = State()
    season = None
    for date, kind, lg, hk, ak, hg, ag, neutral in _events(cutoff):
        s = _season_of(date)
        if season is not None and s != season:  # new season: a third of the way back to the league average
            for key in st.att:
                st.att[key] *= 1 - SEASON_REGRESS
                st.dfn[key] *= 1 - SEASON_REGRESS
        season = s
        _init(st, hk, s)
        _init(st, ak, s)
        h = 0.0 if neutral else HOME
        if kind == 0:
            st.seen_season.setdefault(lg, set()).add(s)
            mu = st.mu.setdefault(lg, math.log(1.35))
            lh = math.exp(mu + h + st.att[hk] - st.dfn[ak])
            la = math.exp(mu + st.att[ak] - st.dfn[hk])
            eh, ea = hg - lh, ag - la
            st.mu[lg] = mu + K_MU * (eh + ea) / 2
        else:
            lgh, lga = st.league_of[hk], st.league_of[ak]
            lh = math.exp(st.mu_ucl + h + st.att[hk] + st.lg_att.get(lgh, -0.2) - st.dfn[ak] - st.lg_def.get(lga, -0.2))
            la = math.exp(st.mu_ucl + st.att[ak] + st.lg_att.get(lga, -0.2) - st.dfn[hk] - st.lg_def.get(lgh, -0.2))
            eh, ea = hg - lh, ag - la
            st.mu_ucl += K_MU * (eh + ea) / 2
            if lgh != lga:
                st.lg_att[lgh] = st.lg_att.get(lgh, -0.2) + k_league * eh
                st.lg_def[lga] = st.lg_def.get(lga, -0.2) - k_league * eh
                st.lg_att[lga] = st.lg_att.get(lga, -0.2) + k_league * ea
                st.lg_def[lgh] = st.lg_def.get(lgh, -0.2) - k_league * ea
        st.att[hk] += k * eh
        st.dfn[ak] -= k * eh
        st.att[ak] += k * ea
        st.dfn[hk] -= k * ea
        st.n[hk] += 1
        st.n[ak] += 1
    return st


@dataclass
class EloTeams:
    """Team strengths for UCL matches from an Elo State, by UEFA team id."""
    st: State
    key_of: dict[int, str]
    pot_of: dict[int, int]
    home: float = HOME
    scale: float = UCL_SCALE

    @property
    def mu(self) -> float:
        return self.st.mu_ucl

    def rating(self, tid: int) -> tuple[float, float]:
        key = self.key_of.get(tid)
        if key is None:
            return LEAGUE_PRIOR["NOR"], LEAGUE_PRIOR["NOR"]
        lg = key.split(":")[0]
        if key not in self.st.att:  # never played before the cutoff
            v = NO_DATA_CLUB if ":#" in key else 0.0
            return v + self.st.lg_att.get(lg, -0.2), v + self.st.lg_def.get(lg, -0.2)
        c = 1.0 if ":#" in key else self.scale  # no-data clubs have only UCL results already on the UCL scale
        return (c * self.st.att[key] + self.st.lg_att.get(lg, -0.2),
                c * self.st.dfn[key] + self.st.lg_def.get(lg, -0.2))

    def lambdas(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        ah, dh = self.rating(home_id)
        aa, da = self.rating(away_id)
        h = 0.0 if neutral else self.home
        return math.exp(self.mu + h + ah - da), math.exp(self.mu + aa - dh)

    def avg_for(self, tid: int) -> tuple[float, float]:
        a, d = self.rating(tid)
        oa = float(np.mean([self.rating(t)[0] for t in self.pot_of]))
        od = float(np.mean([self.rating(t)[1] for t in self.pot_of]))
        hs = math.exp(self.mu + self.home / 2)
        return hs * math.exp(a - od), hs * math.exp(oa - d)

    def matches_behind(self, tid: int) -> tuple[int, str]:
        """(matches rated, league) for the app's 'what it rests on'."""
        key = self.key_of.get(tid, "")
        return self.st.n.get(key, 0), key.split(":")[0]


def key_of_ids(matches: pd.DataFrame) -> dict[int, str]:
    cm = club_map()
    code_key = {r.team_code: (f"{r.league}:{r.source_name}" if r.source_name else f"{r.league}:#{r.team_code}")
                for r in cm.itertuples()}
    out = {}
    for side in ("home", "away"):
        for tid, code in zip(matches[f"{side}_id"], matches[f"{side}_code"]):
            if code in code_key:
                out[int(tid)] = code_key[code]
    return out


@lru_cache(maxsize=64)
def _cached(cutoff_iso: str, k: float, k_league: float) -> State:
    return run(pd.Timestamp(cutoff_iso), k, k_league)


def teams_at(cutoff: pd.Timestamp, matches: pd.DataFrame, pot_of: dict[int, int],
             k: float = K, k_league: float = K_LEAGUE, scale: float | None = None) -> EloTeams:
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    return EloTeams(_cached(cutoff.isoformat(), k, k_league), key_of_ids(matches), dict(pot_of),
                    scale=UCL_SCALE if scale is None else scale)


if __name__ == "__main__":
    m = pd.read_parquet(feed.DATA / "history" / f"matches_{feed.TOUR}.parquet")
    from ucl.model import pots

    t = teams_at(pd.Timestamp.now(tz="UTC"), m, pots(m))
    rows = []
    for tid, key in t.key_of.items():
        if tid in t.pot_of:
            f, a = t.avg_for(tid)
            rows.append((f - a, key, round(f, 2), round(a, 2), t.st.n.get(key, 0)))
    for r in sorted(rows, reverse=True):
        print(f"{r[1]:28s} scores {r[2]:.2f} concedes {r[3]:.2f} v average UCL side ({r[4]} matches)")
    print("league offsets (attack, defence):", {k: (round(t.st.lg_att[k], 2), round(t.st.lg_def[k], 2)) for k in LEAGUE_PRIOR})
