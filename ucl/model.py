"""Expected points (xP) per player per UCL match.

Two layers:

1. Team strength: a Poisson goals model fitted on every finished UCL match
   of 2025/26 and 2026/27 (2025/26 at half weight). Each club has an attack
   and a defence rating, pulled towards the average of its seeding pot so a
   club with few matches isn't judged on one result. When bookmaker odds are
   available for a match (The Odds API, `soccer_uefa_champs_league`), the
   goals they imply are blended in (ODDS_WEIGHT). ClubElo, the PL model's
   extra strength source, is behind a Cloudflare challenge and isn't used.

2. Players: chance of playing and of playing 60+ minutes from his club's
   recent UCL matches (recent ones count more), UEFA's injury/suspension
   status for the next matchday, and per-90 rates for every scoring action
   (goals, assists, balls recovered, saves, Player of the Match, cards,
   penalties ...), shrunk towards the average for his position, with
   2025/26 at half weight. Goals and assists scale with how many goals his
   club should score in that match; clean sheets, goals conceded and saves
   with how many the opponent should score. Each expected count is turned
   into points with the exact rules in ucl.rules (including the "every 3"
   and "every 2" steps, worked out from the Poisson distribution).

`predict(tour, as_of_md)` uses only matches before `as_of_md`, so the
backtest (ucl.backtest) can replay the season without looking ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ucl import elo, feed, rules

HIST = feed.DATA / "history"
PREV_SEASON_WEIGHT = 0.5
RECENCY = 0.8  # weight of each older club match in the minutes model
SHRINK_90S = 3.0  # prior strength for per-90 rates, in full matches
TEAM_SIGMA = 0.35  # how far a club's rating may move from its pot's average
ODDS_WEIGHT = 0.6
STATUS_NEXT = {"I": 0.05, "S": 0.0, "NIS": 0.0, "D": 0.6}
STATUS_LATER = {"I": 0.5, "S": 1.0, "NIS": 0.0, "D": 0.9}
UNLIKELY_NEXT = 0.35  # "Unlikely to start next game"
MAX_N = 25

# Switchable refinements (ucl.factors backtests each one on its own; only
# the ones that improved the unseen 2026/27 matchdays without clearly
# hurting 2025/26 are on -- see data/ucl/factor_tests.json).
CFG: dict = {
    "elo_weight": 0.0,  # share of the Elo team ratings (ucl.elo) in each match's expected goals; rest = UCL-only Poisson fit
    "shrink_90s": SHRINK_90S,  # prior strength for per-90 rates, in full matches (how slowly UCL minutes take over)
    "prior_price": 0.0,  # attacking prior x (price / position's typical price) ** this
    "prior_team": 1.0,  # attacking prior x (team's expected goals / average) ** this
    "prior_team_elo": True,  # ... with the team's expected goals from the Elo ratings (all matches) only
    "elo_weight_knockout": 0.5,  # share of Elo in the knockout simulation's expected goals
}
ATTACK_COLS = ("gS", "gA", "gOB", "mom")


def load(tour: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(HIST / f"matches_{tour}.parquet")
    p = pd.read_parquet(HIST / f"players_{tour}.parquet")
    a_path = HIST / f"appearances_{tour}.parquet"
    a = pd.read_parquet(a_path) if a_path.exists() else pd.DataFrame()
    if len(a):
        a = a[a["mid"].isin(m.loc[m["finished"], "mid"])].copy()  # drop listed-but-unplayed fixtures
        if "sS" in a:
            a["saves"] = a["sS"]  # shots saved
    return m, p, a


def pot_no(s) -> int:
    try:
        return int(str(s).split()[-1])
    except (ValueError, IndexError):
        return 3


# ------------------------------------------------------------ team model

@dataclass
class TeamModel:
    mu: float
    home: float
    att: dict[int, float]
    dfn: dict[int, float]
    pot_att: np.ndarray
    pot_def: np.ndarray
    pot_of: dict[int, int]

    def rating(self, tid: int) -> tuple[float, float]:
        p = self.pot_of.get(tid, 3) - 1
        return self.pot_att[p] + self.att.get(tid, 0.0), self.pot_def[p] + self.dfn.get(tid, 0.0)

    def lambdas(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        ah, dh = self.rating(home_id)
        aa, da = self.rating(away_id)
        h = 0.0 if neutral else self.home
        return math.exp(self.mu + h + ah - da), math.exp(self.mu + aa - dh)

    def avg_for(self, tid: int) -> tuple[float, float]:
        """Goals the club scores / concedes against an average opponent, half at home."""
        a, d = self.rating(tid)
        oa = float(np.mean([self.rating(t)[0] for t in self.pot_of]))
        od = float(np.mean([self.rating(t)[1] for t in self.pot_of]))
        hs = math.exp(self.mu + self.home / 2)
        return hs * math.exp(a - od), hs * math.exp(oa - d)


def fit_teams(matches: pd.DataFrame, pot_of: dict[int, int]) -> TeamModel:
    m = matches[matches["finished"]].dropna(subset=["home_goals", "away_goals"])
    teams = sorted(set(m["home_id"]) | set(m["away_id"]) | set(pot_of))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = m["home_id"].map(ix).to_numpy(dtype=int)
    ai = m["away_id"].map(ix).to_numpy(dtype=int)
    hg = m["home_goals"].to_numpy(float)
    ag = m["away_goals"].to_numpy(float)
    w = m["weight"].to_numpy(float) if "weight" in m else np.ones(len(m))
    neutral = (m["stage"].str.lower().str.contains("final") & ~m["stage"].str.lower().str.contains("semi|quarter")).to_numpy()
    hp = np.array([pot_of.get(t, 3) - 1 for t in m["home_id"]], dtype=int)
    ap = np.array([pot_of.get(t, 3) - 1 for t in m["away_id"]], dtype=int)

    def unpack(x):
        return x[0], x[1], np.r_[0.0, x[2:5]], np.r_[0.0, x[5:8]], x[8:8 + n], x[8 + n:8 + 2 * n]

    def nll(x):
        mu, home, pa, pdf, a, d = unpack(x)
        hh = np.where(neutral, 0.0, home)
        lh = mu + hh + pa[hp] + a[hi] - pdf[ap] - d[ai]
        la = mu + pa[ap] + a[ai] - pdf[hp] - d[hi]
        ll = w * (hg * lh - np.exp(lh) + ag * la - np.exp(la))
        pen = (np.sum(a ** 2) + np.sum(d ** 2)) / (2 * TEAM_SIGMA ** 2)
        return -ll.sum() + pen

    x0 = np.zeros(8 + 2 * n)
    x0[0] = math.log(1.4)
    x0[1] = 0.2
    res = minimize(nll, x0, method="L-BFGS-B")
    mu, home, pa, pdf, a, d = unpack(res.x)
    return TeamModel(float(mu), float(home), {t: float(a[i]) for t, i in ix.items()},
                     {t: float(d[i]) for t, i in ix.items()}, pa, pdf, dict(pot_of))


# -------------------------------------------------------- Poisson helpers

_N = np.arange(MAX_N + 1)
_LOGFACT = np.array([math.lgamma(k + 1) for k in _N])


def pois_pmf(lam: float) -> np.ndarray:
    lam = max(lam, 1e-9)
    return np.exp(_N * math.log(lam) - lam - _LOGFACT)


def e_floor(lam: float, k: int) -> float:
    """E[floor(X / k)] for X ~ Poisson(lam)."""
    return float(np.sum((_N // k) * pois_pmf(lam)))


# --------------------------------------------------------- player rates

RATE_COLS = ["gS", "gA", "bR", "saves", "yC", "rC", "pE", "pM", "pS", "pC", "oG", "gOB"]
_BASE = {"yC": 0.12, "rC": 0.005, "pE": 0.01, "pM": 0.004, "pS": 0.0, "pC": 0.01, "oG": 0.004, "saves": 0.0}
# used only when there is no history at all (e.g. replaying MD1 with no stored past season)
DEFAULT_PRIOR = {
    rules.GK: {**_BASE, "gS": 0.0, "gA": 0.003, "bR": 2.0, "saves": 3.0, "pS": 0.03, "gOB": 0.0, "mom": 0.03},
    rules.DEF: {**_BASE, "gS": 0.05, "gA": 0.06, "bR": 4.0, "gOB": 0.005, "mom": 0.03},
    rules.MID: {**_BASE, "gS": 0.13, "gA": 0.13, "bR": 3.5, "gOB": 0.025, "mom": 0.05},
    rules.FWD: {**_BASE, "gS": 0.38, "gA": 0.12, "bR": 1.5, "gOB": 0.03, "pE": 0.02, "pM": 0.01, "mom": 0.08},
}


def player_rates(apps: pd.DataFrame, pos_of: dict[int, int], shrink: float = SHRINK_90S,
                 mult: dict[int, float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Shrunk per-90 rates per player (apps carry a 'w' weight column).

    Each rate is pulled towards its position's average with the weight of
    `shrink` full matches; `mult` (pid -> factor) scales that average for
    the attacking actions (ATTACK_COLS), so the pull is towards what this
    player's price and team suggest rather than towards a generic player."""
    a = apps.copy()
    a["w90"] = a["w"] * a["oF"] / 90.0
    for c in RATE_COLS:
        a[c + "_w"] = a[c] * a["w"]
    a["mom_w"] = a["mOM"] * a["w"]
    a["w_app"] = a["w"] * (a["oF"] > 0)
    g = a.groupby("pid").agg(n90=("w90", "sum"), napp=("w_app", "sum"), mom=("mom_w", "sum"),
                             **{c: (c + "_w", "sum") for c in RATE_COLS})
    g["pos"] = [pos_of.get(int(p), int(apps.loc[apps.pid == p, "pos"].iloc[0])) for p in g.index]
    prior = {}
    for pos, sub in g.groupby("pos"):
        tot90 = max(sub["n90"].sum(), 1e-6)
        prior[pos] = {c: sub[c].sum() / tot90 for c in RATE_COLS}
        prior[pos]["mom"] = sub["mom"].sum() / max(sub["napp"].sum(), 1e-6)
    out = pd.DataFrame(index=g.index)
    mlt = pd.Series([float((mult or {}).get(int(p), 1.0)) for p in g.index], index=g.index)
    for c in RATE_COLS:
        pr = g["pos"].map(lambda p: prior.get(p, prior.get(rules.MID, {})).get(c, 0.0))
        if c in ATTACK_COLS:
            pr = pr * mlt
        out[c] = (g[c] + shrink * pr) / (g["n90"] + shrink)
    pr = g["pos"].map(lambda p: prior.get(p, {}).get("mom", 0.03)) * mlt
    out["mom"] = (g["mom"] + shrink * pr) / (g["napp"] + shrink)
    out["pos"] = g["pos"]
    out["n90"] = g["n90"]
    return out, prior


def prior_mult(players: pd.DataFrame, apps: pd.DataFrame, team_goals: dict[int, float],
               price_pow: float, team_pow: float) -> dict[int, float]:
    """pid -> factor on his position's attacking prior, from price and team strength.

    Price: his latest price in a match before the cutoff, else the listed one.
    Normalised so the factors average 1 per position over the players who play."""
    if not price_pow and not team_pow:
        return {}
    last_price = apps.sort_values("md").groupby("pid")["price"].last().to_dict() if len(apps) and "price" in apps else {}
    df = players[["pid", "team_id", "pos", "price"]].copy()
    df["pid"] = df["pid"].astype(int)
    df["price"] = [float(last_price.get(p, pr) or 4.5) for p, pr in zip(df["pid"], df["price"].fillna(4.5))]
    tg = pd.Series(team_goals)
    df["tgoal"] = df["team_id"].astype(int).map(tg).fillna(tg.mean() if len(tg) else 1.4)
    played = set(apps.loc[apps["oF"] > 0, "pid"].astype(int)) if len(apps) else set()
    out = {}
    for _, sub in df.groupby("pos"):
        ref = sub[sub["pid"].isin(played)] if sub["pid"].isin(played).sum() >= 10 else sub
        m = (sub["price"] / ref["price"].median()) ** price_pow * (sub["tgoal"] / ref["tgoal"].mean()) ** team_pow
        norm = float(m[sub["pid"].isin(ref["pid"])].mean()) or 1.0
        out.update({int(p): float(v / norm) for p, v in zip(sub["pid"], m)})
    return out


class BlendTeams:
    """Expected goals as a weighted mix of two team models (same interface)."""

    def __init__(self, a, b, w_b: float):
        self.a, self.b, self.w = a, b, w_b
        self.pot_of = a.pot_of

    def lambdas(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        x, y = self.a.lambdas(home_id, away_id, neutral), self.b.lambdas(home_id, away_id, neutral)
        return (1 - self.w) * x[0] + self.w * y[0], (1 - self.w) * x[1] + self.w * y[1]

    def rating(self, tid: int) -> tuple[float, float]:
        x, y = self.a.rating(tid), self.b.rating(tid)
        return (1 - self.w) * x[0] + self.w * y[0], (1 - self.w) * x[1] + self.w * y[1]

    def avg_for(self, tid: int) -> tuple[float, float]:
        x, y = self.a.avg_for(tid), self.b.avg_for(tid)
        return (1 - self.w) * x[0] + self.w * y[0], (1 - self.w) * x[1] + self.w * y[1]


def minutes_model(players: pd.DataFrame, matches: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Chance of playing / 60+ minutes and minutes when playing, from each
    player's club's finished matches this season (recent ones weigh more)."""
    fin = matches[matches["finished"]].sort_values("kickoff")
    team_matches: dict[int, list[int]] = {}
    for r in fin.itertuples():
        team_matches.setdefault(int(r.home_id), []).append(int(r.mid))
        team_matches.setdefault(int(r.away_id), []).append(int(r.mid))
    mins = {(int(r.pid), int(r.mid)): int(r.oF) for r in apps.itertuples()} if len(apps) else {}
    first_md = apps.groupby("pid")["md"].min().to_dict() if len(apps) else {}
    md_of = dict(zip(fin["mid"], fin["md"]))
    rows = []
    for p in players.itertuples():
        tm = team_matches.get(int(p.team_id), [])
        # a player who joined later isn't penalised for matches before his first one
        start = first_md.get(int(p.pid))
        hist = [mins.get((int(p.pid), mid), 0) for mid in tm]
        if start is not None:
            hist = [h for h, mid in zip(hist, tm) if md_of.get(mid, 99) >= start or h > 0]
        price = float(p.price or 4.5)
        prior_app = float(np.clip(0.15 + (price - 4.5) * 0.22, 0.08, 0.9))
        prior_60 = prior_app * 0.75
        k = len(hist)
        wts = RECENCY ** np.arange(k)[::-1] if k else np.array([])
        w0 = 1.0  # prior worth one match
        app = (np.dot(wts, [h > 0 for h in hist]) + w0 * prior_app) / (wts.sum() + w0)
        s60 = (np.dot(wts, [h >= 60 for h in hist]) + w0 * prior_60) / (wts.sum() + w0)
        played = [h for h in hist if h > 0]
        m_app = float(np.mean(played)) if played else 60.0 * prior_app + 20
        rows.append({"pid": int(p.pid), "p_app": float(app), "p60": float(min(s60, app)),
                     "m_app": float(np.clip(m_app, 15, 95)), "club_matches": k})
    return pd.DataFrame(rows).set_index("pid")


# ---------------------------------------------------------- expected points

def xp_components(pos: int, p_app: float, p60: float, m_app: float, r: dict, lam_for: float, lam_against: float,
                  for_ratio: float, against_ratio: float) -> dict:
    """Expected points for one match from rates and the match's goal expectations."""
    if p_app <= 0:
        return {"xp": 0.0}
    t = m_app / 90.0
    e_g = r["gS"] * t * for_ratio
    e_a = r["gA"] * t * for_ratio
    out = {
        "appear": p_app * rules.APPEARANCE + p60 * rules.SIXTY_PLUS,
        "goals": p_app * e_g * rules.GOAL[pos],
        "outside_box": p_app * r["gOB"] * t * for_ratio * rules.GOAL_OUTSIDE_BOX,
        "assists": p_app * e_a * rules.ASSIST,
        "balls_rec": p_app * e_floor(r["bR"] * t, 3) * rules.BALLS_RECOVERED_PER_3,
        "potm": p_app * min(0.6, r["mom"] * math.sqrt(for_ratio)) * rules.PLAYER_OF_MATCH,
        "cards": p_app * t * (r["yC"] * rules.YELLOW + r["rC"] * rules.RED),
        "pens": p_app * t * (r["pE"] * rules.PEN_WON + r["pM"] * rules.PEN_MISSED + r["pC"] * rules.PEN_CONCEDED),
        "own_goals": p_app * t * r["oG"] * rules.OWN_GOAL,
    }
    on60 = min(m_app, 90.0) / 90.0 if m_app >= 60 else 0.9
    out["clean_sheet"] = p60 * math.exp(-lam_against * on60) * rules.CLEAN_SHEET[pos]
    if rules.CONCEDED_PER_2[pos]:
        out["conceded"] = p_app * e_floor(lam_against * t, 2) * rules.CONCEDED_PER_2[pos]
    if pos == rules.GK:
        out["saves"] = p_app * e_floor(r["saves"] * t * against_ratio, 3) * rules.GK_SAVES_PER_3
        out["pen_saves"] = p_app * t * r["pS"] * rules.GK_PEN_SAVE
    out["xp"] = float(sum(out.values()))
    # per-appearance expectations, for simulating whole points distributions (ucl.sim)
    out.update({"lam_g": e_g, "lam_a": e_a, "lam_br": r["bR"] * t, "lam_sv": r["saves"] * t * against_ratio,
                "p_mom": min(0.6, r["mom"] * math.sqrt(for_ratio)), "lam_ga": lam_against * t,
                "cs_on": on60})
    return out


def status_factor(p, md_offset: int) -> float:
    s = str(getattr(p, "status", "") or "")
    news = str(getattr(p, "news", "") or "")
    if md_offset == 0:
        f = STATUS_NEXT.get(s, 1.0)
        if news.startswith("Unlikely") and s not in STATUS_NEXT:
            f = min(f, UNLIKELY_NEXT)
        return f
    if md_offset == 1:
        return STATUS_LATER.get(s, 1.0) if s != "S" else 1.0
    return 0.0 if s == "NIS" else 1.0


def history(tour: int, as_of_md: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(all matches incl. previous season with weights, this season's matches,
    players, appearances with weights), all strictly before `as_of_md`."""
    m, p, a = load(tour)
    cut = (lambda df: df[df["md"] < as_of_md]) if as_of_md else (lambda df: df)
    m_now = m.copy()
    if as_of_md:
        m_now.loc[m_now["md"] >= as_of_md, ["finished"]] = False
    a_now = cut(a) if len(a) else a
    mats = [m_now.assign(weight=1.0)]
    apps = [a_now.assign(w=1.0)] if len(a_now) else []
    prev = tour - 10
    if (HIST / f"matches_{prev}.parquet").exists():
        pm, _, pa = load(prev)
        mats.append(pm.assign(weight=PREV_SEASON_WEIGHT))
        if len(pa):
            apps.append(pa.assign(w=PREV_SEASON_WEIGHT))
    allm = pd.concat(mats, ignore_index=True)
    alla = pd.concat(apps, ignore_index=True) if apps else pd.DataFrame()
    return allm, m_now, p, alla


def pots(matches: pd.DataFrame) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in matches.sort_values("tour").itertuples():  # this season's pot wins
        out[int(r.home_id)] = pot_no(r.home_pot)
        out[int(r.away_id)] = pot_no(r.away_pot)
    return out


def cutoff_time(m_now: pd.DataFrame, as_of_md: int | None) -> pd.Timestamp:
    """The moment the data must stop: the first kick-off of `as_of_md` (now, live)."""
    if as_of_md:
        k = m_now.loc[m_now["md"] >= as_of_md, "kickoff"]
        if len(k):
            return pd.Timestamp(k.min())
    return pd.Timestamp.now(tz="UTC")


def predict(tour: int = feed.TOUR, as_of_md: int | None = None, mds: list[int] | None = None,
            odds_lambdas: dict[int, tuple[float, float]] | None = None, use_status: bool = True,
            cfg: dict | None = None) -> tuple[pd.DataFrame, TeamModel]:
    """xP for every player in every match of `mds` (default: the not-yet-played
    matchdays of this season), using only data from before `as_of_md`.
    `cfg` overrides CFG (for ucl.factors)."""
    c = {**CFG, **(cfg or {})}
    allm, m_now, players, apps = history(tour, as_of_md)
    pot_of = pots(allm[allm["tour"] == tour]) or pots(allm)
    poisson = team = fit_teams(allm, pot_of)
    elo_teams = None
    if c["elo_weight"] or c["prior_team_elo"] or c["elo_weight_knockout"]:
        elo_teams = elo.teams_at(cutoff_time(m_now, as_of_md), allm[allm["tour"] == tour], pot_of)
    if c["elo_weight"]:
        team = BlendTeams(poisson, elo_teams, c["elo_weight"])
    # the knockout simulation's team model (ucl.knockout), judged on match scores (ucl.factors.team_test)
    team.knockout = BlendTeams(poisson, elo_teams, c["elo_weight_knockout"]) if c["elo_weight_knockout"] else team
    team.elo = elo_teams
    pos_of = dict(zip(players["pid"].astype(int), players["pos"].astype(int)))
    tgoals = {t: (elo_teams if c["prior_team_elo"] else team).avg_for(t)[0] for t in pot_of}
    mult = prior_mult(players, apps[apps["tour"] == tour] if len(apps) else apps, tgoals,
                      c["prior_price"], c["prior_team"])
    rates, prior = player_rates(apps, pos_of, c["shrink_90s"], mult) if len(apps) else (pd.DataFrame(), {})
    cur_apps = apps[apps["tour"] == tour] if len(apps) else apps
    mm = minutes_model(players, m_now, cur_apps)
    if mds is None:
        mds = sorted(m_now.loc[~m_now["finished"], "md"].unique().tolist())
    first = min(mds) if mds else 0
    fx = m_now[m_now["md"].isin(mds)]
    avg = {t: team.avg_for(t) for t in pot_of}
    rows = []
    for g in fx.itertuples():
        neutral = rules.STAGE.get(int(g.md)) == "final"
        lh, la = team.lambdas(int(g.home_id), int(g.away_id), neutral)
        if odds_lambdas and int(g.mid) in odds_lambdas:
            oh, oa = odds_lambdas[int(g.mid)]
            lh, la = ODDS_WEIGHT * oh + (1 - ODDS_WEIGHT) * lh, ODDS_WEIGHT * oa + (1 - ODDS_WEIGHT) * la
        for side, tid, opp, lf, lag, opp_name in ((1, g.home_id, g.away_id, lh, la, g.away_code),
                                                   (0, g.away_id, g.home_id, la, lh, g.home_code)):
            tid = int(tid)
            af, aa = avg.get(tid, (1.4, 1.4))
            for p in players[players["team_id"] == tid].itertuples():
                pid = int(p.pid)
                pos = int(p.pos)
                if pid in rates.index:
                    r = rates.loc[pid].to_dict()
                else:  # no UCL minutes at all: his position's prior, scaled for price and team
                    r = dict(prior.get(pos) or DEFAULT_PRIOR[pos])
                    for col in ATTACK_COLS:
                        r[col] = r.get(col, 0.0) * mult.get(pid, 1.0)
                if not r:
                    continue
                mrow = mm.loc[pid] if pid in mm.index else None
                p_app = float(mrow["p_app"]) if mrow is not None else 0.1
                p60 = float(mrow["p60"]) if mrow is not None else 0.05
                m_app = float(mrow["m_app"]) if mrow is not None else 45.0
                f = status_factor(p, int(g.md) - first) if use_status else 1.0
                if use_status and not int(getattr(p, "active", 1)):
                    f = 0.0
                comp = xp_components(pos, p_app * f, p60 * f, m_app, r, lf, lag, lf / af, lag / aa)
                rows.append({"pid": pid, "md": int(g.md), "gd": int(g.gd), "mid": int(g.mid), "kickoff": g.kickoff,
                             "team_id": tid, "opp_id": int(opp), "opp": opp_name, "home": bool(side),
                             "p_app": p_app * f, "p60": p60 * f, "m_app": m_app, "goals_for": lf,
                             "goals_against": lag, **comp})
    return pd.DataFrame(rows), team
