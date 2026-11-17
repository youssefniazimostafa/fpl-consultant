"""Plain-English notes for the app: how the UCL model works (Analytics),
and what each player's xP rests on (Team page), built from the model's
live settings so they can't drift from what the model actually does."""

from __future__ import annotations

import pandas as pd

from ucl import domestic, model

LEAGUE_NAME = {"ENG": "Premier League", "ESP": "LaLiga", "GER": "Bundesliga", "ITA": "Serie A", "FRA": "Ligue 1",
               "POR": "Primeira Liga", "NED": "Eredivisie", "BEL": "Pro League", "TUR": "Süper Lig",
               "GRE": "Super League", "AUT": "Austrian Bundesliga", "DEN": "Superliga", "NOR": "Eliteserien"}


def how_it_works(cfg: dict | None = None) -> list[str]:
    c = {**model.CFG, **(cfg or {})}
    lines = [
        "Team strength: every club's attack and defence are rated from its Champions League matches "
        "(2025/26 at half weight).",
    ]
    if c["elo_weight"] or c["prior_team_elo"] or c["elo_weight_knockout"]:
        lines.append(
            "A second rating uses every match: domestic league results from 13 leagues and the European ones, "
            "updated match by match like Elo, with each league's strength learned from European matches "
            "between leagues. Clubs from Czechia, Slovakia, Ukraine, Azerbaijan, Kazakhstan and Cyprus "
            "only have their European matches (no free source for their leagues).")
    uses = []
    if c["elo_weight"]:
        uses.append(f"{round(c['elo_weight'] * 100)}% of each match's expected goals")
    if c["prior_team_elo"] and c["prior_team"]:
        uses.append("what a player with few Champions League minutes is expected to do")
    if c["elo_weight_knockout"]:
        uses.append(f"{round(c['elo_weight_knockout'] * 100)}% of the knockout chances")
    if uses:
        joined = uses[0] if len(uses) == 1 else ", ".join(uses[:-1]) + " and " + uses[-1]
        lines.append("That all-matches rating sets " + joined + ".")
    lines += [
        "Players: chance of starting and of playing 60+ minutes from his club's recent matches and UEFA's "
        "injury news; goals, assists, balls recovered, saves and the rest per 90 minutes from his Champions "
        f"League matches, pulled towards a typical player in his position as if he had {c['shrink_90s']:g} "
        "extra average matches. That pull fades as his own minutes grow.",
        "Each expected count becomes points with UEFA's exact scoring rules; goals and assists rise with "
        "how many goals his club should score in that match, clean sheets with how few it should concede. "
        "Bookmaker odds are blended in when available close to the matchday.",
        "Every change to the model is backtested on 2025/26 and on this season's matchdays it hadn't seen, "
        "and kept only if it helps on both.",
    ]
    return lines


class Basis:
    """'4 UCL matches (+ 8 last season) · team rated on 11 Serie A matches this season'."""

    def __init__(self, apps_now: pd.DataFrame, apps_prev: pd.DataFrame, elo_teams=None):
        played = lambda a: a[a["oF"] > 0].groupby("pid")["mid"].nunique().to_dict() if len(a) else {}
        self.now, self.prev = played(apps_now), played(apps_prev)
        self.elo = elo_teams
        self.dom = {}
        d = domestic.load()
        if len(d):
            d = d[d["season"] == domestic.CURRENT]
            for side in ("home", "away"):
                for (lg, club), n in d.groupby(["league", side]).size().items():
                    self.dom[f"{lg}:{club}"] = self.dom.get(f"{lg}:{club}", 0) + int(n)

    def __call__(self, pid: int, team_id: int) -> str:
        n, p = self.now.get(int(pid), 0), self.prev.get(int(pid), 0)
        txt = f"{n} UCL match{'es' if n != 1 else ''} this season" if n else "no UCL minutes yet this season"
        if p:
            txt += f" (+{p} last season)"
        if self.elo is not None:
            key = self.elo.key_of.get(int(team_id), "")
            lg = key.split(":")[0]
            k = self.dom.get(key, 0)
            if k and lg in LEAGUE_NAME:
                txt += f" · team rated on {k} {LEAGUE_NAME[lg]} matches this season, plus its European ones"
            else:
                txt += " · team rated on its European matches only"
        return "Rests on " + txt + "."
