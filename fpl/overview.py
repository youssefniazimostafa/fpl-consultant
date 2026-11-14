"""The desktop app's "FPL overview" page -- one ``overview`` section in
latest.json, about FPL as a whole (not your league). Built by
``fpl.publish`` from the same sources the old site's "Top expected
points", "Biggest misses", "Fixture ticker" and "News flags" panels use,
widened to every player:

* projected points next gameweek: ``xp_by_gameweek`` (+ next opponent from
  ``features``, the fixture ticker's source);
* "slept-on" players: each gameweek's frozen pre-deadline prediction
  (``data/predictions/gw{N}.parquet``, what fpl.scoring's "biggest misses"
  compare against) vs actual points (``player_matches``), with a streak
  count across consecutive gameweeks;
* form and value: actual points per game / per £1m from ``player_matches``;
* injury and team news: FPL's own news text, chance of playing and the
  time it was added (``players``), newest first.

"You own" means your current squad (latest confirmed picks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TOP_PER_POSITION = 12
CAPTAIN_OPTIONS = 5
# slept-on: scored at least this much more than expected, and at least this many points
SLEPT_ON_MIN_ABOVE = 6.0
SLEPT_ON_MIN_POINTS = 8
SLEPT_ON_SHOW = 12
# in form: points per game over the last FORM_WEEKS vs before
FORM_WEEKS = 4
FORM_MIN_GAMES = 3
FORM_MIN_JUMP = 2.0
FORM_SHOW = 12
VALUE_POOL = 80
NEWS_DAYS = 7
NEWS_SHOW = 24


def _card(pid: int, info: dict[int, dict[str, Any]], mine: set[int]) -> dict[str, Any]:
    i = info.get(pid, {})
    cost = i.get("now_cost")
    return {
        "player_id": pid,
        "web_name": i.get("web_name"),
        "position": i.get("position"),
        "team_code": None if pd.isna(i.get("team_code")) else int(i.get("team_code")),
        "team_short": i.get("team_short_name"),
        "price": None if cost is None or pd.isna(cost) else round(float(cost) / 10, 1),
        "you_own": pid in mine,
    }


def _per_gw_actual(matches: pd.DataFrame) -> pd.DataFrame:
    """Actual points and minutes per player per gameweek (doubles summed)."""
    return matches.groupby(["player_id", "gameweek"], as_index=False)[["total_points", "minutes"]].sum()


def _slept_on_by_gw(pred_dir: Path, actual: pd.DataFrame, gameweeks: list[int]) -> dict[int, pd.DataFrame]:
    out = {}
    for gw in gameweeks:
        path = pred_dir / f"gw{gw}.parquet"
        if not path.exists():
            continue
        pred = pd.read_parquet(path).groupby("player_id", as_index=False)["xp"].sum()
        a = actual[actual["gameweek"] == gw][["player_id", "total_points"]]
        m = pred.merge(a, on="player_id", how="inner")
        m["above"] = m["total_points"] - m["xp"]
        hit = m[(m["above"] >= SLEPT_ON_MIN_ABOVE) & (m["total_points"] >= SLEPT_ON_MIN_POINTS)]
        out[gw] = hit.sort_values("above", ascending=False)
    return out


def build_overview(
    *,
    players: pd.DataFrame,
    xp_by_gameweek: pd.DataFrame,
    next_fixture: dict[int, dict[str, Any]],
    matches: pd.DataFrame,
    league_picks: pd.DataFrame,
    team_id: int | None,
    predictions_dir: Path,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if players.empty or xp_by_gameweek.empty:
        return {"available": False, "reason": "players/xp_by_gameweek tables not found -- run fpl.fetch and fpl.xp first."}
    now = now or pd.Timestamp.now("UTC")
    info = players.set_index("id").to_dict("index")
    mine: set[int] = set()
    if team_id is not None and not league_picks.empty:
        lp = league_picks[(league_picks["manager_id"] == team_id)]
        if not lp.empty:
            mine = {int(p) for p in lp[lp["gameweek"] == lp["gameweek"].max()]["player_id"]}

    # --- projected points next gameweek ---
    cols = sorted((c for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp")), key=lambda c: int(c[2:-3]))
    next_col = cols[0]
    next_gw = int(next_col[2:-3])
    proj = xp_by_gameweek[["player_id", next_col]].rename(columns={next_col: "projected"}).dropna()
    proj = proj.sort_values("projected", ascending=False)

    def proj_card(pid: int, xp: float) -> dict[str, Any]:
        fx = next_fixture.get(pid)
        return {
            **_card(pid, info, mine),
            "projected_points": round(float(xp), 1),
            "opponent": f"{fx['opponent_short']} ({'H' if fx['is_home'] else 'A'})" if fx else None,
        }

    top_by_position: dict[str, list[dict[str, Any]]] = {"ALL": []}
    for r in proj.itertuples():
        pid = int(r.player_id)
        pos = info.get(pid, {}).get("position")
        if pos is None:
            continue
        if len(top_by_position["ALL"]) < TOP_PER_POSITION:
            top_by_position["ALL"].append(proj_card(pid, r.projected))
        bucket = top_by_position.setdefault(pos, [])
        if len(bucket) < TOP_PER_POSITION:
            bucket.append(proj_card(pid, r.projected))
    captains = [
        {**proj_card(int(r.player_id), r.projected), "as_captain": round(2 * float(r.projected), 1)}
        for r in proj.head(CAPTAIN_OPTIONS).itertuples()
    ]

    # --- actual points history ---
    slept_on: list[dict[str, Any]] = []
    in_form: list[dict[str, Any]] = []
    value: list[dict[str, Any]] = []
    last_gw = None
    if not matches.empty:
        actual = _per_gw_actual(matches)
        last_gw = int(actual["gameweek"].max())

        # slept-on, with a streak over consecutive gameweeks
        recent = [g for g in range(last_gw - 4, last_gw + 1) if g >= 1]
        lists = _slept_on_by_gw(predictions_dir, actual, recent)
        if last_gw in lists:
            members = {g: set(df["player_id"]) for g, df in lists.items()}
            for r in lists[last_gw].head(SLEPT_ON_SHOW).itertuples():
                pid = int(r.player_id)
                streak, g = 0, last_gw
                while g in members and pid in members[g]:
                    streak, g = streak + 1, g - 1
                slept_on.append({
                    **_card(pid, info, mine), "gameweek": last_gw,
                    "points": int(r.total_points), "expected": round(float(r.xp), 1),
                    "above": round(float(r.above), 1), "weeks_running": streak,
                })

        # in form: points per game (games = played minutes > 0)
        played = actual[actual["minutes"] > 0]
        recent_g = played[played["gameweek"] > last_gw - FORM_WEEKS]
        before_g = played[played["gameweek"] <= last_gw - FORM_WEEKS]
        r_ppg = recent_g.groupby("player_id")["total_points"].agg(["sum", "count"])
        b_ppg = before_g.groupby("player_id")["total_points"].agg(["sum", "count"])
        form = r_ppg.join(b_ppg, lsuffix="_recent", rsuffix="_before", how="inner")
        form = form[(form["count_recent"] >= FORM_MIN_GAMES) & (form["count_before"] >= FORM_MIN_GAMES)]
        form["recent"] = form["sum_recent"] / form["count_recent"]
        form["before"] = form["sum_before"] / form["count_before"]
        form["jump"] = form["recent"] - form["before"]
        form = form[form["jump"] >= FORM_MIN_JUMP].sort_values("jump", ascending=False).head(FORM_SHOW)
        in_form = [
            {**_card(int(pid), info, mine), "recent_ppg": round(float(r.recent), 1), "before_ppg": round(float(r.before), 1),
             "recent_weeks": FORM_WEEKS, "before_from": 1, "before_to": last_gw - FORM_WEEKS, "recent_from": last_gw - FORM_WEEKS + 1}
            for pid, r in form.iterrows()
        ]

        # cheap high scorers: points per £1m at today's price
        last4 = actual[actual["gameweek"] > last_gw - FORM_WEEKS].groupby("player_id")["total_points"].sum()
        last1 = actual[actual["gameweek"] == last_gw].set_index("player_id")["total_points"]
        rows = []
        for pid, pts4 in last4.items():
            card = _card(int(pid), info, mine)
            if not card["price"]:
                continue
            pts1 = int(last1.get(pid, 0))
            rows.append({**card, "points_last_gw": pts1, "points_last_4": int(pts4),
                         "per_m_last_gw": round(pts1 / card["price"], 2), "per_m_last_4": round(float(pts4) / card["price"], 2)})
        by4 = sorted(rows, key=lambda r: -r["per_m_last_4"])[:VALUE_POOL]
        by1 = sorted(rows, key=lambda r: -r["per_m_last_gw"])[:VALUE_POOL]
        keep = {r["player_id"] for r in by4} | {r["player_id"] for r in by1}
        value = [r for r in rows if r["player_id"] in keep]

    # --- injury and team news, newest first ---
    news = []
    if "news_added" in players.columns:
        n = players[players["news"].fillna("").astype(bool)].copy()
        n["added"] = pd.to_datetime(n["news_added"], utc=True, errors="coerce")
        n = n[n["added"] >= now - pd.Timedelta(days=NEWS_DAYS)].sort_values("added", ascending=False).head(NEWS_SHOW)
        for r in n.itertuples():
            chance = r.chance_of_playing_next_round
            news.append({
                **_card(int(r.id), info, mine), "news": r.news, "status": r.status,
                "chance": None if pd.isna(chance) else int(chance), "added": r.added.isoformat(),
            })

    return {
        "available": True,
        "next_gameweek": next_gw,
        "last_gameweek": last_gw,
        "form_weeks": FORM_WEEKS,
        "top_projected": top_by_position,
        "captain_options": captains,
        "slept_on": slept_on,
        "in_form": in_form,
        "value": value,
        "news": news,
    }
