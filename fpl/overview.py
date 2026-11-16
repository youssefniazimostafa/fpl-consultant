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

import math
from pathlib import Path
from typing import Any

import pandas as pd

TOP_PER_POSITION = 12
CAPTAIN_OPTIONS = 10  # the Scout page shows 10, like its other top lists
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


# --------------------------------------------------------------- player lookup

LOOKUP_RECENT = 6  # gameweeks of recent form in a report
LOOKUP_FIXTURES = 5  # gameweeks ahead


def _num(v, nd=2):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def build_lookup(
    *,
    players: pd.DataFrame,
    xp: pd.DataFrame,
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    matches: pd.DataFrame,
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    team_id: int | None,
    transfer_bar: float,
    my_transfers: pd.DataFrame | None = None,
    free_transfers: int = 1,
) -> dict[str, Any]:
    """Every FPL player for the Scout page's Player lookup: one short row each
    for the list, and one compact report each (fixtures, what drives his
    expected points, recent form, value, and for players you don't own the
    best like-for-like swap from your squad). Reads only tables the
    pipeline already built; changes nothing in the model or the plan."""
    if players.empty or xp.empty:
        return {"available": False}
    info = players.set_index("id")
    gws = sorted(int(g) for g in xp["event"].unique())[:LOOKUP_FIXTURES]
    next_gw = gws[0]
    xp_pg = xp[xp["event"].isin(gws)].groupby(["player_id", "event"])["xp"].sum().unstack(fill_value=0.0)
    xp5 = xp_pg.sum(axis=1)

    # your squad, bank and rivals at the latest confirmed picks
    mine: set[int] = set()
    bank = 0.0
    rival_own: dict[int, int] = {}
    n_rivals = 0
    if not league_picks.empty:
        last = league_picks[league_picks["gameweek"] == league_picks["gameweek"].max()]
        if team_id is not None:
            mine = {int(p) for p in last.loc[last["manager_id"] == team_id, "player_id"]}
            h = league_history[(league_history["manager_id"] == team_id)] if not league_history.empty else league_history
            if h is not None and not h.empty:
                bank = float(h.sort_values("gameweek")["bank"].iloc[-1]) / 10
        rivals = last[last["manager_id"] != team_id]
        n_rivals = int(rivals["manager_id"].nunique())
        rival_own = rivals.groupby("player_id")["manager_id"].nunique().astype(int).to_dict()

    # actual form
    played = pd.DataFrame()
    last_gw = 0
    if not matches.empty:
        played = matches.groupby(["player_id", "gameweek"], as_index=False)[
            ["total_points", "minutes", "goals_scored", "assists"]].sum()
        last_gw = int(played["gameweek"].max())
    form = {}
    recent: dict[int, list] = {}
    if not played.empty:
        r4 = played[(played["gameweek"] > last_gw - 4) & (played["minutes"] > 0)]
        g = r4.groupby("player_id")["total_points"].agg(["sum", "count"])
        form = (g["sum"] / g["count"]).to_dict()
        r6 = played[played["gameweek"] > last_gw - LOOKUP_RECENT]
        for pid, grp in r6.groupby("player_id"):
            by = grp.set_index("gameweek")
            recent[int(pid)] = [[gw, *(int(by.loc[gw, c]) if gw in by.index else 0
                                    for c in ("total_points", "minutes", "goals_scored", "assists"))]
                                for gw in range(max(1, last_gw - LOOKUP_RECENT + 1), last_gw + 1)]

    # fixture difficulty per (team, event)
    fdr: dict[tuple[int, int], list] = {}
    for f in fixtures.itertuples() if not fixtures.empty else []:
        if pd.isna(f.event):
            continue
        fdr.setdefault((int(f.team_h), int(f.event)), []).append(int(f.team_h_difficulty))
        fdr.setdefault((int(f.team_a), int(f.event)), []).append(int(f.team_a_difficulty))
    fx_rows = xp[xp["event"].isin(gws)].sort_values(["event", "kickoff_time"])
    fx_by: dict[int, list] = {}
    for r in fx_rows.itertuples():
        fx_by.setdefault(int(r.player_id), []).append([int(r.event), r.opponent_short, "H" if r.is_home else "A", _num(r.xp, 1)])
    feat_next = features[features["event"] == next_gw].groupby("player_id").first() if not features.empty else pd.DataFrame()
    xp_next = xp[xp["event"] == next_gw].groupby("player_id").agg(
        xg=("expected_xg", "sum"), xa=("expected_xa", "sum"), defcon=("defcon_pts", "sum"), bonus=("bonus_pts", "sum"),
        cs_pts=("clean_sheet_pts", "sum"), mins=("expected_minutes", "sum"), p_play=("p_play", "max"), p_60=("p_60", "max"))

    squad = [p for p in mine if p in info.index]
    club_count: dict[int, int] = {}
    for p in squad:
        club_count[int(info.loc[p, "team"])] = club_count.get(int(info.loc[p, "team"]), 0) + 1

    # --- best like-for-like swap, judged like the optimiser's transfer gate ---
    # The team's value each gameweek is the optimiser's objective for a fixed
    # squad: best valid XI + the captain's extra + BENCH_VALUE_WEIGHT x bench,
    # discounted by WEEKLY_DISCOUNT per week. As in the optimiser's per-week
    # gate (fpl.optimise.gate_planned_weeks), the move is compared with
    # holding this week while keeping the option to make the same move next
    # week with the banked free transfer: gain = its value over the horizon
    # minus the best of (0, its value from next week on), minus any hit.
    # It clears the bar when that reaches TRANSFER_THRESHOLD_XP. (The real
    # gate re-solves the whole plan; this checks only this one move.)
    # Budget: your real selling prices.
    from fpl.optimise import BENCH_VALUE_WEIGHT, HIT_COST, WEEKLY_DISCOUNT, estimate_selling_prices

    sell = estimate_selling_prices(set(squad), players, my_transfers if my_transfers is not None else pd.DataFrame()) if squad else {}
    pos_of = {int(p): info.loc[p, "position"] for p in info.index}
    disc = {g: WEEKLY_DISCOUNT ** k for k, g in enumerate(gws)}
    wxp = {(int(p), int(g)): float(v) for (p, g), v in xp_pg.stack().items()} if not xp_pg.empty else {}
    bounds = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

    def week_value(ids: list[int], g: int) -> tuple[float, float]:
        """(objective, XI + captain) for one gameweek."""
        by = {q: sorted((wxp.get((p, g), 0.0) for p in ids if pos_of[p] == q), reverse=True) for q in bounds}
        best_xi = None
        for nd in range(3, 6):
            for nm in range(2, 6):
                nf = 10 - nd - nm
                if not 1 <= nf <= 3 or len(by["DEF"]) < nd or len(by["MID"]) < nm or len(by["FWD"]) < nf:
                    continue
                xi = by["GKP"][:1] + by["DEF"][:nd] + by["MID"][:nm] + by["FWD"][:nf]
                if best_xi is None or sum(xi) > sum(best_xi):
                    best_xi = xi
        if best_xi is None:
            return 0.0, 0.0
        total = sum(sum(v) for v in by.values())
        xi_cap = sum(best_xi) + max(best_xi)
        return xi_cap + BENCH_VALUE_WEIGHT * (total - sum(best_xi)), xi_cap

    def squad_value(ids: list[int]) -> tuple[float, float, float]:
        """(objective over the horizon, objective from next week on, XI + captain this week)"""
        obj, later, week0 = 0.0, 0.0, 0.0
        for g in gws:
            o, x = week_value(ids, g)
            obj += disc[g] * o
            if g == next_gw:
                week0 = x
            else:
                later += disc[g] * o
        return obj, later, week0

    base_obj, base_later, base_w0 = squad_value(squad) if squad else (0.0, 0.0, 0.0)
    # The saved-transfer value (FT_VALUE_XP) cancels out here: holding banks
    # the free transfer and then spends it on the same move next week, so
    # both branches end the horizon with the same free transfers. A hit
    # applies only when there's no free transfer this week.
    ft_cost = 0.0
    hit = HIT_COST if free_transfers < 1 else 0

    def swap(pid: int, pos: str, price: float, club: int) -> dict[str, Any] | None:
        best = None
        for s_id in squad:
            if pos_of[s_id] != pos:
                continue
            s_sell = sell.get(s_id, int(info.loc[s_id, "now_cost"])) / 10
            if price > s_sell + bank + 1e-9:
                continue
            clubs_after = club_count.get(club, 0) - (1 if int(info.loc[s_id, "team"]) == club else 0) + 1
            if clubs_after > 3:
                continue
            obj, later, w0 = squad_value([p for p in squad if p != s_id] + [pid])
            horizon = obj - base_obj
            gain = horizon - max(0.0, later - base_later) - hit
            if best is None or gain > best["gain"]:
                best = {"out_id": int(s_id), "out_name": info.loc[s_id, "web_name"], "out_sell": round(s_sell, 1),
                        "gain": round(gain, 2), "week_gain": round(w0 - base_w0, 2), "horizon_gain": round(horizon, 2),
                        "bank_after": round(s_sell + bank - price, 1)}
        if best:
            best["clears_bar"] = best["gain"] >= transfer_bar
            best["ft_cost"] = round(ft_cost, 2)
            best["hit"] = hit
        return best

    rows, reports = [], {}
    for pid, i in info.iterrows():
        pid = int(pid)
        if i.get("removed") is True:
            continue
        price = float(i["now_cost"]) / 10
        team_id_p = int(i["team"])
        team_fx = fx_by.get(pid, [])
        fixtures_out = [[gw, opp, ha, (fdr.get((team_id_p, gw)) or [None])[0], x] for gw, opp, ha, x in team_fx]
        own = pid in mine
        rows.append([pid, i["web_name"], i.get("team_short_name"), None if pd.isna(i.get("team_code")) else int(i["team_code"]),
                     i["position"], price, _num(xp_pg.loc[pid, next_gw] if pid in xp_pg.index and next_gw in xp_pg.columns else 0, 1),
                     _num(form.get(pid), 1), _num(pd.to_numeric(i.get("selected_by_percent"), errors="coerce"), 1), own])
        fn = feat_next.loc[pid] if pid in feat_next.index else None
        xn = xp_next.loc[pid] if pid in xp_next.index else None
        chance = i.get("chance_of_playing_next_round")
        total = int(i.get("total_points") or 0)
        mins = int(i.get("minutes") or 0)
        starts = int(i.get("starts") or 0)
        rep = {
            "full": f"{i.get('first_name', '')} {i.get('second_name', '')}".strip(),
            "rivals": int(rival_own.get(pid, 0)),
            "chance": None if pd.isna(chance) else int(chance),
            "news": i.get("news") or "",
            "status": i.get("status"),
            "fx": fixtures_out,
            "xp5": _num(xp5.get(pid, 0.0), 1),
            "drv": None if fn is None else {
                # the xP table's chances already include injury news (the features' p_60 assumes he starts)
                "start": _num(xn["p_play"] if xn is not None else fn["p_start"]),
                "p60": _num(xn["p_60"] if xn is not None else fn["p_60"]),
                "goal": _num(1 - math.exp(-float(xn["xg"])) if xn is not None else None),
                "assist": _num(1 - math.exp(-float(xn["xa"])) if xn is not None else None),
                "cs": _num(fn["clean_sheet_prob"]),
                "defcon": _num(xn["defcon"]) if xn is not None else None,
                "bonus": _num(xn["bonus"]) if xn is not None else None,
                "mins": _num(xn["mins"], 0) if xn is not None else None,
                "opp": fn["opponent_short"],
            },
            "recent": recent.get(pid, []),
            "season": {"pts": total, "mins": mins, "g": int(i.get("goals_scored") or 0), "a": int(i.get("assists") or 0),
                       "cs": int(i.get("clean_sheets") or 0), "bonus": int(i.get("bonus") or 0), "starts": starts,
                       "ppg": _num(pd.to_numeric(i.get("points_per_game"), errors="coerce"), 1)},
            "ppm": round(total / price, 1) if price else None,
            "change": round(float(i.get("cost_change_start") or 0) / 10, 1),
        }
        if not own and squad:
            rep["swap"] = swap(pid, i["position"], price, team_id_p)
        reports[str(pid)] = rep
    return {
        "available": True, "next_gameweek": next_gw, "gameweeks": gws, "last_gameweek": last_gw,
        "rivals": n_rivals, "bank": round(bank, 1), "transfer_bar": transfer_bar, "owned": len(mine),
        "free_transfers": free_transfers,
        "columns": ["id", "name", "club", "team_code", "pos", "price", "xp", "form", "sel", "own"],
        "rows": rows, "reports": reports,
    }
