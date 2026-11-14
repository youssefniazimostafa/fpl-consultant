"""The desktop app's "League" page data -- one ``league_page`` section in
latest.json, built by ``fpl.publish`` from the same tables the old site's
"Standing vs rivals", "Transfers", "Improve my team" and "Differentials"
panels use (league_history, league_picks, rival window, free transfers,
rival transfer forecast, chips grid, differentials), plus the counts the
page states in plain words ("owned by 3 of your 9 rivals").

Everything about rivals is their last *confirmed* team: FPL hides a
manager's picks for the coming gameweek until its deadline passes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.decision import CHIP_HALF_BOUNDARY, CHIP_LABELS, CHIP_ORDER

TOP_OWNED = 15
DIFFERENTIALS = 10
DIFFERENTIAL_MAX_RIVALS = 1  # "few or none"
POPULAR_NOT_OWNED = 8

STRATEGY_TEXT = {
    "chasing": "You're chasing 1st, so we're taking a bit more risk: players fewer rivals own.",
    "protecting": "You're protecting your lead, so we're sticking closer to the players your rivals own.",
    "balanced": "You're close to the top with plenty of season left, so we're simply picking the strongest team.",
}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_league_page(
    *,
    team_id: int | None,
    league_id: int | None,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    league_picks: pd.DataFrame,
    players: pd.DataFrame,
    xp_by_gameweek: pd.DataFrame,
    my_leagues: pd.DataFrame,
    rival_window: dict[str, Any],
    free_transfers: dict[str, Any],
    rival_forecast: dict[str, Any],
    chips: dict[str, Any],
    differentials: dict[str, Any],
    strategy: dict[str, Any],
    decision: dict[str, Any],
    season_simulation: dict[str, Any],
) -> dict[str, Any]:
    if team_id is None or league_history.empty or league_managers.empty or league_picks.empty or players.empty:
        return {"available": False, "reason": "league tables not found -- set league_id and run fpl.league first."}

    gw = int(league_history["gameweek"].max())
    pinfo = players.set_index("id")
    name = pinfo["web_name"].to_dict()
    team_code = pinfo["team_code"].to_dict()
    team_short = pinfo["team_short_name"].to_dict()
    position = pinfo["position"].to_dict()
    gw_points_player = pinfo["event_points"].to_dict()
    next_col = None
    if not xp_by_gameweek.empty:
        cols = sorted(c for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp"))
        next_col = min(cols, key=lambda c: int(c[2:-3])) if cols else None
    next_xp = xp_by_gameweek.set_index("player_id")[next_col].to_dict() if next_col else {}

    # --- standings ---
    latest = league_history[league_history["gameweek"] == gw].merge(league_managers, on="manager_id")
    latest = latest.sort_values("total_points", ascending=False).reset_index(drop=True)
    in_area = {int(r["manager_id"]) for r in (rival_window.get("rivals") or []) if r.get("in_window")}
    standings = []
    for i, r in latest.iterrows():
        mid = int(r["manager_id"])
        standings.append({
            "rank": i + 1,
            "manager_id": mid,
            "entry_name": r["entry_name"],
            "player_name": r["player_name"],
            "total_points": int(r["total_points"]),
            # FPL's gameweek points are before transfer hits; show what counts
            "gw_points": int(r["points"]) - int(r.get("event_transfers_cost") or 0),
            "is_you": mid == team_id,
            "in_your_area": mid in in_area and mid != team_id,
        })
    me = next(s for s in standings if s["is_you"])
    leader = standings[0]
    second = standings[1] if len(standings) > 1 else None
    you = {
        "rank": me["rank"],
        "rank_text": _ordinal(me["rank"]),
        "managers": len(standings),
        "total_points": me["total_points"],
        "points_behind_first": leader["total_points"] - me["total_points"] if me["rank"] > 1 else 0,
        "lead": (me["total_points"] - second["total_points"]) if me["rank"] == 1 and second else None,
    }

    # --- your chances ---
    head = (decision or {}).get("headline") or {}
    my_season = next((s for s in (season_simulation.get("standings") or []) if s.get("manager_id") == team_id), None)
    chances = {
        "this_gameweek": head.get("win_chance_this_week"),
        "gameweek": (decision or {}).get("gameweek"),
        "top_by_gameweek": season_simulation.get("target_gameweek"),
        "top_by": my_season.get("p_finish_1st_by_gw19") if my_season else None,
    }

    # --- every manager's last confirmed team ---
    picks = league_picks[league_picks["gameweek"] == league_picks["gameweek"].max()]
    picks_gw = int(league_picks["gameweek"].max())
    ft_by = {int(m["manager_id"]): m.get("free_transfers") for m in (free_transfers.get("managers") or [])}
    fc_by = {int(f["manager_id"]): f for f in (rival_forecast.get("forecasts") or [])}
    grid_by = {int(g["manager_id"]): g for g in ((chips.get("rivals") or {}).get("grid") or [])}
    owned_by: dict[int, set[int]] = {}
    teams: dict[str, Any] = {}
    for mid, rows in picks.groupby("manager_id"):
        mid = int(mid)
        rows = rows.sort_values("position")
        owned_by[mid] = {int(p) for p in rows["player_id"]}
        cards = []
        for r in rows.itertuples():
            pid = int(r.player_id)
            cards.append({
                "player_id": pid, "web_name": name.get(pid, "?"), "team_code": team_code.get(pid),
                "team_short": team_short.get(pid), "position": position.get(pid), "slot": int(r.position),
                "is_captain": bool(r.is_captain), "is_vice_captain": bool(r.is_vice_captain),
                "starting": int(r.position) <= 11, "points": gw_points_player.get(pid) if picks_gw == gw else None,
            })
        chip = rows["active_chip"].dropna().iloc[0] if rows["active_chip"].notna().any() else None
        fc = fc_by.get(mid, {})
        teams[str(mid)] = {
            "gameweek": picks_gw,
            "chip": CHIP_LABELS.get(chip, chip) if chip else None,
            "captain": next((c["web_name"] for c in cards if c["is_captain"]), None),
            "starting_xi": [c for c in cards if c["starting"]],
            "bench": [c for c in cards if not c["starting"]],
            "free_transfers": ft_by.get(mid),
            "likely_in": fc.get("predicted_in"),
            "likely_out": fc.get("predicted_out"),
        }

    # --- rivals in your area ---
    mine = owned_by.get(team_id, set())
    half = "h1" if gw < CHIP_HALF_BOUNDARY else "h2"
    rivals_in_area = []
    for s in standings:
        if not s["in_your_area"]:
            continue
        mid = s["manager_id"]
        theirs = owned_by.get(mid, set())
        g = grid_by.get(mid, {})
        used = [f"{CHIP_LABELS[k]} ({g[f'{k}_{half}'].replace('used ', '')})" for k in CHIP_ORDER
                if isinstance(g.get(f"{k}_{half}"), str) and g[f"{k}_{half}"].startswith("used")]
        left = [CHIP_LABELS[k] for k in CHIP_ORDER if g.get(f"{k}_{half}") == "available"]
        fc = fc_by.get(mid, {})
        rivals_in_area.append({
            "manager_id": mid, "entry_name": s["entry_name"], "player_name": s["player_name"],
            "rank": s["rank"], "points_vs_you": s["total_points"] - me["total_points"],
            "they_have": sorted(name.get(p, "?") for p in theirs - mine),
            "you_have": sorted(name.get(p, "?") for p in mine - theirs),
            "free_transfers": ft_by.get(mid),
            "likely_in": fc.get("predicted_in"), "likely_out": fc.get("predicted_out"),
            "chips_used": used, "chips_left": left,
        })

    # --- who owns what ---
    rivals = [m for m in owned_by if m != team_id]
    count: dict[int, int] = {}
    caps: dict[int, int] = {}
    benched: dict[int, int] = {}
    for r in picks[(picks["manager_id"] != team_id) & (picks["multiplier"] == 0)].itertuples():
        benched[int(r.player_id)] = benched.get(int(r.player_id), 0) + 1
    my_bench = {int(r.player_id) for r in picks[(picks["manager_id"] == team_id) & (picks["multiplier"] == 0)].itertuples()}
    for m in rivals:
        for p in owned_by[m]:
            count[p] = count.get(p, 0) + 1
    for r in picks[(picks["manager_id"] != team_id) & picks["is_captain"]].itertuples():
        caps[int(r.player_id)] = caps.get(int(r.player_id), 0) + 1

    def owned_card(p: int) -> dict[str, Any]:
        return {
            "player_id": p, "web_name": name.get(p, "?"), "team_code": team_code.get(p),
            "team_short": team_short.get(p), "position": position.get(p),
            "rivals_owning": count.get(p, 0), "rivals_captaining": caps.get(p, 0),
            "rivals_benching": benched.get(p, 0),
            "you_own": p in mine, "you_benched": p in my_bench, "projected_points": round(float(next_xp[p]), 1) if p in next_xp else None,
        }

    top_owned = [owned_card(p) for p in sorted(count, key=lambda p: (-count[p], name.get(p, "")))[:TOP_OWNED]]
    not_owned = [p for p in count if p not in mine]
    # whose big week would hurt you most: widely owned AND likely to score
    popular = sorted(not_owned, key=lambda p: -(count[p] * float(next_xp.get(p, 0.0))))[:POPULAR_NOT_OWNED]

    # --- differentials: players you don't own that at most
    # DIFFERENTIAL_MAX_RIVALS of your rivals own (counted league-wide from
    # their latest teams), ranked by next week's projected points. The old
    # site's panel judges ownership within your rival window only, which
    # can call a player 3 of your 9 rivals own a "differential".
    candidates = [p for p in next_xp if p not in mine and count.get(int(p), 0) <= DIFFERENTIAL_MAX_RIVALS]
    diffs = [owned_card(int(p)) for p in sorted(candidates, key=lambda p: -float(next_xp[p]))[:DIFFERENTIALS]]

    # --- leagues you're in (private classic only) ---
    leagues = []
    if not my_leagues.empty:
        private = my_leagues[(my_leagues["league_type"] == "x") & (my_leagues["scoring"] == "c")]
        leagues = [
            {"league_id": int(r.league_id), "name": r.name, "supported": int(r.league_id) == league_id}
            for r in private.itertuples()
        ]
    current = next((lg for lg in leagues if lg["supported"]), None)

    mode = strategy.get("mode") or "balanced"
    return {
        "available": True,
        "gameweek": gw,
        "picks_gameweek": picks_gw,
        "league_id": league_id,
        "league_name": current["name"] if current else None,
        "leagues": leagues,
        "you": you,
        "chances": chances,
        "strategy": {"mode": mode, "text": STRATEGY_TEXT.get(mode, STRATEGY_TEXT["balanced"])},
        "standings": standings,
        "teams": teams,
        "rivals_in_area": rivals_in_area,
        "rivals_count": len(rivals),
        "top_owned": top_owned,
        "popular_not_owned": [owned_card(p) for p in popular],
        "differentials": diffs,
    }
