"""The desktop app's "Planner" page -- one ``planner`` section in
latest.json: the plan for the next few gameweeks at a glance.

Built by ``fpl.publish`` from the same plan the rest of the app uses:

* transfers per week: the chosen plan's transfer table (the one
  ``recommended_team``/``transfer_plan`` are built from), with the incoming
  players' cards from the chosen squad table; week 1 is copied straight
  from the ONE ``decision`` (fpl/decision.py), so it can't disagree with it;
* free transfers per week: the optimiser's own per-week values (saved in
  its transfer table), week 1 from the decision -- never recalculated;
* chips: each chip you still have, placed in its best week from the chip
  plan (``chips.yours.plan``), with used chips shown as used;
* fixtures: your squad's clubs against each gameweek, with FPL's own
  fixture difficulty rating (1 easy - 5 hard), blank and double gameweeks
  marked.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.decision import CHIP_LABELS, CHIP_ORDER

WEEKS_SHOWN = 6
NOTE = ("This plan updates every time the data refreshes. Only this week's moves are final advice; "
        "later weeks will change as news comes in.")


def _card(pid: int | None, info: dict[int, dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    if pid is None or pid not in info:
        return {"player_id": None, "web_name": name, "team_code": None, "team_short": None, "position": None, "price": None}
    i = info[pid]
    return {
        "player_id": pid, "web_name": i["web_name"], "team_code": None if pd.isna(i.get("team_code")) else int(i["team_code"]),
        "team_short": i.get("team_short_name"), "position": i.get("position"),
        "price": None if pd.isna(i.get("now_cost")) else round(float(i["now_cost"]) / 10, 1),
    }


def build_planner(
    *,
    decision: dict[str, Any],
    squad_src: pd.DataFrame,
    transfers_src: pd.DataFrame,
    players: pd.DataFrame,
    gameweeks: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    chips_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    if not decision.get("available") or players.empty or gameweeks.empty:
        return {"available": False, "reason": "No plan in this update yet."}
    gw0 = int(decision["gameweek"])
    week_ids = [gw0 + i for i in range(WEEKS_SHOWN)]
    deadlines = {int(r.gameweek): r.deadline_time for r in gameweeks.itertuples()}
    weeks = [{"gameweek": g, "deadline": deadlines.get(g)} for g in week_ids if g in deadlines]
    info = players.set_index("id").to_dict("index")
    by_name: dict[str, list[int]] = {}
    for pid, i in info.items():
        by_name.setdefault(i["web_name"], []).append(int(pid))
    planned_weeks = sorted(int(g) for g in transfers_src["gameweek"].unique()) if not transfers_src.empty else []

    # --- transfers per week (week 1 = the decision, exactly) ---
    # Free transfers come from the optimiser's own plan (its per-week
    # free_transfers column), never recalculated here.
    transfers: list[dict[str, Any]] = []
    ft_by_week: dict[int, int] = {}
    if not transfers_src.empty and "free_transfers" in transfers_src.columns:
        for r in transfers_src.dropna(subset=["free_transfers"]).itertuples():
            ft_by_week.setdefault(int(r.gameweek), int(r.free_transfers))
    for w in weeks:
        g = w["gameweek"]
        if g == gw0:
            moves = [{"out": t["player_out"], "in": t["player_in"]} for t in decision.get("transfers") or []]
            hits, reason = int(decision.get("hits") or 0), decision.get("hit_reason")
            planned = True
        elif g in planned_weeks:
            rows = transfers_src[transfers_src["gameweek"] == g].dropna(subset=["player_in", "player_out"], how="all")
            ins = squad_src[(squad_src["gameweek"] == g) & squad_src["transferred_in"]] if not squad_src.empty else pd.DataFrame()
            in_ids = {info[int(p)]["web_name"]: int(p) for p in ins.get("player_id", [])}
            prev_squad = set(squad_src[squad_src["gameweek"] == g - 1]["player_id"].astype(int)) if not squad_src.empty else set()
            moves = []
            for r in rows.itertuples():
                in_id = in_ids.get(r.player_in)
                out_ids = [p for p in by_name.get(r.player_out, []) if p in prev_squad] or by_name.get(r.player_out, [])
                moves.append({"out": _card(out_ids[0] if out_ids else None, info, r.player_out),
                              "in": _card(in_id, info, r.player_in)})
            hits = int(rows["hits_taken"].sum()) if not rows.empty else 0
            reason = rows["hit_reason"].dropna().iloc[0] if "hit_reason" in rows and rows["hit_reason"].notna().any() else None
            planned = True
        else:
            moves, hits, reason, planned = [], 0, None, False
        ft = (decision.get("free_transfers") or {}).get("available") if g == gw0 else ft_by_week.get(g)
        transfers.append({
            "gameweek": g, "planned": planned, "moves": moves, "save": planned and not moves,
            "free_transfers": ft if planned else None, "hits": hits, "hit_cost": 4 * hits, "hit_reason": reason,
        })

    # --- chips: best week for each chip you still have; used ones as used ---
    statuses = {c["chip"]: c for c in ((decision.get("chip") or {}).get("statuses") or [])}
    play_now = ((decision.get("chip") or {}).get("play") or {}).get("chip")
    chips = []
    for key in CHIP_ORDER:
        st = statuses.get(key, {})
        if not st.get("available", False):
            chips.append({"chip": key, "label": CHIP_LABELS[key], "status": "used", "note": st.get("note", "Used"), "gameweek": None, "gain": None})
            continue
        rows = [r for r in chips_plan if r.get("chip") == key and r.get("available")]
        best = max(rows, key=lambda r: r["estimated_gain"]) if rows else None
        if best is None:
            chips.append({"chip": key, "label": CHIP_LABELS[key], "status": "later", "note": "Not in the next few weeks", "gameweek": None, "gain": None})
            continue
        g = int(best["gameweek"])
        # the decision is the only word on this week: a chip is "this week" only if it says so
        if g == gw0 and play_now != key:
            g, status = None, "later"
        chips.append({"chip": key, "label": CHIP_LABELS[key], "status": "planned" if g else "later",
                      "gameweek": g, "gain": round(float(best["estimated_gain"]), 1)})

    # --- fixtures for your squad's clubs, FPL's difficulty ---
    squad = [c for c in decision["starting_xi"] + decision["bench"]]
    team_of = {int(r.id): r for r in teams.itertuples()} if not teams.empty else {}
    code_to_id = {int(r.code): int(r.id) for r in teams.itertuples()} if not teams.empty else {}
    clubs: dict[int, dict[str, Any]] = {}
    for c in squad:
        tid = code_to_id.get(c.get("team_code"))
        if tid is None:
            continue
        clubs.setdefault(tid, {"team_short": team_of[tid].short_name, "team_code": int(team_of[tid].code), "players": []})
        clubs[tid]["players"].append(c["web_name"])
    rows_out = []
    fx = fixtures[fixtures["event"].isin(week_ids)] if not fixtures.empty else fixtures
    for tid, club in sorted(clubs.items(), key=lambda kv: -len(kv[1]["players"])):
        cells = {}
        for g in [w["gameweek"] for w in weeks]:
            games = []
            for r in fx[fx["event"] == g].itertuples():
                if r.team_h == tid:
                    games.append({"opponent": team_of[r.team_a].short_name, "home": True, "difficulty": int(r.team_h_difficulty)})
                elif r.team_a == tid:
                    games.append({"opponent": team_of[r.team_h].short_name, "home": False, "difficulty": int(r.team_a_difficulty)})
            cells[str(g)] = {"games": games, "blank": not games, "double": len(games) > 1}
        rows_out.append({**club, "cells": cells})

    return {
        "available": True,
        "note": NOTE,
        "first_gameweek": gw0,
        "weeks": weeks,
        "transfers": transfers,
        "chips": chips,
        "fixtures": rows_out,
    }
