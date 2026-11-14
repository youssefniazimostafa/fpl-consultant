"""Tools for the app's "Live consultant" chat (fpl/chat.py). Every answer
the chat gives must come from these, built on the same tables and the same
optimiser as the rest of the app -- nothing here invents data.

* get_player(name)
* compare_players(a, b) -- side by side + whether swapping is possible
* what_if(transfers_in, transfers_out, captain) -- re-solve the plan with
  that move forced (fpl.optimise.solve_horizon), vs the recommendation
* get_league_context()

Players are matched by FPL web name (fuzzy, so "van hecke" works).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import process

from fpl.fetch import load_config
from fpl.league import resolve_team_id
from fpl.optimise import (
    HIT_COST,
    MAX_PER_CLUB,
    WEEKLY_DISCOUNT,
    build_candidate_pool,
    estimate_selling_prices,
    load_tables,
    lookup_weekly_xp,
    recommend_transfers,
    resolve_free_transfers,
    resolve_my_squad,
    solve_horizon,
    strategy_adjusted,
)

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "site" / "data" / "latest.json"

MODE_MEANING = {
    "balanced": "Balanced: the plan picks the strongest team on projected points and mostly ignores who your rivals own.",
    "chasing": "Chasing: you're well behind, so the plan leans towards players few rivals own (differentials) to catch up.",
    "protecting": "Protecting: you're well ahead, so the plan leans towards players your rivals own, to stop them catching up.",
}


class ToolError(Exception):
    pass


@lru_cache(maxsize=1)
def _ctx() -> dict[str, Any]:
    """The optimiser's inputs, set up exactly as fpl.optimise.main does."""
    config = load_config()
    t = load_tables()
    for extra in ("teams", "fixtures", "xp"):  # not all in load_tables()
        if t.get(extra) is None or t[extra].empty:
            path = ROOT / "data" / "tables" / f"{extra}.parquet"
            t[extra] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    players, xp_by_gw = t["players"], t["xp_by_gameweek"]
    managers = t.get("league_managers", pd.DataFrame())
    team_id = resolve_team_id(config, managers)
    my_squad, hist_bank, gameweek = resolve_my_squad(t["league_picks"], t["league_history"], team_id)
    bank = int(config["bank_override"]) if config.get("bank_override") is not None else hist_bank
    swings = t["league_swings"]
    eo: dict[int, float] = {}
    if not swings.empty:
        eff = swings["weighted_league_eo"].fillna(swings["league_eo"]) if "weighted_league_eo" in swings.columns else swings["league_eo"]
        eo = pd.Series(eff.values, index=swings["player_id"]).to_dict()
    mode = config.get("strategy")
    if mode not in {"chasing", "protecting", "balanced"}:
        mode = t["league_strategy"]["mode"].iloc[0] if not t["league_strategy"].empty else "balanced"
    sell_owned = estimate_selling_prices(my_squad, players, t["my_transfers"])
    avoid = {int(x) for x in (config.get("avoid") or []) if isinstance(x, int)}
    pool = build_candidate_pool(players, xp_by_gw, my_squad, avoid)
    src = players.set_index("id")[["web_name", "position", "team", "now_cost", "status", "chance_of_playing_next_round", "news"]].copy()
    src["chance_of_playing_next_round"] = src["chance_of_playing_next_round"].where(src["chance_of_playing_next_round"].notna(), None)
    info = src.to_dict("index")
    weeks = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    weeks = weeks[: max(min(int(config.get("horizon", 5)), len(weeks)), 1)]
    raw = lookup_weekly_xp(xp_by_gw, weeks, pool)
    return {
        "t": t, "players": players, "xp_by_gw": xp_by_gw, "team_id": team_id, "my_squad": set(my_squad), "bank": bank,
        "mode": mode, "eo": eo, "sell": {p: sell_owned.get(p, info[p]["now_cost"]) for p in pool}, "pool": pool,
        "info": info, "weeks": weeks, "discount": {g: WEEKLY_DISCOUNT ** i for i, g in enumerate(weeks)},
        "raw": raw, "adj": strategy_adjusted(raw, eo, mode),
        "free": resolve_free_transfers(t["league_free_transfers"], team_id),
    }


def _find(name: str) -> int:
    c = _ctx()
    names = {int(pid): r["web_name"] for pid, r in c["info"].items()}
    hit = process.extractOne(name, names, score_cutoff=75)
    if not hit:
        raise ToolError(f"No player called {name!r} in FPL's data.")
    return int(hit[2])


def _player(pid: int) -> dict[str, Any]:
    c = _ctx()
    t, info = c["t"], c["info"][pid]
    pl = c["players"].set_index("id").loc[pid]
    xp = c["xp_by_gw"].set_index("player_id")
    proj = {f"GW{g}": round(float(xp.loc[pid, f"gw{g}_xp"]), 2) for g in c["weeks"]} if pid in xp.index else {}
    teams = t.get("teams", pd.DataFrame())
    short = teams.set_index("id")["short_name"].to_dict() if not teams.empty else {}
    fx = t.get("fixtures", pd.DataFrame())
    fixtures = []
    if not fx.empty:
        tid = int(pl["team"])
        for r in fx[fx["event"].isin(c["weeks"])].sort_values("event").itertuples():
            if r.team_h == tid:
                fixtures.append({"gameweek": int(r.event), "opponent": short.get(r.team_a), "home": True, "fpl_difficulty_1_to_5": int(r.team_h_difficulty)})
            elif r.team_a == tid:
                fixtures.append({"gameweek": int(r.event), "opponent": short.get(r.team_h), "home": False, "fpl_difficulty_1_to_5": int(r.team_a_difficulty)})
    xpt = t.get("xp", pd.DataFrame())
    start = None
    if not xpt.empty:
        nxt = xpt[(xpt["player_id"] == pid) & (xpt["event"] == c["weeks"][0])]
        if not nxt.empty:
            start = round(float(nxt["p_60"].iloc[0]) * 100)
    sw = t["league_swings"]
    srow = sw[sw["player_id"] == pid] if not sw.empty else sw
    chance = info.get("chance_of_playing_next_round")
    return {
        "name": info["web_name"],
        "club": pl.get("team_short_name"),
        "position": info["position"],
        "price_m": round(info["now_cost"] / 10, 1),
        "projected_points": proj,
        "projected_points_next_5": round(sum(proj.values()), 2),
        "chance_of_playing_60_min_next_gw_pct": start,
        "fpl_chance_of_playing_next_round_pct": None if chance is None or pd.isna(chance) else int(chance),
        "fpl_news": info.get("news") or None,
        "fixtures": fixtures,
        "league_ownership_pct": None if srow.empty else round(float(srow["league_ownership_pct"].iloc[0]), 1),
        "overall_ownership_pct": float(pl.get("selected_by_percent") or 0),
        "you_own": pid in c["my_squad"],
    }


def get_player(name: str) -> dict[str, Any]:
    return _player(_find(name))


def compare_players(a: str, b: str) -> dict[str, Any]:
    c = _ctx()
    pa, pb = _find(a), _find(b)
    A, B = _player(pa), _player(pb)
    swaps = []
    for out_id, in_id in ((pa, pb), (pb, pa)):
        if out_id not in c["my_squad"] or in_id in c["my_squad"]:
            continue
        o, i = c["info"][out_id], c["info"][in_id]
        budget = c["bank"] + c["sell"].get(out_id, o["now_cost"])
        club = sum(1 for p in c["my_squad"] if p != out_id and c["info"].get(p, {}).get("team") == i["team"])
        reasons = []
        if o["position"] != i["position"]:
            reasons.append(f"different positions ({o['position']} vs {i['position']})")
        if budget < i["now_cost"]:
            reasons.append(f"£{(i['now_cost'] - budget) / 10:.1f}m short (bank £{c['bank'] / 10:.1f}m + selling price £{c['sell'].get(out_id, o['now_cost']) / 10:.1f}m)")
        if club >= MAX_PER_CLUB:
            reasons.append(f"you'd have {club + 1} players from one club (max {MAX_PER_CLUB})")
        swaps.append({"out": o["web_name"], "in": i["web_name"], "possible": not reasons,
                      "money_left_after_m": round((budget - i["now_cost"]) / 10, 1), "why_not": reasons})
    if not swaps:
        swaps.append({"note": "Swap check needs exactly one of the two in your squad."})
    return {"a": A, "b": B, "swap": swaps}


def _plan_points(plan: dict[str, Any]) -> tuple[float, float]:
    c = _ctx()
    pts = []
    for wk in plan["weeks"]:
        g = wk["gameweek"]
        v = sum(c["raw"].get((p, g), 0.0) for p in wk["xi_ids"])
        if wk["captain_id"] is not None:
            v += c["raw"].get((wk["captain_id"], g), 0.0)
        pts.append(v - HIT_COST * wk["hits"])
    return pts[0], sum(pts)


def what_if(transfers_in: list[str] | None = None, transfers_out: list[str] | None = None, captain: str | None = None) -> dict[str, Any]:
    """Empty transfers = keep your squad this week (no transfer)."""
    c = _ctx()
    args = (c["pool"], c["adj"], c["info"], c["my_squad"], c["sell"], c["bank"], c["free"], c["weeks"], c["discount"])
    base = solve_horizon(*args)  # the recommendation, exactly as fpl.optimise.main
    rec, _, _, _ = recommend_transfers(base, *args, raw_weekly_xp=c["raw"])
    ins = frozenset(_find(n) for n in (transfers_in or []))
    outs = frozenset(_find(n) for n in (transfers_out or []))
    cap = _find(captain) if captain else None
    for p in outs:
        if p not in c["my_squad"]:
            raise ToolError(f"You don't own {c['info'][p]['web_name']}.")
    g0 = c["weeks"][0]
    held = frozenset({g0}) if not ins and not outs else frozenset()
    # exactly the asked-for moves this week: everyone else in your squad stays
    keep = frozenset(c["my_squad"] - outs) | ins
    alt = solve_horizon(*args, held_weeks=held, force_in=keep, force_out=outs, force_captain=cap)
    if alt["status"] != "Optimal":
        raise ToolError("That move isn't possible within FPL's rules (budget, positions or 3 per club).")
    r0, r5 = _plan_points(rec)
    a0, a5 = _plan_points(alt)
    name = lambda ids: sorted(c["info"][p]["web_name"] for p in ids)  # noqa: E731
    wk = alt["weeks"][0]
    return {
        "scenario": {"transfers_in": name(wk["transfers_in"]), "transfers_out": name(wk["transfers_out"]),
                     "captain": c["info"][wk["captain_id"]]["web_name"] if wk["captain_id"] else None,
                     "points_hit": HIT_COST * wk["hits"], "kept_squad_this_week": bool(held)},
        "recommendation": {"transfers_in": name(rec["weeks"][0]["transfers_in"]), "transfers_out": name(rec["weeks"][0]["transfers_out"]),
                           "captain": c["info"][rec["weeks"][0]["captain_id"]]["web_name"]},
        "projected_points_this_week": {"scenario": round(a0, 1), "recommendation": round(r0, 1), "difference": round(a0 - r0, 1)},
        "projected_points_next_5_weeks": {"scenario": round(a5, 1), "recommendation": round(r5, 1), "difference": round(a5 - r5, 1)},
        "note": "Projected points of the best XI + captain each week, minus any points hits; later weeks are re-planned for both.",
    }


def get_league_context() -> dict[str, Any]:
    d = json.loads(LATEST.read_text())
    lp = d.get("league_page") or {}
    mode = (d.get("strategy") or {}).get("mode", "balanced")
    return {
        "you": lp.get("you"), "strategy_mode": mode, "what_the_mode_means": MODE_MEANING.get(mode),
        "chances": lp.get("chances"),
        "rivals_near_you": [{k: r.get(k) for k in ("entry_name", "rank", "points_vs_you", "they_have", "you_have", "free_transfers", "likely_in", "likely_out", "chips_left")}
                            for r in lp.get("rivals_in_area", [])],
        "most_owned_in_league": [{k: p.get(k) for k in ("web_name", "rivals_owning", "rivals_captaining", "you_own")}
                                 for p in lp.get("top_owned", [])[:10]],
    }


TOOLS = {"get_player": get_player, "compare_players": compare_players, "what_if": what_if, "get_league_context": get_league_context}

TOOL_SCHEMAS = [
    {"name": "get_player", "description": "One player's price, club, position, projected points for each of the next 5 gameweeks, chance of playing, fixtures with FPL difficulty, FPL news, league and overall ownership, and whether you own him.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Player's FPL name, e.g. 'Van Hecke'"}}, "required": ["name"], "additionalProperties": False}},
    {"name": "compare_players", "description": "Two players side by side (same fields as get_player) plus whether swapping one you own for the other is possible (budget, bank, position, max 3 per club).",
     "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": False}},
    {"name": "what_if", "description": "Re-solve this week's plan with a move forced, and compare projected points this week and over 5 weeks against the recommendation. Leave transfers_in and transfers_out empty to test keeping your squad (no transfer). captain is optional.",
     "input_schema": {"type": "object", "properties": {"transfers_in": {"type": "array", "items": {"type": "string"}}, "transfers_out": {"type": "array", "items": {"type": "string"}},
                                                        "captain": {"type": "string"}}, "required": ["transfers_in", "transfers_out"], "additionalProperties": False}},
    {"name": "get_league_context", "description": "Your league position, rivals near you (what they own that you don't and vice versa, free transfers, likely moves, chips), the most-owned players in your league, and the strategy mode with its meaning.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]
