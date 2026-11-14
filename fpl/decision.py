"""The ONE recommendation for the upcoming gameweek.

``fpl.publish`` builds a single ``decision`` object here and every panel of
the desktop app's "Your team" page (pitch, bench, transfer card, captain
card, chips card, "Why this team") renders from it -- no panel computes its
own recommendation. ``fpl.check_consistency`` then verifies the published
JSON agrees with itself.

**Transfer vs hold -- the one rule (``choose_plan``).** Two measures exist:

* the optimiser's plan (``fpl.optimise``): strategy-adjusted projected
  points over the next few gameweeks, with a threshold before a transfer
  is worth making;
* the win-chances simulation (``fpl.simulation``): how often each option
  beats every rival *this* gameweek.

They answer different questions. Rule: **in "chasing" mode follow the
option with the better win chances; otherwise follow the higher projected
points (the optimiser).** Chasing means you need to overtake people soon,
so this week's chance of beating them matters most; when balanced or
protecting, steady projected points over several weeks is the better
guide, and a single-week simulation is noisier than a 5-week projection.
A win-chances difference under ``WIN_CHANCE_TIE`` is treated as a tie
(Monte Carlo noise), in which case the optimiser's plan stands.

**Chips -- the one rule (``chip_advice``).** Play an available chip this
week only if this week is its best week in the optimiser's chip plan.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

WIN_CHANCE_TIE = 0.01  # 1 percentage point; 20k draws give ~0.3pp standard error
CAPTAIN_TIE_POINTS = 0.3  # captain options this close are "about the same"
CAPTAIN_RISK_MIN_POINTS = 1.0  # only warn about a rival captain at 1+ point swing
MAX_FREE_TRANSFERS = 5
CHIP_HALF_BOUNDARY = 19  # FPL resets chips after GW19

CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
CHIP_ORDER = ["wildcard", "freehit", "bboost", "3xc"]

TRANSFER = "transfer"
ROLL = "roll"


# ---------------------------------------------------------------------------
# transfer vs hold
# ---------------------------------------------------------------------------


def _plan_captain_row(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    mine = [r for r in rows if r.get("scenario") == scenario]
    if not mine:
        return None
    flagged = [r for r in mine if r.get("is_plan_captain")]
    return flagged[0] if flagged else max(mine, key=lambda r: r["p_finish_1st"])


def choose_plan(
    mode: str | None, optimiser_transfer: bool, sim_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Which option the page recommends, and why. ``sim_rows`` are
    ``simulation_results.json`` rows (scenario labels from fpl.simulation)."""
    from fpl.simulation import HOLD_LABEL, TRANSFER_LABEL

    t_row = _plan_captain_row(sim_rows, TRANSFER_LABEL)
    h_row = _plan_captain_row(sim_rows, HOLD_LABEL)
    p_t = t_row["p_finish_1st"] if t_row else None
    p_h = h_row["p_finish_1st"] if h_row else None
    optimiser_choice = TRANSFER if optimiser_transfer else ROLL

    sim_choice = None
    if p_t is not None and p_h is not None:
        if abs(p_t - p_h) < WIN_CHANCE_TIE:
            sim_choice = "tie"
        else:
            sim_choice = TRANSFER if p_t > p_h else ROLL

    rule = "projected_points"
    choice = optimiser_choice
    if mode == "chasing" and sim_choice in (TRANSFER, ROLL):
        rule = "win_chances"
        choice = sim_choice
    return {
        "choice": choice,
        "rule": rule,
        "mode": mode or "balanced",
        "optimiser_choice": optimiser_choice,
        "simulation_choice": sim_choice,
        "win_chance_transfer": p_t,
        "win_chance_hold": p_h,
        "overridden": choice != optimiser_choice,
    }


# ---------------------------------------------------------------------------
# chips
# ---------------------------------------------------------------------------


def chip_advice(plan_rows: list[dict[str, Any]], gameweek: int) -> dict[str, Any]:
    """The one chip rule: play an available chip now only if this is its
    best week in the plan. Also the best upcoming week otherwise."""
    play = None
    upcoming = None
    urgent: list[str] = []
    for key in CHIP_ORDER:
        rows = [r for r in plan_rows if r.get("chip") == key and r.get("available")]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["estimated_gain"])
        if any(r.get("urgent") for r in rows):
            urgent.append(key)
        if best["gameweek"] == gameweek:
            if play is None or best["estimated_gain"] > play["gain"]:
                play = {"chip": key, "label": CHIP_LABELS[key], "gain": round(float(best["estimated_gain"]), 1)}
        elif upcoming is None or best["estimated_gain"] > upcoming["gain"]:
            upcoming = {
                "chip": key, "label": CHIP_LABELS[key], "gameweek": int(best["gameweek"]),
                "gain": round(float(best["estimated_gain"]), 1),
            }
    return {"play": play, "upcoming": upcoming, "urgent": urgent}


def chip_statuses(grid_row: dict[str, Any] | None, plan_rows: list[dict[str, Any]], gameweek: int) -> list[dict[str, Any]]:
    half = "h1" if gameweek <= CHIP_HALF_BOUNDARY else "h2"
    out = []
    for key in CHIP_ORDER:
        cell = grid_row.get(f"{key}_{half}") if grid_row else None
        if isinstance(cell, str):
            available = cell == "available"
            note = "Available" if available else cell.replace("used", "Used", 1)
        else:
            rows = [r for r in plan_rows if r.get("chip") == key]
            available = any(r.get("available") for r in rows)
            note = ("Available" if available else "Used") if rows else "Unknown"
        out.append({"chip": key, "label": CHIP_LABELS[key], "available": available, "note": note})
    return out


def next_half_chips(grid_row: dict[str, Any] | None, gameweek: int) -> list[dict[str, Any]]:
    """The second set of chips, which FPL unlocks after GW19 -- shown so the
    page lists both halves. Empty once the second half has started (they're
    then "this half's" chips)."""
    if gameweek > CHIP_HALF_BOUNDARY:
        return []
    first = CHIP_HALF_BOUNDARY + 1
    out = []
    for key in CHIP_ORDER:
        cell = grid_row.get(f"{key}_h2") if grid_row else None
        used = isinstance(cell, str) and cell.startswith("used")
        out.append({
            "chip": key, "label": CHIP_LABELS[key], "available": False,
            "note": cell.replace("used", "Used", 1) if used else f"Unlocks in GW{first}",
            "unlocks_gameweek": first,
        })
    return out


# ---------------------------------------------------------------------------
# captain
# ---------------------------------------------------------------------------


def rival_captaincy(league_picks: pd.DataFrame, team_id: int | None) -> dict[str, Any]:
    """From every rival's latest confirmed picks: how many own / captain
    each player. FPL hides next week's picks until the deadline, so this is
    their last known choice (same assumption as fpl.simulation)."""
    if league_picks.empty:
        return {"rivals": 0, "owners": {}, "captains": {}}
    latest = league_picks[league_picks["gameweek"] == league_picks["gameweek"].max()]
    rivals = latest[latest["manager_id"] != team_id]
    owners = rivals.groupby("player_id")["manager_id"].nunique().to_dict()
    caps = rivals[rivals["is_captain"]].groupby("player_id")["manager_id"].nunique().to_dict()
    return {
        "rivals": int(rivals["manager_id"].nunique()),
        "owners": {int(k): int(v) for k, v in owners.items()},
        "captains": {int(k): int(v) for k, v in caps.items()},
    }


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def captain_section(
    xi: list[dict[str, Any]], rc: dict[str, Any], mode: str, xp_next: dict[int, float], web_names: dict[int, str]
) -> dict[str, Any] | None:
    cap = next((p for p in xi if p.get("is_captain")), None)
    vice = next((p for p in xi if p.get("is_vice_captain")), None)
    if cap is None:
        return None
    cap_xp = cap.get("projected_points") or 0.0
    rivals = rc["rivals"]
    owners = lambda pid: rc["owners"].get(pid, 0)  # noqa: E731

    # --- why this captain (tie-break in plain words) ---
    close = sorted(
        (p for p in xi if p is not cap and (p.get("projected_points") or 0.0) >= cap_xp - CAPTAIN_TIE_POINTS),
        key=lambda p: -(p.get("projected_points") or 0.0),
    )
    if not close:
        why = f"{cap['web_name']} is your highest projected scorer this week ({cap_xp:.1f} points)."
        tie_break = "top"
    else:
        alt = close[0]
        lead = f"{_names([cap['web_name']] + [p['web_name'] for p in close])} are projected about the same"
        n_cap, n_alt = owners(cap["player_id"]), owners(alt["player_id"])
        if n_cap < n_alt:
            tie_break = "differential"
            tail = (
                "picked as a differential because you're chasing" if mode == "chasing"
                else "the pick because fewer rivals have him"
            )
            own = f"none of your {rivals} rivals own him" if n_cap == 0 else f"{n_cap} of your {rivals} rivals own him"
            why = f"{lead}; {cap['web_name']} is {tail}: {own}, {n_alt} own {alt['web_name']}."
        elif n_cap > n_alt:
            tie_break = "template"
            tail = "picked to protect your lead" if mode == "protecting" else "picked because more rivals own him, so it's the safer choice"
            why = f"{lead}; {cap['web_name']} is {tail} ({n_cap} of your {rivals} rivals own him)."
        else:
            tie_break = "projection"
            why = f"{lead}; {cap['web_name']} edges it on projected points."

    # --- captaincy risk, only when it's meaningful ---
    fav_id = max(rc["captains"], key=rc["captains"].get) if rc["captains"] else None
    fav_count = rc["captains"].get(fav_id, 0) if fav_id is not None else 0
    xi_ids = {p["player_id"] for p in xi}
    risk: dict[str, Any] = {"show_loss": False, "rivals_favourite": None, "possible_loss": None}
    if fav_id is None:
        risk["text"] = "No clear favourite captain among your rivals, so this pick is low-risk."
    elif fav_id == cap["player_id"]:
        most = "Most" if fav_count * 2 >= rivals else "Several"
        risk["text"] = f"{most} of your rivals are likely to captain him too, so this pick is low-risk."
        risk["rivals_favourite"] = {"player_id": fav_id, "web_name": cap["web_name"], "rivals_captaining": fav_count}
    else:
        fav_name = web_names.get(fav_id, "?")
        fav_xp = float(xp_next.get(fav_id, 0.0))
        owned = fav_id in xi_ids
        # vs a rival who captains him: you get his points once (if you own
        # him) plus your captain doubled; they get him doubled
        loss = (fav_xp - cap_xp) if owned else 2 * (fav_xp - cap_xp)
        risk["rivals_favourite"] = {"player_id": fav_id, "web_name": fav_name, "rivals_captaining": fav_count, "projected_points": round(fav_xp, 2)}
        if loss >= CAPTAIN_RISK_MIN_POINTS:
            risk.update(show_loss=True, possible_loss=round(loss, 1))
            risk["text"] = f"If your rivals captain {fav_name} instead, you could lose about {loss:.0f} points to them."
        elif not owned and fav_count * 2 >= rivals:
            risk["text"] = f"Most of your rivals are likely to captain {fav_name}, who isn't in your team, so a big game from him would cost you."
        else:
            risk["text"] = f"Your rivals' most likely captain is {fav_name}, projected about the same as {cap['web_name']}, so this pick is low-risk."

    return {
        "player_id": cap["player_id"],
        "web_name": cap["web_name"],
        "projected_points": cap_xp,
        "vice_captain_id": vice["player_id"] if vice else None,
        "vice_captain_name": vice["web_name"] if vice else None,
        "tie_break": tie_break,
        "close_alternatives": [p["web_name"] for p in close],
        "why": why,
        "risk": risk,
    }


# ---------------------------------------------------------------------------
# the decision object
# ---------------------------------------------------------------------------


def _quiet(news: str) -> str:
    t = news.strip().rstrip(".")
    return t[0].lower() + t[1:] if len(t) > 1 and t[0].isupper() and t[1].islower() else t


def _pct(p: float | None) -> str:
    return f"{round(p * 100)}%" if p is not None else "?"


def build_decision(
    recommended_team: dict[str, Any],
    transfer_plan: dict[str, Any],
    plan_choice: dict[str, Any],
    horizon_gain: float | None,
    horizon_weeks: int,
    chips_plan: list[dict[str, Any]],
    chips_grid_row: dict[str, Any] | None,
    rival_caps: dict[str, Any],
    xp_next: dict[int, float],
    web_names: dict[int, str],
    strategy: dict[str, Any],
    season_context: dict[str, Any] | None,
) -> dict[str, Any]:
    rt = recommended_team
    if not rt.get("available"):
        return {"available": False, "reason": rt.get("reason", "recommended_team unavailable")}
    gw = int(rt["gameweek"])
    xi, bench = rt["starting_xi"], rt["bench"]
    transfers = rt.get("transfers") or []
    action = TRANSFER if transfers else ROLL
    mode = plan_choice["mode"]

    ft = rt.get("free_transfers_available")
    n = len(transfers)
    hits = int(rt.get("hits_taken") or 0)
    used = min(n, ft) if ft is not None else None
    left = (ft - used) if ft is not None else None
    next_week = min(left + 1, MAX_FREE_TRANSFERS) if left is not None else None

    week0 = [w for w in (transfer_plan.get("weeks") or []) if w.get("gameweek") == gw and w.get("player_out")]
    out_names = {t["player_out"].get("web_name") for t in transfers}
    news = [
        {"web_name": w["player_out"], "news": w["news"], "urgency": w.get("urgency")}
        for w in week0 if w.get("news") and w["player_out"] in out_names
    ]

    captain = captain_section(xi, rival_caps, mode, xp_next, web_names)
    advice = chip_advice(chips_plan, gw)
    statuses = chip_statuses(chips_grid_row, chips_plan, gw)

    projected = sum((p.get("projected_points") or 0.0) for p in xi)
    if captain:
        projected += captain["projected_points"]
    p_t, p_h = plan_choice["win_chance_transfer"], plan_choice["win_chance_hold"]
    win_chance = p_t if action == TRANSFER else p_h

    decision: dict[str, Any] = {
        "available": True,
        "gameweek": gw,
        "action": action,
        "rule": plan_choice["rule"],
        "plan_choice": plan_choice,
        "transfers": transfers,
        "transfer_news": news,
        "hits": hits,
        "hit_cost": hits * 4,
        "hit_reason": rt.get("hit_reason"),
        "free_transfers": {"available": ft, "used": used, "left": left, "next_week": next_week},
        "starting_xi": xi,
        "bench": bench,
        "captain": captain,
        "chip": {
            "play": advice["play"],
            "upcoming": advice["upcoming"],
            "urgent": [CHIP_LABELS[c] for c in advice["urgent"]],
            "use_by_gameweek": CHIP_HALF_BOUNDARY if gw <= CHIP_HALF_BOUNDARY else None,
            "statuses": statuses,
            "next_half": next_half_chips(chips_grid_row, gw),
        },
        "headline": {
            "projected_points": round(projected, 1),
            "horizon_gain": None if horizon_gain is None or (isinstance(horizon_gain, float) and math.isnan(horizon_gain)) else round(float(horizon_gain), 1),
            "horizon_weeks": horizon_weeks,
            "win_chance_this_week": win_chance,
            "win_chance_alternative": p_h if action == TRANSFER else p_t,
            "season_chance": (season_context or {}).get("p_finish_1st_by_target"),
            "season_target_gameweek": (season_context or {}).get("target_gameweek"),
        },
    }
    decision["why"] = why_sentences(decision, strategy)
    return decision


def why_sentences(d: dict[str, Any], strategy: dict[str, Any]) -> list[str]:
    """3-5 short plain-English sentences, all read off the decision."""
    out: list[str] = []
    h = d["headline"]
    pc = d["plan_choice"]
    ft = d["free_transfers"]
    gain = h["horizon_gain"]
    weeks = h["horizon_weeks"]
    p_t, p_h = pc["win_chance_transfer"], pc["win_chance_hold"]

    # 1. the transfer (or roll), in one sentence
    if d["action"] == TRANSFER:
        for t in d["transfers"]:
            o, i = t["player_out"]["web_name"], t["player_in"]["web_name"]
            n = next((x for x in d["transfer_news"] if x["web_name"] == o), None)
            # the news itself is on the transfer card; keep this short
            if n and n["urgency"] == "forced":
                lead = f"Swap {o}, who won't play for you, for {i}"
            elif n and n["urgency"] == "flagged":
                lead = f"Swap {o}, a doubt to play, for {i}"
            else:
                lead = f"Swap {o} for {i}"
            cost = "with no points hit" if d["hits"] == 0 else f"taking a {d['hit_cost']}-point hit"
            out.append(f"{lead}, {cost}.")
    else:
        nxt = f" and have {ft['next_week']} next week" if ft["next_week"] is not None else ""
        out.append(f"Save your free transfer this week{nxt}.")

    # 2. the rule that decided it, in plain words
    gain_txt = f"about {gain:.0f} point{'s' if round(gain) != 1 else ''} over the next {weeks} weeks" if gain is not None else None
    if d["action"] == TRANSFER and pc["rule"] == "projected_points":
        if p_t is not None and p_h is not None:
            if pc["simulation_choice"] == TRANSFER:
                chances = f"and gives better chances of winning this week ({_pct(p_t)} vs {_pct(p_h)} if you hold)"
            elif pc["simulation_choice"] == "tie":
                chances = f"so we're doing it, even though holding gives very similar chances this week ({_pct(p_h)})"
            else:
                chances = f"so we're doing it, even though holding gives slightly better chances this week ({_pct(p_h)} vs {_pct(p_t)})"
            sep = " " if pc["simulation_choice"] == TRANSFER else ", "
            out.append(f"Making the transfer adds {gain_txt or 'projected points'}{sep}{chances}.")
        elif gain_txt:
            out.append(f"Making the transfer adds {gain_txt}.")
    elif d["action"] == TRANSFER:  # chasing, win chances decided
        out.append(f"You're chasing, so we follow the better chances of winning this week ({_pct(p_t)} vs {_pct(p_h)} for holding).")
    elif pc["overridden"]:  # chasing, win chances say hold
        out.append(
            f"You're chasing, so we hold: it gives better chances of winning this week ({_pct(p_h)} vs {_pct(p_t)}), "
            f"even though the transfer adds {gain_txt or 'a little'}."
        )
    elif gain_txt and gain is not None and gain > 0:
        out.append(f"The best transfer adds only {gain_txt}, not enough to use it now.")

    # 3. captain
    cap = d["captain"]
    if cap:
        out.append(f"Captain {cap['web_name']}: {cap['why'][0].lower() + cap['why'][1:]}" if not cap["why"].startswith(cap["web_name"]) else cap["why"])

    # 4. chips
    chip = d["chip"]
    if chip["play"]:
        out.append(f"Play your {chip['play']['label']} now: this is its best week (about {chip['play']['gain']:.0f} extra points).")
    elif chip["upcoming"]:
        u = chip["upcoming"]
        out.append(f"No chip this week; your {u['label']} is worth more in GW{u['gameweek']}.")

    # 5. where you stand (only if room)
    if len(out) < 5 and strategy.get("available") and isinstance(strategy.get("gap_to_first"), (int, float)):
        gap = strategy["gap_to_first"]
        where = f"You're {gap} point{'s' if gap != 1 else ''} off first" if gap > 0 else "You're top of your league"
        sc = h["season_chance"]
        tail = f", with about a {_pct(sc)} chance of being top by GW{h['season_target_gameweek']}" if sc is not None else ""
        out.append(f"{where}{tail}.")
    return out[:5]
