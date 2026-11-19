"""Step 17: today's close calls, scored on your league goals (fpl/goals.py).

Written by ``fpl.simulation`` (which already has the plans, the rivals'
last confirmed XIs and captains, and the transfer options) to
``data/tables/goal_calls.json``; ``fpl.publish`` puts it in latest.json as
``goal_calls`` for the Recommended team captain card, the Planner, the Live
consultant and the Your plan review (tool ``get_goal_chances``).

For the captain: every player in the plan's XI goes through the 1-SD
close-call test against the highest-xP captain; the close ones are
simulated. For transfers: the options of Analytics' "Your options this
week" (the optimiser's move or hold, the next-best moves and hold) against
the optimiser's pick. Each is computed for the goals' target gameweek and
for the other common target (GW19 or the end of the season), so the chat can
answer both.

Display only unless ``fpl.goals.USE_GOAL_AWARE`` is on: then the
optimiser's captain is already the rule's pick (see fpl/optimise.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl import goals as G

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "tables" / "goal_calls.json"
SHOW_OPTIONS = 3
HALF_GW = 19


def _rest_stats(league_history: pd.DataFrame) -> dict[int, tuple[float, float]]:
    stats = league_history.groupby("manager_id")["points"].agg(["mean", "std"])
    stats["std"] = stats["std"].fillna(0.0).clip(lower=G.REST_MIN_SD)
    return {int(m): (float(r["mean"]), float(r["std"])) for m, r in stats.iterrows()}


def league_state(league_history: pd.DataFrame, rival_plans: list[dict[str, Any]], team_id: int) -> dict[str, Any] | None:
    """Totals now and each manager's weekly mean / SD, plus the rivals in
    the simulator's shape (their last confirmed XI and captain)."""
    if league_history.empty:
        return None
    last = league_history[league_history["gameweek"] == league_history["gameweek"].max()]
    totals = last.set_index("manager_id")["total_points"].to_dict()
    if team_id not in totals:
        return None
    rest = _rest_stats(league_history)
    rivals = [{"xi_ids": r["xi_ids"], "captain_id": r["captain_id"], "total": float(totals.get(r["manager_id"], 0)),
               "rest_mean": rest.get(r["manager_id"], (55.0, 16.5))[0], "rest_sd": rest.get(r["manager_id"], (55.0, 16.5))[1]}
              for r in rival_plans if r["manager_id"] in totals]
    me = {"total": float(totals[team_id]), "rest_mean": rest[team_id][0], "rest_sd": rest[team_id][1]}
    return {"rivals": rivals, "me": me, "gameweek": int(last["gameweek"].iloc[0])}


def _sd_map(ids: set[int], xp: dict[int, float], info: dict[int, dict[str, Any]]) -> dict[int, float]:
    return {p: G.error_sd(info.get(p, {}).get("position", "MID"), float(xp.get(p, 0.0))) for p in ids}


def captain_call(
    xi_ids: list[int], xpert_captain: int | None, xp: dict[int, float], info: dict[int, dict[str, Any]],
    state: dict[str, Any], goals: list[dict[str, Any]], rest_weeks: int,
) -> dict[str, Any]:
    name = lambda p: info.get(p, {}).get("web_name", "?")  # noqa: E731
    position = {p: info.get(p, {}).get("position", "MID") for p in xi_ids}
    xp_xi = {p: float(xp.get(p, 0.0)) for p in xi_ids}
    best, rows = G.captain_close_calls(list(xi_ids), xp_xi, position)
    close = [r for r in rows if r["close"]]
    out: dict[str, Any] = {
        "xp_pick": {"player_id": best, "web_name": name(best), "xp": round(xp_xi[best], 2)},
        "xpert_pick": {"player_id": xpert_captain, "web_name": name(xpert_captain) if xpert_captain else None},
        "close": len(close) > 1,
    }
    if len(close) <= 1:
        out.update(verdict="clear", goal_pick=out["xp_pick"], options=[], draws=0)
        return out
    cands = [r["player_id"] for r in close]
    opts = [{"xi_ids": sorted(xi_ids), "captain_id": c, "hits": 0, "future_delta": 0.0} for c in cands]
    ids = set(xi_ids) | {p for r in state["rivals"] for p in r["xi_ids"]}
    sim = G.simulate(opts, state["rivals"], state["me"], {p: float(xp.get(p, 0.0)) for p in ids},
                     _sd_map(ids, xp, info), goals, rest_weeks, ref=0)
    pick, verdict = G.choose(sim, 0)
    options = [{
        "player_id": c, "web_name": name(c), "xp": round(r["xp"], 2), "gap": round(r["gap"], 2), "sd": round(r["sd"], 2),
        "chances": {k: round(v, 4) for k, v in s["chances"].items()}, "score": round(s["score"], 4),
        "diff_vs_xp_pick": round(s["diff_vs_xp_pick"], 4), "diff_se": round(s["diff_se"], 4),
        "is_xp_pick": c == best, "is_xpert_pick": c == xpert_captain, "is_goal_pick": c == cands[pick],
    } for c, r, s in zip(cands, close, sim["rows"])]
    out.update(verdict=verdict, goal_pick={"player_id": cands[pick], "web_name": name(cands[pick])},
               options=options, draws=sim["draws"], close_count=len(close))
    return out


def transfer_call(
    plans: list[tuple[dict[str, Any], bool]], raw: dict[tuple[int, int], float], discount: dict[int, float],
    free_now: int, ft_value: float, xp: dict[int, float], info: dict[int, dict[str, Any]],
    state: dict[str, Any], goals: list[dict[str, Any]], rest_weeks: int, target: int,
) -> dict[str, Any]:
    name = lambda p: info.get(p, {}).get("web_name", "?")  # noqa: E731
    plans = [(p, rec) for p, rec in plans if rec or len(p["weeks"][0]["xi_ids"]) == 11]
    if not plans:
        return {"close": False, "verdict": "clear", "options": []}
    ref_i = next((i for i, (_, rec) in enumerate(plans) if rec), 0)
    order = [ref_i] + [i for i in range(len(plans)) if i != ref_i]
    ps = [plans[i][0] for i in order]
    position = {p: info.get(p, {}).get("position", "MID") for p in info}
    vals = [G.plan_value(p, raw, discount, free_now, ft_value) for p in ps]
    sds = [0.0] + [G.plan_diff_sd(ps[0], p, raw, discount, position) for p in ps[1:]]
    close = [0] + [i for i in range(1, len(ps)) if vals[0] - vals[i] <= G.CLOSE_SD * sds[i]]

    def label(p: dict[str, Any]) -> str:
        w0 = p["weeks"][0]
        if not w0["transfers_in"]:
            return "Save the transfer"
        return ", ".join(f"{name(o)} → {name(i)}" for o, i in zip(sorted(w0["transfers_out"]), sorted(w0["transfers_in"])))

    base = {"options": [{"label": label(p), "value": round(v, 2), "gap": round(vals[0] - v, 2), "sd": round(s, 2),
                         "close": i in close, "is_xpert_pick": i == 0} for i, (p, v, s) in enumerate(zip(ps, vals, sds))]}
    if len(close) <= 1:
        return {"close": False, "verdict": "clear", "xpert_pick": label(ps[0]), "goal_pick": label(ps[0]), **base}
    ref_future = G.future_points(ps[0], raw, target) + ft_value * G.ft_end(ps[0], free_now)
    opts = []
    for i in close:
        w0 = ps[i]["weeks"][0]
        opts.append({"xi_ids": sorted(w0["xi_ids"]), "captain_id": w0["captain_id"],
                     "hits": max(len(w0["transfers_in"]) - free_now, 0),
                     "future_delta": G.future_points(ps[i], raw, target) + ft_value * G.ft_end(ps[i], free_now) - ref_future})
    ids = {p for o in opts for p in o["xi_ids"]} | {p for r in state["rivals"] for p in r["xi_ids"]}
    sim = G.simulate(opts, state["rivals"], state["me"], {p: float(xp.get(p, 0.0)) for p in ids},
                     _sd_map(ids, xp, info), goals, rest_weeks, ref=0, seed=G.SEED + 1)
    pick, verdict = G.choose(sim, 0)
    for j, i in enumerate(close):
        s = sim["rows"][j]
        base["options"][i].update(chances={k: round(v, 4) for k, v in s["chances"].items()}, score=round(s["score"], 4),
                                  diff_vs_xp_pick=round(s["diff_vs_xp_pick"], 4), diff_se=round(s["diff_se"], 4),
                                  is_goal_pick=j == pick)
    return {"close": True, "verdict": verdict, "xpert_pick": label(ps[0]), "goal_pick": label(ps[close[pick]]),
            "draws": sim["draws"], **base}


def _captain_text(call: dict[str, Any], goals: list[dict[str, Any]], when: str) -> tuple[str | None, str]:
    """(the side-by-side line, why XPert picks its captain) in plain words."""
    xp_pick, xpert, goal = call["xp_pick"], call["xpert_pick"], call.get("goal_pick") or {}
    if not call["close"]:
        return None, f"{xpert['web_name']} is clearly ahead on expected points, so no close call."
    opts = call["options"]
    keep = [o for o in opts if o["is_xp_pick"] or o["is_xpert_pick"] or o["is_goal_pick"]]
    for o in sorted(opts, key=lambda o: -o["xp"]):
        if len(keep) >= SHOW_OPTIONS:
            break
        if o not in keep:
            keep.append(o)
    keep.sort(key=lambda o: -o["xp"])
    line = " · ".join(G.option_text(o["web_name"], o["chances"], goals) for o in keep)
    pick_name = xpert["web_name"]
    if G.USE_GOAL_AWARE and call["verdict"] == "goals":
        why = (f"XPert picks {pick_name}: about the same expected points as {xp_pick['web_name']}, "
               f"and a better chance of your goals by {when}.")
    elif call["verdict"] == "goals":
        why = (f"On your goals by {when}, {goal['web_name']} comes out ahead. XPert still picks {pick_name} by today's rule "
               f"(fewer rivals own him breaks near-ties): the goal-based rule isn't used for picks yet because it "
               f"didn't clearly beat today's rule in testing.")
    elif call["verdict"] == "agree":
        why = (f"{xp_pick['web_name']} has the most expected points and also the best chance of your goals by {when}."
               + ("" if pick_name == xp_pick["web_name"] else f" XPert picks {pick_name} by today's rule (fewer rivals own him breaks near-ties)."))
    else:
        why = (f"Too close to call on your goals by {when}: the differences are within the simulation's noise, "
               f"so expected points decide.")
        if pick_name != xp_pick["web_name"]:
            why += f" XPert picks {pick_name} by today's rule (fewer rivals own him breaks near-ties)."
    return line, why


def _one_target(
    target: int, note: str | None, goals: dict[str, Any], plans_by_label: dict[str, dict[str, Any]],
    option_plans: list[tuple[dict[str, Any], bool]], solver: dict[str, Any], state: dict[str, Any], next_gw: int,
) -> dict[str, Any]:
    rest_weeks = max(target - next_gw, 0)
    when = "the end of the season" if target == G.LAST_GW else f"GW{target}"
    caps = {}
    for label, plan in plans_by_label.items():
        w0 = plan["weeks"][0]
        call = captain_call(sorted(w0["xi_ids"]), w0["captain_id"], solver["xp"], solver["info"], state,
                            goals["goals"], rest_weeks)
        call["text"], call["why"] = _captain_text(call, goals["goals"], when)
        caps[label] = call
    tr = transfer_call(option_plans, solver["raw"], solver["discount"], solver["free"], solver["ft_value"],
                       solver["xp"], solver["info"], state, goals["goals"], rest_weeks, target)
    if tr.get("close"):
        shown = [o for o in tr["options"] if o.get("chances")]
        tr["text"] = " · ".join(G.option_text(o["label"], o["chances"], goals["goals"]) for o in shown)
        tr["why"] = ("Too close to call on your goals, so expected points decide." if tr["verdict"] == "too close"
                     else f"{tr['goal_pick']} gives the best chance of your goals by {when}."
                     + ("" if tr["goal_pick"] == tr["xpert_pick"] or G.USE_GOAL_AWARE
                        else f" XPert still recommends {tr['xpert_pick']}: the goal-based rule isn't used for picks yet."))
    return {"target_gameweek": target, "target_note": note, "goals_text": G.describe_goals(goals, target),
            "captain": caps, "transfers": tr}


def build(
    plans_by_label: dict[str, dict[str, Any]], option_plans: list[tuple[dict[str, Any], bool]],
    solver: dict[str, Any], rival_plans: list[dict[str, Any]], league_history: pd.DataFrame, team_id: int,
) -> dict[str, Any]:
    """The goal_calls section. ``solver``: xp (next GW xP by player), raw
    (horizon xP), discount, free, ft_value, info, next_gw."""
    goals = G.load_goals()
    state = league_state(league_history, rival_plans, team_id)
    if state is None or not rival_plans:
        return {"available": False, "reason": "needs your league's standings and rivals' teams"}
    next_gw = solver["next_gw"]
    target, note = G.target_gameweek(goals, next_gw)
    main = _one_target(target, note, goals, plans_by_label, option_plans, solver, state, next_gw)
    other_t = HALF_GW if target == G.LAST_GW and next_gw <= HALF_GW else G.LAST_GW
    other = None
    if other_t != target:
        other = _one_target(other_t, None, goals, plans_by_label, option_plans, solver, state, next_gw)
    return {
        "available": True, "gameweek": next_gw, "goals": goals["goals"], "goals_source": goals["source"],
        "close_sd": G.CLOSE_SD, "noise_se": G.NOISE_SE, "auto_uses_rule": G.USE_GOAL_AWARE,
        "test_result": G.GOAL_TEST_RESULT, **main, "other_target": other,
    }


def summary_for_chat(gc: dict[str, Any], scenario: str | None = None) -> dict[str, Any]:
    """The compact view the Live consultant / plan review tool returns."""
    if not gc or not gc.get("available"):
        return {"available": False, "reason": (gc or {}).get("reason", "not computed yet")}

    def view(t: dict[str, Any]) -> dict[str, Any]:
        cap = t["captain"].get(scenario) if scenario else next(iter(t["captain"].values()), None)
        return {
            "by": t["goals_text"], "note": t.get("target_note"),
            "captain": None if cap is None else {
                "close_call": cap["close"], "xpert_pick": cap["xpert_pick"]["web_name"],
                "highest_expected_points": cap["xp_pick"]["web_name"], "goal_rule_pick": (cap.get("goal_pick") or {}).get("web_name"),
                "verdict": cap["verdict"], "summary": cap.get("text"), "why": cap.get("why"),
                "options": [{k: o[k] for k in ("web_name", "xp", "chances", "diff_vs_xp_pick", "diff_se")} for o in cap.get("options", [])],
            },
            "transfers": {k: t["transfers"].get(k) for k in ("close", "verdict", "xpert_pick", "goal_pick", "text", "why", "options")},
        }

    out = {"gameweek": gc["gameweek"], "goal_rule_used_for_picks": gc["auto_uses_rule"], "how_close_is_defined":
           f"an option within {gc['close_sd']:g} standard deviation of the best on expected points; "
           f"a goal difference counts only beyond {gc['noise_se']:g} simulation standard errors",
           "main": view(gc)}
    if gc.get("other_target"):
        out["other_target"] = view(gc["other_target"])
    return out


def _np_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))
