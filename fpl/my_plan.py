"""The desktop app's "Your plan" card (Planner): your own transfers, captain,
vice-captain and chip for next gameweek, checked and reviewed against the
consultant's recommendation.

Run by the app's backend in the engine copy, never by the web page:

* ``uv run python -m fpl.my_plan context`` -- prints ``PLAN_RESULT {...}``:
  your squad with selling prices, bank, free transfers, every player (price,
  club, position, projected points), the chips you can still play, and the
  recommendation, so the card can validate a plan live.
* ``uv run python -m fpl.my_plan review`` -- reads ``{"plan": {...},
  "model": ...}`` on stdin and prints ``PLAN_RESULT {...}``: the validation,
  the points comparison (the Live consultant's what_if engine, by player
  id), the chances against your rivals (fpl.simulation's shared draws) and,
  when ANTHROPIC_API_KEY is set, the consultant's written review (fpl.chat
  with its grounding rules plus the review instructions below).

Nothing here writes a file: the app keeps the plan in its own data folder
(~/Library/Application Support/FPL Consultant/my_plan.json), outside the
repo and the engine copy, and the pipeline never reads it -- the model's
recommendation can't change because of it.

Plan shape: {"gameweek": 11, "transfers": [{"out": id, "in": id}, ...],
"captain": id, "vice_captain": id, "chip": null | "wildcard" | "freehit" |
"bboost" | "3xc"}.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

import pandas as pd

from fpl import chat_tools as T
from fpl.optimise import HIT_COST, MAX_PER_CLUB, POSITION_QUOTAS
from fpl.simulation import load_rival_plans, simulate_options

CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
NO_HIT_CHIPS = {"wildcard", "freehit"}

REVIEW_RULES = """THIS REQUEST IS A PLAN REVIEW. It overrides the "How to answer" shape above (the grounding rules still apply in full).

The user has saved their own plan for next gameweek. The message gives the plan, its checks, the comparison with the app's recommendation (projected points this week and over 5 weeks, from the same engine as what_if) and the chances against rivals. Use only those numbers and tool results; you may check a player with get_player if you need a fact that isn't there.

Write the review in plain football language, in exactly this order, as five short paragraphs of one or two sentences each (no headings, no bullet lists, bold only a few key names or numbers). Hard limit: 150 words in total; count them and cut detail rather than go over:
1. One line summing up the plan.
2. Its strengths and weaknesses.
3. How it compares with the consultant's recommendation: expected points this week and over 5 weeks, and the chances against your rivals where given. Say plainly when a gap is tiny ("about half a point, basically level").
4. The risks of this plan, and the risks of the consultant's plan.
5. One closing line that leaves the decision to the user, without telling them what to do.
If the plan breaks a rule (budget, 3 per club, squad shape) say so first and keep the rest short. Never change or argue for a different recommendation than the app's.
Rival facts: "rival_context" gives, for each captain and incoming player, how many of your rivals own him and how many captained him last gameweek. Use exactly those counts and never mix them up (owning is not captaining). How ownership works: a player few rivals have gains you ground when he scores and costs you ground when he blanks; a player most rivals have (or captain) changes little either way, because they score or blank with you. Only name strengths, weaknesses and risks you can support from the facts given; never mention something you didn't check (no "I haven't checked the bench")."""


def _latest() -> dict[str, Any]:
    return json.loads(T.LATEST.read_text()) if T.LATEST.exists() else {}


def context() -> dict[str, Any]:
    c = T._ctx()
    players: pd.DataFrame = c["players"]
    xp = c["xp_by_gw"].set_index("player_id")
    weeks = c["weeks"]
    next_col, cols = f"gw{weeks[0]}_xp", [f"gw{g}_xp" for g in weeks]
    sell = T.estimate_selling_prices(c["my_squad"], players, c["t"]["my_transfers"])
    d = _latest()
    dec = d.get("decision") or {}
    statuses = ((dec.get("chip") or {}).get("statuses")) or []
    out_players = []
    for r in players.itertuples():
        pid = int(r.id)
        has = pid in xp.index
        out_players.append({
            "id": pid, "name": r.web_name, "position": r.position, "team": int(r.team),
            "team_short": getattr(r, "team_short_name", None),
            "team_code": None if pd.isna(getattr(r, "team_code", None)) else int(r.team_code),
            "price": int(r.now_cost),
            "xp_next": round(float(xp.loc[pid, next_col]), 2) if has and next_col in xp.columns else 0.0,
            "xp_5": round(float(xp.loc[pid, cols].sum()), 1) if has else 0.0,
            "chance": None if pd.isna(r.chance_of_playing_next_round) else int(r.chance_of_playing_next_round),
            "status": r.status,
        })
    return {
        "ok": True,
        "gameweek": weeks[0],
        "data_stamp": d.get("generated_at"),
        "squad": [{"id": int(p), "sell_price": int(sell.get(p, c["info"][p]["now_cost"]))} for p in sorted(c["my_squad"])],
        "bank": int(c["bank"]),
        "free_transfers": int(c["free"]),
        "players": out_players,
        "chips": [{"chip": s["chip"], "label": s["label"], "available": bool(s["available"])} for s in statuses],
        "rules": {"max_per_club": MAX_PER_CLUB, "quotas": POSITION_QUOTAS, "hit_cost": HIT_COST},
        "recommendation": {
            "transfers": [{"out": t["player_out"].get("player_id"), "in": t["player_in"].get("player_id")}
                          for t in dec.get("transfers") or []],
            "captain": (dec.get("captain") or {}).get("player_id"),
            "vice_captain": next((p.get("player_id") for p in dec.get("starting_xi") or [] if p.get("is_vice_captain")), None),
            "chip": ((dec.get("chip") or {}).get("play") or {}).get("chip") if isinstance((dec.get("chip") or {}).get("play"), dict)
            else (dec.get("chip") or {}).get("play"),
        },
    }


def validate(plan: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """The same checks the card runs live: squad, budget (selling prices),
    max 3 per club, squad shape (so a valid formation exists), captain and
    vice in the squad, free transfers and points hit, chip still available."""
    info = {p["id"]: p for p in ctx["players"]}
    sell = {s["id"]: s["sell_price"] for s in ctx["squad"]}
    squad = set(sell)
    errors: list[str] = []
    moves = [(int(t["out"]), int(t["in"])) for t in plan.get("transfers") or [] if t.get("out") and t.get("in")]
    outs, ins = [o for o, _ in moves], [i for _, i in moves]
    if len(set(outs)) != len(outs) or len(set(ins)) != len(ins):
        errors.append("The same player appears in two transfers.")
    for o, i in moves:
        if o not in squad:
            errors.append(f"You don't own {info.get(o, {}).get('name', o)}.")
        if i in squad:
            errors.append(f"You already own {info[i]['name']}.")
        if o in info and i in info and info[o]["position"] != info[i]["position"]:
            errors.append(f"{info[o]['name']} ({info[o]['position']}) can only be swapped for another {info[o]['position']}.")
    new_squad = (squad - set(outs)) | set(ins)
    bank_after = ctx["bank"] + sum(sell.get(o, 0) for o in outs) - sum(info[i]["price"] for i in ins if i in info)
    if bank_after < 0:
        errors.append(f"Over budget by £{-bank_after / 10:.1f}m.")
    clubs: dict[int, list[str]] = {}
    for p in new_squad:
        if p in info:
            clubs.setdefault(info[p]["team"], []).append(info[p]["name"])
    for names in clubs.values():
        if len(names) > MAX_PER_CLUB:
            errors.append(f"More than {MAX_PER_CLUB} players from one club: {', '.join(sorted(names))}.")
    for pos, n in POSITION_QUOTAS.items():
        if sum(1 for p in new_squad if info.get(p, {}).get("position") == pos) != n:
            errors.append(f"Your squad needs exactly {n} {pos}.")
    cap, vice = plan.get("captain"), plan.get("vice_captain")
    if cap not in new_squad:
        errors.append("Pick a captain from your squad.")
    if vice not in new_squad:
        errors.append("Pick a vice-captain from your squad.")
    if cap is not None and cap == vice:
        errors.append("The captain and vice-captain must be different players.")
    chip = plan.get("chip")
    if chip and not any(c["chip"] == chip and c["available"] for c in ctx["chips"]):
        errors.append(f"You can't play {CHIP_LABELS.get(chip, chip)} this half of the season.")
    hits = 0 if chip in NO_HIT_CHIPS else max(0, len(moves) - ctx["free_transfers"])
    return {"ok": not errors, "errors": errors, "bank_after": bank_after, "hits": hits, "hit_cost": HIT_COST * hits,
            "transfers": len(moves), "free_transfers": ctx["free_transfers"], "squad": sorted(new_squad)}


def _sim_entry(plan_week: dict[str, Any], chip: str | None, hits: int) -> dict[str, Any]:
    xi = sorted(plan_week["squad_ids"] if chip == "bboost" else plan_week["xi_ids"])
    return {"xi_ids": xi, "captain_id": plan_week["captain_id"], "hits": hits}


def review(plan: dict[str, Any], model: str | None, with_ai: bool) -> dict[str, Any]:
    ctx = context()
    check = validate(plan, ctx)
    info = {p["id"]: p for p in ctx["players"]}
    name = lambda p: info.get(p, {}).get("name", "?")  # noqa: E731
    out: dict[str, Any] = {"ok": True, "validation": check, "data_stamp": ctx["data_stamp"], "comparison": None,
                           "chances": None, "review": None, "cost_usd": 0.0}
    if not check["ok"]:
        return out
    moves = [(int(t["out"]), int(t["in"])) for t in plan.get("transfers") or [] if t.get("out") and t.get("in")]
    chip = plan.get("chip")
    with contextlib.redirect_stdout(io.StringIO()):
        cmp, rec, alt = T.compare_plan(frozenset(i for _, i in moves), frozenset(o for o, _ in moves), plan.get("captain"))
    c = T._ctx()
    g0 = c["weeks"][0]
    xp_next = {p["id"]: p["xp_next"] for p in ctx["players"]}
    week = alt["weeks"][0]
    # chips the what_if engine doesn't model: added this week only, and no hit under Wildcard / Free Hit
    chip_bonus = 0.0
    if chip == "bboost":
        chip_bonus = sum(xp_next.get(p, 0.0) for p in week["squad_ids"] - week["xi_ids"])
    elif chip == "3xc" and week["captain_id"] is not None:
        chip_bonus = xp_next.get(week["captain_id"], 0.0)
    if chip in NO_HIT_CHIPS:
        chip_bonus += HIT_COST * week["hits"]
    for k in ("projected_points_this_week", "projected_points_next_5_weeks"):
        cmp[k]["scenario"] = round(cmp[k]["scenario"] + chip_bonus, 1)
        cmp[k]["difference"] = round(cmp[k]["scenario"] - cmp[k]["recommendation"], 1)
    cmp["your_best_xi"] = sorted(name(p) for p in week["xi_ids"])
    cmp["your_bench"] = sorted(name(p) for p in week["squad_ids"] - week["xi_ids"])
    cmp["chip"] = CHIP_LABELS.get(chip) if chip else None
    cmp["vice_captain"] = name(plan.get("vice_captain"))
    if chip == "freehit":
        cmp["note"] += " Free Hit: the 5-week figure keeps these moves (the squad would really revert after one week)."
    out["comparison"] = cmp

    # chances against your rivals (same shared draws as fpl.simulation)
    t = c["t"]
    rivals = load_rival_plans(t["league_picks"], t.get("league_managers", pd.DataFrame()), c["team_id"], xp_next)
    lh = t["league_history"]
    lh = lh[(lh["gameweek"] == lh["gameweek"].max()) & (lh["manager_id"] != c["team_id"])].sort_values("total_points", ascending=False)
    leader = int(lh["manager_id"].iloc[0]) if not lh.empty else None
    if rivals:
        sims = simulate_options(
            [_sim_entry(week, chip, 0 if chip in NO_HIT_CHIPS else week["hits"]), _sim_entry(rec["weeks"][0], None, rec["weeks"][0]["hits"])],
            rivals, xp_next, leader,
        )
        out["chances"] = {"rivals": len(rivals), "yours": sims[0], "consultant": sims[1],
                          "note": "Triple Captain's extra points aren't in the chances." if chip == "3xc" else None}

    if with_ai:
        from fpl.chat import DEFAULT_MODEL, answer  # needs the anthropic package and key only here
        facts = {
            "gameweek": g0,
            "your_plan": {
                "transfers": [{"out": name(o), "in": name(i)} for o, i in moves] or "no transfers (save the free transfer)",
                "captain": name(plan.get("captain")), "vice_captain": name(plan.get("vice_captain")),
                "chip": CHIP_LABELS.get(chip) if chip else "none",
                "points_hit": check["hit_cost"], "bank_after": f"£{check['bank_after'] / 10:.1f}m",
                "free_transfers": check["free_transfers"],
            },
            "comparison_with_recommendation": cmp,
            "chances_against_rivals": out["chances"],
        }
        latest_picks = t["league_picks"][t["league_picks"]["gameweek"] == t["league_picks"]["gameweek"].max()]
        rival_picks = latest_picks[latest_picks["manager_id"] != c["team_id"]]
        n_rivals = int(rival_picks["manager_id"].nunique())
        rec_w0 = rec["weeks"][0]
        people = {plan.get("captain"), rec_w0["captain_id"], *(i for _, i in moves), *rec_w0["transfers_in"]}
        facts["rival_context"] = {
            name(p): {"rivals_owning": int((rival_picks["player_id"] == p).sum()),
                      "rivals_captained_last_gw": int(((rival_picks["player_id"] == p) & rival_picks["is_captain"]).sum()),
                      "of_rivals": n_rivals}
            for p in people if p is not None
        }
        msg = "Review my plan for next gameweek.\n\n" + json.dumps(facts, ensure_ascii=False)
        with contextlib.redirect_stdout(io.StringIO()):
            res = answer([{"role": "user", "content": msg}], model or DEFAULT_MODEL, extra_rules=REVIEW_RULES)
        out["review"] = res.get("answer")
        out["cost_usd"] = res.get("cost_usd", 0.0)
        out["tools"] = [u["name"] for u in res.get("tools", [])]
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "context"
    try:
        if mode == "context":
            with contextlib.redirect_stdout(io.StringIO()):
                result = context()
        else:
            req = json.loads(sys.stdin.read() or "{}")
            result = review(req.get("plan") or {}, req.get("model"), bool(req.get("with_ai")))
    except Exception as e:  # noqa: BLE001 -- the card shows the error instead of a crash
        import anthropic

        msg = str(e)
        if isinstance(e, anthropic.AuthenticationError):
            msg = "Your Anthropic API key was rejected. Check it in Settings."
        elif isinstance(e, anthropic.APIConnectionError):
            msg = "Couldn't reach Anthropic. Are you offline?"
        result = {"ok": False, "error": msg}
    print("PLAN_RESULT " + json.dumps(result, ensure_ascii=False, default=float), flush=True)


if __name__ == "__main__":
    main()
