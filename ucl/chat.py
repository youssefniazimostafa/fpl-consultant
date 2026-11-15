"""The Live consultant for UCL mode. Same contract as fpl.chat: run by the
desktop backend as ``uv run python -m ucl.chat`` in the engine copy with
ANTHROPIC_API_KEY in its environment only (from the Keychain); reads
``{"messages": [...], "model": "..."}`` on stdin, prints one line
``CHAT_RESULT {...}``. Nothing is saved. Grounded in data/ucl/latest.json
and the rules in ucl.rules, through the tools below.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import anthropic
from rapidfuzz import fuzz, process

from fpl.chat import DEFAULT_MODEL, FALLBACK_MODELS, MAX_TOOL_ROUNDS, _cost
from ucl import feed, rules

LATEST = feed.DATA / "latest.json"

RULES = """You are the Live consultant inside XPert, in its Champions League mode (UEFA Champions League Fantasy 2026/27). The user is a keen fantasy player who isn't technical. Turn the app's numbers into reasons a football fan understands.

Grounding (always):
- Answer only from the data below and from tool results. Never invent injuries, news, line-ups or results.
- If the data and tools don't cover something, say "I don't have data on that".
- The game rules are in the data below (get_rules has details). Their own UCL team can't be read (UEFA needs a login), so talk about the recommended squad unless they tell you their players.
- The recommendation is the app's, not yours. Don't change it because the user argues; explain it and let them choose.
- There is no mini-league here: the aim is the most points against the game average.
- No betting or gambling advice.

How to answer: one plain opening sentence, then up to 3 short bullets, then optionally a short follow-up offer. 60-90 words, never over 100 unless asked. Say "expected points" (xP only after that), "matchday", "match day" for each of its days, "free transfer", "hit". British English. Never name your tools."""


def _d() -> dict:
    return json.loads(LATEST.read_text())


def _summary() -> str:
    d = _d()
    t = d["team"]
    keep = {
        "matchday": d["md"], "stage": d["stage"], "deadline_utc": d["deadline"], "budget": d["budget"],
        "club_limit": d["club_limit"],
        "squad": [{k: p.get(k) for k in ("name", "club", "pos", "price", "xp", "day", "opp", "role", "bench_order",
                                          "captain", "vice", "reason")} for p in t["players"]],
        "team_expected_points": t["xp"], "cost": t["cost"], "bank": t["bank"], "formation": t["formation"],
        "captain_options": t["captain_options"][:3], "between_days_flexibility": t["flexibility"],
        "plan": [{k: s[k] for k in ("md", "xp", "ft", "hits", "chip", "captain")} | {
            "in": [p["name"] for p in s["in"]], "out": [p["name"] for p in s["out"]]} for s in d["plan"]],
        "chips": d["chips"]["summary"], "between_days": d["between_days"].get("note"),
        "notes": d["notes"],
    }
    rules_txt = (f"Rules: squad 2 GK/5 DEF/5 MID/3 FWD, €{rules.BUDGET_LEAGUE:.0f}m (€{rules.BUDGET_KNOCKOUT:.0f}m from "
                 "the knockout play-offs), club limit 3 in the league phase, then 4/4/5/6/8. League phase: 2 free "
                 "transfers per matchday, 1 can carry over; unlimited before MD1, the play-offs (MD9) and the round of "
                 "16 (MD11); knockouts 2,3,5,3,5,3,5, no carry-over; -4 per extra transfer. Chips Wildcard and "
                 "Limitless once each, not on MD1, 9 or 11. Captain doubled. Between the days of a matchday: up to 4 "
                 "substitutions and a captain switch; any manual change turns off automatic substitutions; not on MD8 "
                 "or the final.")
    return f"Data generated {d['generated_at']}.\n{rules_txt}\n\nThe app's recommendation:\n{json.dumps(keep, ensure_ascii=False)}"


def _players() -> list[dict]:
    d = _d()
    seen, out = set(), []
    for lst in [d["team"]["players"], *d["scout"].values()]:
        for p in lst:
            if p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
    return out


def get_player(name: str) -> dict:
    ps = _players()
    hit = process.extractOne(name, [p["name"] for p in ps], scorer=fuzz.WRatio)
    if not hit or hit[1] < 60:
        return {"error": f"No player like {name!r} in the app's lists (squad and scout lists)."}
    p = ps[hit[2]]
    return {k: p.get(k) for k in ("name", "club", "pos", "price", "sel", "xp", "opp", "home", "day", "p_play", "p60",
                                  "status", "news", "reason")}


def compare_players(a: str, b: str) -> dict:
    return {"a": get_player(a), "b": get_player(b)}


def scout_list(kind: str) -> list[dict]:
    d = _d()["scout"]
    if kind not in d:
        return [{"error": f"lists: {', '.join(d)}"}]
    return [{k: p.get(k) for k in ("name", "club", "pos", "price", "xp", "sel", "value", "day")} for p in d[kind]]


def knockout_chances(club: str) -> dict:
    rows = _d()["knockout"]
    hit = process.extractOne(club, [r["team"] for r in rows], scorer=fuzz.WRatio)
    return rows[hit[2]] if hit and hit[1] >= 60 else {"error": "club not found"}


def between_days() -> dict:
    return _d()["between_days"]


TOOLS = {"get_player": get_player, "compare_players": compare_players, "scout_list": scout_list,
         "knockout_chances": knockout_chances, "between_days": between_days}
SCHEMAS = [
    {"name": "get_player", "description": "One player's expected points, fixture, match day, price, ownership and news.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}},
    {"name": "compare_players", "description": "Two players side by side.",
     "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                      "required": ["a", "b"], "additionalProperties": False}},
    {"name": "scout_list", "description": "A top-10 list: top_xp, captains, value, form, differentials, most_selected, news.",
     "input_schema": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["top_xp", "captains", "value", "form",
                      "differentials", "most_selected", "news"]}}, "required": ["kind"], "additionalProperties": False}},
    {"name": "knockout_chances", "description": "A club's league-phase table position and chances: top 8, play-offs, out, later rounds, expected matches left.",
     "input_schema": {"type": "object", "properties": {"club": {"type": "string"}}, "required": ["club"], "additionalProperties": False}},
    {"name": "between_days", "description": "The current between-days advice (substitutions, captain switch, expected gain).",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]


def answer(messages: list[dict[str, str]], model: str = DEFAULT_MODEL, extra_rules: str | None = None) -> dict[str, Any]:
    """``extra_rules``: task instructions added after the grounding rules and
    the data (ucl.my_plan's plan review)."""
    client = anthropic.Anthropic(max_retries=1, timeout=120.0)
    system = [{"type": "text", "text": RULES}, {"type": "text", "text": _summary(), "cache_control": {"type": "ephemeral"}}]
    if extra_rules:
        system.append({"type": "text", "text": extra_rules})
    tools = [dict(t, strict=True) for t in SCHEMAS]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    convo: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
    extra: dict[str, Any] = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
    cost, used = 0.0, []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.beta.messages.create(model=model, max_tokens=4000, system=system, tools=tools, messages=convo,
                                           output_config={"effort": "low"}, **extra)
        cost += _cost(model, resp.usage)
        if resp.stop_reason == "refusal":
            return {"ok": True, "answer": "I can't help with that one.", "cost_usd": cost, "tools": used}
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return {"ok": True, "answer": text or "I don't have an answer for that.", "cost_usd": cost, "tools": used}
        convo.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                args = b.input if isinstance(b.input, dict) else {}
                try:
                    out, err = json.dumps(TOOLS[b.name](**args), ensure_ascii=False), False
                except Exception as e:  # noqa: BLE001
                    out, err = f"Tool failed: {e}", True
                used.append({"name": b.name, "input": args, "error": err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        convo.append({"role": "user", "content": results})
    return {"ok": True, "answer": "That needed too many look-ups; try a narrower question.", "cost_usd": cost, "tools": used}


def main() -> None:
    req = json.loads(sys.stdin.read() or "{}")
    try:
        result = answer(req.get("messages") or [], req.get("model") or DEFAULT_MODEL)
    except anthropic.AuthenticationError:
        result = {"ok": False, "error": "Your Anthropic API key was rejected. Check it in Settings."}
    except anthropic.APIConnectionError:
        result = {"ok": False, "error": "Couldn't reach Anthropic. Are you offline?"}
    except anthropic.APIStatusError as e:
        result = {"ok": False, "error": f"Anthropic returned an error ({e.status_code}). Try again shortly."}
    except FileNotFoundError:
        result = {"ok": False, "error": "No UCL data yet. Press Refresh in UCL mode first."}
    print("CHAT_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
