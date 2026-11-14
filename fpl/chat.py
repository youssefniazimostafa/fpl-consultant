"""The app's "Live consultant" chat: one question in, one grounded answer
out. Run by the desktop app's backend (never the web page) as
``uv run python -m fpl.chat`` in the engine copy, with the Anthropic key in
the ANTHROPIC_API_KEY environment variable (read from the macOS Keychain by
the backend; never written anywhere). Reads ``{"messages": [...],
"model": "..."}`` (the conversation so far, text only) on stdin and prints
one JSON line ``{"ok": ..., "answer": ..., "cost_usd": ..., "tools": [...]}``
prefixed with ``CHAT_RESULT `` on stdout.

Grounding: the system prompt carries the app's one recommendation
(``decision``), its "Why this team" text and the strategy mode; anything
else comes from the tools in fpl/chat_tools.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

import anthropic

from fpl.chat_tools import LATEST, MODE_MEANING, TOOL_SCHEMAS, TOOLS, ToolError

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 6
# $ per million tokens: input, output, cache write, cache read
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}  # server-side refusal fallbacks

RULES = """You are the Live consultant inside the FPL Consultant app. You explain the app's own recommendation and data to an FPL manager who knows the game but isn't technical.

Rules:
- Answer only from the data below and from tool results. Quote the actual numbers (projected points, prices, chances, ownership).
- If the data and tools don't cover something, say "I don't have data on that" instead of guessing.
- Never invent injuries, news, stats, line-ups or results. FPL news text comes only from get_player.
- Explain the app's reasoning when it matters, e.g. in balanced mode the plan mostly ignores rival ownership; a captain tie between near-equal options is broken by rival ownership.
- Use what_if for "what if I do / don't do X" questions and compare_players for "X or Y" questions.
- Keep answers short (a few sentences or a short list), in plain football language: "projected points", not xP or model jargon.
- No betting or gambling advice."""


def _summary() -> str:
    d = json.loads(LATEST.read_text())
    dec = d.get("decision") or {}
    mode = (d.get("strategy") or {}).get("mode", "balanced")
    cap = dec.get("captain") or {}
    keep = {
        "gameweek": dec.get("gameweek"),
        "action": dec.get("action"),
        "transfers": [{"out": t["player_out"].get("web_name"), "in": t["player_in"].get("web_name")} for t in dec.get("transfers") or []],
        "points_hit": dec.get("hit_cost"),
        "free_transfers": dec.get("free_transfers"),
        "captain": cap.get("web_name"), "captain_projected_points": cap.get("projected_points"),
        "vice_captain": cap.get("vice_captain_name"), "captain_reason": cap.get("why"),
        "captain_risk": (cap.get("risk") or {}).get("text"),
        "chip": (dec.get("chip") or {}).get("play"),
        "starting_xi": [{"name": p.get("web_name"), "position": p.get("position"), "projected_points": p.get("projected_points")} for p in dec.get("starting_xi") or []],
        "bench_in_order": [p.get("web_name") for p in dec.get("bench") or []],
        "headline": dec.get("headline"),
        "rule_used": dec.get("rule"),
    }
    return (f"Data generated {d.get('generated_at')}.\n\nThe app's recommendation for this gameweek (the ONE decision every page shows):\n"
            f"{json.dumps(keep, ensure_ascii=False)}\n\n\"Why this team\" text shown to the user:\n{' '.join(dec.get('why') or [])}\n\n"
            f"Strategy mode: {mode}. {MODE_MEANING.get(mode, '')}")


def _tools() -> list[dict[str, Any]]:
    tools = [dict(t, strict=True) for t in TOOL_SCHEMAS]
    for t in tools:  # strict mode: every property listed as required; captain may be null
        if t["name"] == "what_if":
            t["input_schema"] = {**t["input_schema"], "required": ["transfers_in", "transfers_out", "captain"],
                                 "properties": {**t["input_schema"]["properties"], "captain": {"type": ["string", "null"]}}}
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _cost(model: str, usage: Any) -> float:
    pin, pout, pw, pr = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return ((usage.input_tokens or 0) * pin + (usage.output_tokens or 0) * pout
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * pw
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * pr) / 1_000_000


def _run_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    if name not in TOOLS:
        return f"Unknown tool {name}", True
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # keep stdout for the result line
            return json.dumps(TOOLS[name](**args), ensure_ascii=False), False
    except ToolError as e:
        return str(e), True
    except Exception as e:  # noqa: BLE001 -- a tool failure becomes an error result, not a crash
        return f"Tool failed: {e}", True


def answer(messages: list[dict[str, str]], model: str = DEFAULT_MODEL) -> dict[str, Any]:
    client = anthropic.Anthropic(max_retries=1, timeout=120.0)  # key from ANTHROPIC_API_KEY
    system = [{"type": "text", "text": RULES}, {"type": "text", "text": _summary(), "cache_control": {"type": "ephemeral"}}]
    convo: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
    extra: dict[str, Any] = {}
    if model in FALLBACK_MODELS:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    cost, used = 0.0, []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.beta.messages.create(
            model=model, max_tokens=4000, system=system, tools=_tools(), messages=convo,
            output_config={"effort": "low"}, **extra,
        )
        cost += _cost(model, resp.usage)
        if resp.stop_reason == "refusal":
            return {"ok": True, "answer": "I can't help with that one.", "cost_usd": cost, "tools": used}
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if resp.stop_reason == "max_tokens":
                text += " (answer cut short)"
            return {"ok": True, "answer": text or "I don't have an answer for that.", "cost_usd": cost, "tools": used}
        convo.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                args = b.input if isinstance(b.input, dict) else {}
                out, is_err = _run_tool(b.name, args)
                used.append({"name": b.name, "input": args, "error": is_err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": is_err})
        convo.append({"role": "user", "content": results})
    return {"ok": True, "answer": "That needed too many look-ups; try a narrower question.", "cost_usd": cost, "tools": used}


def main() -> None:
    req = json.loads(sys.stdin.read() or "{}")
    try:
        result = answer(req.get("messages") or [], req.get("model") or DEFAULT_MODEL)
    except anthropic.AuthenticationError:
        result = {"ok": False, "error": "Your Anthropic API key was rejected. Check it in Settings."}
    except anthropic.PermissionDeniedError:
        result = {"ok": False, "error": "Your Anthropic key isn't allowed to use this model. Pick another in Settings."}
    except anthropic.NotFoundError:
        result = {"ok": False, "error": "That model isn't available to your key. Pick another in Settings."}
    except anthropic.RateLimitError:
        result = {"ok": False, "error": "Anthropic is rate-limiting your key right now. Try again in a minute."}
    except anthropic.APIConnectionError:
        result = {"ok": False, "error": "Couldn't reach Anthropic. Are you offline?"}
    except anthropic.APIStatusError as e:
        result = {"ok": False, "error": f"Anthropic returned an error ({e.status_code}). Try again shortly."}
    print("CHAT_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
