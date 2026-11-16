"""The UCL Planner's "Your plan" card: your own transfers, captain,
vice-captain and chip for the next matchday, checked against the UCL rules
(ucl.rules) and reviewed against XPert's plan.

Same contract as fpl.my_plan, run by the desktop backend in the engine copy:

* ``uv run python -m ucl.my_plan context`` -> ``PLAN_RESULT {...}``: the squad
  the plan starts from, bank, free transfers, every player with expected
  points, the chips you can play this matchday, and XPert's plan.
* ``uv run python -m ucl.my_plan review`` reads ``{"plan": ..., "model": ...,
  "with_ai": ...}`` on stdin -> ``PLAN_RESULT {...}``: the checks, the points
  comparison, the risks of each plan and, with ANTHROPIC_API_KEY in its
  environment (from the Keychain), the Live consultant's written review.

Your own UCL team can't be read (UEFA needs a login), so the plan starts from
XPert's squad for this matchday (the one it recommends you create). Nothing
here writes a file: the app saves the plan in its own data folder
(ucl_plan.json, next to app.log), never in the repo or latest.json, and the
pipeline never reads it.

Plan shape: {"md": 5, "transfers": [{"out": id, "in": id}], "captain": id,
"vice_captain": id, "chip": null | "wildcard" | "limitless"}.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
from typing import Any

import pandas as pd

from ucl import feed, model, rules

CHIP_LABELS = {"wildcard": "Wildcard", "limitless": "Limitless"}
MAX_REVIEW_WORDS = 150  # aim "about 130"; longer gets one rewrite (the same one-retry check as fpl.my_plan)

REVIEW_RULES = """THIS REQUEST IS A PLAN REVIEW. It overrides the "How to answer" shape above (the grounding rules still apply in full).

The user has saved their own plan for the next Champions League matchday. The message gives the plan, its rule checks, the comparison with XPert's plan (expected points this matchday and up to the next unlimited-transfer window, both squads kept as they are) and the risk facts for each plan. Use only those numbers; you may look a player up if you need a fact that isn't there.

Write the review in plain football language, in exactly this order, as five short paragraphs of one or two sentences each (no headings, no bullet lists, bold only a few key names or numbers). Hard limit: 130 words in total:
1. One line summing up the plan.
2. Its strengths and weaknesses.
3. The points comparison with XPert's plan, this matchday and over the horizon. Say plainly when a gap is tiny ("about half a point, basically level").
4. The risks, as two short paragraphs that start exactly "Risks for your plan:" and "Risks for XPert's plan:". Each names the two or three biggest real risks of that plan from its "risk_facts", every one with its number: points given up over the horizon, the chance a starter doesn't play (and the expected number of starters who won't), players or clubs the model has seen in only a few UCL matches ("based on 4 matches"), the captain's chance of playing and whether he can be switched, and how much the between-days flexibility is worth. Never present something that isn't a risk as one (not "all starters should play"); if a plan has little risk of a kind, leave that kind out.
5. One closing line that leaves the decision to the user, without telling them what to do.
Aim for about 130 words in total.
If the plan breaks a rule say so first and keep the rest short. Never argue for a different recommendation than XPert's."""


def _latest() -> dict[str, Any]:
    return json.loads((feed.DATA / "latest.json").read_text())


def _xp() -> pd.DataFrame:
    f = feed.RAW / "xp_latest.parquet"
    if f.exists():
        return pd.read_parquet(f)
    xp, _ = model.predict(feed.TOUR)  # no local UCL refresh yet: recompute from the stored history
    return xp


def context() -> dict[str, Any]:
    d = _latest()
    md = int(d["md"])
    _, players, _ = model.load(feed.TOUR)
    xp = _xp()
    mds = [st["md"] for st in d["plan"]]
    xpm = xp[xp["md"].isin(mds)].groupby(["pid", "md"])["xp"].sum().unstack(fill_value=0.0)
    day = xp[xp["md"] == md].groupby("pid")["gd"].min()
    p_app = xp[xp["md"] == md].groupby("pid")["p_app"].max()
    t = d["team"]
    squad = [p["id"] for p in t["players"]]
    new_team = bool(t.get("transfers", {}).get("new_team"))
    out_players = []
    for p in players[players["active"] == 1].itertuples():
        pid = int(p.pid)
        out_players.append({
            "id": pid, "name": p.name, "club": p.team_code, "club_id": int(p.team_id), "pos": rules.POS_NAME[int(p.pos)],
            "price": float(p.price), "status": p.status or None,
            "xp_next": round(float(xpm.loc[pid, md]) if pid in xpm.index and md in xpm.columns else 0.0, 2),
            "xp_h": round(float(xpm.loc[pid].sum()) if pid in xpm.index else 0.0, 2),
            "gd": int(day.get(pid, 0)), "p_play": round(float(p_app.get(pid, 0.0)), 2),
        })
    vice = next((p["id"] for p in t["players"] if p.get("vice")), None)
    return {
        "ok": True, "md": md, "stage": d["stage"], "data_stamp": d["generated_at"],
        "squad": squad, "bank": t["bank"], "budget": d["budget"], "club_limit": d["club_limit"],
        "free_transfers": None if new_team else d["plan"][0].get("ft"),
        "new_team": new_team, "horizon": mds,
        "days": [{"gd": x["gd"], "label": x["label"]} for x in d["matchday"]["days"]],
        "chips": [{"chip": c, "label": CHIP_LABELS[c], "available": rules.chip_allowed(md, c)} for c in rules.CHIPS],
        "chip_banned": sorted(rules.CHIP_BANNED_MDS),
        "rules": {"shape": {rules.POS_NAME[k]: v for k, v in rules.SQUAD_SHAPE.items()}, "hit_cost": rules.HIT_COST},
        "recommendation": {"transfers": [], "captain": t["captain"], "vice_captain": vice, "chip": None},
        "players": out_players,
    }


def _best_xi(ids: list[int], val: dict[int, float], pos: dict[int, int]) -> list[int]:
    """The XI with the most expected points that's a valid formation."""
    best, best_v = [], -1.0
    by = {q: sorted([p for p in ids if pos[p] == q], key=lambda p: -val.get(p, 0)) for q in rules.POS_NAME}
    for d_, m_, f_ in itertools.product(range(3, 6), range(3, 6), range(1, 4)):
        if 1 + d_ + m_ + f_ != rules.XI_SIZE:
            continue
        xi = by[rules.GK][:1] + by[rules.DEF][:d_] + by[rules.MID][:m_] + by[rules.FWD][:f_]
        v = sum(val.get(p, 0) for p in xi)
        if len(xi) == rules.XI_SIZE and v > best_v:
            best, best_v = xi, v
    return best


def validate(plan: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    info = {p["id"]: p for p in c["players"]}
    moves = [(m["out"], m["in"]) for m in plan.get("transfers") or [] if m.get("out") and m.get("in")]
    squad = [p for p in c["squad"] if p not in {o for o, _ in moves}] + [i for _, i in moves]
    chip = plan.get("chip")
    errors = []
    if any(i not in info for _, i in moves) or any(o not in c["squad"] for o, _ in moves):
        errors.append("A player in the plan isn't available.")
        return {"ok": False, "errors": errors, "squad": squad}
    if len(set(squad)) != len(squad):
        errors.append("The same player appears twice.")
    if any(info[o]["pos"] != info[i]["pos"] for o, i in moves):
        errors.append("Each player out must be replaced in the same position.")
    md = c["md"]
    probs = rules.valid_squad([{"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[info[p]["pos"]] for p in squad if p in info],
                              [info[p]["club_id"] for p in squad if p in info],
                              [info[p]["price"] for p in squad if p in info], md, chip)
    errors += [f"Squad: {x}." for x in probs]
    if chip and not rules.chip_allowed(md, chip):
        errors.append(f"{CHIP_LABELS.get(chip, chip)} can't be played on MD{md}.")
    cap, vice = plan.get("captain"), plan.get("vice_captain")
    if cap not in squad or vice not in squad or cap == vice:
        errors.append("Pick a captain and a different vice-captain from your squad.")
    ft = c["free_transfers"]
    hits = 0 if (ft is None or chip in ("wildcard", "limitless")) else max(0, len(moves) - ft)
    cost = sum(info[p]["price"] for p in squad if p in info)
    return {"ok": not errors, "errors": errors, "squad": squad, "moves": moves, "hits": hits,
            "hit_cost": hits * rules.HIT_COST, "cost": round(cost, 1),
            "bank_after": round(c["budget"] - cost, 1), "free_transfers": ft}


def _score(squad: list[int], captain: int, c: dict[str, Any], hits: int) -> dict[str, Any]:
    info = {p["id"]: p for p in c["players"]}
    pos = {p: {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[info[p]["pos"]] for p in squad}
    nxt = {p: info[p]["xp_next"] for p in squad}
    xi = _best_xi(squad, nxt, pos)
    this = sum(nxt[p] for p in xi) + (nxt.get(captain, 0) if captain in xi else max(nxt[p] for p in xi)) - hits * rules.HIT_COST
    # the horizon, keeping this squad: best XI and captain each matchday, from per-matchday xP
    xp = _xp()
    xpm = xp[xp["md"].isin(c["horizon"])].groupby(["pid", "md"])["xp"].sum()
    total = 0.0
    for m in c["horizon"]:
        val = {p: float(xpm.get((p, m), 0.0)) for p in squad}
        x = _best_xi(squad, val, pos)
        total += sum(val[p] for p in x) + (val[captain] if m == c["md"] and captain in x else max(val[p] for p in x))
    total -= hits * rules.HIT_COST
    return {"this_md": round(this, 2), "horizon": round(total, 2), "xi": xi}


def _evidence() -> tuple[dict[str, int], dict[int, int]]:
    """UCL matches behind the model's view: per club (finished matches in
    2025/26 and 2026/27) and per player (appearances with minutes)."""
    clubs: dict[str, int] = {}
    apps: dict[int, int] = {}
    for tour in (feed.PREV_TOUR, feed.TOUR):
        try:
            m, _, a = model.load(tour)
        except FileNotFoundError:
            continue
        fin = m[m["finished"]]
        for code in list(fin["home_code"]) + list(fin["away_code"]):
            clubs[code] = clubs.get(code, 0) + 1
        if len(a):
            for pid, n in a[a["oF"] > 0].groupby("pid").size().items():
                apps[int(pid)] = apps.get(int(pid), 0) + int(n)
    return clubs, apps


def _risks(squad: list[int], xi: list[int], captain: int, c: dict[str, Any], chip: str | None,
           horizon_gap: float, flex: dict[str, Any], evidence: tuple[dict[str, int], dict[int, int]]) -> dict[str, Any]:
    info = {p["id"]: p for p in c["players"]}
    club_n, app_n = evidence
    may_miss = sorted(((info[p]["name"], round((1 - info[p]["p_play"]) * 100)) for p in xi if info[p]["p_play"] < 0.9),
                      key=lambda x: -x[1])
    expected_missing = round(sum(1 - info[p]["p_play"] for p in xi), 1)
    per_day = {dd["label"]: sum(1 for p in xi if info[p]["gd"] == dd["gd"]) for dd in c["days"]}
    clubs = pd.Series([info[p]["club"] for p in squad]).value_counts()
    thin = sorted({info[p]["club"] for p in xi if club_n.get(info[p]["club"], 0) < 8},
                  key=lambda k: club_n.get(k, 0))
    cap_day = next((dd["label"] for dd in c["days"] if dd["gd"] == info[captain]["gd"]), None)
    last = c["days"][-1]["label"] if c["days"] else None
    return {
        "points_given_up_over_horizon": round(max(0.0, -horizon_gap), 1),
        "starters_who_might_not_play": [f"{n}: {pct}% chance he doesn't play" for n, pct in may_miss[:4]] or "every starter is 90%+ to play",
        "expected_starters_missing": expected_missing,
        "starters_per_day": per_day,
        "little_ucl_evidence": [f"{k}: the model's view rests on {club_n.get(k, 0)} UCL matches" for k in thin] or "none",
        "captain": (f"{info[captain]['name']}: {round(info[captain]['p_play'] * 100)}% to play, plays {cap_day}, "
                    f"{app_n.get(captain, 0)} UCL appearances behind his numbers; "
                    + ("can be switched to a later-day starter if he scores little" if cap_day and cap_day != last
                       else "plays on the last day, so no captain switch to fall back on")),
        "between_days_flexibility_points": flex.get("value", 0.0),
        "clubs_with_3": {k: int(v) for k, v in clubs.items() if v >= 3} or "none",
        "chip": f"uses {CHIP_LABELS[chip]} now, so it's gone for the knockouts" if chip else "keeps both chips",
    }


def _flex(squad: list[int], xi: list[int], captain: int, c: dict[str, Any]) -> dict[str, Any]:
    from ucl import between_days

    info = {p["id"]: p for p in c["players"]}
    pos = {p: {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[info[p]["pos"]] for p in squad}
    bench = [p for p in squad if p not in xi]
    bench = [p for p in bench if pos[p] == 1] + sorted([p for p in bench if pos[p] != 1], key=lambda p: -info[p]["xp_next"])
    xp = _xp()
    xp_md = xp[xp["md"] == c["md"]].copy()
    xp_md["pos"] = xp_md["pid"].map(pos)
    return between_days.flexibility_value({"xi": xi, "bench": bench, "captain": captain}, xp_md, pos, n=800)


def review(plan: dict[str, Any], model_name: str | None, with_ai: bool) -> dict[str, Any]:
    c = context()
    check = validate(plan, c)
    out: dict[str, Any] = {"ok": True, "validation": {"ok": check["ok"], "errors": check["errors"]}, "data_stamp": c["data_stamp"],
                           "checks": {k: check.get(k) for k in ("hits", "hit_cost", "cost", "bank_after", "free_transfers")}}
    if not check["ok"]:
        return out
    info = {p["id"]: p for p in c["players"]}
    name = lambda p: info[p]["name"] if p in info else None  # noqa: E731
    yours = _score(check["squad"], plan["captain"], c, check["hits"])
    rec = c["recommendation"]
    theirs = _score(c["squad"], rec["captain"], c, 0)
    out["comparison"] = {
        "this_md": {"yours": yours["this_md"], "xpert": theirs["this_md"]},
        "horizon": {"yours": yours["horizon"], "xpert": theirs["horizon"],
                    "label": f"MD{c['horizon'][0]}-{c['horizon'][-1]}, each squad kept as it is"},
    }
    ev = _evidence()
    gap = yours["horizon"] - theirs["horizon"]
    out["risks"] = {
        "yours": _risks(check["squad"], yours["xi"], plan["captain"], c, plan.get("chip"), gap,
                        _flex(check["squad"], yours["xi"], plan["captain"], c), ev),
        "xpert": _risks(c["squad"], theirs["xi"], rec["captain"], c, None, -gap,
                        _flex(c["squad"], theirs["xi"], rec["captain"], c), ev),
    }
    if with_ai:
        from ucl.chat import DEFAULT_MODEL, answer

        facts = {
            "matchday": c["md"], "stage": c["stage"],
            "your_plan": {
                "transfers": [{"out": name(o), "in": name(i)} for o, i in check["moves"]] or "no changes to XPert's squad",
                "captain": name(plan["captain"]), "vice_captain": name(plan.get("vice_captain")),
                "chip": CHIP_LABELS.get(plan.get("chip") or "", "none"),
                "points_hit": check["hit_cost"], "money_left": f"€{check['bank_after']:.1f}m",
                "free_transfers": "unlimited (new team)" if c["free_transfers"] is None else c["free_transfers"],
            },
            "xpert_plan": {"captain": name(rec["captain"]), "vice_captain": name(rec["vice_captain"]), "chip": "none (keeps both)"},
            "comparison": out["comparison"],
            "risk_facts": out["risks"],
        }
        convo = [{"role": "user", "content": "Review my plan for the next matchday.\n\n" + json.dumps(facts, ensure_ascii=False)}]
        use = model_name or DEFAULT_MODEL
        with contextlib.redirect_stdout(io.StringIO()):
            res = answer(convo, use, extra_rules=REVIEW_RULES)
        cost = res.get("cost_usd", 0.0)
        text = res.get("answer") or ""
        missing = [h for h in ("Risks for your plan:", "Risks for XPert's plan:") if h not in text]
        n_words = len(text.split())
        retried = False
        if missing or n_words > MAX_REVIEW_WORDS:  # one corrected retry
            retried = True
            fix = ("Rewrite the review: " + ("include the paragraphs starting " + " and ".join(f'"{m}"' for m in missing) + "; " if missing else "")
                   + (f"it is {n_words} words, keep it under 130; " if n_words > MAX_REVIEW_WORDS else "")
                   + "same facts, numbers and five-part shape.")
            with contextlib.redirect_stdout(io.StringIO()):
                res = answer(convo + [{"role": "assistant", "content": text}, {"role": "user", "content": fix}], use, extra_rules=REVIEW_RULES)
            cost += res.get("cost_usd", 0.0)
            text = res.get("answer") or text
        out.update({"review": text, "cost_usd": cost, "model": use, "words": len(text.split()), "retried": retried})
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
        elif isinstance(e, FileNotFoundError):
            msg = "No UCL data yet. Press Refresh in UCL mode first."
        result = {"ok": False, "error": msg}
    print("PLAN_RESULT " + json.dumps(feed.clean(result), ensure_ascii=False, default=float), flush=True)


if __name__ == "__main__":
    main()
