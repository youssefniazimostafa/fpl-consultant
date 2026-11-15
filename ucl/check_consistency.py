"""UCL consistency check on data/ucl/latest.json.

Checks the recommended squad against the rules (ucl.rules), that the
numbers shown agree with each other, and that no secret ended up in the
file. ``--self-test`` breaks a copy in several ways and checks that each
break is caught.

Run with ``uv run python -m ucl.check_consistency [--self-test]``.
"""

from __future__ import annotations

import copy
import json
import re
import sys

from ucl import feed, rules

POS = {"GK": rules.GK, "DEF": rules.DEF, "MID": rules.MID, "FWD": rules.FWD}
SECRET = re.compile(r"(sk-ant-[A-Za-z0-9_-]{8,}|apiKey=|[0-9a-f]{32})")


def check(d: dict) -> list[str]:
    out = []
    md = d.get("md")
    t = d.get("team", {})
    ps = t.get("players", [])
    if len(ps) != rules.SQUAD_SIZE:
        out.append(f"squad has {len(ps)} players")
    probs = rules.valid_squad([POS[p["pos"]] for p in ps], [p["team_id"] for p in ps], [p["price"] for p in ps], md)
    out += [f"squad: {x}" for x in probs]
    xi = [p for p in ps if p.get("role") == "start"]
    bench = sorted([p for p in ps if p.get("role") == "bench"], key=lambda p: p.get("bench_order", 9))
    if not rules.valid_xi([POS[p["pos"]] for p in xi]):
        out.append("starting XI is not a valid formation")
    if [p.get("bench_order") for p in bench] != [1, 2, 3, 4]:
        out.append("bench order is not 1-4")
    elif bench[0]["pos"] != "GK":
        out.append("the reserve keeper should be first on the bench")
    caps = [p for p in xi if p.get("captain")]
    vices = [p for p in xi if p.get("vice")]
    if len(caps) != 1 or len(vices) != 1 or caps[0]["id"] == vices[0]["id"]:
        out.append("need exactly one captain and one different vice-captain in the XI")
    elif caps[0]["id"] != t.get("captain"):
        out.append("captain flag and team captain disagree")
    if len({p["id"] for p in ps}) != len(ps):
        out.append("a player appears twice")
    total = sum(p.get("xp", 0) for p in xi) + (caps[0].get("xp", 0) if caps else 0)
    if abs(total - t.get("xp", -1)) > 0.05:
        out.append(f"team xP {t.get('xp')} != sum of starters + captain {total:.2f}")
    if abs(sum(p["price"] for p in ps) - t.get("cost", -1)) > 0.05:
        out.append("team cost doesn't add up")
    for p in ps:
        if p.get("xp", 0) < 0 or (p.get("xp", 0) > 0 and not p.get("day")):
            out.append(f"{p['name']}: bad xP or no match day")
    plan = d.get("plan", [])
    if plan and plan[0]["md"] != md:
        out.append("plan doesn't start at the current matchday")
    for a, b in zip(plan, plan[1:]):
        if b["md"] != a["md"] + 1:
            out.append("plan matchdays are not consecutive")
    for st in plan[1:]:
        n = len(st["in"])
        if len(st["out"]) != n:
            out.append(f"MD{st['md']}: {n} in but {len(st['out'])} out")
        if st["chip"] is None and st["ft"] is not None and st["hits"] != max(0, n - st["ft"]):
            out.append(f"MD{st['md']}: hits {st['hits']} don't match {n} transfers with {st['ft']} free")
    bd = d.get("between_days", {})
    if len(bd.get("subs", [])) > rules.BETWEEN_DAYS_MAX_SUBS:
        out.append("more than 4 between-days substitutions")
    if (bd.get("subs") or bd.get("captain_switch")) and not rules.between_days_allowed(md):
        out.append("between-days changes advised on a matchday that doesn't allow them")
    for r in d.get("knockout", []):
        s = r["p_top8"] + r["p_playoff"] + r["p_out"]
        if abs(s - 1) > 0.01:
            out.append(f"{r['team']}: league-phase outcomes add to {s:.3f}")
    text = json.dumps(d)
    if SECRET.search(text):
        out.append("something that looks like a key is in latest.json")
    if "my_plan" in text or re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text):
        out.append("personal data (saved plan or an email address) in latest.json")
    return out


def self_test(d: dict) -> list[str]:
    fails = []
    breaks = {
        "two captains": lambda x: [p.update(captain=True) for p in x["team"]["players"] if p.get("role") == "start"][:0],
        "16 players": lambda x: x["team"]["players"].append(copy.deepcopy(x["team"]["players"][-1])),
        "over budget": lambda x: [p.update(price=p["price"] + 3) for p in x["team"]["players"]],
        "bad formation": lambda x: [p.update(role="start") for p in x["team"]["players"] if p.get("bench_order") == 2],
        "a key": lambda x: x.update(oops="sk-ant-abcdefghijklmnop"),
        "5 subs": lambda x: x["between_days"].update(subs=[{}] * 5),
        "wrong xP total": lambda x: x["team"].update(xp=x["team"]["xp"] + 5),
    }
    if check(d):
        fails.append("the unbroken file should pass")
    for name, f in breaks.items():
        x = copy.deepcopy(d)
        f(x)
        if not check(x):
            fails.append(f"not caught: {name}")
    return fails


def main() -> None:
    d = json.loads((feed.DATA / "latest.json").read_text())
    if "--self-test" in sys.argv:
        fails = self_test(d)
        for f in fails:
            print("FAIL:", f)
        print("PASS: UCL consistency self-test (7 breaks caught)" if not fails else "FAIL")
        raise SystemExit(1 if fails else 0)
    probs = check(d)
    for p in probs:
        print("FAIL:", p)
    print(f"PASS: UCL MD{d['md']} recommendation is consistent" if not probs else f"FAIL: {len(probs)} problem(s)")
    raise SystemExit(1 if probs else 0)


if __name__ == "__main__":
    main()
