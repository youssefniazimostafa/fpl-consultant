"""Consistency check for the ONE recommendation (``decision`` in
``site/data/latest.json``, see fpl/decision.py) -- run by the pipeline
right after ``fpl.publish``, and fails loudly (non-zero exit, which fails
the Actions run before the site deploys) if the page would contradict
itself:

1. every incoming player is in the recommended XI or bench;
2. no outgoing player still appears in the XI or bench;
3. the captain and vice-captain are both in the XI, and are different;
4. the captain is the same on the captain card, the pitch badge (the one
   XI card flagged ``is_captain``), the "Why this team" text, and the
   site's ``recommended_team``/``verdict`` panels;
5. hit / free-transfer numbers agree: hits = transfers beyond the free
   ones, used + left = available, and the same hits on the decision,
   ``recommended_team`` and ``verdict``; holding means no transfers;
6. the chip shown as "play this week" matches the chip plan (re-derived
   here independently from ``chips.yours.plan``);
7. the site's captain board highlights the decision's captain, over the
   decision's XI;
8. the site's captaincy-risk panel does too;
9. the site's simulation panel highlights the decision's plan (transfer or
   hold, same captain, same incoming players);
10. the app's Planner shows exactly the decision in its first week (same
   transfers, hits, free transfers and chip), and every later planned
   week's free transfers equal the optimiser's plan.

Run with ``uv run python -m fpl.check_consistency [path/to/latest.json]``.
``--self-test`` breaks each rule once on a copy of the real data and
checks it's caught (and that the unbroken data passes).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "site" / "data" / "latest.json"


def _ids(cards: list[dict[str, Any]]) -> set[int]:
    return {c.get("player_id") for c in cards if c.get("player_id") is not None}


def _expected_chip(plan: list[dict[str, Any]], gw: int) -> str | None:
    """Independent re-derivation of the chip rule: the available chip whose
    best planned week is this week (largest gain if several)."""
    best_now: tuple[float, str] | None = None
    for chip in {r["chip"] for r in plan}:
        rows = [r for r in plan if r["chip"] == chip and r.get("available")]
        if not rows:
            continue
        top = max(rows, key=lambda r: r["estimated_gain"])
        if top["gameweek"] == gw and (best_now is None or top["estimated_gain"] > best_now[0]):
            best_now = (top["estimated_gain"], chip)
    return best_now[1] if best_now else None


def check(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    d = data.get("decision") or {}
    if not d.get("available"):
        return []  # nothing recommended, nothing to contradict
    xi, bench = d["starting_xi"], d["bench"]
    squad = _ids(xi) | _ids(bench)
    transfers = d.get("transfers") or []

    # 1 + 2
    for t in transfers:
        pin, pout = t["player_in"], t["player_out"]
        if pin.get("player_id") not in squad:
            errors.append(f"rule 1: incoming {pin.get('web_name')} isn't in the recommended XI or bench")
        if pout.get("player_id") in squad:
            errors.append(f"rule 2: outgoing {pout.get('web_name')} still appears in the XI or bench")

    # 3
    cap, vice = d.get("captain") or {}, (d.get("captain") or {}).get("vice_captain_id")
    xi_ids = _ids(xi)
    if cap.get("player_id") not in xi_ids:
        errors.append(f"rule 3: captain {cap.get('web_name')} isn't in the starting XI")
    if vice not in xi_ids:
        errors.append(f"rule 3: vice-captain {cap.get('vice_captain_name')} isn't in the starting XI")
    if vice is not None and vice == cap.get("player_id"):
        errors.append("rule 3: captain and vice-captain are the same player")

    # 4
    badges = [c for c in xi if c.get("is_captain")]
    if len(badges) != 1 or badges[0].get("player_id") != cap.get("player_id"):
        errors.append(
            f"rule 4: pitch C badge ({[c.get('web_name') for c in badges]}) doesn't match captain card ({cap.get('web_name')})"
        )
    vbadges = [c for c in xi if c.get("is_vice_captain")]
    if len(vbadges) != 1 or vbadges[0].get("player_id") != vice:
        errors.append(f"rule 4: pitch V badge ({[c.get('web_name') for c in vbadges]}) doesn't match vice-captain")
    why = " ".join(d.get("why") or [])
    if cap.get("web_name") and f"Captain {cap['web_name']}" not in why and not why.count(cap["web_name"]):
        errors.append(f"rule 4: 'Why this team' doesn't mention captain {cap.get('web_name')}")
    for other in xi:
        if other.get("player_id") != cap.get("player_id") and f"Captain {other.get('web_name')}" in why:
            errors.append(f"rule 4: 'Why this team' names {other.get('web_name')} as captain")
    rt_cap = ((data.get("recommended_team") or {}).get("captain") or {}).get("player_id")
    if rt_cap is not None and rt_cap != cap.get("player_id"):
        errors.append("rule 4: recommended_team captain differs from the decision")
    v_cap = ((data.get("verdict") or {}).get("captain") or {}).get("player_id")
    if v_cap is not None and v_cap != cap.get("player_id"):
        errors.append("rule 4: verdict captain differs from the decision")

    # 5
    ft = d["free_transfers"]
    n = len(transfers)
    if ft.get("available") is not None:
        expected_hits = max(0, n - ft["available"])
        if d["hits"] != expected_hits:
            errors.append(f"rule 5: {n} transfer(s) with {ft['available']} free should be {expected_hits} hit(s), decision says {d['hits']}")
        if ft.get("used") is not None and ft["used"] + ft["left"] != ft["available"]:
            errors.append(f"rule 5: free transfers used ({ft['used']}) + left ({ft['left']}) != available ({ft['available']})")
        if ft.get("used") is not None and ft["used"] != min(n, ft["available"]):
            errors.append(f"rule 5: free transfers used ({ft['used']}) doesn't match {n} transfer(s)")
    if d.get("hit_cost") != 4 * d["hits"]:
        errors.append(f"rule 5: hit cost {d.get('hit_cost')} isn't 4 x {d['hits']} hit(s)")
    for name in ("recommended_team", "verdict"):
        other = (data.get(name) or {}).get("hits_taken")
        if other is not None and other != d["hits"]:
            errors.append(f"rule 5: {name} says {other} hit(s), decision says {d['hits']}")
    if (d["action"] == "roll") != (n == 0):
        errors.append(f"rule 5: action '{d['action']}' doesn't match {n} transfer(s)")
    verdict = data.get("verdict") or {}
    if verdict.get("available") and bool(verdict.get("transfer_recommended")) != (d["action"] == "transfer"):
        errors.append("rule 5: verdict and decision disagree on transfer vs hold")

    # 6
    plan = ((data.get("chips") or {}).get("yours") or {}).get("plan") or []
    if plan:
        expected = _expected_chip(plan, d["gameweek"])
        shown = ((d.get("chip") or {}).get("play") or {}).get("chip")
        if shown != expected:
            errors.append(f"rule 6: chip shown as 'play this week' ({shown}) differs from the plan ({expected})")

    # 7 + 8: the site's captain board and captaincy-risk panels
    for rule, name in (("rule 7", "captain_board"), ("rule 8", "captaincy_risk")):
        panel = data.get(name) or {}
        if not panel.get("available"):
            continue
        rows = panel.get("players") or []
        marked = [r for r in rows if r.get("is_optimiser_captain")]
        if len(marked) != 1 or marked[0].get("player_id") != cap.get("player_id"):
            errors.append(f"{rule}: {name} highlights {[r.get('web_name') for r in marked]} as captain, decision says {cap.get('web_name')}")
        if {r.get("player_id") for r in rows} != xi_ids:
            errors.append(f"{rule}: {name} players aren't the decision's starting XI")

    # 9: the site's simulation panel highlights the decision's plan
    sim = data.get("simulation") or {}
    if sim.get("available") and sim.get("plans"):
        rec = sim.get("recommended")
        flagged = [r for r in sim["plans"] if r.get("is_recommended")]
        want = "transfer plan" if d["action"] == "transfer" else "hold (roll free transfer)"
        if rec is None or len(flagged) != 1:
            errors.append(f"rule 9: simulation doesn't mark exactly one recommended plan ({len(flagged)} marked)")
        else:
            if rec.get("scenario") != want:
                errors.append(f"rule 9: simulation highlights '{rec.get('scenario')}', decision is '{d['action']}'")
            if rec.get("captain") != cap.get("web_name"):
                errors.append(f"rule 9: simulation highlights captain {rec.get('captain')}, decision says {cap.get('web_name')}")
            in_names = sorted(t["player_in"].get("web_name") for t in transfers)
            if sorted(rec.get("transfers_in") or []) != in_names:
                errors.append(f"rule 9: simulation plan brings in {rec.get('transfers_in')}, decision brings in {in_names}")

    # 10: the Planner's first week is the decision, exactly
    planner = data.get("planner") or {}
    if planner.get("available"):
        wk = next((w for w in planner.get("transfers") or [] if w.get("gameweek") == d["gameweek"]), None)
        if wk is None:
            errors.append(f"rule 10: the planner has no GW{d['gameweek']} column")
        else:
            p_moves = sorted((m["out"].get("player_id"), m["in"].get("player_id")) for m in wk["moves"])
            d_moves = sorted((t["player_out"].get("player_id"), t["player_in"].get("player_id")) for t in transfers)
            if p_moves != d_moves:
                errors.append(f"rule 10: planner GW{d['gameweek']} transfers {p_moves} differ from the decision {d_moves}")
            if wk.get("hits") != d["hits"]:
                errors.append(f"rule 10: planner GW{d['gameweek']} hits {wk.get('hits')} differ from the decision {d['hits']}")
            if wk.get("free_transfers") != ft.get("available"):
                errors.append(f"rule 10: planner free transfers {wk.get('free_transfers')} differ from the decision {ft.get('available')}")
        # every planned week's free transfers = the optimiser's plan
        plan_ft = {}
        for row in ((data.get("transfer_plan") or {}).get("weeks") or []):
            if row.get("free_transfers") is not None:
                plan_ft.setdefault(row["gameweek"], row["free_transfers"])
        for w in planner.get("transfers") or []:
            if w.get("planned") and w["gameweek"] in plan_ft and w.get("free_transfers") != plan_ft[w["gameweek"]]:
                errors.append(f"rule 10: planner GW{w['gameweek']} free transfers {w.get('free_transfers')} "
                              f"differ from the optimiser's plan {plan_ft[w['gameweek']]}")
        planned_now = [c["chip"] for c in planner.get("chips") or [] if c.get("gameweek") == d["gameweek"]]
        play = ((d.get("chip") or {}).get("play") or {}).get("chip")
        if planned_now != ([play] if play else []):
            errors.append(f"rule 10: planner puts {planned_now or 'no chip'} in GW{d['gameweek']}, the decision says {play or 'no chip'}")
    return errors


# ---------------------------------------------------------------------------
# self-test: break each rule once
# ---------------------------------------------------------------------------


def _breakers(data: dict[str, Any]) -> list[tuple[str, Any]]:
    d = data["decision"]
    xi = d["starting_xi"]
    cap_id = d["captain"]["player_id"]
    non_cap = next(c for c in xi if c["player_id"] != cap_id and not c.get("is_vice_captain"))
    tests = []

    def incoming_missing(x):
        x["decision"]["transfers"] = x["decision"]["transfers"] or [{"player_out": {"player_id": -1, "web_name": "Nobody"}, "player_in": {}}]
        x["decision"]["transfers"][0]["player_in"] = {"player_id": -99, "web_name": "Ghost"}
    tests.append(("rule 1", incoming_missing))

    def outgoing_present(x):
        x["decision"]["transfers"] = x["decision"]["transfers"] or [{"player_in": copy.deepcopy(xi[0]), "player_out": {}}]
        x["decision"]["transfers"][0]["player_out"] = copy.deepcopy(non_cap)
    tests.append(("rule 2", outgoing_present))

    def captain_not_in_xi(x):
        x["decision"]["captain"]["player_id"] = -7
    tests.append(("rule 3", captain_not_in_xi))

    def badge_mismatch(x):
        for c in x["decision"]["starting_xi"]:
            c["is_captain"] = c["player_id"] == non_cap["player_id"]
    tests.append(("rule 4", badge_mismatch))

    def why_wrong_captain(x):
        x["decision"]["why"] = [s.replace(d["captain"]["web_name"], non_cap["web_name"]) for s in x["decision"]["why"]]
    tests.append(("rule 4", why_wrong_captain))

    def hits_disagree(x):
        x["decision"]["hits"] += 1
        x["decision"]["hit_cost"] = 4 * x["decision"]["hits"]
    tests.append(("rule 5", hits_disagree))

    def chip_disagree(x):
        x["decision"]["chip"]["play"] = None if x["decision"]["chip"]["play"] else {"chip": "bboost", "label": "Bench Boost", "gain": 1.0}
    tests.append(("rule 6", chip_disagree))

    def board_wrong_captain(x):
        for r in x["captain_board"]["players"]:
            r["is_optimiser_captain"] = r["player_id"] == non_cap["player_id"]
    tests.append(("rule 7", board_wrong_captain))

    def risk_wrong_captain(x):
        for r in x["captaincy_risk"]["players"]:
            r["is_optimiser_captain"] = r["player_id"] == non_cap["player_id"]
    tests.append(("rule 8", risk_wrong_captain))

    def sim_wrong_plan(x):
        plans = x["simulation"]["plans"]
        other = next(r for r in plans if not r.get("is_recommended"))
        for r in plans:
            r["is_recommended"] = r is other
        x["simulation"]["recommended"] = other
    tests.append(("rule 9", sim_wrong_plan))

    def planner_wrong_week(x):
        wk = x["planner"]["transfers"][0]
        if wk["moves"]:
            wk["moves"] = []
        else:
            wk["moves"] = [{"out": {"player_id": -1}, "in": {"player_id": -2}}]
    tests.append(("rule 10", planner_wrong_week))

    def planner_chip_now(x):
        for c in x["planner"]["chips"]:
            if c["status"] != "used":
                c["gameweek"] = x["decision"]["gameweek"] if not x["decision"]["chip"]["play"] else None
                break
    tests.append(("rule 10", planner_chip_now))

    def planner_ft_drift(x):
        later = [w for w in x["planner"]["transfers"] if w.get("planned") and w["gameweek"] != x["decision"]["gameweek"]]
        later[0]["free_transfers"] = (later[0]["free_transfers"] or 0) + 1
    tests.append(("rule 10", planner_ft_drift))
    return tests


def self_test(data: dict[str, Any]) -> bool:
    ok = True
    base = check(data)
    print(f"  unbroken data: {'PASS' if not base else 'FAIL ' + str(base)}")
    ok &= not base
    for rule, breaker in _breakers(data):
        broken = copy.deepcopy(data)
        breaker(broken)
        errs = check(broken)
        caught = any(e.startswith(rule) for e in errs)
        print(f"  broke {rule} via {breaker.__name__}: {'caught' if caught else 'NOT CAUGHT'} -- {errs[:2]}")
        ok &= caught
    return ok


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else DEFAULT_PATH
    data = json.loads(path.read_text())
    if "--self-test" in sys.argv:
        print(f"Consistency check self-test on {path}:")
        ok = self_test(data)
        print("PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)
        return
    errors = check(data)
    if errors:
        print(f"FAIL: the recommendation in {path} contradicts itself:")
        for e in errors:
            print(f"  ! {e}")
        sys.exit(1)
    d = data.get("decision") or {}
    print(f"PASS: one consistent recommendation for GW{d.get('gameweek')} ({d.get('action', 'n/a')}, "
          f"captain {(d.get('captain') or {}).get('web_name')}, chip {((d.get('chip') or {}).get('play') or {}).get('label', 'none')})")


if __name__ == "__main__":
    main()
