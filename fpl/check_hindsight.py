"""Validates the hindsight on the Last gameweek / Team of the week pages
for every finished gameweek up to ``--to`` (default 10), replaying each as
if it were the latest:

* best possible >= your own picks scored under the same rules (no chip,
  no hit) -- weeks where you played a chip or took a hit are listed
  separately, since your real score there isn't comparable;
* team of the week >= best possible, and its squad is legal (budget,
  2/5/5/3, max 3 per club, valid formation);
* the gap parts add up exactly to best possible - your score;
* every player's breakdown reconciles with FPL's official points (the
  page's own reconciliation; it refuses to publish otherwise).

Run with ``uv run python -m fpl.check_hindsight [--to 10]``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

from fpl.check_consultant_scoring import _day_dir
from fpl.fetch import TABLES_DIR, get, load_config
from fpl.last_gameweek import build_last_gameweek
from fpl.league import resolve_team_id
from fpl.optimise import FORMATION_BOUNDS, MAX_PER_CLUB, POSITION_QUOTAS
from fpl.publish import hindsight_inputs

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    to_gw = int(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else 10
    d = _day_dir()
    boot = get("bootstrap-static/", d)
    hist = pd.read_parquet(TABLES_DIR / "league_history.parquet")
    mgr = pd.read_parquet(TABLES_DIR / "league_managers.parquet")
    picks = pd.read_parquet(TABLES_DIR / "league_picks.parquet")
    fixtures = pd.read_parquet(TABLES_DIR / "fixtures.parquet")
    team_id = resolve_team_id(load_config(), mgr)
    team_of = {int(e["id"]): int(e["team"]) for e in boot["elements"]}
    ok = True
    for gw in range(1, to_gw + 1):
        b = copy.deepcopy(boot)
        for e in b["events"]:
            e["finished"] = e["id"] <= gw
        r = build_last_gameweek(team_id=team_id, get_json=lambda p, b=b: b if p == "bootstrap-static/" else get(p, d),
                                league_history=hist, league_managers=mgr, predictions_dir=ROOT / "data" / "predictions",
                                fixtures=fixtures, hindsight=hindsight_inputs(picks, team_id))
        if not r["available"]:
            print(f"GW{gw}: page not available ({r['reason']})")
            ok = False
            continue
        t, bp = r["team_of_the_week"], r["best_possible"]
        squad = t["starting_xi"] + t["bench"]
        pos = {p: sum(c["position"] == p for c in squad) for p in POSITION_QUOTAS}
        clubs = pd.Series([team_of[c["player_id"]] for c in squad]).value_counts().max()
        xi_pos = {p: sum(c["position"] == p for c in t["starting_xi"]) for p in FORMATION_BOUNDS}
        legal = (t["cost"] <= 100.0 and pos == POSITION_QUOTAS and clubs <= MAX_PER_CLUB and len(t["starting_xi"]) == 11
                 and all(lo <= xi_pos[p] <= hi for p, (lo, hi) in FORMATION_BOUNDS.items()))
        line = f"GW{gw}: TOTW {t['total']} ({t['formation']}, £{t['cost']}m, C {t['captain_name']}, legal {legal})"
        ok &= legal and t["available"]
        if bp.get("available"):
            parts = bp["parts"]
            sums = sum(parts.values()) == bp["best"] - bp["you"]
            ge_rules = bp["best"] >= bp["your_rules_score"]
            ge_totw = t["total"] >= bp["best"]
            special = bool(bp["chip"]) or bp["hits"] > 0
            line += (f" | best {bp['best']} vs you {bp['you']} (same rules {bp['your_rules_score']}), FT {bp['free_transfers']}"
                     f" | parts {parts} add up {sums} | best>=yours {ge_rules} | TOTW>=best {ge_totw}"
                     + (f" | CHIP/HIT WEEK ({bp['chip'] or ''} hits {bp['hits']})" if special else ""))
            ok &= sums and ge_totw and (ge_rules or special)
        else:
            line += f" | best possible: {bp.get('reason')}"
        print(line)
    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
