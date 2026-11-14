"""Validates fpl.consultant's scorer against FPL's own numbers.

1. Your GW10 (no frozen recommendation exists for it): your *actual* GW10
   team, bench order, captain, vice, chip and hits -- all from the FPL API
   -- fed to the scorer as if they were the recommendation. It must
   reproduce your real GW10 net points exactly.
2. Every league manager, every finished gameweek up to ``--to`` (default
   10): the same replay, compared with FPL's official net points and FPL's
   own list of automatic substitutions. Reports the real cases where an
   auto-sub happened and where a captain played 0 minutes.
3. Synthetic cases for the rules: an auto-sub that has to skip a bench
   player to keep a valid formation, a goalkeeper sub, a captain who
   didn't play (vice takes the armband), Triple Captain passing to the
   vice, Bench Boost (no subs), and hits.

Run with ``uv run python -m fpl.check_consultant_scoring [--to 10]``.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from fpl.consultant import actual_from_api, live_points, positions_from_bootstrap, score_team
from fpl.fetch import RAW_DIR, TABLES_DIR, get, load_config
from fpl.league import resolve_team_id


def _day_dir():
    d = RAW_DIR / pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def replay(team_id: int, gw: int, live: dict, position: dict) -> dict[str, Any] | None:
    picks = get(f"entry/{team_id}/event/{gw}/picks/", _day_dir())
    if not picks:
        return None
    mine = actual_from_api(picks, [], gw)
    s = score_team(mine["xi"], mine["bench"], mine["captain"], mine["vice"], mine["chip"], mine["hits"], live, position)
    captain_missed = live.get(mine["captain"], {}).get("minutes", 0) == 0
    # Compare what decides the score: the same bench players come on, and
    # everyone taken off played 0 minutes. *Which* of several 0-minute
    # starters FPL pairs with a sub depends on the original pick order,
    # which FPL's post-gameweek picks no longer show -- and can't change
    # the score (0 minutes = 0 points; the armband passes to the vice).
    zero = lambda p: live.get(p, {}).get("minutes", 0) == 0  # noqa: E731
    subs_ok = (
        sorted(i for _, i in s["autosubs"]) == sorted(i for _, i in mine["fpl_autosubs"])
        and all(zero(o) for o, _ in s["autosubs"])
    )
    return {
        "team_id": team_id, "gameweek": gw, "official": mine["net_points"], "scorer": s["points"],
        "autosubs_ok": subs_ok,
        "autosubs": s["autosubs"], "fpl_autosubs": mine["fpl_autosubs"],
        "captain_missed": captain_missed, "chip": mine["chip"], "hits": mine["hits"],
    }


def synthetic() -> list[tuple[str, bool, str]]:
    pos = {1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "MID",
           10: "FWD", 11: "FWD", 12: "GKP", 13: "DEF", 14: "MID", 15: "FWD"}
    xi, bench = list(range(1, 12)), [12, 13, 14, 15]
    base = {p: {"points": 2, "minutes": 90} for p in pos}
    results = []

    # a 3-5-2 loses a DEF (#2, 0 min): first bench outfielder (#13 DEF) comes in
    live = {**base, 2: {"points": 0, "minutes": 0}, 13: {"points": 6, "minutes": 90}}
    s = score_team(xi, bench, 10, 11, None, 0, live, pos)
    results.append(("auto-sub: DEF out, first bench player in", s["autosubs"] == [(2, 13)] and s["points"] == 10 * 2 + 6 + 2, str(s["autosubs"]) + f" {s['points']}"))

    # formation: DEF #2 doesn't play and bench DEF #13 didn't play either; the
    # MID (#14) can't come in (only 2 DEF left), so the FWD (#15) can't either
    # -> no valid sub, 10 players score
    live = {**base, 2: {"points": 0, "minutes": 0}, 13: {"points": 0, "minutes": 0}}
    s = score_team(xi, bench, 10, 11, None, 0, live, pos)
    results.append(("auto-sub skipped to keep 3 DEF", s["autosubs"] == [] and s["points"] == 10 * 2 + 2, str(s["autosubs"]) + f" {s['points']}"))

    # a MID doesn't play: bench DEF #13 is first in order and keeps a valid formation
    live = {**base, 5: {"points": 0, "minutes": 0}, 13: {"points": 1, "minutes": 30}}
    s = score_team(xi, bench, 10, 11, None, 0, live, pos)
    results.append(("auto-sub uses bench order", s["autosubs"] == [(5, 13)], str(s["autosubs"])))

    # goalkeeper only for goalkeeper
    live = {**base, 1: {"points": 0, "minutes": 0}, 12: {"points": 3, "minutes": 90}}
    s = score_team(xi, bench, 10, 11, None, 0, live, pos)
    results.append(("GK replaced only by the bench GK", s["autosubs"] == [(1, 12)], str(s["autosubs"])))

    # captain didn't play: vice gets the armband (bench #13/#14 didn't play
    # either, so the FWD #15 is the sub)
    live = {**base, 10: {"points": 0, "minutes": 0}, 11: {"points": 8, "minutes": 90},
            13: {"points": 0, "minutes": 0}, 14: {"points": 0, "minutes": 0}, 15: {"points": 1, "minutes": 20}}
    s = score_team(xi, bench, 10, 11, None, 0, live, pos)
    expected = 9 * 2 + 8 * 2 + 1  # 9 others, vice doubled, FWD sub in
    results.append(("captain 0 min -> vice captains", s["armband"] == 11 and s["autosubs"] == [(10, 15)] and s["points"] == expected,
                    f"armband={s['armband']} subs={s['autosubs']} pts={s['points']} expected {expected}"))

    # Triple Captain passes to the vice
    s = score_team(xi, bench, 10, 11, "3xc", 0, live, pos)
    results.append(("Triple Captain passes to the vice", s["points"] == 9 * 2 + 8 * 3 + 1, str(s["points"])))

    # Bench Boost: all 15 count, no subs
    live = {**base, 2: {"points": 0, "minutes": 0}}
    s = score_team(xi, bench, 10, 11, "bboost", 0, live, pos)
    results.append(("Bench Boost: bench counts, no subs", s["autosubs"] == [] and s["points"] == 14 * 2 + 2, str(s["points"])))

    # hits
    s = score_team(xi, bench, 10, 11, None, 2, base, pos)
    results.append(("two hits = -8", s["points"] == 11 * 2 + 2 - 8, str(s["points"])))
    return results


def main() -> None:
    to_gw = int(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else 10
    bootstrap = get("bootstrap-static/", _day_dir())
    position = positions_from_bootstrap(bootstrap)
    team_id = resolve_team_id(load_config(), pd.read_parquet(TABLES_DIR / "league_managers.parquet"))
    managers = pd.read_parquet(TABLES_DIR / "league_managers.parquet")["manager_id"].astype(int).tolist()
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished") and e["id"] <= to_gw]
    ok = True

    print("1. Your GW10, scored as if it were the recommendation:")
    live10 = live_points(get("event/10/live/", _day_dir()))
    r = replay(team_id, 10, live10, position)
    match = r["official"] == r["scorer"]
    ok &= match
    print(f"   FPL says {r['official']} net points, scorer says {r['scorer']} -> {'MATCH' if match else 'MISMATCH'}"
          f" (chip {r['chip']}, hits {r['hits']}, auto-subs {r['fpl_autosubs'] or 'none'})")

    print(f"2. Every league manager, GW{min(finished)}-{max(finished)}:")
    rows = []
    for gw in finished:
        live = live_points(get(f"event/{gw}/live/", _day_dir()))
        for m in managers:
            rr = replay(m, gw, live, position)
            if rr:
                rows.append(rr)
    df = pd.DataFrame(rows)
    pts_ok = (df["official"] == df["scorer"])
    subs_ok = df["autosubs_ok"]
    print(f"   {len(df)} manager-gameweeks: points match {pts_ok.sum()}/{len(df)}, auto-subs match {subs_ok.sum()}/{len(df)}")
    for _, x in df[~pts_ok | ~subs_ok].iterrows():
        print(f"   ! MISMATCH manager {x.team_id} GW{x.gameweek}: FPL {x.official} vs scorer {x.scorer}; "
              f"subs FPL {x.fpl_autosubs} vs scorer {x.autosubs}")
    ok &= bool(pts_ok.all() and subs_ok.all())
    with_subs = df[df["fpl_autosubs"].map(len) > 0]
    missed = df[df["captain_missed"]]
    print(f"   real auto-sub cases: {len(with_subs)} (all {'matched' if (with_subs.official == with_subs.scorer).all() else 'NOT matched'})")
    for _, x in with_subs.head(3).iterrows():
        print(f"     e.g. manager {x.team_id} GW{x.gameweek}: subs {x.fpl_autosubs}, FPL {x.official} = scorer {x.scorer}")
    print(f"   real captain-played-0-minutes cases: {len(missed)} (all {'matched' if (missed.official == missed.scorer).all() else 'NOT matched'})")
    for _, x in missed.head(3).iterrows():
        print(f"     e.g. manager {x.team_id} GW{x.gameweek}: FPL {x.official} = scorer {x.scorer}")
    chips = df[df["chip"].notna()]
    print(f"   chip weeks: {len(chips)} ({', '.join(sorted(set(chips['chip'])))}), all {'matched' if (chips.official == chips.scorer).all() else 'NOT matched'}; "
          f"weeks with hits: {(df['hits'] > 0).sum()}")

    print("3. Synthetic rule tests:")
    for name, passed, detail in synthetic():
        ok &= passed
        print(f"   [{'PASS' if passed else 'FAIL'}] {name} ({detail})")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
