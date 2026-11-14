"""The desktop app's "Last gameweek" page -- one ``last_gameweek`` section
in latest.json: where your points came from in the latest finished
gameweek, and how each player did against what was expected.

Nothing is re-implemented here:

* your picks, captain, vice, chip, hits and official points come from the
  FPL API (``entry/{id}/event/{gw}/picks/``), read through
  ``fpl.consultant.actual_from_api`` (which undoes FPL's automatic subs to
  get the team you picked);
* the team "as played" (auto-subs, armband) is ``fpl.consultant.score_team``
  -- the same scorer validated against FPL's official points for every
  league manager, GW1-10 (fpl/check_consultant_scoring.py);
* per-player points and the reasons come from ``event/{gw}/live/``'s own
  ``explain`` breakdown (FPL's scoring categories, not ours);
* "expected" is the frozen pre-deadline prediction
  (``data/predictions/gw{N}.parquet``), the same one fpl.scoring measures.

The section is only published when every number reconciles: the page
total must equal your official points, and each player's breakdown must
add up to his points -- otherwise it says so instead of showing wrong
numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from fpl.consultant import ELEMENT_TYPE_POSITION, actual_from_api, live_points, positions_from_bootstrap, score_team

def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# FPL's explain identifiers -> plain words (the points always come from FPL)
ITEM_LABELS = {
    "minutes": lambda v: f"{v} min",
    "goals_scored": lambda v: _plural(v, "goal", "goals"),
    "assists": lambda v: _plural(v, "assist", "assists"),
    "clean_sheets": lambda v: "clean sheet",
    "goals_conceded": lambda v: _plural(v, "goal conceded", "goals conceded"),
    "saves": lambda v: _plural(v, "save", "saves"),
    "bonus": lambda v: f"{v} bonus",
    "defensive_contribution": lambda v: _plural(v, "defensive contribution", "defensive contributions"),
    "yellow_cards": lambda v: "yellow card" if v == 1 else f"{v} yellow cards",
    "red_cards": lambda v: "red card",
    "own_goals": lambda v: "own goal" if v == 1 else f"{v} own goals",
    "penalties_saved": lambda v: "penalty saved" if v == 1 else f"{v} penalties saved",
    "penalties_missed": lambda v: "penalty missed" if v == 1 else f"{v} penalties missed",
}


# Table columns on the Last gameweek page, and the short count shown
# under each points bubble. Anything not listed goes in "Other".
ITEM_COLUMNS = {
    "minutes": "minutes", "goals_scored": "goals", "assists": "assists", "clean_sheets": "clean_sheet",
    "bonus": "bonus", "defensive_contribution": "defensive", "saves": "saves",
}
ITEM_SHORT = {
    "minutes": lambda v: f"{v} min",
    "goals_scored": lambda v: _plural(v, "goal", "goals"),
    "assists": lambda v: _plural(v, "assist", "assists"),
    "clean_sheets": lambda v: "clean sheet",
    "bonus": lambda v: f"{v} bonus",
    "defensive_contribution": lambda v: _plural(v, "action", "actions"),
    "saves": lambda v: _plural(v, "save", "saves"),
    "goals_conceded": lambda v: f"{v} conceded",
    "yellow_cards": lambda v: "yellow card",
    "red_cards": lambda v: "red card",
    "own_goals": lambda v: "own goal",
    "penalties_missed": lambda v: "pen. missed",
    "penalties_saved": lambda v: "pen. saved",
}


def _match_items(fixture_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One match's scoring items in FPL's order, repeated items combined,
    zero-point items dropped (they don't change the total)."""
    combined: dict[str, dict[str, int]] = {}
    for st in fixture_stats:
        c = combined.setdefault(st["identifier"], {"value": 0, "points": 0})
        c["value"] += int(st["value"])
        c["points"] += int(st["points"])
    items = []
    for ident, c in combined.items():
        if c["points"] == 0:
            continue
        label = ITEM_LABELS.get(ident, lambda v, i=ident: f"{v} {i.replace('_', ' ')}")(c["value"])
        short = ITEM_SHORT.get(ident, lambda v, i=ident: i.replace("_", " "))(c["value"])
        items.append({"label": label, "short": short, "column": ITEM_COLUMNS.get(ident, "other"),
                      "identifier": ident, "value": c["value"], "points": c["points"]})
    return items


CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


def _reason(minutes: int, started: bool, stats: dict[str, dict[str, int]]) -> str:
    """Plain words from FPL's own breakdown, e.g. "Played 90 min, 1 assist,
    clean sheet, 2 bonus" or "Came on for 20 min, no returns"."""
    if minutes == 0:
        return "Didn't play"
    parts = [f"Played {minutes} min" if started else f"Came on for {minutes} min"]
    def v(k: str) -> int:
        return stats.get(k, {}).get("value", 0)
    if v("goals_scored"):
        parts.append(f"{v('goals_scored')} goal{'s' if v('goals_scored') > 1 else ''}")
    if v("assists"):
        parts.append(f"{v('assists')} assist{'s' if v('assists') > 1 else ''}")
    if stats.get("clean_sheets", {}).get("points", 0) > 0:
        parts.append("clean sheet")
    if v("saves") and stats.get("saves", {}).get("points", 0):
        parts.append(f"{v('saves')} saves")
    if v("defensive_contribution") and stats.get("defensive_contribution", {}).get("points", 0):
        parts.append("defensive bonus")
    if v("bonus"):
        parts.append(f"{v('bonus')} bonus")
    if stats.get("goals_conceded", {}).get("points", 0) < 0:
        parts.append(f"conceded {v('goals_conceded')}")
    if v("yellow_cards"):
        parts.append("yellow card")
    if v("red_cards"):
        parts.append("red card")
    if v("own_goals"):
        parts.append("own goal")
    if v("penalties_missed"):
        parts.append("missed a penalty")
    if v("penalties_saved"):
        parts.append("saved a penalty")
    if len(parts) == 1:
        parts.append("no returns")
    return ", ".join(parts)


def build_last_gameweek(
    *,
    team_id: int | None,
    get_json: Callable[[str], Any],
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    predictions_dir: Path,
    fixtures: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if team_id is None:
        return {"available": False, "reason": "No team ID set, so there's no gameweek to show."}
    bootstrap = get_json("bootstrap-static/")
    finished = [e for e in (bootstrap or {}).get("events", []) if e.get("finished")]
    if not finished:
        return {"available": False, "reason": "No gameweek has finished yet."}
    event = max(finished, key=lambda e: e["id"])
    gw = int(event["id"])
    picks = get_json(f"entry/{team_id}/event/{gw}/picks/")
    live_json = get_json(f"event/{gw}/live/")
    if not picks or not live_json:
        return {"available": False, "reason": f"Your GW{gw} team or the GW{gw} results aren't available from FPL yet."}

    mine = actual_from_api(picks, [], gw)  # the team you picked (FPL's auto-subs undone)
    position = positions_from_bootstrap(bootstrap)
    live = live_points(live_json)
    s = score_team(mine["xi"], mine["bench"], mine["captain"], mine["vice"], mine["chip"], mine["hits"], live, position)

    info = {int(e["id"]): e for e in bootstrap["elements"]}
    teams = {int(t["id"]): t for t in bootstrap["teams"]}
    breakdown: dict[int, dict[str, dict[str, int]]] = {}
    starts = {int(el["id"]): int(el["stats"].get("starts") or 0) > 0 for el in live_json["elements"]}
    fixture_teams: dict[int, tuple[int, int]] = {}
    if fixtures is not None and not fixtures.empty:
        fixture_teams = {int(r.id): (int(r.team_h), int(r.team_a)) for r in fixtures.itertuples()}
    team_of_player = {int(e["id"]): int(e["team"]) for e in bootstrap["elements"]}

    def _opponent(pid: int, fixture_id: int | None) -> str | None:
        ha = fixture_teams.get(int(fixture_id)) if fixture_id is not None else None
        if not ha:
            return None
        mine_t = team_of_player.get(pid)
        opp = ha[1] if ha[0] == mine_t else ha[0]
        home = ha[0] == mine_t
        return f"{teams.get(opp, {}).get('short_name', '?')} ({'H' if home else 'A'})"

    matches_by_player: dict[int, list[dict[str, Any]]] = {}
    for el in live_json["elements"]:
        agg: dict[str, dict[str, int]] = {}
        matches = []
        for fx in el.get("explain", []):
            for st in fx["stats"]:
                a = agg.setdefault(st["identifier"], {"points": 0, "value": 0})
                a["points"] += st["points"]
                a["value"] += st["value"]
            items = _match_items(fx["stats"])
            matches.append({"items": items, "total": sum(i["points"] for i in items),
                            "opponent": _opponent(int(el["id"]), fx.get("fixture"))})
        breakdown[int(el["id"])] = agg
        matches_by_player[int(el["id"])] = [m for m in matches if m["items"]]

    expected: dict[int, float] = {}
    pred_path = predictions_dir / f"gw{gw}.parquet"
    if pred_path.exists():
        expected = pd.read_parquet(pred_path).groupby("player_id")["xp"].sum().to_dict()

    multiplier = 3 if mine["chip"] == "3xc" else 2
    subs_in = {i for _, i in s["autosubs"]}
    subs_out = {o for o, _ in s["autosubs"]}

    def card(pid: int, on_pitch: bool) -> dict[str, Any]:
        e = info[pid]
        pts = live.get(pid, {}).get("points", 0)
        mins = live.get(pid, {}).get("minutes", 0)
        mult = multiplier if (on_pitch and pid == s["armband"]) else 1
        stats = breakdown.get(pid, {})
        exp = expected.get(pid)
        return {
            "player_id": pid,
            "web_name": e["web_name"],
            "position": ELEMENT_TYPE_POSITION[e["element_type"]],
            "team_code": e.get("team_code"),
            "team_short": teams.get(e["team"], {}).get("short_name"),
            "points": pts,
            "counted_points": pts * mult if on_pitch else (pts if mine["chip"] == "bboost" else 0),
            "multiplier": mult,
            "minutes": mins,
            "expected": None if exp is None else round(float(exp), 1),
            "surprise": None if exp is None else round(pts - float(exp), 1),
            "reason": _reason(mins, starts.get(pid, True), stats),
            "breakdown_points": sum(v["points"] for v in stats.values()),
            # FPL's own items per match ("85 min +2 · 1 goal +4 ... = 11")
            "matches": matches_by_player.get(pid, []),
            "is_captain": pid == mine["captain"],
            "is_vice_captain": pid == mine["vice"],
            "had_armband": pid == s["armband"],
            "subbed_on": pid in subs_in,
            "subbed_off": pid in subs_out,
        }

    xi = [card(p, True) for p in s["scoring_ids"]]
    bench_ids = [p for p in mine["bench"] if p not in subs_in] + [p for p in mine["xi"] if p in subs_out]
    bench = [card(p, mine["chip"] == "bboost") for p in bench_ids]
    # the list: everyone who played or was due to play
    listed = sorted(
        xi + [c for c in bench if c["subbed_off"]],
        key=lambda c: -(c["surprise"] if c["surprise"] is not None else float("-inf")),
    )

    # --- reconcile before publishing anything ---
    problems = []
    if s["points"] != mine["net_points"]:
        problems.append(f"page total {s['points']} != your official GW{gw} points {mine['net_points']}")
    for c in xi + bench:
        if c["breakdown_points"] != c["points"]:
            problems.append(f"{c['web_name']}: breakdown adds to {c['breakdown_points']}, not {c['points']}")
        for m in c["matches"]:
            if sum(i["points"] for i in m["items"]) != m["total"]:
                problems.append(f"{c['web_name']}: a match's items don't add up to {m['total']}")
        if sum(m["total"] for m in c["matches"]) != c["points"]:
            problems.append(f"{c['web_name']}: shown items add to {sum(m['total'] for m in c['matches'])}, not {c['points']}")
        if c["counted_points"] != c["points"] * c["multiplier"] and c in xi:
            problems.append(f"{c['web_name']}: captain step doesn't match")
    if problems:
        return {"available": False, "gameweek": gw, "reason": "Last gameweek's numbers didn't add up, so they aren't shown: " + "; ".join(problems)}

    # --- headline strip ---
    league_rank = None
    league_size = None
    if not league_history.empty:
        wk = league_history[league_history["gameweek"] == gw].copy()
        if not wk.empty:
            wk["net"] = wk["points"] - wk["event_transfers_cost"].fillna(0)
            wk = wk.sort_values("net", ascending=False).reset_index(drop=True)
            me = wk.index[wk["manager_id"] == team_id]
            league_size = len(wk)
            if len(me):
                mine_net = int(wk.loc[me[0], "net"])
                league_rank = int((wk["net"] > mine_net).sum()) + 1  # ties share a rank
    cap = next((c for c in xi if c["had_armband"]), None)

    # --- plain summary ---
    by_pos: dict[str, int] = {}
    for c in xi:
        by_pos[c["position"]] = by_pos.get(c["position"], 0) + c["counted_points"]
    pos_word = {"GKP": "your goalkeeper", "DEF": "your defenders", "MID": "your midfielders", "FWD": "your forwards"}
    top_pos = max(by_pos, key=by_pos.get)
    with_exp = [c for c in listed if c["surprise"] is not None]
    summary = [
        f"Most of your {s['gross_points']} points came from {pos_word[top_pos]} ({by_pos[top_pos]})"
        + (f", and the armband on {cap['web_name']} was worth {cap['counted_points']}." if cap else "."),
    ]
    if with_exp:
        best, worst = with_exp[0], with_exp[-1]
        if best["surprise"] > 0:
            summary.append(f"Biggest win: {best['web_name']} scored {best['points']} against an expected {best['expected']:.0f}.")
        if worst["surprise"] < 0:
            summary.append(f"Biggest letdown: {worst['web_name']} scored {worst['points']} against an expected {worst['expected']:.0f}.")
    if mine["hits"]:
        summary.append(f"Your points hit cost {4 * mine['hits']}.")

    return {
        "available": True,
        "gameweek": gw,
        "total": s["points"],
        "gross": s["gross_points"],
        "hit_cost": 4 * mine["hits"],
        "fpl_average": event.get("average_entry_score"),
        "fpl_highest": event.get("highest_score"),
        "league_rank": league_rank,
        "league_size": league_size,
        "captain_points": cap["counted_points"] if cap else 0,
        "captain_name": cap["web_name"] if cap else None,
        "captain_stood_in": bool(cap and not cap["is_captain"]),
        "chip": CHIP_LABELS.get(mine["chip"], mine["chip"]) if mine["chip"] else None,
        "autosubs": [{"off": info[o]["web_name"], "on": info[i]["web_name"]} for o, i in s["autosubs"]],
        "has_expected": bool(expected),
        "starting_xi": xi,
        "bench": bench,
        "players": listed,
        "summary": summary,
    }
