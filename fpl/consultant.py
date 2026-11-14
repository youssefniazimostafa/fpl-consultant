""""You vs the consultant": freeze each gameweek's recommendation, then
score it with real points once the gameweek is over, next to what you
actually scored.

1. ``freeze_recommendation`` (called by ``fpl.publish``): every run before
   a gameweek's deadline writes that gameweek's ``decision`` to
   ``data/recommendations/gw{N}.json``, overwriting it; once the deadline
   has passed the file is never touched again -- the same frozen-file
   pattern as ``data/predictions/gw{N}.parquet``.
2. ``score_team``: FPL's own scoring for a 15-man pick -- captain x2 (x3
   with Triple Captain); the vice-captain takes the armband if the captain
   played 0 minutes; automatic substitutions in bench order for starters
   who played 0 minutes, keeping a valid formation (goalkeeper only for
   goalkeeper; at least 3 DEF, 2 MID, 1 FWD); the bench only counts with
   Bench Boost (and then nobody is auto-subbed); minus 4 per hit.
3. ``score_gameweek`` (called by ``fpl.scoring``): scores the frozen
   recommendation with ``event/{gw}/live/`` and compares it with what you
   actually did, all read from the FPL API -- your picks, captain, chip,
   points and transfer cost from ``entry/{id}/event/{gw}/picks/`` and your
   transfers from ``entry/{id}/transfers/``. Nothing is entered by hand.
   One row per gameweek goes to ``data/accuracy/consultant_vs_you.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RECOMMENDATIONS_DIR = ROOT / "data" / "recommendations"
CSV_PATH = ROOT / "data" / "accuracy" / "consultant_vs_you.csv"
FIRST_TRACKED_GW = 11
HIT_COST = 4
ELEMENT_TYPE_POSITION = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FORMATION_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}

CSV_COLUMNS = [
    "gameweek", "consultant_points", "my_points", "difference",
    "followed_transfers", "followed_captain",
    "consultant_total", "my_total", "total_difference",
    "consultant_captain", "my_captain", "consultant_transfers", "my_transfers",
    "consultant_autosubs", "consultant_chip", "my_chip", "scored_at",
]


# ---------------------------------------------------------------------------
# 1. freeze
# ---------------------------------------------------------------------------


def frozen_recommendation(decision: dict[str, Any], frozen_at: str) -> dict[str, Any]:
    """The parts of the decision needed to score it later."""
    def slim(p: dict[str, Any]) -> dict[str, Any]:
        return {"player_id": p.get("player_id"), "web_name": p.get("web_name"), "position": p.get("position")}

    cap = decision.get("captain") or {}
    chip = (decision.get("chip") or {}).get("play")
    return {
        "gameweek": decision["gameweek"],
        "frozen_at": frozen_at,
        "action": decision["action"],
        "starting_xi": [slim(p) for p in decision["starting_xi"]],
        "bench": [slim(p) for p in decision["bench"]],  # in bench order
        "captain_id": cap.get("player_id"),
        "captain_name": cap.get("web_name"),
        "vice_captain_id": cap.get("vice_captain_id"),
        "vice_captain_name": cap.get("vice_captain_name"),
        "transfers_in": [slim(t["player_in"]) for t in decision.get("transfers") or []],
        "transfers_out": [slim(t["player_out"]) for t in decision.get("transfers") or []],
        "hits": decision.get("hits", 0),
        "chip": chip["chip"] if chip else None,
    }


def freeze_recommendation(decision: dict[str, Any], gameweeks: pd.DataFrame, now: pd.Timestamp | None = None) -> Path | None:
    """Write data/recommendations/gw{N}.json unless gw N's deadline has passed."""
    if not decision.get("available"):
        return None
    gw = int(decision["gameweek"])
    now = now or pd.Timestamp.now("UTC")
    if not gameweeks.empty:
        row = gameweeks[gameweeks["gameweek"] == gw]
        if not row.empty and now >= pd.Timestamp(row["deadline_time"].iloc[0]):
            return None  # frozen -- never touched after the deadline
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECOMMENDATIONS_DIR / f"gw{gw}.json"
    path.write_text(json.dumps(frozen_recommendation(decision, now.isoformat()), indent=2, ensure_ascii=False))
    return path


# ---------------------------------------------------------------------------
# 2. FPL scoring for one 15-man pick
# ---------------------------------------------------------------------------


def _valid_formation(positions: list[str]) -> bool:
    counts = {p: positions.count(p) for p in FORMATION_MIN}
    return counts["GKP"] == 1 and all(counts[p] >= n for p, n in FORMATION_MIN.items() if p != "GKP")


def score_team(
    xi: list[int],
    bench: list[int],
    captain: int | None,
    vice: int | None,
    chip: str | None,
    hits: int,
    live: dict[int, dict[str, int]],
    position: dict[int, str],
) -> dict[str, Any]:
    """``xi`` in FPL pick order (1-11), ``bench`` in bench order (12-15);
    ``live`` maps player id -> {"points", "minutes"}; ``position`` maps id
    -> GKP/DEF/MID/FWD. Returns net points (after hits) and the details."""
    def played(p: int) -> bool:
        return live.get(p, {}).get("minutes", 0) > 0

    def pts(p: int) -> int:
        return live.get(p, {}).get("points", 0)

    scoring = list(xi)
    subs: list[tuple[int, int]] = []
    if chip == "bboost":
        scoring = list(xi) + list(bench)
    else:
        remaining = list(bench)
        for starter in xi:
            if played(starter):
                continue
            for b in remaining:
                if not played(b):
                    continue
                if (position[starter] == "GKP") != (position[b] == "GKP"):
                    continue  # goalkeeper only for goalkeeper
                trial = [b if p == starter else p for p in scoring]
                if _valid_formation([position[p] for p in trial]):
                    scoring = trial
                    subs.append((starter, b))
                    remaining.remove(b)
                    break

    multiplier = 3 if chip == "3xc" else 2
    armband = None
    if captain is not None and played(captain):
        armband = captain
    elif vice is not None and played(vice):
        armband = vice

    total = sum(pts(p) for p in scoring)
    if armband is not None and armband in scoring:
        total += (multiplier - 1) * pts(armband)
    net = total - HIT_COST * hits
    return {"points": net, "gross_points": total, "autosubs": subs, "armband": armband, "scoring_ids": scoring}


# ---------------------------------------------------------------------------
# 3. score a finished gameweek: the consultant vs you
# ---------------------------------------------------------------------------


def live_points(live_json: dict[str, Any]) -> dict[int, dict[str, int]]:
    return {
        int(e["id"]): {"points": int(e["stats"]["total_points"]), "minutes": int(e["stats"]["minutes"])}
        for e in live_json["elements"]
    }


def positions_from_bootstrap(bootstrap: dict[str, Any]) -> dict[int, str]:
    return {int(e["id"]): ELEMENT_TYPE_POSITION[e["element_type"]] for e in bootstrap["elements"]}


def actual_from_api(picks_json: dict[str, Any], transfers_json: list[dict[str, Any]], gw: int) -> dict[str, Any]:
    """What you actually did in ``gw``, straight from the FPL API.

    For a finished gameweek FPL returns the picks *after* its automatic
    substitutions (the subbed-in player sits in the XI). Undo them, so this
    is the team you actually picked before the deadline -- the scorer then
    re-applies the substitutions itself."""
    by_element = {p["element"]: dict(p) for p in picks_json["picks"]}
    for s in reversed(picks_json.get("automatic_subs", [])):
        a, b = by_element.get(s["element_in"]), by_element.get(s["element_out"])
        if a and b:
            a["position"], b["position"] = b["position"], a["position"]
    picks = sorted(by_element.values(), key=lambda p: p["position"])
    eh = picks_json["entry_history"]
    cost = int(eh.get("event_transfers_cost") or 0)
    moves = [t for t in transfers_json if t.get("event") == gw]
    return {
        "xi": [p["element"] for p in picks[:11]],
        "bench": [p["element"] for p in picks[11:]],
        "captain": next((p["element"] for p in picks if p["is_captain"]), None),
        "vice": next((p["element"] for p in picks if p["is_vice_captain"]), None),
        "chip": picks_json.get("active_chip"),
        "hits": cost // HIT_COST,
        "net_points": int(eh["points"]) - cost,  # FPL's entry_history points are before hits
        "transfers_in": sorted(t["element_in"] for t in moves),
        "transfers_out": sorted(t["element_out"] for t in moves),
        "fpl_autosubs": [(s["element_out"], s["element_in"]) for s in picks_json.get("automatic_subs", [])],
    }


def score_recommendation(rec: dict[str, Any], live: dict[int, dict[str, int]], position: dict[int, str]) -> dict[str, Any]:
    return score_team(
        [p["player_id"] for p in rec["starting_xi"]],
        [p["player_id"] for p in rec["bench"]],
        rec.get("captain_id"),
        rec.get("vice_captain_id"),
        rec.get("chip"),
        int(rec.get("hits") or 0),
        live,
        position,
    )


def compare(rec: dict[str, Any], mine: dict[str, Any], live: dict[int, dict[str, int]],
            position: dict[int, str], names: dict[int, str]) -> dict[str, Any]:
    c = score_recommendation(rec, live, position)
    rec_in = sorted(p["player_id"] for p in rec["transfers_in"])
    rec_out = sorted(p["player_id"] for p in rec["transfers_out"])
    followed_transfers = rec_in == mine["transfers_in"] and rec_out == mine["transfers_out"]

    def moves(ins: list[int], outs: list[int]) -> str:
        return "; ".join(f"{names.get(o, o)}>{names.get(i, i)}" for o, i in zip(outs, ins)) or "none"

    return {
        "gameweek": rec["gameweek"],
        "consultant_points": c["points"],
        "my_points": mine["net_points"],
        "difference": c["points"] - mine["net_points"],
        "followed_transfers": "yes" if followed_transfers else "no",
        "followed_captain": "yes" if rec.get("captain_id") == mine["captain"] else "no",
        "consultant_captain": rec.get("captain_name"),
        "my_captain": names.get(mine["captain"], mine["captain"]),
        "consultant_transfers": moves(rec_in, rec_out),
        "my_transfers": moves(mine["transfers_in"], mine["transfers_out"]),
        "consultant_autosubs": "; ".join(f"{names.get(o, o)}>{names.get(i, i)}" for o, i in c["autosubs"]) or "none",
        "consultant_chip": rec.get("chip") or "none",
        "my_chip": mine["chip"] or "none",
    }


def upsert_row(row: dict[str, Any], path: Path | None = None) -> pd.DataFrame:
    """Add/replace this gameweek's row and recompute running totals since GW11."""
    path = path or CSV_PATH  # resolved at call time, not import time
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=CSV_COLUMNS)
    df = df[df["gameweek"] != row["gameweek"]]
    df = pd.concat([df, pd.DataFrame([{**row, "scored_at": pd.Timestamp.now("UTC").isoformat()}])], ignore_index=True)
    df = df.sort_values("gameweek").reset_index(drop=True)
    tracked = df["gameweek"] >= FIRST_TRACKED_GW
    df["consultant_total"] = df["consultant_points"].where(tracked, 0).cumsum()
    df["my_total"] = df["my_points"].where(tracked, 0).cumsum()
    df["total_difference"] = df["consultant_total"] - df["my_total"]
    df = df[[c for c in CSV_COLUMNS if c in df.columns]]
    df.to_csv(path, index=False)
    return df


def score_gameweek(gw: int, team_id: int, get_json, bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Score gw's frozen recommendation against your real gameweek, if both
    exist. ``get_json(path)`` fetches from the FPL API (fpl.fetch.get)."""
    rec_path = RECOMMENDATIONS_DIR / f"gw{gw}.json"
    if not rec_path.exists():
        return {"available": False, "reason": f"no frozen recommendation for GW{gw}"}
    event = next((e for e in bootstrap["events"] if e["id"] == gw), None)
    if not event or not event.get("finished"):
        return {"available": False, "reason": f"GW{gw} hasn't finished yet"}
    rec = json.loads(rec_path.read_text())
    live_json = get_json(f"event/{gw}/live/")
    picks_json = get_json(f"entry/{team_id}/event/{gw}/picks/")
    transfers_json = get_json(f"entry/{team_id}/transfers/") or []
    if not live_json or not picks_json:
        return {"available": False, "reason": f"live points or your GW{gw} picks unavailable from the FPL API"}
    names = {int(e["id"]): e["web_name"] for e in bootstrap["elements"]}
    row = compare(rec, actual_from_api(picks_json, transfers_json, gw), live_points(live_json),
                  positions_from_bootstrap(bootstrap), names)
    upsert_row(row)
    return {"available": True, **row}


# ---------------------------------------------------------------------------
# 4. published section
# ---------------------------------------------------------------------------


def build_section(path: Path | None = None) -> dict[str, Any]:
    path = path or CSV_PATH
    if not path.exists():
        return {"available": True, "started": False, "first_gameweek": FIRST_TRACKED_GW, "last_week": None,
                "totals": None, "history": []}
    df = pd.read_csv(path)
    df = df[df["gameweek"] >= FIRST_TRACKED_GW].sort_values("gameweek")
    if df.empty:
        return {"available": True, "started": False, "first_gameweek": FIRST_TRACKED_GW, "last_week": None,
                "totals": None, "history": []}
    hist = json.loads(df.to_json(orient="records"))
    last = hist[-1]
    return {
        "available": True,
        "started": True,
        "first_gameweek": FIRST_TRACKED_GW,
        "last_week": {k: last[k] for k in ("gameweek", "consultant_points", "my_points", "difference",
                                           "followed_transfers", "followed_captain")},
        "totals": {"gameweeks": len(hist), "consultant": last["consultant_total"], "you": last["my_total"],
                   "difference": last["total_difference"]},
        "history": hist,
    }
