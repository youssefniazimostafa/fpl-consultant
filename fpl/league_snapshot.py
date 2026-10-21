"""League snapshot -- turns a hand-transcribed ``data/manual/league_snapshot.
json`` (screenshots of your mini-league, since its API endpoint isn't
wired up) into the exact same table shapes ``fpl.fetch`` would have built
from a real ``league_id``: ``league_managers`` / ``league_picks`` /
``league_history`` / ``league_chips``. ``fpl.league`` can't tell the
difference, so every Step 8 metric (ownership, EO, swing, chips grid,
strategy) and everything downstream in Step 9/10 works unchanged.

**Why a snapshot at all.** FPL only shows a rival's picks for the
gameweek that just finished -- never the one you're about to play. Your
own squad, by contrast, is always the live, current one (already
reflecting transfers since the last deadline). So the two halves of this
snapshot are deliberately asof different moments: rival squads carry
``pts`` for the gameweek that just finished; your squad carries ``opp``
(next fixture) for the gameweek that's about to be played. Both get
stored under the *same* ``gameweek`` label in the converted tables below
-- not because they happened in the same week, but because every
downstream function (``build_swings_table``, ``resolve_my_squad``, ...)
only uses that label as a join key to ask "who do I have on now vs. what
did the league just do", which is exactly the comparison this feature is
for.

**Validation (Step A)**, kept as a standing regression test since the
spec notes all 9 rivals currently pass it:

* every name is matched to an ``element`` id via (web_name, club,
  position) against ``players.parquet`` -- never guessed. An unmatched or
  ambiguous name raises, with the complete list of problems, not just the
  first one.
* a rival's starting-XI ``pts`` (plus the bench, only if they played
  Bench Boost) must sum to their ``gw`` score on the leaderboard.
* a snapshot ``club`` that disagrees with the player's current API team
  is a warning (API wins), not a hard failure -- shirts in old screenshots
  go stale.

Run with `uv run python -m fpl.league_snapshot` to validate the snapshot
on its own, independent of the rest of the pipeline.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.fetch import RAW_DIR, get

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "data" / "manual" / "league_snapshot.json"
TABLES_DIR = ROOT / "data" / "tables"

# The snapshot's position labels match the screenshots' GK/DEF/MID/FWD;
# players.parquet (from bootstrap-static's element_type) spells the
# keeper slot "GKP".
POSITION_ALIASES = {"GK": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

CHIP_NAMES = ["wildcard", "freehit", "bboost", "3xc"]
CHIP_HALF_BOUNDARY = 19
TOTAL_GAMEWEEKS = 38

# Multiplier for each on-pitch outcome -- 0 bench / 1 start / 2 captain /
# 3 triple captain -- matching Step 8's own definition exactly.
MULTIPLIER_BY_ROLE = {None: 1, "V": 1, "C": 2, "TC": 3}


class SnapshotError(ValueError):
    """Raised with every matching/validation problem found, not just the
    first one -- "fails loudly with a clear list", per the spec."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())


def resolve_gameweek() -> int:
    """The current gameweek, from bootstrap-static's ``events[].is_current``
    -- cached under today's raw dir exactly like ``fpl.fetch`` caches
    everything else, so this costs nothing extra on a day ``fpl.fetch``
    already ran."""
    day_dir = RAW_DIR / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = get("bootstrap-static/", day_dir)
    if bootstrap is None:
        raise RuntimeError("bootstrap-static/ returned nothing -- check network/API status")
    for event in bootstrap["events"]:
        if event.get("is_current"):
            return int(event["id"])
    # Between seasons / before GW1: fall back to one before the next event.
    for event in bootstrap["events"]:
        if event.get("is_next"):
            return max(int(event["id"]) - 1, 1)
    raise RuntimeError("could not resolve the current gameweek from bootstrap-static")


# ---------------------------------------------------------------------------
# Step A.2/A.3 -- name matching
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    name = name.replace("ß", "ss")
    stripped = unicodedata.normalize("NFD", name)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").lower()


def _build_indexes(
    players: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], list[int]], dict[tuple[str, str], list[int]]]:
    """(name, club, position) -> ids, and a looser (name, position) -> ids
    fallback for when a transcribed club is stale (players do change
    clubs mid-season)."""
    by_triple: dict[tuple[str, str, str], list[int]] = {}
    by_name_pos: dict[tuple[str, str], list[int]] = {}
    for row in players.itertuples():
        name = _normalise(row.web_name)
        by_triple.setdefault((name, row.team_short_name, row.position), []).append(row.id)
        by_name_pos.setdefault((name, row.position), []).append(row.id)
    return by_triple, by_name_pos


def match_players(
    snapshot: dict[str, Any], players: pd.DataFrame
) -> tuple[dict[tuple[str, str, str], int], list[str]]:
    """Match every (name, club, pos) appearing anywhere in the snapshot to
    a player id. Returns the match map plus a list of club-mismatch
    warnings (API team wins, snapshot club is just a note). Raises
    ``SnapshotError`` with every unmatched/ambiguous name at once if any
    are found.
    """
    by_triple, by_name_pos = _build_indexes(players)
    web_name = players.set_index("id")["team_short_name"]

    matches: dict[tuple[str, str, str], int] = {}
    problems: list[str] = []
    warnings: list[str] = []

    def entries() -> list[tuple[str, str, str]]:
        seen: set[tuple[str, str, str]] = set()
        for squad in snapshot["squads"].values():
            for p in squad["xi"] + squad["bench"]:
                pos = POSITION_ALIASES.get(p["pos"], p["pos"])
                seen.add((p["name"], p["club"], pos))
        return sorted(seen)

    for name, club, pos in entries():
        key = (name, club, pos)
        norm = _normalise(name)
        candidates = by_triple.get((norm, club, pos), [])
        if len(candidates) == 1:
            matches[key] = candidates[0]
            continue
        if len(candidates) > 1:
            problems.append(
                f"ambiguous: name={name!r} club={club!r} pos={pos!r} -> ids {candidates}"
            )
            continue

        # No exact (name, club, pos) hit -- maybe the snapshot's club is
        # stale (player transferred). Retry ignoring club.
        fallback = by_name_pos.get((norm, pos), [])
        if len(fallback) == 1:
            pid = fallback[0]
            matches[key] = pid
            actual_club = web_name.get(pid, "?")
            warnings.append(
                f"club mismatch: {name!r} transcribed as {club!r}, API has them at "
                f"{actual_club!r} (id {pid}) -- using the API's club"
            )
            continue

        problems.append(
            f"no match: name={name!r} club={club!r} pos={pos!r}"
            + (f" (candidates ignoring club: {fallback})" if fallback else "")
        )

    if problems:
        raise SnapshotError(
            f"{len(problems)} player(s) in the snapshot could not be matched:\n  "
            + "\n  ".join(problems)
        )
    return matches, warnings


# ---------------------------------------------------------------------------
# Step A.4 -- gameweek points regression test
# ---------------------------------------------------------------------------


def validate_gw_totals(snapshot: dict[str, Any]) -> pd.DataFrame:
    """Each rival's XI pts (+ bench, only if they played Bench Boost) must
    equal their ``gw`` on the leaderboard. Raises on any mismatch -- a
    standing regression test, since the spec notes all 9 rivals currently
    pass it."""
    standings = {s["team"]: s for s in snapshot["standings"]}
    rows = []
    for team, squad in snapshot["squads"].items():
        if standings[team].get("is_me"):
            continue  # your squad has no pts yet -- it's for the upcoming gameweek
        xi_pts = sum(p.get("pts", 0) for p in squad["xi"])
        bench_pts = sum(p.get("pts", 0) for p in squad["bench"])
        computed = xi_pts + (bench_pts if squad.get("chip") == "bboost" else 0)
        expected = standings[team]["gw"]
        rows.append(
            {
                "team": team,
                "xi_pts": xi_pts,
                "bench_pts": bench_pts,
                "chip": squad.get("chip"),
                "computed_gw": computed,
                "expected_gw": expected,
                "ok": computed == expected,
            }
        )
    result = pd.DataFrame(rows)
    failures = result[~result["ok"]]
    if not failures.empty:
        raise SnapshotError(
            "gameweek points regression test failed for: "
            + ", ".join(
                f"{r.team} (computed {r.computed_gw} != expected {r.expected_gw})"
                for r in failures.itertuples()
            )
        )
    return result


# ---------------------------------------------------------------------------
# Step B helper -- effective multiplier per squad entry, including the
# vice-captain-elevation case (nominal captain auto-subbed out of the XI
# because he didn't play, e.g. Lowkerkezenuinely's Cherki -> João Pedro)
# ---------------------------------------------------------------------------


def _effective_multipliers(squad: dict[str, Any]) -> dict[tuple[str, str], int]:
    """(name, club) -> multiplier, for every XI and bench entry in one
    squad, applying the vice-elevation rule generically: if the nominal
    captain ends up on the bench (didn't play), the vice -- if he's in the
    XI -- becomes the *effective* captain (multiplier 2) instead."""
    xi_keys = {(p["name"], p["club"]) for p in squad["xi"]}
    mult: dict[tuple[str, str], int] = {}

    captain_on_bench = any(p.get("role") == "C" for p in squad["bench"])
    vice_in_xi = next((p for p in squad["xi"] if p.get("role") == "V"), None)

    for p in squad["xi"]:
        key = (p["name"], p["club"])
        role = p.get("role")
        if role == "V" and captain_on_bench:
            mult[key] = 2  # vice elevated to effective captain
        else:
            mult[key] = MULTIPLIER_BY_ROLE.get(role, 1)
    for p in squad["bench"]:
        mult[(p["name"], p["club"])] = 0  # bench is always 0, nominal captain included

    if captain_on_bench and vice_in_xi is None:
        # No vice in the XI to take over -- shouldn't happen in practice,
        # but don't silently lose the captain multiplier if it does.
        raise SnapshotError(
            f"squad has its captain benched with no vice in the XI to elevate: {squad}"
        )
    return mult


# ---------------------------------------------------------------------------
# Step B/C -- convert to league_managers / league_picks / league_history /
# league_chips shaped DataFrames
# ---------------------------------------------------------------------------


def build_league_tables(
    players: pd.DataFrame, snapshot: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """Convert the manual snapshot into the same four tables
    ``fpl.fetch`` would have written from a real ``league_id``. Returns
    ``(league_managers, league_picks, league_history, league_chips,
    my_manager_id, gameweek)``.

    ``manager_id`` is each manager's leaderboard ``rank`` -- stable,
    readable, and already unique by construction.
    """
    snapshot = snapshot or load_snapshot()
    validate_gw_totals(snapshot)  # raise before touching anything else
    matches, warnings = match_players(snapshot, players)
    for w in warnings:
        print(f"  [league_snapshot] {w}")

    gameweek = resolve_gameweek()
    team_short_name = players.set_index("id")["team_short_name"]

    standings_by_team = {s["team"]: s for s in snapshot["standings"]}
    manager_id_by_team = {s["team"]: int(s["rank"]) for s in snapshot["standings"]}
    my_team = next(s["team"] for s in snapshot["standings"] if s.get("is_me"))
    my_manager_id = manager_id_by_team[my_team]

    managers_rows, history_rows, picks_rows = [], [], []
    for team, s in standings_by_team.items():
        manager_id = manager_id_by_team[team]
        managers_rows.append(
            {"manager_id": manager_id, "entry_name": team, "player_name": s["manager"]}
        )
        history_rows.append(
            {
                "manager_id": manager_id,
                "gameweek": gameweek,
                "total_points": s["total"],
                # Bank isn't visible in a screenshot; override it with
                # config.json's bank_override for accurate budget math.
                "bank": 0,
                "event_transfers": 0,
                "event_transfers_cost": 0,
            }
        )

        squad = snapshot["squads"][team]
        effective_mult = _effective_multipliers(squad) if team != my_team else None
        for slot_type, group in (("xi", squad["xi"]), ("bench", squad["bench"])):
            for p in group:
                pos = POSITION_ALIASES.get(p["pos"], p["pos"])
                key = (p["name"], p["club"], pos)
                player_id = matches[key]
                if team == my_team:
                    multiplier = MULTIPLIER_BY_ROLE.get(p.get("role"), 1) if slot_type == "xi" else 0
                else:
                    multiplier = effective_mult[(p["name"], p["club"])]
                picks_rows.append(
                    {
                        "manager_id": manager_id,
                        "player_id": player_id,
                        "gameweek": gameweek,
                        "multiplier": multiplier,
                        "team": team_short_name.get(player_id, p["club"]),
                    }
                )

    chips_rows = []
    for c in snapshot.get("chips_used", []):
        manager_id = manager_id_by_team.get(c["team"])
        if manager_id is None:
            continue
        chips_rows.append(
            {"manager_id": manager_id, "name": c["chip"], "event": c["gameweek"] or gameweek}
        )

    league_managers = pd.DataFrame(managers_rows)
    league_history = pd.DataFrame(history_rows)
    league_picks = pd.DataFrame(picks_rows)
    league_chips = pd.DataFrame(chips_rows, columns=["manager_id", "name", "event"])
    return league_managers, league_picks, league_history, league_chips, my_manager_id, gameweek


# ---------------------------------------------------------------------------
# Step C -- chips grid with honest "unknown" states (manual mode only has
# visibility starting the week this snapshot began being tracked)
# ---------------------------------------------------------------------------


def build_chips_grid(
    league_chips: pd.DataFrame, league_managers: pd.DataFrame, gameweek: int
) -> pd.DataFrame:
    """Like ``fpl.league.build_chips_grid``, but a slot we have no record
    of is "unknown" rather than "available" unless we can *prove* it's
    still available -- i.e. its half hasn't started yet. Tracking is
    assumed to have started at the earliest gameweek seen in
    ``league_chips`` (or this gameweek, for a brand-new snapshot)."""
    tracking_started = int(league_chips["event"].min()) if not league_chips.empty else gameweek
    manager_ids = sorted(league_managers["manager_id"])
    grid = pd.DataFrame({"manager_id": manager_ids})

    slot_columns = []
    for chip in CHIP_NAMES:
        for half in (1, 2):
            col = f"{chip}_h{half}"
            slot_columns.append(col)
            lo, hi = (1, CHIP_HALF_BOUNDARY) if half == 1 else (CHIP_HALF_BOUNDARY + 1, TOTAL_GAMEWEEKS)
            if league_chips.empty:
                used_gw = pd.Series(dtype=float)
            else:
                used = league_chips[
                    (league_chips["name"] == chip) & league_chips["event"].between(lo, hi)
                ]
                used_gw = used.set_index("manager_id")["event"]

            def label(manager_id: int) -> str:
                gw = used_gw.get(manager_id)
                if pd.notna(gw):
                    return f"used GW{int(gw)}"
                if lo > gameweek:
                    return "available"  # that half genuinely hasn't started for anyone
                if lo >= tracking_started:
                    return "available"  # we've been watching since this half began
                return "unknown"  # could have been used before tracking started

            grid[col] = grid["manager_id"].apply(label)

    grid["chips_known_available"] = (grid[slot_columns] == "available").sum(axis=1)
    grid["chips_unknown"] = (grid[slot_columns] == "unknown").sum(axis=1)
    return grid.merge(league_managers, on="manager_id", how="left")[
        ["manager_id", "entry_name", "player_name", *slot_columns, "chips_known_available", "chips_unknown"]
    ]


# ---------------------------------------------------------------------------
# standalone validation entry point
# ---------------------------------------------------------------------------


def main() -> None:
    players_path = TABLES_DIR / "players.parquet"
    if not players_path.exists():
        print(f"{players_path} not found -- run `uv run python -m fpl.fetch` first.")
        return
    players = pd.read_parquet(players_path)

    snapshot = load_snapshot()
    print(f"Loaded {SNAPSHOT_PATH} with {len(snapshot['squads'])} squads.")

    totals = validate_gw_totals(snapshot)
    print("\nStep A.4 regression test -- rival GW points reconcile with the leaderboard:")
    print(totals.to_string(index=False))

    matches, warnings = match_players(snapshot, players)
    print(f"\nMatched {len(matches)} unique (name, club, pos) entries to player ids.")
    for w in warnings:
        print(f"  {w}")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
