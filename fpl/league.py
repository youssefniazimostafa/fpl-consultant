"""League intelligence -- turns ``league_picks.parquet``, ``league_history.
parquet`` and ``league_chips.parquet`` (Step 6's fetch, scoped to your mini-
league) into exactly where you can gain on your rivals and where they can
hurt you.

FPL hides a rival's picks for the *next* gameweek until its deadline passes,
so every number here is necessarily as of the last deadline that has
already happened (the latest gameweek present in ``league_picks``) plus
whatever the transfer-habit stats below predict from history.

Five tables are built, one per metric in the spec:

* **league ownership / EO / your swing** (``league_swings.parquet``) --
  for every player any rival or you own at that gameweek: the share of
  rivals who have him in their 15 (``league_ownership_pct``), the average
  of rivals' multipliers on him (0 bench / 1 start / 2 captain / 3 triple
  captain -- ``league_eo``), your own multiplier, and the swing
  (``your_multiplier - league_eo``). Positive swing means you gain when he
  hauls; negative means you lose ground.
* **chips remaining** (``league_chips_grid.parquet``) -- one row per
  manager, one column per chip-half slot (4 chip types x 2 season halves =
  8 slots), showing the gameweek it was used in or "available".
* **transfer habits** (``league_transfer_habits.parquet``) -- hits taken,
  transfers per gameweek, and how often an incoming player had just hauled
  (scored in that gameweek's league-wide top decile) the week before a
  rival bought him.
* **squad overlap** (``league_squad_overlap.parquet``) -- how many of each
  rival's 15 you currently share, who's unique to them (their threats to
  you) and unique to you.
* **strategy mode** (``league_strategy.parquet``, one row) -- chasing /
  protecting / balanced, from your points gap to first place and the
  gameweeks remaining (see ``determine_strategy`` for the exact rule), with
  a manual override via ``config.json``'s ``strategy`` key.

Your *expected* gain never depends on any of this -- it only depends on
your own xP (Step 6). League data changes **risk**, not expectation: a
chasing team wants swings in its favour (low league EO, differentials),
a protecting team wants to close off rivals' swings (high league EO, match
the league's captain), and a balanced team just maximises xP and uses EO
only to break ties.

Run with `uv run python -m fpl.league`.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import fpl.league_snapshot as league_snapshot
from fpl.fetch import load_config

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
MANUAL_SNAPSHOT_PATH = ROOT / "data" / "manual" / "league_snapshot.json"

# FPL chip rules (2024/25 onward): each of the 4 chip types can be played
# once in each half of the season, split at this gameweek boundary. This
# boundary (and the chip name list) lives in bootstrap-static's own
# ``chips`` entries (start_event/stop_event per half), but fpl.fetch
# doesn't persist that list -- it's hardcoded here instead, documented the
# same way other season-structure invariants are elsewhere in this project
# (e.g. xp.py's SAVES_PER_POINT), since it has been GW19/20 every season
# since the split was introduced.
CHIP_HALF_BOUNDARY = 19
CHIP_NAMES = ["wildcard", "freehit", "bboost", "3xc"]
TOTAL_GAMEWEEKS = 38

# A gameweek score at or above this percentile (across everyone who played
# that week) counts as a "haul" for transfer-chasing detection -- adaptive
# to that week's scoring environment rather than a fixed points cutoff.
HAUL_PERCENTILE = 0.90

# Heuristic: roughly how many points a well-timed differential pick or
# captain swings per gameweek relative to the field. Used only to decide
# how "comfortable" a points gap or lead is once scaled by gameweeks
# remaining -- not part of the xP model itself.
TYPICAL_WEEKLY_SWING = 6

STRATEGY_MODES = {"chasing", "protecting", "balanced"}

# Rank-weighted EO (an extension beyond the spec's plain average): a rival
# ``RANK_WEIGHT_SCALE`` points away from you in the table counts ~37% as
# much as one level with you, and it decays smoothly from there --
# ``exp(-|points_gap_to_you| / RANK_WEIGHT_SCALE)``. This is what lets one
# "balanced" mode already behave like "chase the rival just above you,
# protect against the one just below" without a second, asymmetric mode:
# a rival a few points above or below you dominates the weighted average,
# while someone ten places back barely moves it.
RANK_WEIGHT_SCALE = 30

REQUIRED_TABLES = [
    "players", "league_picks", "league_history", "league_chips", "league_managers",
]


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in REQUIRED_TABLES:
        path = TABLES_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- set team_id and league_id in config.json "
                "and run `uv run python -m fpl.fetch` first."
            )
        tables[name] = pd.read_parquet(path)
    return tables


def latest_gameweek(league_picks: pd.DataFrame) -> int:
    """The most recent gameweek with picks data -- i.e. the last deadline
    that has already passed, which is as current as this view can ever be."""
    return int(league_picks["gameweek"].max())


def chip_half(event: int) -> int:
    return 1 if event <= CHIP_HALF_BOUNDARY else 2


# ---------------------------------------------------------------------------
# league ownership / EO / your swing
# ---------------------------------------------------------------------------


def build_swings_table(
    league_picks: pd.DataFrame, players: pd.DataFrame, team_id: int, gameweek: int
) -> pd.DataFrame:
    """League ownership, league EO, your multiplier and your swing for
    every player anyone in the league (rival or you) owns at ``gameweek``."""
    gw_picks = league_picks[league_picks["gameweek"] == gameweek]
    rivals = gw_picks[gw_picks["manager_id"] != team_id]
    n_rivals = rivals["manager_id"].nunique()
    if n_rivals == 0:
        raise ValueError(
            f"no rival picks found at gameweek {gameweek} -- check league_id/team_id in config.json"
        )

    league_eo = rivals.groupby("player_id")["multiplier"].sum() / n_rivals
    league_ownership = rivals.groupby("player_id")["manager_id"].nunique() / n_rivals
    your_multiplier = gw_picks.loc[gw_picks["manager_id"] == team_id].set_index("player_id")["multiplier"]

    player_ids = sorted(set(league_eo.index) | set(your_multiplier.index))
    table = pd.DataFrame({"player_id": player_ids})
    table["league_ownership_pct"] = (table["player_id"].map(league_ownership).fillna(0.0) * 100).round(1)
    table["league_eo"] = table["player_id"].map(league_eo).fillna(0.0).round(2)
    table["your_multiplier"] = table["player_id"].map(your_multiplier).fillna(0).astype(int)
    table["swing"] = (table["your_multiplier"] - table["league_eo"]).round(2)

    info = players[["id", "web_name", "team_short_name", "position", "selected_by_percent"]].rename(
        columns={"id": "player_id", "selected_by_percent": "overall_ownership_pct"}
    )
    table = table.merge(info, on="player_id", how="left")
    table["overall_ownership_pct"] = pd.to_numeric(table["overall_ownership_pct"], errors="coerce")

    columns = [
        "player_id", "web_name", "team_short_name", "position",
        "your_multiplier", "league_ownership_pct", "overall_ownership_pct",
        "league_eo", "swing",
    ]
    return table[columns].sort_values("swing").reset_index(drop=True)


def build_rival_weights(
    league_history: pd.DataFrame, team_id: int, gameweek: int
) -> dict[int, float]:
    """``exp(-|points_gap_to_you| / RANK_WEIGHT_SCALE)`` for every rival at
    ``gameweek`` -- a rival close to you in the table (above *or* below)
    weighs more than one far away, regardless of direction."""
    totals = league_history[league_history["gameweek"] == gameweek].set_index("manager_id")["total_points"]
    if team_id not in totals.index:
        raise ValueError(f"manager {team_id} has no league_history row at gameweek {gameweek}")
    your_points = totals[team_id]
    return {
        int(manager_id): float(np.exp(-abs(points - your_points) / RANK_WEIGHT_SCALE))
        for manager_id, points in totals.items()
        if manager_id != team_id
    }


def build_weighted_swings_table(
    league_picks: pd.DataFrame,
    team_id: int,
    gameweek: int,
    rival_weights: dict[int, float],
) -> pd.DataFrame:
    """Rank-weighted league EO and swing -- the same shape as
    ``build_swings_table``'s ``league_eo``/``swing``, but each rival's
    multiplier is weighted by closeness to you in the standings (see
    ``build_rival_weights``) instead of counted equally. Meant to be
    merged onto ``build_swings_table``'s output as extra columns, not used
    standalone, so it only returns the two new ones plus the join key.
    """
    gw_picks = league_picks[league_picks["gameweek"] == gameweek]
    rivals = gw_picks[gw_picks["manager_id"] != team_id].copy()
    rivals = rivals[rivals["manager_id"].isin(rival_weights)]
    rivals["weight"] = rivals["manager_id"].map(rival_weights)
    total_weight = sum(rival_weights.values())
    if total_weight == 0:
        raise ValueError("all rival weights are zero -- check league_history/team_id")

    weighted = (
        rivals.assign(weighted_mult=rivals["multiplier"] * rivals["weight"])
        .groupby("player_id")["weighted_mult"]
        .sum()
        / total_weight
    )
    your_multiplier = gw_picks.loc[gw_picks["manager_id"] == team_id].set_index("player_id")["multiplier"]

    player_ids = sorted(set(weighted.index) | set(your_multiplier.index))
    table = pd.DataFrame({"player_id": player_ids})
    table["weighted_league_eo"] = table["player_id"].map(weighted).fillna(0.0).round(2)
    your_mult = table["player_id"].map(your_multiplier).fillna(0).astype(int)
    table["weighted_swing"] = (your_mult - table["weighted_league_eo"]).round(2)
    return table[["player_id", "weighted_league_eo", "weighted_swing"]]


# ---------------------------------------------------------------------------
# chips remaining
# ---------------------------------------------------------------------------


def build_chips_grid(
    league_chips: pd.DataFrame, league_history: pd.DataFrame, league_managers: pd.DataFrame
) -> pd.DataFrame:
    """One row per manager, one column per chip-half slot, showing the
    gameweek a chip was used in or "available" if that slot is unspent --
    i.e. who can Bench Boost or Triple Captain into the next double."""
    manager_ids = sorted(set(league_history["manager_id"]) | set(league_managers["manager_id"]))
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
            grid[col] = grid["manager_id"].map(used_gw).apply(
                lambda gw: f"used GW{int(gw)}" if pd.notna(gw) else "available"
            )

    grid["chips_remaining"] = (grid[slot_columns] == "available").sum(axis=1)
    return grid.merge(league_managers, on="manager_id", how="left")[
        ["manager_id", "entry_name", "player_name", *slot_columns, "chips_remaining"]
    ]


# ---------------------------------------------------------------------------
# transfer habits
# ---------------------------------------------------------------------------


def build_transfer_habits(
    league_history: pd.DataFrame,
    league_picks: pd.DataFrame,
    player_matches: pd.DataFrame,
    league_managers: pd.DataFrame,
) -> pd.DataFrame:
    """Per-manager hits, transfer volume, and how often an incoming player
    had just hauled (top-``HAUL_PERCENTILE`` score) the gameweek before they
    bought him -- the "chases last week's haul" pattern the spec asks for.
    """
    habits = (
        league_history.groupby("manager_id")
        .agg(
            gameweeks_played=("gameweek", "nunique"),
            total_transfers=("event_transfers", "sum"),
            total_hits_cost=("event_transfers_cost", "sum"),
        )
        .reset_index()
    )
    habits["hits_taken"] = habits["total_hits_cost"] // 4
    habits["transfers_per_gw"] = (
        habits["total_transfers"] / habits["gameweeks_played"].replace(0, np.nan)
    ).round(2)

    chase_summary = _build_chase_summary(league_picks, player_matches)
    habits = habits.merge(chase_summary, on="manager_id", how="left")
    habits["transfers_in_tracked"] = habits["transfers_in_tracked"].fillna(0).astype(int)
    habits["chase_rate"] = habits["chase_rate"].round(2)

    return habits.merge(league_managers, on="manager_id", how="left")[
        [
            "manager_id", "entry_name", "player_name", "gameweeks_played",
            "total_transfers", "transfers_per_gw", "hits_taken",
            "transfers_in_tracked", "chase_rate",
        ]
    ]


def _build_chase_summary(league_picks: pd.DataFrame, player_matches: pd.DataFrame) -> pd.DataFrame:
    """For every player newly in a manager's squad vs. the prior gameweek
    (skipping any gap in the data), whether he scored in that gameweek's
    top decile the week before -- i.e. whether the manager chased a haul.
    """
    haul_cutoff = player_matches.groupby("gameweek")["total_points"].quantile(HAUL_PERCENTILE)
    pm = player_matches.merge(haul_cutoff.rename("haul_cutoff"), left_on="gameweek", right_index=True, how="left")
    pm["is_haul"] = pm["total_points"] >= pm["haul_cutoff"]
    haul_lookup = pm.set_index(["player_id", "gameweek"])["is_haul"]

    rows: list[dict[str, Any]] = []
    for manager_id, grp in league_picks.groupby("manager_id"):
        squads = grp.groupby("gameweek")["player_id"].apply(set)
        gws = sorted(squads.index)
        for prev_gw, gw in zip(gws, gws[1:]):
            if gw != prev_gw + 1:
                continue  # don't treat a gap in the data as a transfer
            for player_id in squads[gw] - squads[prev_gw]:
                chased = bool(haul_lookup.get((player_id, prev_gw), False))
                rows.append({"manager_id": manager_id, "chased_haul": chased})

    if not rows:
        return pd.DataFrame(columns=["manager_id", "transfers_in_tracked", "chase_rate"])

    chase_df = pd.DataFrame(rows)
    summary = chase_df.groupby("manager_id")["chased_haul"].agg(["count", "sum"]).reset_index()
    summary = summary.rename(columns={"count": "transfers_in_tracked", "sum": "chased_haul_count"})
    summary["chase_rate"] = summary["chased_haul_count"] / summary["transfers_in_tracked"]
    return summary[["manager_id", "transfers_in_tracked", "chase_rate"]]


# ---------------------------------------------------------------------------
# squad overlap
# ---------------------------------------------------------------------------


def build_squad_overlap(
    league_picks: pd.DataFrame,
    players: pd.DataFrame,
    league_managers: pd.DataFrame,
    team_id: int,
    gameweek: int,
) -> pd.DataFrame:
    """Head-to-head squad overlap with each rival at ``gameweek``: who
    you're really competing with, and who's unique to each side."""
    gw_picks = league_picks[league_picks["gameweek"] == gameweek]
    your_squad = set(gw_picks.loc[gw_picks["manager_id"] == team_id, "player_id"])
    names = players.set_index("id")["web_name"]

    rows = []
    for manager_id, grp in gw_picks[gw_picks["manager_id"] != team_id].groupby("manager_id"):
        rival_squad = set(grp["player_id"])
        shared = your_squad & rival_squad
        unique_to_rival = rival_squad - your_squad
        unique_to_you = your_squad - rival_squad
        rows.append(
            {
                "manager_id": manager_id,
                "overlap_count": len(shared),
                "shared_players": ", ".join(sorted(names.get(p, str(p)) for p in shared)),
                "unique_to_rival": ", ".join(sorted(names.get(p, str(p)) for p in unique_to_rival)),
                "unique_to_you": ", ".join(sorted(names.get(p, str(p)) for p in unique_to_you)),
            }
        )
    overlap = pd.DataFrame(rows).sort_values("overlap_count", ascending=False).reset_index(drop=True)
    return overlap.merge(league_managers, on="manager_id", how="left")[
        ["manager_id", "entry_name", "player_name", "overlap_count", "shared_players", "unique_to_rival", "unique_to_you"]
    ]


# ---------------------------------------------------------------------------
# strategy mode
# ---------------------------------------------------------------------------


def determine_strategy(
    league_history: pd.DataFrame, team_id: int, gameweek: int, config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Chasing / protecting / balanced, from your points gap to first place
    and the gameweeks remaining. ``config.json``'s ``strategy`` key
    overrides this outright when set to one of ``STRATEGY_MODES``.

    The rule: scale the gap (if you trail) or lead margin (if you're the
    leader) by the gameweeks remaining (floored at 1), against
    ``TYPICAL_WEEKLY_SWING``. Dividing by gameweeks remaining captures both
    halves of the spec's "chasing: far behind, or many gameweeks gone" in
    one number -- a fixed gap matters more to close the fewer gameweeks are
    left, and symmetrically a fixed lead gets easier to protect as the
    season runs out, without needing two separate rules.
    """
    override = config.get("strategy")
    if override in STRATEGY_MODES:
        return override, {"source": "config override"}

    totals = league_history[league_history["gameweek"] == gameweek].set_index("manager_id")["total_points"]
    if team_id not in totals.index:
        raise ValueError(f"manager {team_id} has no league_history row at gameweek {gameweek}")

    your_points = totals[team_id]
    ranked = totals.sort_values(ascending=False)
    leader_points = ranked.iloc[0]
    second_points = ranked.iloc[1] if len(ranked) > 1 else leader_points
    gameweeks_remaining = max(TOTAL_GAMEWEEKS - gameweek, 1)
    gap_to_first = leader_points - your_points  # 0 if you're (joint) first

    if gap_to_first > 0:
        risk_per_gw = gap_to_first / gameweeks_remaining
        mode = "chasing" if risk_per_gw > TYPICAL_WEEKLY_SWING else "balanced"
    else:
        lead_margin = your_points - second_points
        risk_per_gw = lead_margin / gameweeks_remaining
        mode = "protecting" if risk_per_gw > TYPICAL_WEEKLY_SWING else "balanced"

    detail = {
        "source": "auto",
        "your_points": int(your_points),
        "leader_points": int(leader_points),
        "second_points": int(second_points),
        "gap_to_first": int(gap_to_first),
        "gameweeks_remaining": gameweeks_remaining,
        "risk_per_gw": round(float(risk_per_gw), 2),
    }
    return mode, detail


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def write_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / name
    df.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(df)} rows)")


def _cross_check_snapshot(
    players: pd.DataFrame, league_picks: pd.DataFrame, league_managers: pd.DataFrame, gameweek: int
) -> None:
    """Step F: when ``league_id`` is set, the live API is authoritative and
    the manual snapshot (if one also exists) is only cross-checked against
    it here -- any mismatch is a warning about a stale/mistranscribed
    snapshot, never a reason to stop, since the live data wins either way.
    """
    try:
        snapshot = league_snapshot.load_snapshot()
        matches, _ = league_snapshot.match_players(snapshot, players)
        gw_picks = league_picks[league_picks["gameweek"] == gameweek]
        live_squads = (
            gw_picks.merge(league_managers, on="manager_id").groupby("entry_name")["player_id"].apply(set)
        )
        mismatched = 0
        for team, squad in snapshot["squads"].items():
            if team not in live_squads.index:
                continue
            snapshot_ids = {
                matches[(p["name"], p["club"], league_snapshot.POSITION_ALIASES.get(p["pos"], p["pos"]))]
                for p in squad["xi"] + squad["bench"]
            }
            if snapshot_ids != live_squads[team]:
                mismatched += 1
        verdict = "snapshot squads match the live API" if mismatched == 0 else (
            f"{mismatched} squad(s) differ from the live API -- snapshot may be stale"
        )
        print(f"  [snapshot cross-check] {verdict}")
    except Exception as exc:  # cross-check is best-effort only, never fatal
        print(f"  [snapshot cross-check] skipped ({exc})")


def resolve_team_id(config: dict[str, Any], league_managers: pd.DataFrame) -> int | None:
    """Step F: ``team_id`` is explicit in config.json in live-API mode, but
    manual-snapshot mode derives "you" from the snapshot and flags that
    row with ``is_me`` in ``league_managers`` -- so downstream steps
    (fpl.optimise, fpl.publish) can resolve it from there instead of
    requiring it in config.json at all."""
    team_id = config.get("team_id")
    if team_id is not None:
        return int(team_id)
    if "is_me" in league_managers.columns:
        me = league_managers[league_managers["is_me"]]
        if not me.empty:
            return int(me["manager_id"].iloc[0])
    return None


def main() -> None:
    config = load_config()
    league_id = config.get("league_id")

    if league_id is not None:
        # Step F: league_id set -> live API picks are authoritative.
        team_id = config.get("team_id")
        if team_id is None:
            print("No team_id set in config.json; skipping league intelligence (nothing to compare against).")
            return
        team_id = int(team_id)

        tables = load_tables()
        players, league_picks, league_history, league_chips, league_managers = (
            tables["players"], tables["league_picks"], tables["league_history"],
            tables["league_chips"], tables["league_managers"],
        )
        if league_picks.empty or league_history.empty:
            print(
                "league_picks/league_history are empty -- set league_id in config.json "
                "and run `uv run python -m fpl.fetch` first."
            )
            return
        gameweek = latest_gameweek(league_picks)
        source = "the live API (league_id set in config.json)"
        if MANUAL_SNAPSHOT_PATH.exists():
            _cross_check_snapshot(players, league_picks, league_managers, gameweek)
    else:
        # Step F: no league_id -> fall back to the manual snapshot, if any.
        players_path = TABLES_DIR / "players.parquet"
        if not players_path.exists() or not MANUAL_SNAPSHOT_PATH.exists():
            print(
                "No league_id set in config.json, and no data/manual/league_snapshot.json "
                "found either; skipping league intelligence."
            )
            return
        players = pd.read_parquet(players_path)
        league_managers, league_picks, league_history, league_chips, team_id, gameweek = (
            league_snapshot.build_league_tables(players)
        )
        source = "the manual snapshot (no league_id set in config.json)"

        # Persist these to data/tables/ exactly like fpl.fetch does for the
        # live path, so fpl.optimise/fpl.publish need zero changes to find
        # them -- an is_me flag lets optimise.py resolve "my" manager_id
        # without requiring team_id in config.json at all in this mode.
        league_managers = league_managers.assign(is_me=league_managers["manager_id"] == team_id)
        write_table(league_managers, "league_managers.parquet")
        write_table(league_picks, "league_picks.parquet")
        write_table(league_history, "league_history.parquet")
        write_table(league_chips, "league_chips.parquet")

    print(f"Building league intelligence as of gameweek {gameweek}, from {source}...")

    swings = build_swings_table(league_picks, players, team_id, gameweek)
    rival_weights = build_rival_weights(league_history, team_id, gameweek)
    weighted = build_weighted_swings_table(league_picks, team_id, gameweek, rival_weights)
    swings = swings.merge(weighted, on="player_id", how="left")
    write_table(swings, "league_swings.parquet")

    if league_id is not None:
        grid = build_chips_grid(league_chips, league_history, league_managers)
    else:
        grid = league_snapshot.build_chips_grid(league_chips, league_managers, gameweek)
    write_table(grid, "league_chips_grid.parquet")

    player_matches_path = TABLES_DIR / "player_matches.parquet"
    if player_matches_path.exists():
        player_matches = pd.read_parquet(player_matches_path)
        habits = build_transfer_habits(league_history, league_picks, player_matches, league_managers)
        write_table(habits, "league_transfer_habits.parquet")
    else:
        print(f"  skipping transfer habits: {player_matches_path} not found")

    overlap = build_squad_overlap(league_picks, players, league_managers, team_id, gameweek)
    write_table(overlap, "league_squad_overlap.parquet")

    mode, detail = determine_strategy(league_history, team_id, gameweek, config)
    strategy_row = pd.DataFrame([{"gameweek": gameweek, "mode": mode, **detail}])
    write_table(strategy_row, "league_strategy.parquet")

    print(f"\nStrategy mode: {mode.upper()} ({detail['source']})")
    if detail["source"] == "auto":
        print(
            f"  you: {detail['your_points']} pts, leader: {detail['leader_points']} pts, "
            f"gap to first: {detail['gap_to_first']}, {detail['gameweeks_remaining']} GWs remaining "
            f"-> {detail['risk_per_gw']} pts/GW at stake"
        )

    print("\nCheckpoint: 10 players with the largest negative swing for you "
          f"(gameweek {gameweek}):")
    print(swings.head(10).to_string(index=False))

    print("\nCheckpoint: chips-remaining grid for every rival:")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
