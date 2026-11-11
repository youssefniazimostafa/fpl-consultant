"""Step 12 -- rival transfer model, now real.

This used to be a scaffold: with no ``league_id`` configured, the only
history of rivals' squads was a single hand-transcribed ``data/
manual_snapshots/<gw>.json`` archive, and one data point isn't a habit.
Step 12 sets ``league_id``/``my_entry_id`` in ``config.json``, which makes
``fpl.fetch`` pull every manager's picks for every gameweek played so far
straight from the live API (``league_picks.parquet``) -- that *is* enough
history to diff, so this module now does what its old docstring described
as a future plan:

1. ``build_rival_transfer_history`` -- diffs each manager's squad between
   consecutive gameweeks in ``league_picks.parquet`` to get every transfer
   in/out, per manager, per gameweek.
2. ``build_rival_habits`` -- turns that history (plus ``fpl.league``'s own
   hits/chase-rate stats, so the two don't disagree) into three per-rival
   habit numbers: how often they take a hit, how often an incoming player
   had just hauled ("chases form"), and how often an outgoing player had
   already gone a gameweek with 0 minutes before being sold ("sells
   flagged players").
3. ``forecast_rival_transfers`` -- for whichever gameweek you ask about,
   picks each rival's single most-likely incoming and outgoing player
   using those three habit numbers plus this gameweek's actual signals
   (who just hauled league-wide, who's currently flagged in ``players.
   chance_of_playing_next_round``) -- each forecast line names the signal
   it's based on, so "no clear signal" is a real, visible answer rather
   than a guess dressed up as one.

This does **not** fold the forecast into ``fpl.league``'s own
``league_eo`` -- that calculation is validated against live standings
(see ``fpl.league.validate_standings``) and a forecast swap is, by
definition, not yet a confirmed pick. Instead the forecast is published as
its own advisory table that the dashboard's differentials/threats panels
can show alongside (not instead of) the confirmed EO.

Run with `uv run python -m fpl.rival_transfers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl.league import HAUL_PERCENTILE, build_transfer_habits

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"

# A rival needs at least this many tracked transfers before their habit
# rates mean anything -- otherwise one early hit or one lucky haul-chase
# would read as a 100%/0% "habit", which is noise, not a pattern.
MIN_TRACKED_TRANSFERS_FOR_HABIT = 3

# How confident a habit rate needs to be before the forecast actually
# names a player, rather than reporting "no clear signal".
CONFIDENCE_THRESHOLD = 0.34  # roughly "did it at least 1 time in 3"


def build_rival_transfer_history(league_picks: pd.DataFrame) -> pd.DataFrame:
    """Every transfer in/out, per manager, per gameweek -- diffs each
    manager's squad (``league_picks``) against their squad the gameweek
    before. A gap in the data (a missing gameweek) is skipped rather than
    read as a transfer, same rule ``fpl.league``'s chase-summary uses."""
    rows: list[dict[str, Any]] = []
    for manager_id, grp in league_picks.groupby("manager_id"):
        squads = grp.groupby("gameweek")["player_id"].apply(set)
        gws = sorted(squads.index)
        for prev_gw, gw in zip(gws, gws[1:]):
            if gw != prev_gw + 1:
                continue
            for player_id in squads[gw] - squads[prev_gw]:
                rows.append({"manager_id": manager_id, "gameweek": gw, "player_id": player_id, "direction": "in"})
            for player_id in squads[prev_gw] - squads[gw]:
                rows.append({"manager_id": manager_id, "gameweek": gw, "player_id": player_id, "direction": "out"})
    return pd.DataFrame(rows, columns=["manager_id", "gameweek", "player_id", "direction"])


def build_rival_habits(
    transfer_history: pd.DataFrame,
    league_history: pd.DataFrame,
    league_picks: pd.DataFrame,
    player_matches: pd.DataFrame,
    league_managers: pd.DataFrame,
) -> pd.DataFrame:
    """Per-rival habit rates: ``hit_rate`` (share of gameweeks played with
    a hit taken), ``chase_rate`` (share of incoming players who'd just
    hauled -- reuses ``fpl.league.build_transfer_habits`` so the two
    tables never disagree on this number), and ``sells_flagged_rate``
    (share of outgoing players who'd already played 0 minutes the
    gameweek before being sold -- the closest signal this project has to
    "was already looking injured/benched" without a historical news
    archive)."""
    base = build_transfer_habits(league_history, league_picks, player_matches, league_managers)
    base["hit_rate"] = (base["hits_taken"] / base["gameweeks_played"].replace(0, np.nan)).round(2)

    minutes_by_player_gw = player_matches.set_index(["player_id", "gameweek"])["minutes"]
    out_rows = transfer_history[transfer_history["direction"] == "out"].copy()
    if not out_rows.empty:
        out_rows["prev_gw"] = out_rows["gameweek"] - 1
        out_rows["was_benched_or_out"] = [
            minutes_by_player_gw.get((pid, gw), np.nan) == 0
            for pid, gw in zip(out_rows["player_id"], out_rows["prev_gw"])
        ]
        sells_flagged = (
            out_rows.groupby("manager_id")["was_benched_or_out"]
            .agg(transfers_out_tracked="count", sells_flagged_rate="mean")
            .reset_index()
        )
        sells_flagged["sells_flagged_rate"] = sells_flagged["sells_flagged_rate"].round(2)
    else:
        sells_flagged = pd.DataFrame(columns=["manager_id", "transfers_out_tracked", "sells_flagged_rate"])

    habits = base.merge(sells_flagged, on="manager_id", how="left")
    habits["transfers_out_tracked"] = habits["transfers_out_tracked"].fillna(0).astype(int)
    return habits


def forecast_rival_transfers(
    habits: pd.DataFrame,
    league_picks: pd.DataFrame,
    players: pd.DataFrame,
    player_matches: pd.DataFrame,
    free_transfers: pd.DataFrame,
    gameweek: int,
) -> pd.DataFrame:
    """For each rival, the single most-likely incoming and outgoing player
    next gameweek, from this gameweek's actual signals (who just hauled
    league-wide, who's currently flagged) weighted by that rival's own
    tracked habit rate -- a rival whose ``chase_rate``/``sells_flagged_rate``
    isn't both tracked on enough transfers (``MIN_TRACKED_TRANSFERS_FOR_HABIT``)
    and above ``CONFIDENCE_THRESHOLD`` gets "no clear signal" instead of a
    guess. Only rivals with >= 1 free transfer get a forecast at all --
    nobody can buy without one.
    """
    current_squads = league_picks[league_picks["gameweek"] == gameweek].groupby("manager_id")["player_id"].apply(set)

    haul_cutoff = player_matches.groupby("gameweek")["total_points"].quantile(HAUL_PERCENTILE)
    this_gw_matches = player_matches[player_matches["gameweek"] == gameweek]
    haulers = (
        this_gw_matches[this_gw_matches["total_points"] >= haul_cutoff.get(gameweek, np.inf)]
        .merge(players[["id", "web_name", "form"]], left_on="player_id", right_on="id", how="left")
        .sort_values("total_points", ascending=False)
    )

    flagged = players[
        players["chance_of_playing_next_round"].notna() & (players["chance_of_playing_next_round"] < 100)
    ].set_index("id")

    rows = []
    for row in habits.itertuples():
        ft_row = free_transfers[free_transfers["manager_id"] == row.manager_id]
        has_ft = not ft_row.empty and ft_row["free_transfers"].iloc[0] >= 1
        squad = current_squads.get(row.manager_id, set())

        predicted_in, in_reason = None, "no clear signal"
        enough_data = row.transfers_in_tracked >= MIN_TRACKED_TRANSFERS_FOR_HABIT
        if has_ft and enough_data and row.chase_rate >= CONFIDENCE_THRESHOLD:
            candidates = haulers[~haulers["player_id"].isin(squad)]
            if not candidates.empty:
                pick = candidates.iloc[0]
                predicted_in = pick["web_name"]
                in_reason = (
                    f"hauled {int(pick['total_points'])} pts this GW; chases hauls "
                    f"{row.chase_rate:.0%} of the time ({row.transfers_in_tracked} tracked)"
                )

        predicted_out, out_reason = None, "no clear signal"
        enough_out_data = row.transfers_out_tracked >= MIN_TRACKED_TRANSFERS_FOR_HABIT
        if has_ft and enough_out_data and row.sells_flagged_rate >= CONFIDENCE_THRESHOLD:
            owned_flagged = flagged[flagged.index.isin(squad)]
            if not owned_flagged.empty:
                worst = owned_flagged.sort_values("chance_of_playing_next_round").iloc[0]
                predicted_out = worst["web_name"]
                out_reason = (
                    f"{int(worst['chance_of_playing_next_round'])}% chance of playing; sells flagged "
                    f"players {row.sells_flagged_rate:.0%} of the time ({row.transfers_out_tracked} tracked)"
                )

        rows.append({
            "manager_id": row.manager_id,
            "entry_name": row.entry_name,
            "player_name": row.player_name,
            "free_transfers": int(ft_row["free_transfers"].iloc[0]) if not ft_row.empty else None,
            "predicted_in": predicted_in,
            "predicted_in_reason": in_reason,
            "predicted_out": predicted_out,
            "predicted_out_reason": out_reason,
        })
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / name
    df.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(df)} rows)")


def main() -> None:
    required = ["league_picks", "league_history", "league_managers", "player_matches", "players", "league_free_transfers"]
    missing = [name for name in required if not (TABLES_DIR / f"{name}.parquet").exists()]
    if missing:
        print(f"Rival transfer model: skipped -- missing {missing} (run fpl.fetch and fpl.league first).")
        return

    league_picks = pd.read_parquet(TABLES_DIR / "league_picks.parquet")
    league_history = pd.read_parquet(TABLES_DIR / "league_history.parquet")
    league_managers = pd.read_parquet(TABLES_DIR / "league_managers.parquet")
    player_matches = pd.read_parquet(TABLES_DIR / "player_matches.parquet")
    players = pd.read_parquet(TABLES_DIR / "players.parquet")
    free_transfers = pd.read_parquet(TABLES_DIR / "league_free_transfers.parquet")

    if league_picks["gameweek"].nunique() < 2:
        print("Rival transfer model: skipped -- fewer than 2 gameweeks of league_picks, nothing to diff yet.")
        return

    gameweek = int(league_picks["gameweek"].max())
    history = build_rival_transfer_history(league_picks)
    write_table(history, "league_rival_transfer_history.parquet")

    habits = build_rival_habits(history, league_history, league_picks, player_matches, league_managers)
    write_table(habits, "league_rival_habits.parquet")

    forecast = forecast_rival_transfers(habits, league_picks, players, player_matches, free_transfers, gameweek)
    write_table(forecast, "league_rival_transfer_forecast.parquet")

    print(f"\nCheckpoint: rival transfer forecast for the gameweek after {gameweek}:")
    print(forecast[["entry_name", "free_transfers", "predicted_in", "predicted_out"]].to_string(index=False))


if __name__ == "__main__":
    main()
