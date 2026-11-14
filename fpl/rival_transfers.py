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
3. ``forecast_rival_transfers`` -- each rival's single most likely move,
   from their own habits and squad: whether they move at all (transfers a
   week so far), who goes out (flagged, or their weakest starter), and who
   comes in (same position, within their budget; recent haulers for
   haul-chasers, best expected points otherwise). Each forecast line names
   the reason it's based on.

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


# A rival who made fewer than this many transfers per gameweek so far is
# forecast to save the free transfer rather than make a move.
MIN_TRANSFERS_PER_GW_TO_MOVE = 0.5
# "Recent haul" window for haul-chasers: the last N gameweeks.
HAUL_WINDOW_GWS = 2
MAX_PER_CLUB = 3


def forecast_rival_transfers(
    habits: pd.DataFrame,
    league_picks: pd.DataFrame,
    players: pd.DataFrame,
    player_matches: pd.DataFrame,
    free_transfers: pd.DataFrame,
    gameweek: int,
    xp_by_gameweek: pd.DataFrame | None = None,
    league_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Each rival's single most likely move next gameweek, built from that
    rival's own habits and squad (Step 15) -- not one league-wide "whoever
    just hauled" pick for everyone:

    1. Will they move at all? Nobody can without a free transfer; a rival
       who has made fewer than MIN_TRANSFERS_PER_GW_TO_MOVE transfers a
       gameweek so far is forecast to save it.
    2. Who goes out: a flagged player in their squad if they usually sell
       flagged players (``sells_flagged_rate``, as before); otherwise
       their starter with the lowest expected points next gameweek.
    3. Who comes in, for that position, affordable with that player's price
       plus their bank, keeping 3-per-club: a haul-chaser (``chase_rate``
       >= CONFIDENCE_THRESHOLD on enough tracked transfers) gets the best
       recent hauler (last HAUL_WINDOW_GWS gameweeks); everyone else the
       player with the most expected points over the next few gameweeks.
    Every forecast names the reason it's based on."""
    current_squads = league_picks[league_picks["gameweek"] == gameweek].groupby("manager_id")["player_id"].apply(set)
    starters = league_picks[(league_picks["gameweek"] == gameweek) & (league_picks["multiplier"] > 0)]
    starters = starters.groupby("manager_id")["player_id"].apply(set)
    info = players.set_index("id")
    price = info["now_cost"].to_dict()
    pos = info["position"].to_dict()
    club = info["team"].to_dict()
    name = info["web_name"].to_dict()
    available = set(info.index[info["status"].isin(["a"])]) if "status" in info.columns else set(info.index)

    xp_next: dict[int, float] = {}
    xp_total: dict[int, float] = {}
    if xp_by_gameweek is not None and not xp_by_gameweek.empty:
        cols = sorted((c for c in xp_by_gameweek.columns if c.startswith("gw") and c.endswith("_xp")), key=lambda c: int(c[2:-3]))
        x = xp_by_gameweek.set_index("player_id")
        if cols:
            xp_next = x[cols[0]].fillna(0.0).to_dict()
        if "xp_total" in x.columns:
            xp_total = x["xp_total"].fillna(0.0).to_dict()
    bank: dict[int, int] = {}
    if league_history is not None and not league_history.empty and "bank" in league_history.columns:
        lh = league_history[league_history["gameweek"] == league_history["gameweek"].max()]
        bank = {int(m): int(b) for m, b in zip(lh["manager_id"], lh["bank"].fillna(0))}

    haul_cutoff = player_matches.groupby("gameweek")["total_points"].quantile(HAUL_PERCENTILE)
    recent = player_matches[player_matches["gameweek"] > gameweek - HAUL_WINDOW_GWS]
    recent = recent[recent["total_points"] >= recent["gameweek"].map(haul_cutoff)]
    haul_points = recent.groupby("player_id")["total_points"].max().to_dict()

    flagged = players[
        players["chance_of_playing_next_round"].notna() & (players["chance_of_playing_next_round"] < 100)
    ].set_index("id")

    rows = []
    for row in habits.itertuples():
        ft_row = free_transfers[free_transfers["manager_id"] == row.manager_id]
        ft = int(ft_row["free_transfers"].iloc[0]) if not ft_row.empty else None
        squad = current_squads.get(row.manager_id, set())
        xi = starters.get(row.manager_id, set())
        rate = float(row.transfers_per_gw) if pd.notna(getattr(row, "transfers_per_gw", np.nan)) else 0.0
        base = {"manager_id": row.manager_id, "entry_name": row.entry_name, "player_name": row.player_name,
                "free_transfers": ft}
        if not ft or not squad:
            rows.append({**base, "predicted_in": None, "predicted_in_reason": "no free transfer",
                         "predicted_out": None, "predicted_out_reason": "no free transfer"})
            continue
        if rate < MIN_TRANSFERS_PER_GW_TO_MOVE:
            why = f"saves transfers: {rate:.1f} a week so far"
            rows.append({**base, "predicted_in": None, "predicted_in_reason": why,
                         "predicted_out": None, "predicted_out_reason": why})
            continue

        # who goes out
        out_id, out_reason = None, None
        enough_out = row.transfers_out_tracked >= MIN_TRACKED_TRANSFERS_FOR_HABIT
        owned_flagged = flagged[flagged.index.isin(squad)]
        if not owned_flagged.empty and enough_out and (row.sells_flagged_rate or 0) >= CONFIDENCE_THRESHOLD:
            worst = owned_flagged.sort_values("chance_of_playing_next_round").iloc[0]
            out_id = int(worst.name)
            out_reason = (f"{int(worst['chance_of_playing_next_round'])}% chance of playing; sells flagged "
                          f"players {row.sells_flagged_rate:.0%} of the time")
        elif not owned_flagged.empty:
            worst = owned_flagged.sort_values("chance_of_playing_next_round").iloc[0]
            out_id = int(worst.name)
            out_reason = f"{int(worst['chance_of_playing_next_round'])}% chance of playing"
        elif xi:
            out_id = min(xi, key=lambda p: xp_next.get(p, 0.0))
            out_reason = f"weakest starter by expected points ({xp_next.get(out_id, 0.0):.1f})"

        # who comes in
        in_id, in_reason = None, None
        if out_id is not None:
            budget = price.get(out_id, 0) + bank.get(int(row.manager_id), 0)
            clubs: dict[Any, int] = {}
            for p in squad - {out_id}:
                clubs[club.get(p)] = clubs.get(club.get(p), 0) + 1
            fits = [p for p in info.index if p not in squad and pos.get(p) == pos.get(out_id)
                    and price.get(p, 10**6) <= budget and p in available
                    and clubs.get(club.get(p), 0) < MAX_PER_CLUB]
            chases = (row.transfers_in_tracked >= MIN_TRACKED_TRANSFERS_FOR_HABIT
                      and pd.notna(row.chase_rate) and row.chase_rate >= CONFIDENCE_THRESHOLD)
            haulers = [p for p in fits if p in haul_points]
            if chases and haulers:
                in_id = max(haulers, key=lambda p: (haul_points[p], xp_total.get(p, 0.0)))
                in_reason = (f"hauled {int(haul_points[in_id])} pts recently; chases hauls "
                             f"{row.chase_rate:.0%} of the time ({row.transfers_in_tracked} tracked)")
            elif fits:
                in_id = max(fits, key=lambda p: xp_total.get(p, 0.0))
                in_reason = f"most expected points in that position within budget (£{budget / 10:.1f}m)"
        rows.append({
            **base,
            "predicted_in": name.get(in_id) if in_id is not None else None,
            "predicted_in_reason": in_reason or "no clear signal",
            "predicted_out": name.get(out_id) if out_id is not None else None,
            "predicted_out_reason": out_reason or "no clear signal",
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

    xp_path = TABLES_DIR / "xp_by_gameweek.parquet"
    xp_by_gameweek = pd.read_parquet(xp_path) if xp_path.exists() else None
    forecast = forecast_rival_transfers(habits, league_picks, players, player_matches, free_transfers, gameweek,
                                        xp_by_gameweek=xp_by_gameweek, league_history=league_history)
    write_table(forecast, "league_rival_transfer_forecast.parquet")

    print(f"\nCheckpoint: rival transfer forecast for the gameweek after {gameweek}:")
    print(forecast[["entry_name", "free_transfers", "predicted_in", "predicted_out"]].to_string(index=False))


if __name__ == "__main__":
    main()
