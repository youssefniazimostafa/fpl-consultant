"""Decision engine -- turns Step 6's xP (``xp_by_gameweek.parquet``) and Step
8's league intelligence (``league_swings.parquet``, ``league_strategy.
parquet``, ``league_chips_grid.parquet``) into concrete moves: transfers,
starting XI, captain, and chip timing.

Rather than looping over swaps (which can't jointly satisfy budget, the
2/5/5/3 squad, max-3-per-club and a valid formation for anything beyond a
single swap), every decision is one MILP solved with PuLP. The same model
handles a single gameweek and a multi-week plan: a "single gameweek" is
just a horizon of length 1.

**The model.** For each candidate player ``p`` and each gameweek ``g`` in
the horizon: ``squad[p,g]`` (in the 15), ``xi[p,g]`` (starting), ``cap[p,g]``
(captain), ``transfer_in[p,g]``/``transfer_out[p,g]`` (bought/sold that
week). Squad evolves by a linear recursion,
``squad[p,g] = squad[p,g-1] + transfer_in[p,g] - transfer_out[p,g]``,
instead of re-deriving "currently owned" from last week's solution (which
would multiply two binaries together and stop the model being linear).
Cash works the same way: a ``bank[g]`` variable recurses from ``bank[g-1]``
plus sale proceeds minus purchase cost, with ``bank[g] >= 0`` standing in
for the whole budget constraint (an overspend would force bank negative,
which isn't allowed) -- there's no separate "squad value <= budget"
inequality to maintain alongside it.

Free transfers roll over the same way, with one trick: the naive recursion
``ft[g] <= ft[g-1] - used[g-1] + 1`` can go negative when a hit was taken
the week before. But ``hits[g-1]`` is already pinned to its true minimum
(``max(0, used[g-1] - ft[g-1])``) by the ``-4 x hits`` objective penalty, so
adding it back, ``ft[g] <= ft[g-1] - used[g-1] + hits[g-1] + 1``, computes
the correct rollover for free with no new binaries.

Value optimised: ``xp`` adjusted for strategy mode and league EO (Steps 6
and 8 meet here) -- ``xp * (1 + lambda * (EO_CENTER - league_eo))``, with
``lambda`` positive when chasing (reward low-EO differentials) and
negative when protecting (reward high-EO, close off rivals' swings).

**Extensions, in the order the spec asks for:**

1. Multi-week horizon (``solve_horizon``, horizon length from
   ``config.json``'s ``horizon``), discounted ~0.9/week since later
   predictions are less certain.
2. Transfer threshold (``recommend_transfers``): a move is only recommended
   if holding this week (while still letting the model move in a later
   week instead) loses at least ``TRANSFER_THRESHOLD_XP`` of horizon value.
3. Selling prices (``estimate_selling_prices``): purchase price (from
   ``my_transfers.parquet``, falling back to the season-start price) plus
   half the rise, rounded down -- the real FPL rule -- unless
   ``config.json``'s ``bank_override`` is set.
4. Chip planner (``plan_chips``): Bench Boost and Triple Captain read
   straight off the baseline plan; Wildcard/Free Hit re-solve a single
   week with (effectively) unlimited transfers and compare to the
   baseline's value that week. Flags first-half chips as urgent as GW19
   approaches.
5. Captain board (``build_captain_board``): the XI ranked by xP, with a
   chase-mode bonus for differentials (low league captaincy) and a
   protect-mode flag on the league's most-backed captain.

**Simplifications, documented rather than hidden:** sell prices are frozen
for the whole horizon (no in-window price-rise modelling); Wildcard and
Free Hit get the *same* single-week gain estimate, since modelling a
Wildcard's effect on *future* weeks would need a full re-plan from that
week forward -- out of scope here.

Run with `uv run python -m fpl.optimise`, optionally `--gameweek N` to
check legality/reasoning for a gameweek you remember (the Checkpoint): this
resolves your squad as of that gameweek instead of the latest one, but xP
is always *today's* forecast, so treat that run as "would this squad have
been legal then", not "what would I have been told to do back then".
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pulp

from fpl.fetch import load_config
from fpl.league import CHIP_HALF_BOUNDARY, CHIP_NAMES, chip_half

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"

# ---------------------------------------------------------------------------
# FPL squad rules
# ---------------------------------------------------------------------------
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_CLUB = 3
POSITION_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
# min, max of each position allowed in the starting XI (always 1 GK).
FORMATION_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

# A new signing costs 4 points unless covered by a free transfer -- the
# real FPL hit cost, unchanged since the game's inception.
HIT_COST = 4
MAX_FREE_TRANSFERS = 5
# A large-but-finite stand-in for "unlimited transfers" (Wildcard/Free Hit):
# big enough that hits can never bind, since a 15-player pool can't need
# more than SQUAD_SIZE transfers in one week anyway.
UNLIMITED_FREE_TRANSFERS = 20

# Later weeks' predictions are less certain -- shrink their weight in the
# objective by this factor per week out, compounding.
WEEKLY_DISCOUNT = 0.9
# A bench player's value still matters a little (bench boost, autosubs on
# a blank), just far less than a starter's -- same 0.1 weighting the spec's
# own pseudocode uses.
BENCH_VALUE_WEIGHT = 0.1
# Only recommend a transfer if it beats rolling (optionally moving in a
# later week instead) by at least this much xP over the whole horizon --
# the midpoint of the spec's "about 1.5-2 xP" band.
TRANSFER_THRESHOLD_XP = 1.75

# Keeps the MILP tractable: the candidate pool is your 15 plus the top-N
# per position by horizon xP, not the full ~668-player universe.
TRANSFER_POOL_PER_POSITION = 40

# Player statuses that should never be bought *new* -- injured/suspended/
# unavailable. An already-owned player keeps his variable regardless (the
# model can still choose to sell him; his low xP already pushes that way).
UNAVAILABLE_STATUSES = {"u", "i", "s"}

# Adjusts xP toward (chasing) or away from (protecting) differentials.
# EO_CENTER=0.5 is the spec's own example midpoint; balanced mode uses a
# tiny lambda purely to break ties among otherwise-equal players.
EO_CENTER = 0.5
LAMBDA_BY_MODE = {"chasing": 0.15, "protecting": -0.15, "balanced": 0.02}

# First-half chips (GW1-19) get flagged "urgent" once this many gameweeks
# remain before the CHIP_HALF_BOUNDARY cutoff.
CHIP_URGENCY_WINDOW = 3

REQUIRED_TABLES = ["players", "xp_by_gameweek"]
OPTIONAL_TABLES = [
    "league_picks", "league_history", "league_swings", "league_strategy",
    "league_chips_grid", "my_transfers",
]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_TABLES:
        path = TABLES_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `uv run python -m fpl.fetch` and "
                "`uv run python -m fpl.xp` first."
            )
        tables[name] = pd.read_parquet(path)
    for name in OPTIONAL_TABLES:
        path = TABLES_DIR / f"{name}.parquet"
        tables[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return tables


def resolve_my_squad(
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    team_id: int,
    as_of_gameweek: int | None = None,
) -> tuple[set[int], int, int]:
    """Your current 15 and bank (tenths of GBP 1m), as of the last deadline
    that passed -- or ``as_of_gameweek`` if given (see module docstring's
    caveat about checking a past gameweek)."""
    gameweek = as_of_gameweek or int(league_picks["gameweek"].max())
    my_picks = league_picks[
        (league_picks["gameweek"] == gameweek) & (league_picks["manager_id"] == team_id)
    ]
    if my_picks.empty:
        raise ValueError(f"no picks found for manager {team_id} at gameweek {gameweek}")
    squad_ids = set(my_picks["player_id"])

    hist_row = league_history[
        (league_history["gameweek"] == gameweek) & (league_history["manager_id"] == team_id)
    ]
    bank = int(hist_row["bank"].iloc[0]) if not hist_row.empty else 0
    return squad_ids, bank, gameweek


# ---------------------------------------------------------------------------
# selling prices (extension 3)
# ---------------------------------------------------------------------------


def estimate_selling_prices(
    my_squad_ids: set[int], players: pd.DataFrame, my_transfers: pd.DataFrame
) -> dict[int, int]:
    """Purchase price + half the rise, rounded down (the real FPL rule),
    for every player you currently own. Purchase price comes from your most
    recent buy of him in ``my_transfers.parquet``; players held since
    before GW1 (no transfer-in record) fall back to the season-start price
    implied by ``now_cost - cost_change_start``."""
    info = players.set_index("id")
    now_cost = info["now_cost"]
    season_start_price = now_cost - info["cost_change_start"]

    purchase_price: dict[int, int] = {}
    if not my_transfers.empty and "element_in" in my_transfers.columns:
        buys = (
            my_transfers.sort_values("time")
            .groupby("element_in")["element_in_cost"]
            .last()
        )
        purchase_price = buys.to_dict()

    sell_price: dict[int, int] = {}
    for p in my_squad_ids:
        cur = int(now_cost.get(p, 0))
        buy = int(purchase_price.get(p, season_start_price.get(p, cur)))
        # Price fell since purchase: full loss passed through, no floor.
        # Price rose: keep purchase price plus half the rise, rounded down.
        sell_price[p] = cur if cur < buy else buy + (cur - buy) // 2
    return sell_price


# ---------------------------------------------------------------------------
# candidate pool + strategy-adjusted value
# ---------------------------------------------------------------------------


def build_candidate_pool(
    players: pd.DataFrame, xp_by_gw: pd.DataFrame, my_squad_ids: set[int], avoid_ids: set[int]
) -> list[int]:
    """Your 15 (minus anything on the ``avoid`` list -- that forces a sale,
    since a dropped player simply has no variable to be "kept" with) plus
    the top ``TRANSFER_POOL_PER_POSITION`` tradeable players per position by
    horizon xP. Keeps the MILP to ~150-250 players instead of the full
    universe, solvable in seconds rather than minutes."""
    eligible = players[~players["id"].isin(avoid_ids)].copy()
    is_owned = eligible["id"].isin(my_squad_ids)
    tradeable = eligible[is_owned | ~eligible["status"].isin(UNAVAILABLE_STATUSES)]
    tradeable = tradeable.merge(
        xp_by_gw[["player_id", "xp_total"]], left_on="id", right_on="player_id", how="left"
    )
    tradeable["xp_total"] = tradeable["xp_total"].fillna(0.0)

    pool_ids = set(my_squad_ids) & set(tradeable["id"])
    for pos in POSITION_QUOTAS:
        top = tradeable[tradeable["position"] == pos].sort_values("xp_total", ascending=False)
        pool_ids |= set(top.head(TRANSFER_POOL_PER_POSITION)["id"])
    return sorted(int(p) for p in pool_ids)


def lookup_weekly_xp(
    xp_by_gw: pd.DataFrame, weeks: list[int], pool_ids: list[int]
) -> dict[tuple[int, int], float]:
    """Raw (unadjusted) xP for every (player, gameweek) pair in the pool."""
    indexed = xp_by_gw.set_index("player_id")
    weekly: dict[tuple[int, int], float] = {}
    for p in pool_ids:
        row = indexed.loc[p] if p in indexed.index else None
        for g in weeks:
            col = f"gw{g}_xp"
            value = float(row[col]) if row is not None and col in row and pd.notna(row[col]) else 0.0
            weekly[(p, g)] = value
    return weekly


def strategy_adjusted(
    raw_weekly_xp: dict[tuple[int, int], float], league_eo: dict[int, float], mode: str
) -> dict[tuple[int, int], float]:
    """``xp * (1 + lambda * (EO_CENTER - league_eo))`` -- where Steps 6 and
    8 meet. Positive lambda (chasing) rewards low-EO differentials;
    negative lambda (protecting) rewards high-EO, template picks."""
    lam = LAMBDA_BY_MODE.get(mode, 0.0)
    adjusted = {}
    for (p, g), xp in raw_weekly_xp.items():
        eo = league_eo.get(p, 0.0)
        adjusted[(p, g)] = xp * (1 + lam * (EO_CENTER - eo))
    return adjusted


# ---------------------------------------------------------------------------
# the MILP (core + extension 1: multi-week horizon)
# ---------------------------------------------------------------------------


def solve_horizon(
    pool_ids: list[int],
    weekly_value: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    my_squad_ids: set[int],
    sell_price: dict[int, int],
    initial_bank: int,
    free_transfers_now: int,
    weeks: list[int],
    discount: dict[int, float],
    held_weeks: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """One MILP covering every gameweek in ``weeks`` at once. A single
    gameweek is just ``weeks`` of length 1 -- there's no separate
    single-week code path.

    ``held_weeks``: gameweeks where transfers are forced to zero (used by
    ``recommend_transfers`` to test "hold this week, keep flexibility to
    move later" against the free solve).
    """
    # pulp >=4 ties every LpVariable to the LpProblem that will solve it
    # (created via ``m.add_variable``, not a free-standing constructor), so
    # the problem has to exist before any variable does.
    m = pulp.LpProblem("fpl_horizon", pulp.LpMaximize)

    squad = {(p, g): m.add_variable(f"squad_{p}_{g}", cat="Binary") for p in pool_ids for g in weeks}
    xi = {(p, g): m.add_variable(f"xi_{p}_{g}", cat="Binary") for p in pool_ids for g in weeks}
    cap = {(p, g): m.add_variable(f"cap_{p}_{g}", cat="Binary") for p in pool_ids for g in weeks}
    tin = {(p, g): m.add_variable(f"in_{p}_{g}", cat="Binary") for p in pool_ids for g in weeks}
    tout = {(p, g): m.add_variable(f"out_{p}_{g}", cat="Binary") for p in pool_ids for g in weeks}
    hits = {g: m.add_variable(f"hits_{g}", lowBound=0, cat="Integer") for g in weeks}
    bank = {g: m.add_variable(f"bank_{g}", lowBound=0, cat="Continuous") for g in weeks}
    ft = {
        g: m.add_variable(f"ft_{g}", lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
        for g in weeks[1:]
    }

    # objective: discounted (XI + captain's extra multiplier + a sliver of
    # bench value) each week, minus hit cost at full (undiscounted) weight
    # -- a hit is a certain deduction the week you take it, not a
    # prediction, so it shouldn't shrink just because it's further out.
    value_terms = []
    for g in weeks:
        week_value = pulp.lpSum(weekly_value[(p, g)] * (xi[(p, g)] + cap[(p, g)]) for p in pool_ids)
        bench_value = BENCH_VALUE_WEIGHT * pulp.lpSum(weekly_value[(p, g)] * squad[(p, g)] for p in pool_ids)
        value_terms.append(discount[g] * (week_value + bench_value))
    m += pulp.lpSum(value_terms) - HIT_COST * pulp.lpSum(hits[g] for g in weeks)

    clubs = {pool_info[p]["team"] for p in pool_ids}
    for i, g in enumerate(weeks):
        prev_squad = {p: (1 if p in my_squad_ids else 0) for p in pool_ids} if i == 0 else squad
        prev_g = None if i == 0 else weeks[i - 1]
        prev_bank = initial_bank if i == 0 else bank[prev_g]
        prev_ft = free_transfers_now if i == 0 else ft[g]

        m += pulp.lpSum(squad[(p, g)] for p in pool_ids) == SQUAD_SIZE
        for pos, n in POSITION_QUOTAS.items():
            m += pulp.lpSum(squad[(p, g)] for p in pool_ids if pool_info[p]["position"] == pos) == n
        for club in clubs:
            m += pulp.lpSum(squad[(p, g)] for p in pool_ids if pool_info[p]["team"] == club) <= MAX_PER_CLUB

        m += pulp.lpSum(xi[(p, g)] for p in pool_ids) == XI_SIZE
        m += pulp.lpSum(cap[(p, g)] for p in pool_ids) == 1
        for pos, (lo, hi) in FORMATION_BOUNDS.items():
            count = pulp.lpSum(xi[(p, g)] for p in pool_ids if pool_info[p]["position"] == pos)
            m += count >= lo
            m += count <= hi
        for p in pool_ids:
            m += xi[(p, g)] <= squad[(p, g)]
            m += cap[(p, g)] <= xi[(p, g)]

            prev = prev_squad[p] if i == 0 else prev_squad[(p, prev_g)]
            m += squad[(p, g)] == prev + tin[(p, g)] - tout[(p, g)]
            m += tin[(p, g)] <= 1 - prev
            m += tout[(p, g)] <= prev
            if g in held_weeks:
                m += tin[(p, g)] == 0
                m += tout[(p, g)] == 0

        transfers_used = pulp.lpSum(tin[(p, g)] for p in pool_ids)
        m += hits[g] >= transfers_used - prev_ft

        m += bank[g] == prev_bank + pulp.lpSum(
            sell_price[p] * tout[(p, g)] - pool_info[p]["now_cost"] * tin[(p, g)] for p in pool_ids
        )

        if i < len(weeks) - 1:
            nxt = weeks[i + 1]
            m += ft[nxt] <= prev_ft - transfers_used + hits[g] + 1

    stats = m.solve(pulp.HiGHS(msg=False))

    weekly_results = []
    for g in weeks:
        squad_ids = {p for p in pool_ids if squad[(p, g)].varValue and squad[(p, g)].varValue > 0.5}
        xi_ids = {p for p in pool_ids if xi[(p, g)].varValue and xi[(p, g)].varValue > 0.5}
        cap_ids = [p for p in pool_ids if cap[(p, g)].varValue and cap[(p, g)].varValue > 0.5]
        captain_id = cap_ids[0] if cap_ids else None
        transfers_in = {p for p in pool_ids if tin[(p, g)].varValue and tin[(p, g)].varValue > 0.5}
        transfers_out = {p for p in pool_ids if tout[(p, g)].varValue and tout[(p, g)].varValue > 0.5}
        xi_value = sum(weekly_value[(p, g)] for p in xi_ids)
        if captain_id is not None:
            xi_value += weekly_value[(captain_id, g)]
        weekly_results.append(
            {
                "gameweek": g,
                "squad_ids": squad_ids,
                "xi_ids": xi_ids,
                "captain_id": captain_id,
                "transfers_in": transfers_in,
                "transfers_out": transfers_out,
                "hits": int(round(hits[g].varValue or 0)),
                "bank_after": float(bank[g].varValue or 0),
                "xi_value": xi_value,
            }
        )

    return {
        "status": stats.status.name,
        "objective": pulp.value(m.objective),
        "weeks": weekly_results,
    }


# ---------------------------------------------------------------------------
# transfer threshold (extension 2)
# ---------------------------------------------------------------------------


def recommend_transfers(
    baseline: dict[str, Any],
    pool_ids: list[int],
    weekly_value: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    my_squad_ids: set[int],
    sell_price: dict[int, int],
    initial_bank: int,
    free_transfers_now: int,
    weeks: list[int],
    discount: dict[int, float],
) -> tuple[dict[str, Any], float, bool]:
    """Only recommend this week's transfer if it beats holding (while still
    letting the model move in a *later* week instead) by at least
    ``TRANSFER_THRESHOLD_XP`` over the horizon -- "spot the right week to
    move" means testing whether *this* week is that week, not whether
    moving beats never moving at all."""
    week0 = weeks[0]
    if not baseline["weeks"][0]["transfers_in"]:
        return baseline, 0.0, False

    held = solve_horizon(
        pool_ids, weekly_value, pool_info, my_squad_ids, sell_price,
        initial_bank, free_transfers_now, weeks, discount, held_weeks=frozenset({week0}),
    )
    gain = (baseline["objective"] or 0.0) - (held["objective"] or 0.0)
    if gain < TRANSFER_THRESHOLD_XP:
        return held, gain, False
    return baseline, gain, True


# ---------------------------------------------------------------------------
# chip planner (extension 4)
# ---------------------------------------------------------------------------


def plan_chips(
    plan: dict[str, Any],
    pool_ids: list[int],
    weekly_value: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    my_squad_ids: set[int],
    sell_price: dict[int, int],
    initial_bank: int,
    weeks: list[int],
    team_chips_row: dict[str, Any] | None,
) -> pd.DataFrame:
    """Bench Boost / Triple Captain gains read straight off ``plan``.
    Wildcard / Free Hit re-solve that single week with (effectively)
    unlimited transfers from the squad/bank the baseline plan would have
    entering it, and compare to the baseline's actual value that week --
    both get the *same* estimate here (see module docstring)."""
    rows = []
    for i, g in enumerate(weeks):
        wk = plan["weeks"][i]
        bench_ids = wk["squad_ids"] - wk["xi_ids"]
        bb_gain = sum(weekly_value[(p, g)] for p in bench_ids)
        tc_gain = weekly_value[(wk["captain_id"], g)] if wk["captain_id"] is not None else 0.0

        prev_squad_ids = my_squad_ids if i == 0 else plan["weeks"][i - 1]["squad_ids"]
        prev_bank = initial_bank if i == 0 else plan["weeks"][i - 1]["bank_after"]
        wc_solve = solve_horizon(
            pool_ids, weekly_value, pool_info, prev_squad_ids, sell_price,
            prev_bank, UNLIMITED_FREE_TRANSFERS, [g], {g: 1.0},
        )
        wc_gain = wc_solve["weeks"][0]["xi_value"] - wk["xi_value"]

        half = chip_half(g)
        gains = {"wildcard": wc_gain, "freehit": wc_gain, "bboost": bb_gain, "3xc": tc_gain}
        for chip in CHIP_NAMES:
            col = f"{chip}_h{half}"
            available = team_chips_row is None or team_chips_row.get(col) == "available"
            urgent = bool(
                available and half == 1 and (CHIP_HALF_BOUNDARY - g) <= CHIP_URGENCY_WINDOW
            )
            rows.append(
                {
                    "gameweek": g, "chip": chip, "half": half,
                    "available": available, "estimated_gain": round(gains[chip], 2),
                    "urgent": urgent,
                }
            )
    return pd.DataFrame(rows).sort_values(["chip", "gameweek"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# captain board (extension 5)
# ---------------------------------------------------------------------------


def build_captain_board(
    week0: dict[str, Any],
    raw_weekly_xp: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    mode: str,
    league_eo: dict[int, float],
) -> pd.DataFrame:
    """The XI ranked by xP for next gameweek. Chase mode adds a bonus for
    differentials (low league captaincy); protect mode flags the league's
    most-backed captain so you can match it and close off that swing."""
    g = week0["gameweek"]
    rows = []
    for p in week0["xi_ids"]:
        xp = raw_weekly_xp.get((p, g), 0.0)
        eo = league_eo.get(p, 0.0)
        rows.append(
            {
                "player_id": p,
                "web_name": pool_info[p]["web_name"],
                "position": pool_info[p]["position"],
                "xp": round(xp, 2),
                "league_eo": round(eo, 2),
                "is_optimiser_captain": p == week0["captain_id"],
            }
        )
    board = pd.DataFrame(rows)
    if board.empty:
        return board

    lam = LAMBDA_BY_MODE.get(mode, 0.0)
    board["board_score"] = (board["xp"] * (1 + lam * (EO_CENTER - board["league_eo"]))).round(2)
    if mode == "protecting" and not board.empty:
        most_backed_eo = board["league_eo"].max()
        board["league_most_backed"] = board["league_eo"] == most_backed_eo
    else:
        board["league_most_backed"] = False
    return board.sort_values("board_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# legality (the Checkpoint)
# ---------------------------------------------------------------------------


def validate_squad(
    squad_ids: set[int],
    xi_ids: set[int],
    captain_id: int | None,
    pool_info: dict[int, dict[str, Any]],
    bank_after: float,
) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if len(squad_ids) != SQUAD_SIZE:
        violations.append(f"squad size is {len(squad_ids)}, expected {SQUAD_SIZE}")
    for pos, n in POSITION_QUOTAS.items():
        count = sum(1 for p in squad_ids if pool_info[p]["position"] == pos)
        if count != n:
            violations.append(f"{pos} count is {count}, expected {n}")
    clubs = Counter(pool_info[p]["team"] for p in squad_ids)
    for club, count in clubs.items():
        if count > MAX_PER_CLUB:
            violations.append(f"club {club} has {count} players, max {MAX_PER_CLUB}")
    if len(xi_ids) != XI_SIZE:
        violations.append(f"XI size is {len(xi_ids)}, expected {XI_SIZE}")
    if not xi_ids <= squad_ids:
        violations.append("XI contains a player not in the squad")
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        count = sum(1 for p in xi_ids if pool_info[p]["position"] == pos)
        if not (lo <= count <= hi):
            violations.append(f"XI has {count} {pos}, formation needs {lo}-{hi}")
    if captain_id is None or captain_id not in xi_ids:
        violations.append("captain is not in the XI")
    if bank_after < -1e-6:
        violations.append(f"bank went negative ({bank_after})")
    return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def write_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / name
    df.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(df)} rows)")


def build_squad_plan_table(
    plan: dict[str, Any], pool_info: dict[int, dict[str, Any]], weeks: list[int]
) -> pd.DataFrame:
    rows = []
    for wk in plan["weeks"]:
        g = wk["gameweek"]
        for p in wk["squad_ids"]:
            role = "captain" if p == wk["captain_id"] else ("xi" if p in wk["xi_ids"] else "bench")
            rows.append(
                {
                    "gameweek": g, "player_id": p, "web_name": pool_info[p]["web_name"],
                    "position": pool_info[p]["position"], "role": role,
                    "transferred_in": p in wk["transfers_in"],
                }
            )
    return pd.DataFrame(rows)


def build_transfers_table(
    plan: dict[str, Any], pool_info: dict[int, dict[str, Any]], gain: float, recommended: bool
) -> pd.DataFrame:
    """The full horizon's transfer plan (the "6-week transfer plan" panel),
    one row per planned move in every week -- not just this week's. Only
    gameweek 0's move has actually been checked against the threshold (see
    ``recommend_transfers``); later weeks are the model's *current* plan
    for that week and can shift as news lands closer to its own deadline,
    so ``horizon_gain_xp``/``recommended`` are only populated for week 0."""
    rows = []
    for i, wk in enumerate(plan["weeks"]):
        ins, outs = sorted(wk["transfers_in"]), sorted(wk["transfers_out"])
        for j in range(max(len(ins), len(outs), 1)):
            rows.append(
                {
                    "gameweek": wk["gameweek"],
                    "player_out": pool_info[outs[j]]["web_name"] if j < len(outs) else None,
                    "player_in": pool_info[ins[j]]["web_name"] if j < len(ins) else None,
                    "hits_taken": wk["hits"] if j == 0 else 0,
                    "horizon_gain_xp": round(gain, 2) if i == 0 else None,
                    "recommended": recommended if i == 0 else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gameweek", type=int, default=None,
        help="check legality/reasoning as of this gameweek instead of the latest one",
    )
    args = parser.parse_args()

    config = load_config()
    team_id = config.get("team_id")
    if team_id is None:
        print("No team_id set in config.json; nothing to optimise.")
        return
    team_id = int(team_id)

    tables = load_tables()
    players, xp_by_gw = tables["players"], tables["xp_by_gameweek"]
    league_picks, league_history = tables["league_picks"], tables["league_history"]
    if league_picks.empty or league_history.empty:
        print(
            "league_picks/league_history not found -- set league_id in config.json "
            "and run `uv run python -m fpl.fetch` then `uv run python -m fpl.league` first."
        )
        return

    my_squad_ids, hist_bank, gameweek = resolve_my_squad(
        league_picks, league_history, team_id, args.gameweek
    )
    bank = config.get("bank_override")
    bank = int(bank) if bank is not None else hist_bank
    print(f"Optimising from gameweek {gameweek} with {len(my_squad_ids)} owned players, bank £{bank/10:.1f}m")

    league_eo: dict[int, float] = {}
    if not tables["league_swings"].empty:
        league_eo = tables["league_swings"].set_index("player_id")["league_eo"].to_dict()

    mode = config.get("strategy")
    if mode not in {"chasing", "protecting", "balanced"}:
        mode = tables["league_strategy"]["mode"].iloc[0] if not tables["league_strategy"].empty else "balanced"

    sell_price_owned = estimate_selling_prices(my_squad_ids, players, tables["my_transfers"])
    avoid_ids = {int(x) for x in (config.get("avoid") or []) if isinstance(x, int)}
    pool_ids = build_candidate_pool(players, xp_by_gw, my_squad_ids, avoid_ids)

    pool_info = players.set_index("id")[["web_name", "position", "team", "now_cost"]].to_dict("index")
    now_cost_map = {p: pool_info[p]["now_cost"] for p in pool_ids}
    sell_price = {p: sell_price_owned.get(p, now_cost_map[p]) for p in pool_ids}

    weeks = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    horizon = max(min(int(config.get("horizon", 5)), len(weeks)), 1)
    weeks = weeks[:horizon]
    discount = {g: WEEKLY_DISCOUNT**i for i, g in enumerate(weeks)}

    raw_weekly_xp = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    adj_weekly_value = strategy_adjusted(raw_weekly_xp, league_eo, mode)

    effective_free = max(int(config.get("free_transfers", 1)) - int(config.get("transfers_made", 0)), 0)

    print(f"Strategy mode: {mode.upper()}. Solving a {len(weeks)}-week horizon over a {len(pool_ids)}-player pool...")
    baseline = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )
    final_plan, gain, transfer_recommended = recommend_transfers(
        baseline, pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount,
    )

    team_chips_row = None
    if not tables["league_chips_grid"].empty:
        row = tables["league_chips_grid"][tables["league_chips_grid"]["manager_id"] == team_id]
        if not row.empty:
            team_chips_row = row.iloc[0].to_dict()
    chip_plan = plan_chips(
        final_plan, pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, weeks, team_chips_row,
    )

    captain_board = build_captain_board(final_plan["weeks"][0], raw_weekly_xp, pool_info, mode, league_eo)

    write_table(build_squad_plan_table(final_plan, pool_info, weeks), "optimise_squad.parquet")
    write_table(build_transfers_table(final_plan, pool_info, gain, transfer_recommended), "optimise_transfers.parquet")
    write_table(chip_plan, "optimise_chip_plan.parquet")
    write_table(captain_board, "optimise_captain_board.parquet")

    wk0 = final_plan["weeks"][0]
    ok, violations = validate_squad(wk0["squad_ids"], wk0["xi_ids"], wk0["captain_id"], pool_info, wk0["bank_after"])

    print(f"\nCheckpoint: legality check for gameweek {wk0['gameweek']}'s recommendation -- "
          f"{'PASS' if ok else 'FAIL'}")
    for v in violations:
        print(f"  ! {v}")

    print("\nReasoning:")
    print(f"  solver status: {final_plan['status']}, objective: {final_plan['objective']:.2f}")
    if wk0["transfers_in"] or wk0["transfers_out"]:
        out_names = ", ".join(pool_info[p]["web_name"] for p in sorted(wk0["transfers_out"]))
        in_names = ", ".join(pool_info[p]["web_name"] for p in sorted(wk0["transfers_in"]))
        print(f"  transfer: OUT {out_names} -> IN {in_names} ({wk0['hits']} hit(s)), "
              f"horizon gain {gain:.2f}xP vs holding -> {'recommended' if transfer_recommended else 'below threshold, holding instead'}")
    else:
        print("  no transfer recommended this week -- roll the free transfer.")
    captain_name = pool_info[wk0["captain_id"]]["web_name"] if wk0["captain_id"] else "none"
    print(f"  captain: {captain_name}")
    if not captain_board.empty:
        print("\nCaptain board (top 5):")
        print(captain_board.head(5).to_string(index=False))

    print("\nChips-remaining grid (your recommendations this horizon):")
    print(chip_plan.to_string(index=False))


if __name__ == "__main__":
    main()
