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
from fpl.league import CHIP_HALF_BOUNDARY, CHIP_NAMES, chip_half, resolve_team_id

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
# objective by this factor per week out, compounding. Step 13 rule 4
# calibration (fpl.transfer_calibration stage 2, 2025-26 replay, 3
# starting squads): 0.9 scored best (1873 mean season points; 0.7-0.85
# 1833-1855; 0.95-1.0 1782-1790), so it stays at 0.9.
WEEKLY_DISCOUNT = 0.9
# A bench player's value still matters a little (bench boost, autosubs on
# a blank), just far less than a starter's -- same 0.1 weighting the spec's
# own pseudocode uses.
BENCH_VALUE_WEIGHT = 0.1
# Only recommend a transfer if it beats rolling (optionally moving in a
# later week instead) by at least this much xP over the whole horizon --
# the midpoint of the spec's "about 1.5-2 xP" band.
TRANSFER_THRESHOLD_XP = 1.75

# ---------------------------------------------------------------------------
# Free-transfer rules (Step 13). Each one is switchable and calibrated by
# fpl/transfer_calibration.py: a replay of the whole 2025-26 backtest
# season (forecasts built only from data before each deadline), 1 free
# transfer a week banking up to 5, no hits, no chips, the optimiser's own
# captain, from 3 starting squads; then re-run unchanged on 2024-25 and
# 2023-24, which the grid never saw. A rule is on only if it beat today's
# optimiser on the unseen seasons too. Figures: mean season-points
# difference vs the old optimiser, +- paired SE over the 3 squads.
# ---------------------------------------------------------------------------
# 1. Value of a saved free transfer: every free transfer carried out of the
#    horizon (capped at MAX_FREE_TRANSFERS, so a roll that would be wasted
#    at the cap is worth nothing) adds this many points to the objective,
#    so a move must beat holding by more than it. ON: V=1.5 (the stage-1
#    pick) on its own, keeping the 1.75 threshold: +5 +-39 on 2025-26,
#    +31 +-17 on 2024-25, +75 +-7 on 2023-24; neighbours V=0.5-2 are
#    positive on 5 of 6 unseen-season runs, so 1.5 isn't a lucky spike.
USE_FT_VALUE = True
FT_VALUE_XP = 1.5
# 2. Uncertainty margin: a move must gain more than k x the standard error
#    of (incoming - outgoing) over the horizon, from the per-player-week
#    residual SDs below (by position and projected-points band). With it
#    off, the legacy TRANSFER_THRESHOLD_XP applies instead. OFF: alone,
#    k=0.3 gave +12 / +40 / +41 on the three seasons, but k=0.2 and 0.5
#    flip sign on the unseen seasons (+41/-7, 0/-51), and combined with
#    rule 1 (the stage-1 best, 1873 vs 1820 in-sample) it fell to +1 +-28
#    and -18 +-28 out of sample -- tuned noise, not a real effect.
USE_UNCERTAINTY_MARGIN = False
UNCERTAINTY_K = 0.3
# Upper edges of the projected-points bands for RESIDUAL_SD (last is open).
XP_BANDS = (2.0, 4.0, 6.0)
# SD of (actual - projected) points for one player-week, by position and
# band: the 2025-26 backtest's one-week-ahead residuals pooled with this
# season's live accuracy residuals (fpl.transfer_calibration.residual_sd_table).
# Bands: <2, 2-4, 4-6, 6+ xP (6+ has <20 rows per position, so it reuses 4-6).
RESIDUAL_SD: dict[str, tuple[float, float, float, float]] = {
    "GKP": (0.76, 2.70, 3.37, 3.37),
    "DEF": (1.63, 3.18, 3.86, 3.86),
    "MID": (1.52, 3.06, 3.76, 3.76),
    "FWD": (1.71, 3.39, 4.57, 4.57),
}
# 3. Reliability: in these modes each starter's value is reduced by
#    RELIABILITY_WEIGHT x the SD of his own points distribution (from the
#    xP components: rare goals are high-variance, appearances and clean
#    sheets less so). Captaincy and the shown xP are untouched; chasing
#    mode takes no penalty. OFF: weights 0.05 / 0.1 / 0.2 never beat 0 in
#    the 2025-26 replay at any discount (best 1858 vs 1873 without it).
USE_RELIABILITY = False
RELIABILITY_WEIGHT = 0.0
RELIABILITY_MODES = {"balanced", "protecting"}
# 4. Horizon discount: each later week is weighted WEEKLY_DISCOUNT^i (see
#    above), and a player-week whose fixture has no betting-odds coverage
#    (Elo + xG only) is weighted by NO_ODDS_DISCOUNT on top. OFF: can't be
#    calibrated -- the backtest seasons have no historical odds at all, and
#    the live prediction logs don't record which fixtures had odds.
USE_NO_ODDS_DISCOUNT = False
NO_ODDS_DISCOUNT = 1.0

# Keeps the MILP tractable: the candidate pool is your 15 plus the top-N
# per position by horizon xP, not the full ~668-player universe.
TRANSFER_POOL_PER_POSITION = 40

# Player statuses that should never be bought *new* -- injured/suspended/
# unavailable. An already-owned player keeps his variable regardless (the
# model can still choose to sell him; his low xP already pushes that way).
UNAVAILABLE_STATUSES = {"u", "i", "s"}

# Step 12: a hit (-4/extra transfer) is never justified by pure value --
# only by necessity. These are the same statuses ``UNAVAILABLE_STATUSES``
# uses for "can't be bought", reused here for "can be sold with a hit";
# a plain doubtful ("d") player only counts if his own
# ``chance_of_playing_next_round`` has dropped to this level or below.
HIT_JUSTIFY_STATUSES = {"u", "i", "s"}
HIT_JUSTIFY_DOUBTFUL_CHANCE = 25
# The hit must pay for itself: the incoming player's combined XI output
# over this many future gameweeks has to beat holding (bench covering the
# unavailable starter) by more than ``HIT_COST`` points *per hit taken* --
# "projected points over the next 3 gameweeks" from the spec, using raw
# (undiscounted, strategy-unadjusted) xP so the bar is a plain points
# comparison, not the horizon objective's discounted/EO-adjusted value.
HIT_JUSTIFY_WINDOW = 3

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
    "league_picks", "league_history", "league_managers", "league_swings",
    "league_strategy", "league_chips_grid", "league_free_transfers", "my_transfers",
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


def resolve_free_transfers(league_free_transfers: pd.DataFrame, team_id: int) -> int:
    """Step 12: free transfers available right now, from ``fpl.league``'s
    computed ``league_free_transfers.parquet`` (walks your whole transfer
    history -- see ``fpl.league._free_transfers_for_manager``) -- replaces
    the old hand-typed ``config.json`` ``free_transfers``/``transfers_made``
    pair, which only ever covered "me" and had to be updated by hand every
    week. Falls back to the standard 1 free transfer if the table is
    missing or doesn't have a row for ``team_id`` (e.g. manual-snapshot
    mode with no chips/history tracked yet), rather than raising.
    """
    if league_free_transfers.empty:
        return 1
    row = league_free_transfers[league_free_transfers["manager_id"] == team_id]
    return int(row["free_transfers"].iloc[0]) if not row.empty else 1


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
    no_hit_weeks: frozenset[int] = frozenset(),
    bench_weight: float = BENCH_VALUE_WEIGHT,
    cost_tiebreak: float = 0.0,
    force_in: frozenset[int] = frozenset(),
    force_out: frozenset[int] = frozenset(),
    force_captain: int | None = None,
    ft_value: float = 0.0,
    risk_sd: dict[tuple[int, int], float] | None = None,
    risk_weight: float = 0.0,
    fixed_moves: dict[int, tuple[frozenset[int], frozenset[int]]] | None = None,
) -> dict[str, Any]:
    """One MILP covering every gameweek in ``weeks`` at once. A single
    gameweek is just ``weeks`` of length 1 -- there's no separate
    single-week code path.

    ``held_weeks``: gameweeks where transfers are forced to zero (used by
    ``recommend_transfers`` to test "hold this week, keep flexibility to
    move later" against the free solve).

    ``no_hit_weeks``: gameweeks where transfers are allowed but capped at
    the free transfers available -- no points hit permitted. Step 12's
    hit-justification policy (``justify_hits``) re-solves with this set
    whenever the unconstrained solve's hit isn't earned by necessity, so
    the model still finds the best *free* transfer (or correctly rolls)
    instead of just forcing a hold.

    ``bench_weight`` / ``cost_tiebreak`` (hindsight solves in
    fpl/hindsight.py): with real points the bench doesn't count, so those
    solves pass ``bench_weight=0`` and a tiny ``cost_tiebreak`` per 0.1m of
    squad cost, which picks the cheapest bench among equally good teams.
    Defaults leave the forward-looking plan unchanged.

    ``force_in`` / ``force_out`` / ``force_captain`` (the app chat's
    "what if" tool, fpl/chat_tools.py): pin those players into / out of the
    first week's squad and the first week's captain; everything else is
    solved as usual. Empty by default.

    ``ft_value`` (Step 13 rule 1): points credited per free transfer
    carried out of the horizon (``ft_end``, capped at MAX_FREE_TRANSFERS).
    ``risk_sd`` / ``risk_weight`` (rule 3): each starter's value is reduced
    by ``risk_weight`` x his points SD that week; the captain's extra
    multiplier is not, so captaincy stays a pure xP choice. Both default
    to off.

    ``fixed_moves`` ({gameweek: (ins, outs)}, ``gate_planned_weeks``): those
    weeks' transfers are pinned to exactly these players (empty = hold).
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
        bench_value = bench_weight * pulp.lpSum(weekly_value[(p, g)] * squad[(p, g)] for p in pool_ids)
        value_terms.append(discount[g] * (week_value + bench_value))
        if cost_tiebreak:
            value_terms.append(-cost_tiebreak * pulp.lpSum(pool_info[p]["now_cost"] * squad[(p, g)] for p in pool_ids))
        if risk_weight and risk_sd:
            value_terms.append(
                -discount[g] * risk_weight * pulp.lpSum(risk_sd.get((p, g), 0.0) * xi[(p, g)] for p in pool_ids)
            )
    ft_end = None
    if ft_value:
        ft_end = m.add_variable("ft_end", lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
        value_terms.append(ft_value * ft_end)
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
        if g in no_hit_weeks:
            m += hits[g] <= 0

        m += bank[g] == prev_bank + pulp.lpSum(
            sell_price[p] * tout[(p, g)] - pool_info[p]["now_cost"] * tin[(p, g)] for p in pool_ids
        )

        if i < len(weeks) - 1:
            nxt = weeks[i + 1]
            m += ft[nxt] <= prev_ft - transfers_used + hits[g] + 1
        elif ft_end is not None:
            m += ft_end <= prev_ft - transfers_used + hits[g] + 1

    for g, (ins, outs) in (fixed_moves or {}).items():
        if g not in weeks:
            continue
        for p in pool_ids:
            m += tin[(p, g)] == (1 if p in ins else 0)
            m += tout[(p, g)] == (1 if p in outs else 0)

    g0 = weeks[0]
    for p in force_in:
        m += squad[(p, g0)] == 1
    for p in force_out:
        m += squad[(p, g0)] == 0
    if force_captain is not None:
        m += cap[(force_captain, g0)] == 1

    stats = m.solve(pulp.HiGHS(msg=False))

    weekly_results = []
    # Free transfers available at the start of each week, from the plan's
    # own free-transfer constraint (ft[next] <= prev - used + hits + 1,
    # capped at MAX_FREE_TRANSFERS) taken at its tight value with the
    # solved transfers and hits. ft[g] itself is only an upper bound in the
    # MILP, so its solved value can sit lower when it doesn't bind; this is
    # what the plan actually has. Saved per week so the app's Planner reads
    # it instead of recalculating.
    available_ft = free_transfers_now
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
                "free_transfers": available_ft,
            }
        )
        week_hits = int(round(hits[g].varValue or 0))
        available_ft = min(MAX_FREE_TRANSFERS, max(available_ft - len(transfers_in) + week_hits + 1, 1))

    return {
        "status": stats.status.name,
        "objective": pulp.value(m.objective),
        "weeks": weekly_results,
    }


# ---------------------------------------------------------------------------
# transfer threshold (extension 2)
# ---------------------------------------------------------------------------


def is_flagged_unavailable(player_row: dict[str, Any]) -> bool:
    """Step 12: a sold player justifies a hit only if he's genuinely
    unavailable -- status ``i``/``s``/``u`` (injured/suspended/unavailable),
    or ``d`` (doubtful) with his own ``chance_of_playing_next_round`` down
    at or below ``HIT_JUSTIFY_DOUBTFUL_CHANCE``. A plain doubtful at, say,
    75% is a risk to manage, not a forced sale."""
    status = player_row.get("status")
    if status in HIT_JUSTIFY_STATUSES:
        return True
    if status == "d":
        chance = player_row.get("chance_of_playing_next_round")
        return chance is not None and chance <= HIT_JUSTIFY_DOUBTFUL_CHANCE
    return False


def justify_hits(
    plan: dict[str, Any],
    held: dict[str, Any],
    pool_info: dict[int, dict[str, Any]],
    raw_weekly_xp: dict[tuple[int, int], float],
    weeks: list[int],
) -> tuple[bool, list[str]]:
    """Step 12 rule 3: a hit is allowed only if it's earned by necessity,
    never by pure value.

    1. Every "paid" transfer (beyond the free transfers available) must be
       selling a player who is both a regular starter in the *held* XI
       (never a bench player -- a flagged 4th/5th-choice pick never
       justifies a hit) and genuinely unavailable (``is_flagged_
       unavailable``). Free transfers aren't restricted this way -- you get
       those regardless of necessity.
    2. Even then, the swap(s) have to pay for themselves: the combined XI
       output over the next ``HIT_JUSTIFY_WINDOW`` gameweeks (raw,
       undiscounted xP -- a plain points comparison) must beat holding (and
       using the bench to cover the gap) by more than ``HIT_COST`` points
       *per hit taken*.

    Returns ``(justified, reasons)`` -- ``reasons`` names every flagged
    starter sold and the net gain, for the site's hit warning.
    """
    week0 = plan["weeks"][0]
    hits = week0["hits"]
    if hits <= 0:
        return True, []

    held_xi_ids = held["weeks"][0]["xi_ids"]
    flagged_outs = [
        p for p in week0["transfers_out"]
        if p in held_xi_ids and is_flagged_unavailable(pool_info.get(p, {}))
    ]
    # Every paid transfer (there are ``hits`` of them) must be covered by
    # a flagged, nailed-starter sale -- the free transfer(s) can still be
    # spent on a pure value swap, only the *extra* paid one(s) can't.
    if len(flagged_outs) < hits:
        return False, []

    window = [g for g in weeks[:HIT_JUSTIFY_WINDOW]]

    def xi_points(candidate: dict[str, Any]) -> float:
        total = 0.0
        for wk in candidate["weeks"]:
            if wk["gameweek"] not in window:
                continue
            total += sum(raw_weekly_xp.get((p, wk["gameweek"]), 0.0) for p in wk["xi_ids"])
            if wk["captain_id"] is not None:
                total += raw_weekly_xp.get((wk["captain_id"], wk["gameweek"]), 0.0)
        return total

    net_gain = xi_points(plan) - xi_points(held)
    required = HIT_COST * hits
    if net_gain <= required:
        return False, []

    reasons = [
        f"{pool_info[p]['web_name']} ({pool_info[p].get('status')}"
        + (
            f", {pool_info[p]['chance_of_playing_next_round']:.0f}% chance of playing"
            if pool_info[p].get("chance_of_playing_next_round") is not None
            else ""
        )
        + f") -- replacing him gains {net_gain:.1f} pts over the next {len(window)} GWs, "
          f"beating the {required}-point hit cost"
        for p in flagged_outs
    ]
    return True, reasons


# Default for recommend_transfers' ``uncertainty_k``: "use the switch above".
FROM_CONFIG = object()


def residual_sd(position: str, xp: float) -> float:
    """Typical error of one player-week's projection (RESIDUAL_SD) for his
    position and projected-points band."""
    band = sum(1 for edge in XP_BANDS if xp >= edge)
    return RESIDUAL_SD.get(position, RESIDUAL_SD["MID"])[band]


def transfer_margin(
    week0: dict[str, Any],
    weeks: list[int],
    discount: dict[int, float],
    raw_weekly_xp: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    uncertainty_k: float | None,
) -> float:
    """Step 13 rule 2: how much a week-0 move must gain over holding.
    ``uncertainty_k`` None -> the legacy flat TRANSFER_THRESHOLD_XP;
    otherwise k x the standard error of (players in - players out) over
    the horizon, treating each player-week's error as independent with
    the RESIDUAL_SD of his position and band, weighted like the objective
    (``discount``)."""
    if uncertainty_k is None:
        return TRANSFER_THRESHOLD_XP
    var = 0.0
    for p in week0["transfers_in"] | week0["transfers_out"]:
        pos = pool_info[p]["position"]
        for g in weeks:
            var += (discount[g] * residual_sd(pos, raw_weekly_xp.get((p, g), 0.0))) ** 2
    return uncertainty_k * var**0.5


def outcome_sd_by_week(xp_rows: pd.DataFrame, scoring: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Step 13 rule 3: SD of each player's points in each gameweek, from
    ``fpl.xp``'s per-fixture components treated as independent counts --
    appearance (0/1/2), clean sheet (Bernoulli), goals and assists
    (Poisson x points each), defensive contribution (Bernoulli, 2 pts),
    bonus (~1-3 pts), saves, cards and goals conceded (small counts). A
    defender relying on rare goals has a wide spread; one whose xP is
    appearances, clean sheets and defensive contributions has a narrow
    one. Double gameweeks add the variances."""
    if xp_rows.empty:
        return {}
    rules = scoring.set_index("position")
    df = xp_rows
    pos = df["position"]
    pts_goal = pos.map(rules["points_per_goal"]).astype(float)
    pts_assist = pos.map(rules["points_per_assist"]).astype(float)
    pts_cs = pos.map(rules["points_per_clean_sheet"]).astype(float)
    app = df["appearance_pts"]
    var = (
        (df["p_play"] - df["p_60"]) * 1.0 + df["p_60"] * 4.0 - app**2  # appearance: 1 pt or 2 pts
        + df["expected_goal_pts"] * pts_goal                            # Poisson goals
        + df["expected_assist_pts"] * pts_assist                        # Poisson assists
        + (df["clean_sheet_pts"] * pts_cs - df["clean_sheet_pts"] ** 2).clip(lower=0)
        + (df["defcon_pts"] * 2.0 - df["defcon_pts"] ** 2).clip(lower=0)
        + (df["bonus_pts"] * 2.0 - df["bonus_pts"] ** 2).clip(lower=0)
        + df["saves_pts"] + df["conceded_pts"].abs() + df["card_pts"].abs()
    ).clip(lower=0)
    total = var.groupby([df["player_id"], df["event"]]).sum()
    return {(int(p), int(g)): float(v**0.5) for (p, g), v in total.items()}


def no_odds_weeks(xp_rows: pd.DataFrame) -> set[tuple[int, int]]:
    """Player-weeks with at least one fixture whose lambdas came from Elo +
    xG only (``source != "odds"`` in xp.parquet)."""
    if xp_rows.empty or "source" not in xp_rows.columns:
        return set()
    flagged = xp_rows[xp_rows["source"] != "odds"]
    return {(int(p), int(g)) for p, g in zip(flagged["player_id"], flagged["event"])}


def apply_no_odds_discount(
    weekly_value: dict[tuple[int, int], float], uncovered: set[tuple[int, int]], factor: float
) -> dict[tuple[int, int], float]:
    """Step 13 rule 4b: weight a player-week without odds coverage by
    ``factor`` in the objective (shown xP is unchanged)."""
    if factor == 1.0 or not uncovered:
        return weekly_value
    return {k: v * factor if k in uncovered else v for k, v in weekly_value.items()}


def rule_solve_kwargs(mode: str, risk_sd: dict[tuple[int, int], float] | None = None) -> dict[str, Any]:
    """The Step 13 switches as solve_horizon / recommend_transfers keyword
    arguments, so every caller that reproduces the recommendation (main,
    chat what-ifs, the season simulation, check_transfer_policy) applies
    exactly the same rules."""
    use_risk = USE_RELIABILITY and mode in RELIABILITY_MODES and bool(risk_sd)
    return {
        "ft_value": FT_VALUE_XP if USE_FT_VALUE else 0.0,
        "risk_sd": risk_sd if use_risk else None,
        "risk_weight": RELIABILITY_WEIGHT if use_risk else 0.0,
    }


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
    raw_weekly_xp: dict[tuple[int, int], float] | None = None,
    ft_value: float = 0.0,
    risk_sd: dict[tuple[int, int], float] | None = None,
    risk_weight: float = 0.0,
    uncertainty_k: float | None | object = FROM_CONFIG,
    no_hit_weeks: frozenset[int] = frozenset(),
    gate_later_weeks: bool = True,
) -> tuple[dict[str, Any], float, bool, list[str]]:
    """This week's move (``_recommend_week0``: the threshold and hit rules
    below), then -- unless ``gate_later_weeks`` is False (the season
    replay, which only ever plays week 0) -- every later planned week is
    put through the same bar by ``gate_planned_weeks``, so the plan never
    shows a move that would be held once its week arrives. Week 0's
    recommendation is decided first and is never changed by the gate."""
    if uncertainty_k is FROM_CONFIG:
        uncertainty_k = UNCERTAINTY_K if USE_UNCERTAINTY_MARGIN else None
    rules = {"ft_value": ft_value, "risk_sd": risk_sd, "risk_weight": risk_weight}
    args = (pool_ids, weekly_value, pool_info, my_squad_ids, sell_price, initial_bank, free_transfers_now, weeks, discount)
    plan, gain, recommended, hit_reasons = _recommend_week0(
        baseline, *args, raw_weekly_xp=raw_weekly_xp, uncertainty_k=uncertainty_k, no_hit_weeks=no_hit_weeks, **rules
    )
    if gate_later_weeks and len(weeks) > 1:
        moved_before = bool(plan["weeks"][0]["transfers_in"])
        plan = gate_planned_weeks(
            plan, *args, raw_weekly_xp=raw_weekly_xp, uncertainty_k=uncertainty_k, no_hit_weeks=no_hit_weeks, **rules
        )
        if moved_before and not plan["weeks"][0]["transfers_in"]:  # every week-0 move failed the per-move bar
            recommended, hit_reasons = False, []
        elif plan["weeks"][0]["hits"] == 0:
            hit_reasons = []
    return plan, gain, recommended, hit_reasons


def pair_moves(ins: set[int], outs: set[int], pool_info: dict[int, dict[str, Any]]) -> list[tuple[int, int]]:
    """A week's transfers as (out, in) swaps. FPL's 2/5/5/3 quotas force
    every sale to be replaced within its own position, so the pairing is
    within position (most expensive with most expensive when a week has
    two swaps in one position)."""
    pairs = []
    for pos in POSITION_QUOTAS:
        o = sorted((p for p in outs if pool_info[p]["position"] == pos), key=lambda p: -pool_info[p]["now_cost"])
        n = sorted((p for p in ins if pool_info[p]["position"] == pos), key=lambda p: -pool_info[p]["now_cost"])
        pairs.extend(zip(o, n))
    return pairs


def _filter_week_moves(
    fixed: dict[int, tuple[frozenset[int], frozenset[int]]],
    g: int,
    i: int,
    moves: list[tuple[int, int]],
    full: dict[str, Any],
    args: tuple,
    weeks: list[int],
    discount: dict[int, float],
    raw: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    uncertainty_k: float | None,
    solve: Any,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """Step 13 per-move bar. With every earlier week pinned and later weeks
    free: each move's own gain = the plan with all of this week's moves
    minus the same plan without that one move (its transfer banked, so the
    saved-transfer value counts). If any move is below ``transfer_margin``,
    drop the weakest, re-test the rest, repeat. A move that can't be
    removed on its own (its sale funds another buy, or the club cap needs
    it) is judged as a bundle against holding the whole week. Returns the
    kept moves and one record per move considered."""
    pin = lambda ms: {**fixed, g: (frozenset(m[1] for m in ms), frozenset(m[0] for m in ms))}  # noqa: E731
    hold_obj = None
    records: dict[tuple[int, int], dict[str, Any]] = {}
    kept = list(moves)
    cur = full
    while kept:
        gains = {}
        for m in kept:
            rest = [x for x in kept if x != m]
            alt = solve(pin(rest))
            bundled = alt["status"] != "Optimal"
            if bundled:
                if hold_obj is None:
                    hold_obj = solve(pin([]))["objective"] or 0.0
                other = hold_obj
            else:
                other = alt["objective"] or 0.0
            required = transfer_margin(
                {"transfers_in": {m[1]}, "transfers_out": {m[0]}}, weeks[i:], discount, raw, pool_info, uncertainty_k
            )
            gains[m] = ((cur["objective"] or 0.0) - other, required, bundled)
        failing = [m for m in kept if gains[m][0] < gains[m][1] - 1e-9]
        for m in kept:
            gain, required, bundled = gains[m]
            records[m] = {"out": m[0], "in": m[1], "gain": gain, "required": required,
                          "passed": m not in failing, "bundled": bundled}
        if not failing:
            break
        worst = min(failing, key=lambda m: gains[m][0] - gains[m][1])
        kept.remove(worst)
        records[worst]["passed"] = False
        if kept:
            cur = solve(pin(kept))
    return kept, [records[m] for m in moves if m in records]


def gate_planned_weeks(
    plan: dict[str, Any],
    pool_ids: list[int],
    weekly_value: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    my_squad_ids: set[int],
    sell_price: dict[int, int],
    initial_bank: int,
    free_transfers_now: int,
    weeks: list[int],
    discount: dict[int, float],
    raw_weekly_xp: dict[tuple[int, int], float] | None = None,
    uncertainty_k: float | None = None,
    no_hit_weeks: frozenset[int] = frozenset(),
    **solve_kwargs: Any,
) -> dict[str, Any]:
    """Every move of every week of ``plan`` through the same bar, in order.

    Week 0: ``_recommend_week0`` already tested the week as a whole
    (threshold + hit rules); with two or more moves, each must now also
    clear the bar by itself (``_filter_week_moves``), so a strong move
    can't carry a weak one. Then for each later week g, with every earlier
    week pinned: solve with g free (later weeks free too, exactly like
    week 0's test), and put each of g's moves through the per-move bar.
    The bar is ``transfer_margin`` (1.75 xP today); the saved-transfer
    value counts automatically, since the comparison plan keeps that free
    transfer and both objectives include ``ft_value``. A dropped move's
    transfer is banked, so later weeks' free-transfer counts go up. Later
    weeks may not take hits: whether a hit is justified (a flagged starter)
    can only be judged when that week arrives.

    Each week gets ``gate``: {"gain", "required", "passed", "moves":
    [per-move records], "best_in", "best_out"} -- ``best_*`` is the
    strongest rejected move of a week that ends up holding; ``gate`` is
    None when the solve wanted no move that week."""
    args = (pool_ids, weekly_value, pool_info, my_squad_ids, sell_price, initial_bank, free_transfers_now, weeks, discount)
    later_no_hits = frozenset(no_hit_weeks) | frozenset(weeks[1:])
    raw = raw_weekly_xp or weekly_value

    def solve(fixed: dict[int, tuple[frozenset[int], frozenset[int]]], **kw: Any) -> dict[str, Any]:
        return solve_horizon(*args, fixed_moves=fixed, no_hit_weeks=later_no_hits, **solve_kwargs, **kw)

    def week_gate(fixed, g, i, wk, full, base_gate=None):
        moves = pair_moves(set(wk["transfers_in"]), set(wk["transfers_out"]), pool_info)
        if len(moves) == 1 and base_gate is not None:  # week 0, one move: its own gain is the week's gain
            kept = moves if base_gate["passed"] else []
            records = [{"out": moves[0][0], "in": moves[0][1], "gain": base_gate["gain"],
                        "required": base_gate["required"], "passed": base_gate["passed"], "bundled": False}]
        else:
            kept, records = _filter_week_moves(fixed, g, i, moves, full, args, weeks, discount, raw,
                                               pool_info, uncertainty_k, solve)
        rejected = [r for r in records if not r["passed"]]
        best = max(rejected, key=lambda r: r["gain"]) if rejected and not kept else None
        # week-level gain: week 0 keeps its whole-week gain vs holding (the
        # verdict shows it); a later week shows its weakest kept move's own
        # gain, or the best rejected move's when it holds
        if base_gate is not None:
            week_gain = base_gate["gain"]
        elif kept:
            week_gain = min(r["gain"] for r in records if r["passed"])
        else:
            week_gain = best["gain"]
        gate = {
            "gain": week_gain, "required": records[0]["required"] if records else None,
            "passed": bool(kept), "moves": records,
            "best_in": [best["in"]] if best else [], "best_out": [best["out"]] if best else [],
        }
        return kept, gate

    gates: dict[int, dict[str, Any] | None] = {}
    w0 = plan["weeks"][0]
    g0 = weeks[0]
    base_gate = w0.get("gate")
    fixed: dict[int, tuple[frozenset[int], frozenset[int]]] = {g0: (frozenset(w0["transfers_in"]), frozenset(w0["transfers_out"]))}
    if w0["transfers_in"] or w0["transfers_out"]:
        kept, gate = week_gate({}, g0, 0, w0, plan, base_gate=base_gate or {"gain": 0.0, "required": TRANSFER_THRESHOLD_XP, "passed": True})
        fixed[g0] = (frozenset(m[1] for m in kept), frozenset(m[0] for m in kept))
        gates[g0] = gate
    else:
        gates[g0] = base_gate
    for i, g in enumerate(weeks[1:], start=1):
        free = solve(fixed)
        wk = free["weeks"][i]
        if not wk["transfers_in"] and not wk["transfers_out"]:
            fixed[g] = (frozenset(), frozenset())
            gates[g] = None
            continue
        kept, gates[g] = week_gate(fixed, g, i, wk, free)
        fixed[g] = (frozenset(m[1] for m in kept), frozenset(m[0] for m in kept))
    gated = solve(fixed)
    if gated["status"] != "Optimal":  # can't happen (every pinned week was feasible), but never lose the plan
        return plan
    for wk in gated["weeks"]:
        wk["gate"] = gates.get(wk["gameweek"])
    gated["required_gain"] = plan.get("required_gain")
    gated["objective_ungated"] = plan.get("objective")
    return gated


def _recommend_week0(
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
    raw_weekly_xp: dict[tuple[int, int], float] | None = None,
    ft_value: float = 0.0,
    risk_sd: dict[tuple[int, int], float] | None = None,
    risk_weight: float = 0.0,
    uncertainty_k: float | None | object = FROM_CONFIG,
    no_hit_weeks: frozenset[int] = frozenset(),
) -> tuple[dict[str, Any], float, bool, list[str]]:
    """(Called by ``recommend_transfers``.) Only recommend this week's transfer if it beats holding (while still
    letting the model move in a *later* week instead) by at least
    ``TRANSFER_THRESHOLD_XP`` over the horizon -- "spot the right week to
    move" means testing whether *this* week is that week, not whether
    moving beats never moving at all.

    Step 12 rule 3: before that threshold check, a baseline that wants a
    hit is first checked against ``justify_hits``. An unjustified hit gets
    re-solved with that week's hits forced to zero (``no_hit_weeks``) --
    the model still finds its best *free* transfer, it just can't buy its
    way past the necessity bar -- so "at most your free transfers, unless
    a flagged nailed starter justifies more" holds regardless of what the
    unconstrained solve would have picked.

    Step 13: ``ft_value`` / ``risk_sd`` / ``risk_weight`` are passed to
    every re-solve (give the baseline the same ones, see
    ``rule_solve_kwargs``), and the bar the gain must clear is
    ``transfer_margin`` -- the uncertainty margin when rule 2 is on, the
    legacy threshold otherwise. ``uncertainty_k`` defaults to the
    configured switch; the season replay passes its own. ``no_hit_weeks``
    applies to every re-solve too (the replay forbids hits everywhere).
    """
    if uncertainty_k is FROM_CONFIG:
        uncertainty_k = UNCERTAINTY_K if USE_UNCERTAINTY_MARGIN else None
    rules = {"ft_value": ft_value, "risk_sd": risk_sd, "risk_weight": risk_weight}
    week0 = weeks[0]
    if not baseline["weeks"][0]["transfers_in"]:
        return baseline, 0.0, False, []

    held = solve_horizon(
        pool_ids, weekly_value, pool_info, my_squad_ids, sell_price,
        initial_bank, free_transfers_now, weeks, discount, held_weeks=frozenset({week0}),
        no_hit_weeks=no_hit_weeks, **rules,
    )

    plan = baseline
    hit_reasons: list[str] = []
    if plan["weeks"][0]["hits"] > 0 and raw_weekly_xp is not None:
        justified, hit_reasons = justify_hits(plan, held, pool_info, raw_weekly_xp, weeks)
        if not justified:
            plan = solve_horizon(
                pool_ids, weekly_value, pool_info, my_squad_ids, sell_price,
                initial_bank, free_transfers_now, weeks, discount, no_hit_weeks=no_hit_weeks | {week0},
                **rules,
            )
            hit_reasons = []

    if not plan["weeks"][0]["transfers_in"]:
        return plan, 0.0, False, hit_reasons

    gain = (plan["objective"] or 0.0) - (held["objective"] or 0.0)
    required = transfer_margin(
        plan["weeks"][0], weeks, discount, raw_weekly_xp or weekly_value, pool_info, uncertainty_k
    )
    plan["required_gain"] = required
    held["required_gain"] = required
    gate = {"gain": gain, "required": required, "passed": gain >= required,
            "best_in": sorted(plan["weeks"][0]["transfers_in"]), "best_out": sorted(plan["weeks"][0]["transfers_out"])}
    plan["weeks"][0]["gate"] = gate
    held["weeks"][0]["gate"] = gate
    if gain < required:
        return held, gain, False, []
    return plan, gain, True, hit_reasons


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
    raw_league_eo: dict[int, float] | None = None,
) -> pd.DataFrame:
    """The XI ranked by xP for next gameweek. Chase mode adds a bonus for
    differentials (low league captaincy); protect mode flags the league's
    most-backed captain so you can match it and close off that swing.
    ``league_eo`` drives the scoring (Step 8's rank-weighted EO when
    available, preferring close rivals over the whole league); ``raw_eo``
    is the plain league-wide EO, shown alongside for comparison only."""
    g = week0["gameweek"]
    raw_league_eo = raw_league_eo or league_eo
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
                "raw_league_eo": round(raw_league_eo.get(p, 0.0), 2),
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


def build_captaincy_risk(
    week0: dict[str, Any],
    raw_weekly_xp: dict[tuple[int, int], float],
    pool_info: dict[int, dict[str, Any]],
    league_picks: pd.DataFrame,
    league_history: pd.DataFrame,
    league_managers: pd.DataFrame,
    team_id: int,
    gameweek: int,
    top_n: int = 3,
) -> pd.DataFrame:
    """Step D.5: for every XI player (a candidate captain), the expected
    point swing against each of the ``top_n`` highest-ranked *other*
    managers if he scores his raw xP -- ``(your_multiplier -
    rival_multiplier) * xp``. A popular captain who several top rivals
    already have barely closes the gap (their multiplier is also 2, so the
    swing is near zero or negative against them); a differential swings
    the full value. This is what turns "5/9 rivals captained Haaland" into
    a number: captaining him yourself mostly cancels out against whichever
    of those 5 are near the top, while captaining someone none of them has
    gains the full ``xp`` against every one of them.
    """
    if league_history.empty or league_picks.empty:
        return pd.DataFrame()

    totals = league_history[league_history["gameweek"] == gameweek].set_index("manager_id")["total_points"]
    rival_totals = totals.drop(index=team_id, errors="ignore").sort_values(ascending=False)
    top_rivals = rival_totals.index[:top_n].tolist()
    if not top_rivals:
        return pd.DataFrame()

    names = league_managers.set_index("manager_id")["entry_name"] if not league_managers.empty else pd.Series(dtype=str)
    rival_labels = {r: names.get(r, f"manager {r}") for r in top_rivals}

    gw_picks = league_picks[league_picks["gameweek"] == gameweek]
    rival_mult = (
        gw_picks[gw_picks["manager_id"].isin(top_rivals)]
        .pivot_table(index="player_id", columns="manager_id", values="multiplier", fill_value=0)
    )

    g = week0["gameweek"]
    rows = []
    for p in week0["xi_ids"]:
        xp = raw_weekly_xp.get((p, g), 0.0)
        your_multiplier = 2 if p == week0["captain_id"] else 1
        row = {
            "player_id": p,
            "web_name": pool_info[p]["web_name"],
            "xp": round(xp, 2),
            "is_optimiser_captain": p == week0["captain_id"],
        }
        total_swing = 0.0
        for rival in top_rivals:
            rival_multiplier = float(rival_mult.loc[p, rival]) if p in rival_mult.index and rival in rival_mult.columns else 0.0
            swing = (your_multiplier - rival_multiplier) * xp
            row[f"swing_vs_{rival_labels[rival]}"] = round(swing, 2)
            total_swing += swing
        row["total_swing_vs_top"] = round(total_swing, 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_swing_vs_top", ascending=False).reset_index(drop=True)


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
    plan: dict[str, Any],
    pool_info: dict[int, dict[str, Any]],
    weeks: list[int],
    raw_weekly_xp: dict[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """Step 12 additions: ``is_vice_captain`` (the highest-xP XI player who
    isn't the captain -- the same "next man up" tie-break the real game
    uses) and ``bench_order`` (the reserve keeper is always first/unordered,
    then outfield subs ranked by xP, highest first -- the autosub priority
    the site's "Substitutes" strip labels "GKP, 1., 2., 3." the same way
    the FPL site itself does)."""
    raw_weekly_xp = raw_weekly_xp or {}
    rows = []
    for wk in plan["weeks"]:
        g = wk["gameweek"]
        bench_ids = sorted(wk["squad_ids"] - wk["xi_ids"], key=lambda p: -raw_weekly_xp.get((p, g), 0.0))
        bench_gk = [p for p in bench_ids if pool_info[p]["position"] == "GKP"]
        bench_outfield = [p for p in bench_ids if pool_info[p]["position"] != "GKP"]
        bench_order = {p: 0 for p in bench_gk}
        bench_order.update({p: i + 1 for i, p in enumerate(bench_outfield)})

        non_captain_xi = [p for p in wk["xi_ids"] if p != wk["captain_id"]]
        vice_captain_id = (
            max(non_captain_xi, key=lambda p: raw_weekly_xp.get((p, g), 0.0)) if non_captain_xi else None
        )

        for p in wk["squad_ids"]:
            role = "captain" if p == wk["captain_id"] else ("xi" if p in wk["xi_ids"] else "bench")
            rows.append(
                {
                    "gameweek": g, "player_id": p, "web_name": pool_info[p]["web_name"],
                    "position": pool_info[p]["position"], "role": role,
                    "transferred_in": p in wk["transfers_in"],
                    "is_vice_captain": p == vice_captain_id,
                    "bench_order": bench_order.get(p),
                }
            )
    return pd.DataFrame(rows)


def build_transfers_table(
    plan: dict[str, Any],
    pool_info: dict[int, dict[str, Any]],
    gain: float,
    recommended: bool,
    hit_reasons: list[str] | None = None,
) -> pd.DataFrame:
    """The full horizon's transfer plan (the "6-week transfer plan" panel),
    one row per planned move in every week -- not just this week's. Every
    week's move has been checked against the same bar as week 0
    (``gate_planned_weeks``): ``horizon_gain_xp`` / ``required_gain_xp`` /
    ``recommended`` per week, and a held week's ``held_best_out`` /
    ``held_best_in`` name the move it turned down. Later weeks can still
    shift as news lands closer to their own deadlines.
    ``hit_reason`` (Step 12): the hit-justification explanation for
    gameweek 0's hit(s) -- only ever populated when ``recommend_transfers``
    actually recommends one, since an unjustified hit is re-solved away
    before this table is built (see ``justify_hits``)."""
    hit_reason_text = "; ".join(hit_reasons) if hit_reasons else None
    names = lambda ids: ", ".join(pool_info[p]["web_name"] for p in ids) or None  # noqa: E731
    rows = []
    for i, wk in enumerate(plan["weeks"]):
        pairs = pair_moves(set(wk["transfers_in"]), set(wk["transfers_out"]), pool_info)
        outs, ins = [o for o, _ in pairs], [n for _, n in pairs]
        # Every week has been through the same bar (gate_planned_weeks):
        # its gain vs holding and the bar it needed; a held week also
        # names the best move it turned down, for the Planner's reason.
        g8 = wk.get("gate")
        week_gain = round(gain, 2) if i == 0 else (round(g8["gain"], 2) if g8 else None)
        week_rec = recommended if i == 0 else (bool(g8["passed"]) if g8 else False)
        rejected = g8 is not None and not g8["passed"]
        own = {(r["out"], r["in"]): r for r in (g8 or {}).get("moves") or []}
        for j in range(max(len(ins), len(outs), 1)):
            mv = own.get((outs[j], ins[j])) if j < len(outs) and j < len(ins) else None
            rows.append(
                {
                    "gameweek": wk["gameweek"],
                    "player_out": pool_info[outs[j]]["web_name"] if j < len(outs) else None,
                    "player_in": pool_info[ins[j]]["web_name"] if j < len(ins) else None,
                    "hits_taken": wk["hits"] if j == 0 else 0,
                    "horizon_gain_xp": week_gain,
                    # this move's own gain over the same week without it (the per-move bar)
                    "move_gain_xp": round(mv["gain"], 2) if mv else (round(g8["gain"], 2) if rejected and j == 0 else None),
                    "required_gain_xp": round(g8["required"], 2) if g8 and g8.get("required") is not None else None,
                    "recommended": week_rec,
                    "held_best_out": names(g8["best_out"]) if rejected and j == 0 else None,
                    "held_best_in": names(g8["best_in"]) if rejected and j == 0 else None,
                    "hit_reason": hit_reason_text if (i == 0 and j == 0) else None,
                    # free transfers available at the start of this week, per the plan
                    "free_transfers": wk.get("free_transfers"),
                }
            )
    return pd.DataFrame(rows)


def load_xp_rows(weeks: list[int]) -> pd.DataFrame:
    """Per-fixture xP components (xp.parquet) for the horizon -- the
    reliability SDs and the odds-coverage flags (Step 13) need them."""
    path = TABLES_DIR / "xp.parquet"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_parquet(path)
    return rows[rows["event"].isin(weeks)]


def risk_sd_for(xp_rows: pd.DataFrame) -> dict[tuple[int, int], float]:
    path = TABLES_DIR / "scoring_rules.parquet"
    if xp_rows.empty or not path.exists():
        return {}
    return outcome_sd_by_week(xp_rows, pd.read_parquet(path))


def rule_inputs(
    raw_weekly_xp: dict[tuple[int, int], float], league_eo: dict[int, float], mode: str, weeks: list[int]
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """The objective values and Step 13 solver arguments every caller that
    reproduces the recommendation uses: strategy-adjusted xP with rule 4's
    no-odds weighting, plus ``rule_solve_kwargs`` (rules 1 and 3)."""
    xp_rows = load_xp_rows(weeks)
    value = apply_no_odds_discount(
        strategy_adjusted(raw_weekly_xp, league_eo, mode),
        no_odds_weeks(xp_rows), NO_ODDS_DISCOUNT if USE_NO_ODDS_DISCOUNT else 1.0,
    )
    return value, rule_solve_kwargs(mode, risk_sd_for(xp_rows) if USE_RELIABILITY else None)


def describe_rules(mode: str) -> str:
    parts = [
        f"saved-FT value {FT_VALUE_XP:g}" if USE_FT_VALUE else "saved-FT value off",
        f"uncertainty margin k={UNCERTAINTY_K:g}" if USE_UNCERTAINTY_MARGIN
        else f"uncertainty margin off (flat {TRANSFER_THRESHOLD_XP:g} xP threshold)",
        (f"reliability {RELIABILITY_WEIGHT:g} x SD" if mode in RELIABILITY_MODES else "reliability n/a in chasing")
        if USE_RELIABILITY else "reliability off",
        f"weekly discount {WEEKLY_DISCOUNT:g}"
        + (f", no-odds weeks x{NO_ODDS_DISCOUNT:g}" if USE_NO_ODDS_DISCOUNT else ", no-odds discount off"),
    ]
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gameweek", type=int, default=None,
        help="check legality/reasoning as of this gameweek instead of the latest one",
    )
    args = parser.parse_args()

    config = load_config()
    tables = load_tables()
    players, xp_by_gw = tables["players"], tables["xp_by_gameweek"]
    league_picks, league_history = tables["league_picks"], tables["league_history"]
    league_managers = tables.get("league_managers", pd.DataFrame())

    team_id = resolve_team_id(config, league_managers)
    if team_id is None:
        print("No team_id set in config.json, and no manual snapshot to derive it from; nothing to optimise.")
        return
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

    # Step D: prefer the rank-weighted EO (close rivals dominate) over the
    # plain league-wide EO when fpl.league has computed it; the raw column
    # is kept too, purely for display in the captain board/risk tables.
    league_eo: dict[int, float] = {}
    raw_league_eo: dict[int, float] = {}
    swings = tables["league_swings"]
    if not swings.empty:
        raw_league_eo = swings.set_index("player_id")["league_eo"].to_dict()
        if "weighted_league_eo" in swings.columns:
            effective = swings["weighted_league_eo"].fillna(swings["league_eo"])
            league_eo = pd.Series(effective.values, index=swings["player_id"]).to_dict()
        else:
            league_eo = dict(raw_league_eo)

    mode = config.get("strategy")
    if mode not in {"chasing", "protecting", "balanced"}:
        mode = tables["league_strategy"]["mode"].iloc[0] if not tables["league_strategy"].empty else "balanced"

    sell_price_owned = estimate_selling_prices(my_squad_ids, players, tables["my_transfers"])
    avoid_ids = {int(x) for x in (config.get("avoid") or []) if isinstance(x, int)}
    pool_ids = build_candidate_pool(players, xp_by_gw, my_squad_ids, avoid_ids)

    pool_source = players.set_index("id")[
        ["web_name", "position", "team", "now_cost", "status", "chance_of_playing_next_round", "news"]
    ].copy()
    # NaN -> None for the hit-justification check (``is_flagged_unavailable``
    # compares ``chance_of_playing_next_round`` with ``<=``, which silently
    # (and correctly) treats real NaN as "not flagged" -- but keeping it as
    # an actual ``None`` makes that explicit rather than relying on NaN's
    # comparison quirk, and avoids a stray ``nan`` reaching the JSON/site).
    pool_source["chance_of_playing_next_round"] = pool_source["chance_of_playing_next_round"].where(
        pool_source["chance_of_playing_next_round"].notna(), None
    )
    pool_info = pool_source.to_dict("index")
    now_cost_map = {p: pool_info[p]["now_cost"] for p in pool_ids}
    sell_price = {p: sell_price_owned.get(p, now_cost_map[p]) for p in pool_ids}

    weeks = sorted(int(c[2:-3]) for c in xp_by_gw.columns if c.startswith("gw") and c.endswith("_xp"))
    horizon = max(min(int(config.get("horizon", 5)), len(weeks)), 1)
    weeks = weeks[:horizon]
    discount = {g: WEEKLY_DISCOUNT**i for i, g in enumerate(weeks)}

    raw_weekly_xp = lookup_weekly_xp(xp_by_gw, weeks, pool_ids)
    adj_weekly_value, rules = rule_inputs(raw_weekly_xp, league_eo, mode, weeks)

    effective_free = resolve_free_transfers(tables["league_free_transfers"], team_id)

    print(f"Strategy mode: {mode.upper()}. Solving a {len(weeks)}-week horizon over a {len(pool_ids)}-player pool...")
    print(f"  free-transfer rules: {describe_rules(mode)}")
    baseline = solve_horizon(
        pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, **rules,
    )
    final_plan, gain, transfer_recommended, hit_reasons = recommend_transfers(
        baseline, pool_ids, adj_weekly_value, pool_info, my_squad_ids, sell_price,
        bank, effective_free, weeks, discount, raw_weekly_xp=raw_weekly_xp, **rules,
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

    captain_board = build_captain_board(
        final_plan["weeks"][0], raw_weekly_xp, pool_info, mode, league_eo, raw_league_eo,
    )
    captaincy_risk = build_captaincy_risk(
        final_plan["weeks"][0], raw_weekly_xp, pool_info,
        league_picks, league_history, league_managers, team_id, gameweek,
    )

    write_table(build_squad_plan_table(final_plan, pool_info, weeks, raw_weekly_xp), "optimise_squad.parquet")
    write_table(
        build_transfers_table(final_plan, pool_info, gain, transfer_recommended, hit_reasons),
        "optimise_transfers.parquet",
    )
    write_table(chip_plan, "optimise_chip_plan.parquet")
    write_table(captain_board, "optimise_captain_board.parquet")
    write_table(captaincy_risk, "optimise_captaincy_risk.parquet")

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
              f"horizon gain {gain:.2f}xP vs holding (needs {final_plan.get('required_gain', TRANSFER_THRESHOLD_XP):.2f}) "
              f"-> {'recommended' if transfer_recommended else 'below threshold, holding instead'}")
        for reason in hit_reasons:
            print(f"  ! hit justified: {reason}")
    else:
        held_gain = f" (best move gained {gain:.2f}xP, needed {final_plan.get('required_gain', 0):.2f})" if gain else ""
        print(f"  no transfer recommended this week -- roll the free transfer{held_gain}.")
    captain_name = pool_info[wk0["captain_id"]]["web_name"] if wk0["captain_id"] else "none"
    print(f"  captain: {captain_name}")
    print("\nPer-move bar, every planned week (own gain vs the bar):")
    for wk in final_plan["weeks"]:
        g8 = wk.get("gate") or {}
        recs = g8.get("moves") or []
        if not recs:
            print(f"  GW{wk['gameweek']}: no move wanted ({wk['free_transfers']} FT)")
        for r in recs:
            print(f"  GW{wk['gameweek']}: {pool_info[r['out']]['web_name']} -> {pool_info[r['in']]['web_name']} "
                  f"{r['gain']:+.2f} vs {r['required']:.2f} -> {'keep' if r['passed'] else 'drop, bank the transfer'}"
                  f"{' (bundled)' if r['bundled'] else ''}")
    if not captain_board.empty:
        print("\nCaptain board (top 5):")
        print(captain_board.head(5).to_string(index=False))
    if not captaincy_risk.empty:
        print(f"\nCaptaincy risk vs the top {min(3, len(captaincy_risk.columns) - 4)} rivals (top 5 by total swing):")
        print(captaincy_risk.head(5).to_string(index=False))

    print("\nChips-remaining grid (your recommendations this horizon):")
    print(chip_plan.to_string(index=False))


if __name__ == "__main__":
    main()
