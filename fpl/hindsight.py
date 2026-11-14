"""Hindsight for a finished gameweek, scored with real points.

* ``team_of_the_week``: the best 15-man squad anyone could legally have
  picked -- £100.0m at that gameweek's prices (element-summary history
  ``value``), 2/5/5/3, max 3 per club, a valid formation, best XI + captain
  (x2), no chips. Solved with fpl.optimise.solve_horizon (the same MILP as
  the forward plan) from an empty squad with 15 "free" transfers, bench
  weight 0 and a tiny squad-cost tie-break so the bench is the cheapest
  valid one.
* ``best_possible``: the best *you* could have done with the moves you
  actually had: your squad and bank at the deadline (before that week's
  transfers), up to your free transfers that week (fpl.league's rule), no
  hits, no chips, FPL selling prices (fpl.optimise.estimate_selling_prices
  with prices and your transfer history as of that gameweek). Same solver.
  The gap to your score is split in a fixed order -- your lineup, then
  your captain, then your transfers (plus your chip/hit, if any) -- so the
  parts add up exactly.

Points are FPL's own (event/{gw}/live); selection uses
fpl.consultant.score_team for your own team.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from fpl.consultant import ELEMENT_TYPE_POSITION, score_team
from fpl.league import _free_transfers_for_manager
from fpl.optimise import FORMATION_BOUNDS, estimate_selling_prices, solve_horizon

BUDGET = 1000  # £100.0m in FPL's 0.1m units
COST_TIEBREAK = 1e-4  # per 0.1m of squad cost: < 0.1 point over a whole squad
KEEP_TIEBREAK = 1e-3  # prefer keeping a player over an equal-points swap


def _pool(bootstrap: dict[str, Any], gw_price: dict[int, int]) -> tuple[list[int], dict[int, dict[str, Any]]]:
    info = {}
    for e in bootstrap["elements"]:
        pid = int(e["id"])
        info[pid] = {
            "web_name": e["web_name"],
            "position": ELEMENT_TYPE_POSITION[e["element_type"]],
            "team": int(e["team"]),
            # price that gameweek; players with no match history (no
            # minutes all season) keep today's price -- they score 0 anyway
            "now_cost": int(gw_price.get(pid, e["now_cost"])),
        }
    return sorted(info), info


def gw_prices(matches: pd.DataFrame, gw: int) -> dict[int, int]:
    if matches.empty:
        return {}
    wk = matches[matches["gameweek"] == gw].sort_values(["player_id", "kickoff_time"])
    return {int(r.player_id): int(r.value) for r in wk.drop_duplicates("player_id").itertuples()}


def team_of_the_week(gw: int, bootstrap: dict[str, Any], points: dict[int, int], gw_price: dict[int, int]) -> dict[str, Any]:
    pool, info = _pool(bootstrap, gw_price)
    # tiny id tie-break so equal choices always resolve the same way
    value = {(p, gw): float(points.get(p, 0)) - 1e-7 * p for p in pool}
    plan = solve_horizon(pool, value, info, set(), {p: info[p]["now_cost"] for p in pool}, BUDGET, 15,
                         [gw], {gw: 1.0}, bench_weight=0.0, cost_tiebreak=COST_TIEBREAK)
    wk = plan["weeks"][0]
    xi, squad, cap = sorted(wk["xi_ids"]), sorted(wk["squad_ids"]), wk["captain_id"]
    total = sum(points.get(p, 0) for p in xi) + (points.get(cap, 0) if cap else 0)
    return {"status": plan["status"], "xi": xi, "bench": [p for p in squad if p not in xi], "captain": cap,
            "total": total, "cost": sum(info[p]["now_cost"] for p in squad), "info": info}


def best_xi(squad: list[int], points: dict[int, int], position: dict[int, str], captain: int | None = None,
            choose_captain: bool = True) -> tuple[int, list[int], int | None]:
    """Best valid XI from a fixed 15 (enumerating FPL formations), with the
    given captain kept (forced into the XI), or the best captain if
    ``choose_captain``. Returns (points, xi, captain)."""
    by_pos = {pos: sorted((p for p in squad if position[p] == pos), key=lambda p: (-points.get(p, 0), p))
              for pos in FORMATION_BOUNDS}
    best: tuple[int, list[int], int | None] | None = None
    ranges = [range(lo, hi + 1) for lo, hi in (FORMATION_BOUNDS[k] for k in ("GKP", "DEF", "MID", "FWD"))]
    for n in product(*ranges):
        if sum(n) != 11:
            continue
        xi: list[int] = []
        ok = True
        for pos, k in zip(("GKP", "DEF", "MID", "FWD"), n):
            cand = by_pos[pos]
            if captain is not None and position.get(captain) == pos and captain in cand:
                pick = [captain] + [p for p in cand if p != captain][: k - 1]
            else:
                pick = cand[:k]
            if len(pick) < k:
                ok = False
                break
            xi += pick
        if not ok:
            continue
        cap = captain if captain is not None else (max(xi, key=lambda p: (points.get(p, 0), -p)) if choose_captain else None)
        score = sum(points.get(p, 0) for p in xi) + (points.get(cap, 0) if cap in xi else 0)
        if best is None or score > best[0]:
            best = (score, sorted(xi), cap)
    return best or (0, [], None)


def best_possible(
    *, gw: int, team_id: int, bootstrap: dict[str, Any], points: dict[int, int], live: dict[int, dict[str, int]],
    gw_price: dict[int, int], league_picks: pd.DataFrame, league_history: pd.DataFrame,
    my_transfers: pd.DataFrame, chip_events: set[int], deadline: pd.Timestamp, mine: dict[str, Any], net_points: int,
) -> dict[str, Any]:
    if gw <= 1:
        return {"available": False, "reason": "GW1 was your starting squad, so there were no transfers to compare."}
    picks = league_picks[league_picks["manager_id"] == team_id]
    hist = league_history[league_history["manager_id"] == team_id].sort_values("gameweek")
    # squad and bank going into this deadline: last week's (or the week
    # before, if last week was a Free Hit, which reverts the squad)
    prev = gw - 1
    prev_chip = picks[picks["gameweek"] == prev]["active_chip"].dropna()
    if not prev_chip.empty and prev_chip.iloc[0] == "freehit":
        prev = gw - 2
    before = sorted(int(p) for p in picks[picks["gameweek"] == prev]["player_id"])
    bank_row = hist[hist["gameweek"] == prev]
    if len(before) != 15 or bank_row.empty:
        return {"available": False, "reason": f"Your GW{prev} squad isn't available."}
    bank = int(bank_row["bank"].iloc[0])
    ft = _free_transfers_for_manager(hist[hist["gameweek"] < gw], chip_events)

    pool, info = _pool(bootstrap, gw_price)
    position = {p: info[p]["position"] for p in pool}
    # selling prices by FPL's rule, at this gameweek's prices and your buys so far
    players = pd.DataFrame([{"id": e["id"], "now_cost": info[e["id"]]["now_cost"],
                             "cost_change_start": info[e["id"]]["now_cost"] - (e["now_cost"] - e["cost_change_start"])}
                            for e in bootstrap["elements"]])
    tx = my_transfers
    if not tx.empty and "time" in tx.columns:
        tx = tx[pd.to_datetime(tx["time"], utc=True) < deadline]
    sell = estimate_selling_prices(set(before), players, tx)
    sell_price = {p: sell.get(p, info[p]["now_cost"]) for p in pool}
    value = {(p, gw): float(points.get(p, 0)) + (KEEP_TIEBREAK if p in before else 0.0) - 1e-7 * p for p in pool}
    plan = solve_horizon(pool, value, info, set(before), sell_price, bank, ft, [gw], {gw: 1.0},
                         no_hit_weeks=frozenset({gw}), bench_weight=0.0)
    wk = plan["weeks"][0]
    z_squad = sorted(wk["squad_ids"])
    z, z_xi, z_cap = best_xi(z_squad, points, position)

    # your own choices, the same way, in a fixed order
    actual_squad = mine["xi"] + mine["bench"]
    x_rules = score_team(mine["xi"], mine["bench"], mine["captain"], mine["vice"], None, 0, live, position)
    armband = x_rules["armband"]
    # your squad, best lineup, your (effective) captain kept; no captain
    # bonus if neither your captain nor vice played
    lineup_pts, _, _ = best_xi(actual_squad, points, position, captain=armband, choose_captain=False)
    captain_pts, _, _ = best_xi(actual_squad, points, position)
    parts = {
        "chip_hit": x_rules["points"] - net_points,  # undoing your chip and/or hit
        "lineup": lineup_pts - x_rules["points"],
        "captain": captain_pts - lineup_pts,
        "transfer": z - captain_pts,
    }
    fielded = set(x_rules["scoring_ids"])
    return {
        "available": True,
        "status": plan["status"],
        "free_transfers": ft,
        "bank": bank,
        "squad_before": before,
        "transfers_out": sorted(set(before) - set(z_squad)),
        "transfers_in": sorted(set(z_squad) - set(before)),
        "xi": z_xi,
        "captain": z_cap,
        "your_captain": armband,
        "lineup_in": sorted(set(z_xi) - fielded),
        "lineup_out": sorted(fielded - set(z_xi)),
        "best": z,
        "you": net_points,
        "your_rules_score": x_rules["points"],
        "parts": parts,
        "chip": mine["chip"],
        "hits": mine["hits"],
    }
