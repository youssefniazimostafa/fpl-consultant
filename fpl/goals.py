"""Goal-aware close calls (Step 17).

When two captain (or transfer) options are close on expected points,
choose the one that gives the best simulated chance of reaching your league
goals instead of a fixed ownership tie-breaker.

1. **Goals** (app Settings, passed to the pipeline as FPL_GOALS; the
   published site uses the defaults): a weighted list such as "finish 1st"
   (weight 2) and "stay in the top 2" (weight 1), and "by when" (a gameweek
   or the end of the season). The score of an option is the weighted
   average of its goal chances.

2. **Close call**: option B is close to the best option A when
   ``xP(A) - xP(B) <= CLOSE_SD x SD(points(A) - points(B))``, with each
   player-week's SD the measured projection error (``optimise.RESIDUAL_SD``,
   by position and xP band) and the players' errors independent. Only close
   options are simulated; a clearly better option on expected points always
   wins.

3. **Simulation**: every player in your XI and in each rival's last
   confirmed XI gets one shared draw per simulation (so a player you and a
   rival both own moves both of you together), from a negative binomial
   with mean = his xP and SD = his measured error SD (fatter-tailed than
   the Poisson the weekly panels use, so a captain haul is possible).
   Rivals keep their last confirmed XI and captain. The weeks after this one,
   up to the target gameweek, come from each manager's own weekly-score
   history (Normal(mean, SD) per week, as ``simulate_season_standings``),
   plus, for transfer options, the option's own projected points difference
   over the rest of the horizon. Every option is scored on the same draws,
   so differences between options are paired. Draws grow from 20,000 to at
   most 200,000 until the best option's lead over the expected-points pick
   is more than ``NOISE_SE`` standard errors; if it never is, the call is
   "too close to call" and expected points decide.

Not modelled: same-club correlation (a clean sheet that hits two defenders),
rivals' transfers and captain changes before this deadline, chips.

Whether Auto uses the rule is ``USE_GOAL_AWARE``, decided by
``fpl/goal_calibration.py`` (see the block below).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

LAST_GW = 38
CLOSE_SD = 1.0          # close = within 1 SD of the expected-points difference (fixed before testing)
NOISE_SE = 2.0          # a goal-chance difference counts only beyond 2 simulation SEs
DRAW_STEPS = (20_000, 50_000, 100_000, 200_000)
CHUNK = 10_000
SEED = 17
REST_MIN_SD = 8.0       # same floor as fpl.simulation.SEASON_SIM_MIN_STD

GOAL_KINDS = {"first": 1, "top2": 2, "top3": 3}
GOAL_LABELS = {"first": "finish 1st", "top2": "top 2", "top3": "top 3"}
SHORT_LABELS = {"first": "to win", "top2": "top 2", "top3": "top 3"}
DEFAULT_GOALS: dict[str, Any] = {
    "goals": [{"kind": "first", "weight": 2.0}, {"kind": "top2", "weight": 1.0}],
    "by": "end",
}

# ---------------------------------------------------------------------------
# Auto switch. Rule fixed before running fpl/goal_calibration.py: on only if,
# on each of 2025-26, 2024-25 and 2023-24 (>= 200 simulated 10-manager
# leagues each, paired), the goal-aware rule beats today's balanced mode on
# the weighted goal chance (2/3 P(1st) + 1/3 P(top 2), the Settings
# defaults) by more than 1 SE, neither P(1st) nor P(top 2) is worse by more
# than 1 SE, AND average points are not worse by more than 1 SE.
# Result: see GOAL_TEST_RESULT below.
# ---------------------------------------------------------------------------
USE_GOAL_AWARE = False
GOAL_TEST_RESULT = (
    "2025-26 (200 leagues): weighted goal +0.003 +-0.020, P(1st) +0.000 +-0.019, P(top 2) +0.010 +-0.033, "
    "points +2.6 +-2.6 -> fails (not > 1 SE); 2024-25 (200): weighted goal +0.263 +-0.030, P(1st) +0.240 +-0.033, "
    "P(top 2) +0.310 +-0.035, points +35.2 +-2.7 -> passes; 2023-24: not completed. Fails the all-three-seasons "
    "rule on 2025-26, so Auto keeps balanced mode's 0.02 ownership tie-breaker; the goal chances are display only."
)


def load_goals(raw: str | None = None) -> dict[str, Any]:
    """Goals from FPL_GOALS (the app's Settings), else the defaults.
    Anything malformed falls back to the defaults rather than failing."""
    text = raw if raw is not None else os.environ.get("FPL_GOALS")
    if not text:
        return {**DEFAULT_GOALS, "source": "default"}
    try:
        v = json.loads(text)
        goals = [{"kind": str(g["kind"]), "weight": float(g["weight"])} for g in v.get("goals", [])]
        goals = [g for g in goals if g["kind"] in GOAL_KINDS and g["weight"] > 0]
        by = v.get("by", "end")
        if by != "end":
            by = int(by)
            if not 1 <= by <= LAST_GW:
                raise ValueError(by)
        if not goals:
            raise ValueError("no goals")
        return {"goals": goals, "by": by, "source": "settings"}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {**DEFAULT_GOALS, "source": "default (Settings value unreadable)"}


def target_gameweek(goals: dict[str, Any], next_gw: int) -> tuple[int, str | None]:
    """(target GW, note). A "by GW19" goal that has already passed falls
    back to the end of the season, with a note saying so."""
    by = goals.get("by", "end")
    if by == "end":
        return LAST_GW, None
    if int(by) < next_gw:
        return LAST_GW, f"GW{by} has passed, so this uses the end of the season"
    return int(by), None


def describe_goals(goals: dict[str, Any], target: int) -> str:
    parts = [f"{GOAL_LABELS[g['kind']]} (weight {g['weight']:g})" for g in goals["goals"]]
    when = "the end of the season" if target == LAST_GW else f"GW{target}"
    return f"{', '.join(parts)} by {when}"


# ---------------------------------------------------------------------------
# close calls
# ---------------------------------------------------------------------------


def error_sd(position: str, xp: float, table: dict[str, tuple[float, ...]] | None = None) -> float:
    """Measured SD of one player-week's (actual - projected) points."""
    from fpl.optimise import RESIDUAL_SD, XP_BANDS

    t = table or RESIDUAL_SD
    band = sum(1 for edge in XP_BANDS if xp >= edge)
    row = t.get(position, t.get("MID", RESIDUAL_SD["MID"]))
    return float(row[min(band, len(row) - 1)])


def captain_close_calls(
    xi_ids: list[int], xp: dict[int, float], position: dict[int, str], table: dict | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """(best-xP captain, every XI player with his gap and the SD of the
    difference, flagged close or not). Captaining B instead of A changes
    your score by points(B) - points(A)."""
    best = max(xi_ids, key=lambda p: (xp.get(p, 0.0), -p))
    sd_best = error_sd(position[best], xp.get(best, 0.0), table)
    rows = []
    for p in sorted(xi_ids, key=lambda q: -xp.get(q, 0.0)):
        gap = xp.get(best, 0.0) - xp.get(p, 0.0)
        sd = 0.0 if p == best else math.hypot(sd_best, error_sd(position[p], xp.get(p, 0.0), table))
        rows.append({"player_id": p, "xp": xp.get(p, 0.0), "gap": gap, "sd": sd,
                     "close": p == best or gap <= CLOSE_SD * sd})
    return best, rows


def ft_end(plan: dict[str, Any], free_now: int) -> int:
    """Free transfers carried out of the plan's last week (the optimiser's
    ``ft_end``: what its saved-FT value credits)."""
    from fpl.optimise import MAX_FREE_TRANSFERS

    ft = free_now
    for wk in plan["weeks"]:
        n_in = len(wk["transfers_in"])
        ft = 1 if n_in > ft else min(MAX_FREE_TRANSFERS, ft - n_in + 1)
    return ft


def plan_value(
    plan: dict[str, Any], raw: dict[tuple[int, int], float], discount: dict[int, float],
    free_now: int, ft_value: float,
) -> float:
    """An option's expected points as the optimiser weighs them: discounted
    XI + captain xP over the horizon, minus week-0 hits, plus the saved-FT
    value for the free transfers carried out of the horizon."""
    from fpl.optimise import HIT_COST

    tot = 0.0
    for wk in plan["weeks"]:
        g = wk["gameweek"]
        pts = sum(raw.get((p, g), 0.0) for p in wk["xi_ids"])
        if wk["captain_id"] is not None:
            pts += raw.get((wk["captain_id"], g), 0.0)
        tot += discount[g] * pts
    hits = max(len(plan["weeks"][0]["transfers_in"]) - free_now, 0)
    return tot - HIT_COST * hits + ft_value * ft_end(plan, free_now)


def plan_diff_sd(
    a: dict[str, Any], b: dict[str, Any], raw: dict[tuple[int, int], float], discount: dict[int, float],
    position: dict[int, str], table: dict | None = None,
) -> float:
    """SD of the horizon points difference between two plans: every player
    in one plan's week-g XI but not the other's, each with his measured
    error SD, weighted like the objective."""
    var = 0.0
    for wa, wb in zip(a["weeks"], b["weeks"]):
        g = wa["gameweek"]
        for p in set(wa["xi_ids"]) ^ set(wb["xi_ids"]):
            var += (discount[g] * error_sd(position[p], raw.get((p, g), 0.0), table)) ** 2
    return var ** 0.5


def future_points(plan: dict[str, Any], raw: dict[tuple[int, int], float], target: int) -> float:
    """Projected XI + captain points in the plan's weeks after this one, up
    to the target gameweek (the transfer options' rest-of-season shift)."""
    tot = 0.0
    for wk in plan["weeks"][1:]:
        g = wk["gameweek"]
        if g > target:
            break
        tot += sum(raw.get((p, g), 0.0) for p in wk["xi_ids"])
        if wk["captain_id"] is not None:
            tot += raw.get((wk["captain_id"], g), 0.0)
    return tot


# ---------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------


def _nb_params(mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.clip(mean, 0.0, None)
    v = np.maximum(sd**2, m + 1e-6)
    live = m > 1e-6
    r = np.where(live, m**2 / (v - m), 1.0)
    p = np.where(live, m / v, 1.0)
    return r, p, live


def simulate(
    options: list[dict[str, Any]],
    rivals: list[dict[str, Any]],
    me: dict[str, Any],
    xp: dict[int, float],
    sd: dict[int, float],
    goals: list[dict[str, Any]],
    rest_weeks: int,
    ref: int = 0,
    seed: int = SEED,
    steps: tuple[int, ...] = DRAW_STEPS,
) -> dict[str, Any]:
    """Goal chances for each option on shared draws.

    ``options``: {xi_ids, captain_id, hits, future_delta}; ``rivals``:
    {xi_ids, captain_id, total, rest_mean, rest_sd}; ``me``: {total,
    rest_mean, rest_sd}. ``ref`` is the expected-points pick. Returns per
    option the chance of each goal, the weighted score, its paired
    difference from ``ref`` and that difference's SE, and the draws used."""
    ids = sorted({p for o in options + rivals for p in o["xi_ids"]})
    index = {p: i for i, p in enumerate(ids)}
    n_ids = len(ids)

    def matrix(teams: list[dict[str, Any]]) -> np.ndarray:
        M = np.zeros((n_ids, len(teams)))
        for j, t in enumerate(teams):
            for p in t["xi_ids"]:
                M[index[p], j] += 1
            if t.get("captain_id") in index:
                M[index[t["captain_id"]], j] += 1
        return M

    Mo, Mr = matrix(options), matrix(rivals)
    r, p, live = _nb_params(np.array([xp.get(i, 0.0) for i in ids]), np.array([sd.get(i, 0.0) for i in ids]))
    w = np.array([g["weight"] for g in goals], dtype=float)
    w = w / w.sum()
    need = [GOAL_KINDS[g["kind"]] for g in goals]
    hits = np.array([o.get("hits", 0) for o in options], dtype=float)
    delta = np.array([o.get("future_delta", 0.0) for o in options], dtype=float)
    r_tot = np.array([x["total"] for x in rivals], dtype=float)
    r_mu = np.array([x["rest_mean"] for x in rivals], dtype=float) * rest_weeks
    r_sd = np.array([x["rest_sd"] for x in rivals], dtype=float) * math.sqrt(max(rest_weeks, 0))
    m_mu, m_sd = me["rest_mean"] * rest_weeks, me["rest_sd"] * math.sqrt(max(rest_weeks, 0))

    rng = np.random.default_rng(seed)
    n_opt = len(options)
    met_sum = np.zeros((n_opt, len(goals)))
    d_sum, d_sq = np.zeros(n_opt), np.zeros(n_opt)
    pts_sum = np.zeros(n_opt)
    n = 0
    target = 0
    for step in steps:
        target = step
        while n < target:
            k = min(CHUNK, target - n)
            draws = np.zeros((k, n_ids))
            if live.any():
                draws[:, live] = rng.negative_binomial(r[live], p[live], size=(k, int(live.sum())))
            week_me = draws @ Mo - 4.0 * hits
            rival_final = r_tot + draws @ Mr + rng.normal(r_mu, np.maximum(r_sd, 0.0), size=(k, len(rivals)))
            my_final = me["total"] + week_me + delta + rng.normal(m_mu, max(m_sd, 0.0), size=(k, 1))
            above = (rival_final[:, None, :] > my_final[:, :, None]).sum(axis=2)  # draws x options
            met = np.stack([above < nk for nk in need], axis=2)                    # draws x options x goals
            score = (met * w).sum(axis=2)
            met_sum += met.sum(axis=0)
            d = score - score[:, [ref]]
            d_sum += d.sum(axis=0)
            d_sq += (d**2).sum(axis=0)
            pts_sum += week_me.sum(axis=0)
            n += k
        mean_d = d_sum / n
        se = np.sqrt(np.maximum(d_sq / n - mean_d**2, 0.0) / n)
        best = int(np.argmax(mean_d))
        decided = (best != ref and mean_d[best] > NOISE_SE * se[best]) or (
            best == ref and all(mean_d[j] < -NOISE_SE * se[j] for j in range(n_opt) if j != ref)
        )
        if decided:
            break
    chances = met_sum / n
    mean_d = d_sum / n
    se = np.sqrt(np.maximum(d_sq / n - mean_d**2, 0.0) / n)
    score = chances @ w
    rows = [{
        "chances": {g["kind"]: float(chances[i, j]) for j, g in enumerate(goals)},
        "score": float(score[i]), "diff_vs_xp_pick": float(mean_d[i]), "diff_se": float(se[i]),
        "week_points": float(pts_sum[i] / n),
    } for i in range(n_opt)]
    return {"rows": rows, "draws": n}


def choose(sim: dict[str, Any], ref: int) -> tuple[int, str]:
    """(pick, verdict). verdict: "goals" (a close option has a goal chance
    more than NOISE_SE SEs better than the expected-points pick),
    "agree" (the expected-points pick is also clearly best for the goals),
    or "too close" (no clear difference: expected points decide)."""
    rows = sim["rows"]
    best = max(range(len(rows)), key=lambda i: (rows[i]["diff_vs_xp_pick"], i == ref))
    if best != ref and rows[best]["diff_vs_xp_pick"] > NOISE_SE * rows[best]["diff_se"]:
        return best, "goals"
    others = [i for i in range(len(rows)) if i != ref]
    if others and all(rows[i]["diff_vs_xp_pick"] < -NOISE_SE * rows[i]["diff_se"] for i in others):
        return ref, "agree"
    return ref, "too close"


def option_text(name: str, chances: dict[str, float], goals: list[dict[str, Any]]) -> str:
    """"Haaland: 52.1% top 2, 18.3% to win" (goals in reverse weight order
    as in the example: the secondary goal first reads more naturally)."""
    parts = [f"{100 * chances[g['kind']]:.1f}% {SHORT_LABELS[g['kind']]}"
             for g in sorted(goals, key=lambda g: g["weight"])]
    return f"{name}: {', '.join(parts)}"
