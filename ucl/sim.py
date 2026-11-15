"""Simulated points for whole matchdays (for captain choice, the value of
between-days flexibility and the between-days advice).

Each player's points are drawn from his own match model (ucl.model): plays
or not, 60+ minutes or not, goals, assists, balls recovered, saves and goals
conceded from Poisson draws, Player of the Match as a coin with his chance;
the small items (cards, penalties, own goals, goals from outside the box)
are added at their expected value. Players are drawn independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl import rules

N = 4000


def sample(xp: pd.DataFrame, n: int = N, seed: int = 11) -> np.ndarray:
    """Points draws, shape (n, len(xp)), for the rows of one matchday."""
    rng = np.random.default_rng(seed)
    k = len(xp)
    pos = xp["pos"].to_numpy(int)
    p_app = xp["p_app"].clip(0, 1).to_numpy()
    p60 = (xp["p60"] / xp["p_app"].where(xp["p_app"] > 0, 1)).clip(0, 1).to_numpy()
    app = rng.random((n, k)) < p_app
    sixty = app & (rng.random((n, k)) < p60)
    g = rng.poisson(np.tile(xp["lam_g"].fillna(0).to_numpy(), (n, 1)))
    a = rng.poisson(np.tile(xp["lam_a"].fillna(0).to_numpy(), (n, 1)))
    br = rng.poisson(np.tile(xp["lam_br"].fillna(0).to_numpy(), (n, 1)))
    sv = rng.poisson(np.tile(xp["lam_sv"].fillna(0).to_numpy(), (n, 1)))
    ga = rng.poisson(np.tile(xp["lam_ga"].fillna(0).to_numpy(), (n, 1)))
    mom = rng.random((n, k)) < xp["p_mom"].fillna(0).to_numpy()
    goal = np.array([rules.GOAL[p] for p in pos])
    cs = np.array([rules.CLEAN_SHEET[p] for p in pos])
    con = np.array([rules.CONCEDED_PER_2[p] for p in pos])
    gk = pos == rules.GK
    small = sum(xp.get(c, pd.Series(0, index=xp.index)).fillna(0).to_numpy()
                for c in ("cards", "pens", "own_goals", "outside_box", "pen_saves"))
    small = np.where(p_app > 0, small / np.maximum(p_app, 1e-9), 0)
    pts = (app * rules.APPEARANCE + sixty * rules.SIXTY_PLUS + goal * g + rules.ASSIST * a + br // 3
           + rules.PLAYER_OF_MATCH * mom + cs * (sixty & (ga == 0)) + con * (ga // 2)
           + gk * (sv // 3) + small)
    return np.where(app, pts, 0.0)
