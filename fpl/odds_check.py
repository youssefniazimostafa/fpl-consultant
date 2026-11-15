"""Step 17: is Shin's method (fpl.odds.devig_shin) more accurate than simple
normalisation at turning bookmaker odds into probabilities?

Joins every logged match in data/odds_log/ (both versions, written by
fpl.odds on each run) to its result in fixtures.parquet, using the latest
log taken before kick-off, and scores both versions on the match result
(home / draw / away) and on over/under 2.5 goals:

* log loss: minus the average log of the probability given to what happened
  (lower is better);
* Brier score: the average squared gap between the probabilities and what
  happened (lower is better).

Switch Shin on (fpl.odds.ODDS_METHOD) only if it's better on both markets.

Run with ``uv run python -m fpl.odds_check``.
"""

from __future__ import annotations

import math

import pandas as pd

from fpl.odds import ODDS_LOG_DIR
from fpl.publish import TABLES_DIR

MIN_GAMEWEEKS = 5  # fewer gameweeks than this is too little to decide anything


def load_log() -> pd.DataFrame:
    files = sorted(ODDS_LOG_DIR.glob("odds_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def scored_matches() -> pd.DataFrame:
    log = load_log()
    fx_path = TABLES_DIR / "fixtures.parquet"
    if log.empty or not fx_path.exists():
        return pd.DataFrame()
    fx = pd.read_parquet(fx_path)
    fx = fx[fx["finished"].astype(bool)].copy()
    fx["kickoff"] = pd.to_datetime(fx["kickoff_time"], utc=True)
    log["logged"] = pd.to_datetime(log["logged_at"], utc=True)
    m = log.merge(fx, left_on=["home_team_id", "away_team_id"], right_on=["team_h", "team_a"], how="inner")
    m = m[m["logged"] < m["kickoff"]]
    m = m.sort_values("logged").groupby("id", as_index=False).last()
    m["result"] = [("home" if h > a else "away" if a > h else "draw") for h, a in zip(m["team_h_score"], m["team_a_score"])]
    m["over"] = (m["team_h_score"] + m["team_a_score"]) > 2.5
    return m


def score(m: pd.DataFrame, method: str) -> dict[str, float]:
    eps = 1e-12
    ll1 = br1 = ll2 = br2 = 0.0
    for r in m.itertuples():
        p = {"home": getattr(r, f"p_home_win_{method}"), "draw": getattr(r, f"p_draw_{method}"),
             "away": getattr(r, f"p_away_win_{method}")}
        ll1 -= math.log(max(p[r.result], eps))
        br1 += sum((p[k] - (1.0 if k == r.result else 0.0)) ** 2 for k in p)
        po = getattr(r, f"p_over_2_5_{method}")
        ll2 -= math.log(max(po if r.over else 1 - po, eps))
        br2 += (po - (1.0 if r.over else 0.0)) ** 2
    n = len(m)
    return {"result_log_loss": ll1 / n, "result_brier": br1 / n, "goals_log_loss": ll2 / n, "goals_brier": br2 / n}


def main() -> None:
    m = scored_matches()
    log = load_log()
    print(f"Logged odds: {len(log)} rows in {len(list(ODDS_LOG_DIR.glob('odds_*.parquet')))} file(s); "
          f"finished matches with odds logged before kick-off: {len(m)}"
          + (f" across {m['event'].nunique()} gameweek(s)" if not m.empty else ""))
    if m.empty or m["event"].nunique() < MIN_GAMEWEEKS:
        print(f"Too few played gameweeks with stored odds (need {MIN_GAMEWEEKS}): leave Shin's method off; "
              "both versions keep being logged.")
        if m.empty:
            return
    rows = {meth: score(m, meth) for meth in ("normalise", "shin")}
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
