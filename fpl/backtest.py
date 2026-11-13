"""Backtest ``fpl.xp``'s model against a full past season's actual results --
proof the xP model beats simple baselines before trusting it with transfers.

For every gameweek of the most recent *complete* season in the cloned
vaastav repo (``merged_gw.csv``), player rates, the minutes model and
fixture lambdas are rebuilt using *only* rows strictly before that
gameweek -- no result or form from the gameweek being predicted, or any
later one, ever leaks into its own prediction -- using the exact same
recency-weighted, shrunk-to-position-average machinery ``fpl.features``
and ``fpl.xp`` use live (imported directly, not reimplemented, so the
backtest can't quietly drift from what actually runs in production). The
one deliberate exception is ``p_start``: it's a recency-weighted share of
the player's own historical rows, same as live, but *not* shrunk toward a
position average, since a nailed-on starter's own evidence shouldn't be
diluted toward a rotation-prone position's typical rate. Each gameweek is
then scored with ``fpl.xp.build_xp``'s own formula against
three baselines: FPL's own pre-match ``xP`` (vaastav's name for FPL's
``ep_next`` snapshot, captured before the deadline), the player's own
season points-per-game so far, and his last-5-gameweek average.

Scope note: the live pipeline's fixture lambdas blend Elo, betting-odds
implied goals and rolling Understat xG (Step 4). Reproducing a full
season of those external signals isn't practical here -- the Odds API has
no free historical access, and ClubElo's roster (this fictional season's
20 clubs) doesn't cover every club from a real past Premier League season
(Burnley, West Ham and Wolves played last season but aren't mapped in
``team_aliases.csv``). Instead ``lambda_for``/``lambda_against`` come from
a shrunk, recency-weighted rolling attack/defence strength per team,
computed directly from ``merged_gw.csv``'s player-level ``expected_goals``
-- the same signal Step 4b's rolling Understat strength captures, just
read from the file already on disk instead of a fresh API pull. So this
backtest exercises the part of the model Step 6 called "the heart of the
project": the minutes model, per-90 rate shrinkage/decay, and the
Poisson-based scoring formula -- not the external-data blend.

Data-quality note on the FPL ep_next baseline: for 27 of this season's 38
gameweeks, the cloned repo's per-gameweek ``xP{N}.csv`` scrape file is
missing, and ``merged_gw.csv``'s ``xP`` column is backfilled with a flat
0.0 for every player that gameweek. ``find_degenerate_fpl_xp_gws`` detects
and nulls these out so the FPL-xP baseline is only scored on the 11
gameweeks it actually has a prediction for. Even on those, the cloned
repo's own ``DATA_DICTIONARY.md``/``README.md`` flag ``xP`` as sourced
from ``ep_this`` (scraped *after* each gameweek ends) rather than the
pre-deadline ``ep_next`` managers actually see, with a documented risk of
post-match lookahead bias -- so treat the FPL-xP numbers here as an
optimistic upper bound, not a clean baseline.

Reporting note: pooled (all-rows) Spearman is dominated by the ~60% of
player-gameweeks where nobody plays, which every baseline ranks near zero
correctly but ``ppg``/``last5`` do most easily since they're literally
built from recent playing time. The predicted-likely-starter subset
(``p_start > 0.5``), top-N hit rate and captain accuracy are reported
separately and are the metrics that should gate any change to the model,
not the pooled figure.

Run with `uv run python -m fpl.backtest`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl.features import HALF_LIFE_MATCHES, SHRINKAGE_MATCHES, add_recency_weight, build_minutes_factor, build_player_rates
from fpl.xp import POSITION_MAP, build_prior_season_rates, build_xp, find_prior_season_dir

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
RESULTS_LOG_PATH = Path(__file__).resolve().parent / "backtest_results_log.csv"
PRIOR_CACHE_PATH = Path(__file__).resolve().parent / "backtest_prior_season_player_map.csv"

MIN_BACKTEST_GW = 2  # need at least one prior gameweek of history to predict anything
LAMBDA_SHRINKAGE_MATCHES = SHRINKAGE_MATCHES  # same handful of league-average pseudo-matches as player rates
CALIBRATION_BUCKETS = [0, 2, 4, 6, np.inf]
BASELINE_COLUMNS = {"xp": "our model", "fpl_xp": "FPL's ep_next", "ppg": "season PPG", "last5": "last-5 average"}


# ---------------------------------------------------------------------------
# loading last season's gameweek-by-gameweek data
# ---------------------------------------------------------------------------


def load_backtest_season() -> tuple[Path, pd.DataFrame]:
    """The most recent *complete* season in the cloned vaastav repo -- the
    same one ``fpl.xp`` blends toward as "last season" live, so backtesting
    it tells you how that exact blend logic would have performed."""
    season_dir = find_prior_season_dir()
    if season_dir is None:
        raise RuntimeError("no complete prior season found under data/external/fantasy-premier-league/data")

    gw = pd.read_csv(season_dir / "gws" / "merged_gw.csv")
    teams = pd.read_csv(season_dir / "teams.csv")[["id", "name"]]
    opponent_name = teams.set_index("id")["name"]

    # a handful of rows are byte-for-byte duplicated in the source file (not
    # real double gameweeks -- those have two *different* fixture ids); drop
    # them so they don't get double-counted below.
    gw = gw.drop_duplicates(subset=["element", "GW", "fixture"])

    gw["position"] = gw["position"].map(POSITION_MAP)
    gw = gw.dropna(subset=["position"]).copy()
    gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"])
    for col in ["expected_goals", "expected_assists"]:
        gw[col] = pd.to_numeric(gw[col], errors="coerce").fillna(0.0)
    gw = gw.rename(columns={"element": "player_id", "team": "team_name", "xP": "fpl_xp"})
    gw["opponent_name"] = gw["opponent_team"].map(opponent_name)
    return season_dir, gw


def build_players_like(gw: pd.DataFrame) -> pd.DataFrame:
    """A minimal ``players``-shaped frame (``id``, ``position``) covering
    every player who ever appeared this season -- all ``fpl.features``'
    rate-building functions need."""
    players = (
        gw.groupby("player_id")
        .agg(position=("position", "first"), web_name=("name", "first"))
        .reset_index()
        .rename(columns={"player_id": "id"})
    )
    players["chance_of_playing_next_round"] = 100.0  # no historical injury-news feed to draw on
    return players


# ---------------------------------------------------------------------------
# fixture strength: rolling, shrunk team attack/defence from merged_gw itself
# ---------------------------------------------------------------------------


def build_team_match_pairs(gw: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match it played, with its own match xG and the
    opponent's (i.e. what it conceded, expectation-wise) side by side."""
    team_match = gw.groupby(["fixture", "GW", "team_name"])["expected_goals"].sum().reset_index(name="xg_for")
    pairs = team_match.merge(team_match, on=["fixture", "GW"], suffixes=("", "_opp"))
    pairs = pairs[pairs["team_name"] != pairs["team_name_opp"]].copy()
    return pairs.rename(columns={"xg_for_opp": "xg_against"})[["GW", "team_name", "xg_for", "xg_against"]]


def team_strength_asof(pairs: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    """Recency-weighted (same half-life as player rates) attack/defence
    strength per team from matches strictly before ``target_gw``, shrunk
    toward the league average with a few pseudo-matches so one noisy early
    result doesn't swing a team's rating wildly."""
    hist = pairs[pairs["GW"] < target_gw]
    league_avg_for = hist["xg_for"].mean() if not hist.empty else np.nan
    league_avg_against = hist["xg_against"].mean() if not hist.empty else np.nan

    teams = pairs["team_name"].unique()
    if hist.empty or pd.isna(league_avg_for):
        return pd.DataFrame({"attack": np.nan, "defence": np.nan}, index=teams)

    hist = hist.sort_values(["team_name", "GW"], ascending=[True, False]).copy()
    matches_ago = hist.groupby("team_name").cumcount()
    hist["w"] = 0.5 ** (matches_ago / HALF_LIFE_MATCHES)

    w_for = (hist["w"] * hist["xg_for"]).groupby(hist["team_name"]).sum()
    w_against = (hist["w"] * hist["xg_against"]).groupby(hist["team_name"]).sum()
    w_sum = hist.groupby("team_name")["w"].sum()

    k = LAMBDA_SHRINKAGE_MATCHES
    attack = (w_for + league_avg_for * k) / (w_sum + k)
    defence = (w_against + league_avg_against * k) / (w_sum + k)
    out = pd.DataFrame({"attack": attack, "defence": defence}).reindex(teams)
    out["attack"] = out["attack"].fillna(league_avg_for)
    out["defence"] = out["defence"].fillna(league_avg_against)
    out.attrs["league_avg_for"] = league_avg_for
    return out


def match_lambdas(team_name: pd.Series, opponent_name: pd.Series, strength: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """lambda_for(T vs O) = attack(T) * defence(O) / league_avg, the same
    multiplicative attack-index * defence-index combination
    ``fpl.features``'s Elo/xG fallback uses, just built from this season's
    own rolling team strength instead of Elo."""
    league_avg = strength.attrs.get("league_avg_for", np.nan)
    attack = strength["attack"]
    defence = strength["defence"]
    lambda_for = team_name.map(attack) * opponent_name.map(defence) / league_avg
    lambda_against = opponent_name.map(attack) * team_name.map(defence) / league_avg
    return lambda_for, lambda_against


# ---------------------------------------------------------------------------
# per-gameweek "as of" features
# ---------------------------------------------------------------------------


def build_p_start_asof(pm_hist: pd.DataFrame, players_like: pd.DataFrame) -> pd.Series:
    """Recency-weighted share of a player's own historical rows (every
    registered squad player gets a row each gameweek, 0 minutes and all, so
    this doubles as "starts over the team's matches") where he started --
    simpler than ``fpl.features.build_p_start``'s team-schedule lookup, but
    the same decay philosophy.

    Deliberately *not* shrunk toward the position average like the other
    rate features: a nailed-on starter's own evidence (e.g. 19 starts in 19
    matches) should stand on its own, not get pulled down toward a
    position's typical (rotation-diluted) start rate. A player with no
    history yet gets a neutral 0.5 rather than 0 -- no evidence either way,
    not evidence of never playing."""
    position_by_player = players_like.set_index("id")["position"]
    if pm_hist.empty:
        return pd.Series(0.5, index=position_by_player.index).rename_axis("player_id")
    pm_hist = add_recency_weight(pm_hist)
    w = pm_hist["weight"]
    num = (w * pm_hist["starts"]).groupby(pm_hist["player_id"]).sum()
    den = w.groupby(pm_hist["player_id"]).sum()
    p_start = (num / den).reindex(position_by_player.index)
    return p_start.fillna(0.5).clip(0, 1).rename_axis("player_id")


def build_features_asof(gw: pd.DataFrame, players_like: pd.DataFrame, pairs: pd.DataFrame, target_gw: int) -> pd.DataFrame | None:
    """Everything ``fpl.xp.build_xp`` needs for ``target_gw``, built from
    only ``GW < target_gw`` history plus ``target_gw``'s own (publicly
    known in advance) fixture list -- never its results."""
    hist = gw[gw["GW"] < target_gw]
    actual = gw[gw["GW"] == target_gw]
    if actual.empty:
        return None

    pm_hist = hist.rename(columns={"team_name": "team_id"})
    rates = build_player_rates(pm_hist, players_like)
    minutes_factor = build_minutes_factor(pm_hist, players_like).rename("minutes_factor").rename_axis("player_id")
    p_start = build_p_start_asof(pm_hist, players_like).rename("p_start")

    strength = team_strength_asof(pairs, target_gw)

    # one row per fixture, not per player -- a double gameweek keeps both of
    # a player's matches here (each with its own opponent/lambda), and they
    # get summed back together per player after scoring, in run_backtest.
    fixture_meta = actual[
        ["player_id", "team_name", "opponent_name", "was_home", "fixture", "GW", "kickoff_time", "value",
         "total_points", "fpl_xp", "minutes"]
    ]
    fixture_meta = fixture_meta.rename(
        columns={"GW": "event", "was_home": "is_home", "fixture": "fixture_id", "value": "now_cost",
                 "minutes": "actual_minutes"}
    )

    features = (
        players_like.rename(columns={"id": "player_id"})
        .merge(rates, on="player_id", how="left")
        .merge(minutes_factor, on="player_id", how="left")
        .merge(p_start, on="player_id", how="left")
        .merge(fixture_meta, on="player_id", how="inner")  # only players with an actual fixture this gameweek
    )

    lambda_for, lambda_against = match_lambdas(features["team_name"], features["opponent_name"], strength)
    features["lambda_for"] = lambda_for.to_numpy()
    features["lambda_against"] = lambda_against.to_numpy()
    features["clean_sheet_prob"] = np.exp(-features["lambda_against"])
    features["team_short"] = features["team_name"]
    features["opponent_short"] = features["opponent_name"]
    features["source"] = "backtest_rolling_xg"
    return features


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def attach_baselines(features: pd.DataFrame, hist: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    """FPL's own pre-match ``xP``, season points-per-game so far, and the
    last-5-gameweek average -- all computed from strictly prior rounds."""
    features = features.copy()
    features["ppg"] = features["player_id"].map(hist.groupby("player_id")["total_points"].mean()).fillna(0.0)

    last5 = (
        hist[hist["GW"] >= target_gw - 5]
        .sort_values("GW")
        .groupby("player_id")["total_points"]
        .mean()
    )
    features["last5"] = features["player_id"].map(last5).fillna(0.0)
    return features


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def mean_absolute_error(df: pd.DataFrame, pred_col: str) -> float:
    return (df[pred_col] - df["total_points"]).abs().mean()


def spearman_by_position(df: pd.DataFrame, pred_col: str, min_n: int = 5) -> dict[str, float]:
    out = {}
    for pos, group in df.groupby("position"):
        if len(group) >= min_n and group[pred_col].nunique() > 1 and group["total_points"].nunique() > 1:
            corr, _ = spearmanr(group[pred_col], group["total_points"])
            out[pos] = corr
    return out


def calibration_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    buckets = pd.cut(df[pred_col], CALIBRATION_BUCKETS, right=False)
    return df.groupby(buckets, observed=True).agg(
        n=(pred_col, "size"), mean_pred=(pred_col, "mean"), mean_actual=("total_points", "mean")
    )


def topn_hit_rate(df: pd.DataFrame, pred_col: str, n: int, min_group: int = 3) -> float:
    """Of each model's top-``n`` picks per position per gameweek (or all of
    a smaller pool), what share were actually in the real top-``n`` for
    that position/gameweek by ``total_points``. More decision-relevant than
    pooled Spearman: this is literally "how often would this model's
    shortlist have contained the gameweek's best performers"."""
    df = df.dropna(subset=[pred_col])
    hits, total = 0, 0
    for (_pos, _event), group in df.groupby(["position", "event"]):
        if len(group) < min_group:
            continue
        k = min(n, len(group))
        actual_top = set(group.nlargest(k, "total_points")["player_id"])
        pred_top = set(group.nlargest(k, pred_col)["player_id"])
        hits += len(actual_top & pred_top)
        total += k
    return hits / total if total else float("nan")


def captain_accuracy(df: pd.DataFrame, pred_col: str) -> float:
    """Mean points scored by each model's single highest-ranked player
    (across the whole pool, not per position) per gameweek -- i.e. "if you
    captained whoever this model rated highest every week, how many points
    did that actually return"."""
    points = []
    for _event, group in df.groupby("event"):
        group = group.dropna(subset=[pred_col])
        if group.empty:
            continue
        points.append(group.loc[group[pred_col].idxmax(), "total_points"])
    return float(np.mean(points)) if points else float("nan")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def find_degenerate_fpl_xp_gws(gw: pd.DataFrame) -> set[int]:
    """Gameweeks where vaastav's per-gameweek ``xP{N}.csv`` scrape file is
    missing from this clone of the repo, so ``merged_gw.csv``'s ``xP``
    column (renamed ``fpl_xp``) is backfilled with a flat 0.0 for *every*
    player that gameweek -- not a real FPL ``ep_next`` prediction. A
    genuine snapshot always has some spread (even heavily-benched players
    get a small non-zero xP), so "every single row is exactly 0.0" is a
    reliable signature of a missing source file rather than real data
    (verified against this clone: it matches the missing-``xPN.csv`` list
    exactly, 27 of 38 gameweeks). Scoring the FPL-xP baseline against these
    gameweeks would unfairly crater its measured accuracy against a vector
    of zeros it never actually produced."""
    all_zero = gw.groupby("GW")["fpl_xp"].apply(lambda s: (s == 0).all())
    return set(all_zero[all_zero].index)


def run_backtest() -> pd.DataFrame:
    """One row per player per backtested gameweek with our model's xP,
    all three baselines and the actual points scored. ``fpl_xp`` is NaN
    for gameweeks where the source data was missing (see
    ``find_degenerate_fpl_xp_gws``) so the FPL-xP baseline is only scored
    against gameweeks it actually had a real prediction for."""
    season_dir, gw = load_backtest_season()
    degenerate_fpl_xp_gws = find_degenerate_fpl_xp_gws(gw)
    players_like = build_players_like(gw)
    pairs = build_team_match_pairs(gw)
    scoring = pd.read_parquet(TABLES_DIR / "scoring_rules.parquet")

    prior_season_dir = find_prior_season_dir(before=season_dir.name)
    full_names = gw.groupby("player_id")["name"].first()
    prior = build_prior_season_rates(full_names, season_dir=prior_season_dir, cache_path=PRIOR_CACHE_PATH)

    last_gw = int(gw["GW"].max())
    results = []
    for target_gw in range(MIN_BACKTEST_GW, last_gw + 1):
        hist = gw[gw["GW"] < target_gw]
        features = build_features_asof(gw, players_like, pairs, target_gw)
        if features is None:
            continue
        features = attach_baselines(features, hist, target_gw)

        xp = build_xp(features, scoring, players_like, gw_so_far=target_gw - 1, prior=prior)
        xp = xp.merge(
            features[["player_id", "fixture_id", "total_points", "fpl_xp", "ppg", "last5", "actual_minutes", "p_start"]],
            on=["player_id", "fixture_id"],
            how="left",
        )
        # a double gameweek leaves two fixture-rows per player here; fold them
        # back into one row per player per gameweek, summing the per-match
        # quantities (predicted and actual points both add up across two
        # matches) and keeping the single season-level baselines as is.
        # p_start doesn't vary per fixture within a gameweek, so "max" is
        # just a way to collapse the (identical) duplicate rows.
        xp = xp.groupby(["player_id", "event"], as_index=False).agg(
            web_name=("web_name", "first"),
            position=("position", "first"),
            xp=("xp", "sum"),
            total_points=("total_points", "sum"),
            fpl_xp=("fpl_xp", "sum"),
            ppg=("ppg", "first"),
            last5=("last5", "first"),
            actual_minutes=("actual_minutes", "sum"),
            p_start=("p_start", "max"),
        )
        results.append(xp)

    results = pd.concat(results, ignore_index=True)
    # re-null after the fact (rather than relying on NaN surviving the
    # groupby sum above, which would silently turn an all-NaN double
    # gameweek back into 0.0) so degenerate gameweeks are excluded from
    # every FPL-xP metric, not just the ones computed on raw per-row data.
    results.loc[results["event"].isin(degenerate_fpl_xp_gws), "fpl_xp"] = np.nan
    return results


def summarize(results: pd.DataFrame) -> dict:
    """MAE, rank correlation by position, and calibration for every model,
    both over the full backtested season and split before/after the GW6
    blend cutoff so the "don't overfit early" blend's effect is visible.

    Pooled (all-rows) Spearman is dominated by the ~60% of player-gameweeks
    where nobody plays -- trivially rankable near zero by any model, and
    especially easy for ``ppg``/``last5`` since they *are* recent playing
    time. It's kept for reference only; the predicted-starter subset
    (``p_start > 0.5``), top-N hit rate and captain accuracy below are what
    actually drives transfer/captain decisions and should be the bar for
    adopting any change."""
    summary = {"n_gameweeks": results["event"].nunique(), "n_predictions": len(results)}
    likely_starters = results[results["p_start"] > 0.5]

    for col in BASELINE_COLUMNS:
        subset = results.dropna(subset=[col])
        summary[f"mae_{col}"] = mean_absolute_error(subset, col)
        summary[f"spearman_{col}"] = spearman_by_position(subset, col)
        early = subset[subset["event"] <= 6]
        late = subset[subset["event"] > 6]
        summary[f"spearman_{col}_early"] = spearman_by_position(early, col) if not early.empty else {}
        summary[f"spearman_{col}_late"] = spearman_by_position(late, col) if not late.empty else {}

        likely_subset = likely_starters.dropna(subset=[col])
        summary[f"spearman_{col}_likely_starter"] = spearman_by_position(likely_subset, col)
        summary[f"top10_{col}"] = topn_hit_rate(results, col, 10)
        summary[f"top20_{col}"] = topn_hit_rate(results, col, 20)
        summary[f"captain_{col}"] = captain_accuracy(results, col)

    summary["calibration_xp"] = calibration_table(results, "xp")
    return summary


def print_report(summary: dict) -> None:
    print(f"\nBacktested {summary['n_gameweeks']} gameweeks, {summary['n_predictions']} player-gameweek predictions.\n")

    print("Mean absolute error (lower is better):")
    for col, label in BASELINE_COLUMNS.items():
        print(f"  {label:<16} {summary[f'mae_{col}']:.3f}")

    print("\nSpearman rank correlation by position, full season, all rows (higher is better,")
    print("reference only -- dominated by non-playing players, see predicted-starter table below):")
    positions = sorted({p for col in BASELINE_COLUMNS for p in summary[f"spearman_{col}"]})
    header = "  " + "".join(f"{p:>8}" for p in positions)
    print(f"{'':<16}{header}")
    for col, label in BASELINE_COLUMNS.items():
        row = summary[f"spearman_{col}"]
        cells = "".join(f"{row.get(p, float('nan')):8.3f}" for p in positions)
        print(f"  {label:<14}{cells}")

    print("\nSpearman rank correlation by position, predicted likely starters only (p_start > 0.5) --")
    print("this is the subset that actually drives transfer/captain decisions:")
    print(f"{'':<16}{header}")
    for col, label in BASELINE_COLUMNS.items():
        row = summary[f"spearman_{col}_likely_starter"]
        cells = "".join(f"{row.get(p, float('nan')):8.3f}" for p in positions)
        print(f"  {label:<14}{cells}")

    print("\nTop-N hit rate (share of each model's top-N picks per position/gameweek that were")
    print("actually in the real top-N) and captain accuracy (points from each model's #1 overall")
    print("pick per gameweek):")
    print(f"{'':<16}{'top10':>8}{'top20':>8}{'captain':>10}")
    for col, label in BASELINE_COLUMNS.items():
        print(f"  {label:<14}{summary[f'top10_{col}']:8.3f}{summary[f'top20_{col}']:8.3f}"
              f"{summary[f'captain_{col}']:10.2f}")

    print("\nSpearman rank correlation by position -- GW<=6 (blend window) vs GW>6:")
    for col, label in BASELINE_COLUMNS.items():
        early = summary[f"spearman_{col}_early"]
        late = summary[f"spearman_{col}_late"]
        early_avg = np.mean(list(early.values())) if early else float("nan")
        late_avg = np.mean(list(late.values())) if late else float("nan")
        print(f"  {label:<16} GW<=6 avg: {early_avg:.3f}   GW>6 avg: {late_avg:.3f}")

    print("\nCalibration (our model), predicted xP bucket vs actual points:")
    print(summary["calibration_xp"].to_string())

    our_likely = summary["spearman_xp_likely_starter"]
    our_overall = np.mean(list(our_likely.values())) if our_likely else float("nan")
    ppg_likely = summary["spearman_ppg_likely_starter"]
    ppg_overall = np.mean(list(ppg_likely.values())) if ppg_likely else float("nan")
    verdict = "BEATS" if our_overall > ppg_overall else "DOES NOT beat"
    print(f"\nCheckpoint: our model {verdict} season-PPG on predicted-likely-starter rank "
          f"correlation ({our_overall:.3f} vs {ppg_overall:.3f}) -- this, not the pooled figure, "
          "is the adoption bar.")
    print("Note: FPL's ep_next (fpl_xp) numbers above exclude gameweeks where vaastav's source file")
    print("was missing (backfilled with 0 for everyone), but the surviving values are themselves")
    print("sourced from ep_this and may carry post-match lookahead bias per the cloned repo's own")
    print("DATA_DICTIONARY.md/README caveat -- treat fpl_xp's numbers as an optimistic upper bound,")
    print("not a clean baseline.")


def log_results(summary: dict) -> None:
    row = {
        "run_at": pd.Timestamp.now("UTC").isoformat(),
        "n_gameweeks": summary["n_gameweeks"],
        "n_predictions": summary["n_predictions"],
        "half_life_matches": HALF_LIFE_MATCHES,
        "shrinkage_matches": SHRINKAGE_MATCHES,
    }
    for col in BASELINE_COLUMNS:
        row[f"mae_{col}"] = summary[f"mae_{col}"]
        row[f"spearman_{col}_overall"] = np.mean(list(summary[f"spearman_{col}"].values())) if summary[f"spearman_{col}"] else np.nan
        likely = summary[f"spearman_{col}_likely_starter"]
        row[f"spearman_{col}_likely_starter"] = np.mean(list(likely.values())) if likely else np.nan
        row[f"top10_{col}"] = summary[f"top10_{col}"]
        row[f"top20_{col}"] = summary[f"top20_{col}"]
        row[f"captain_{col}"] = summary[f"captain_{col}"]
    log_df = pd.DataFrame([row])
    if RESULTS_LOG_PATH.exists():
        log_df = pd.concat([pd.read_csv(RESULTS_LOG_PATH), log_df], ignore_index=True)
    log_df.to_csv(RESULTS_LOG_PATH, index=False)


def main() -> None:
    print("Running backtest over the most recent complete season...")
    results = run_backtest()
    path = TABLES_DIR / "backtest_results.parquet"
    results.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(results)} rows)")

    summary = summarize(results)
    print_report(summary)
    log_results(summary)
    print(f"\nAppended this run's summary to {RESULTS_LOG_PATH}")


if __name__ == "__main__":
    main()
