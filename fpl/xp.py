"""Expected points (xP) -- builds ``data/tables/xp.parquet``, one row per
player per upcoming fixture (and a per-player-per-gameweek rollup for
doubles/blanks), scoring FPL's rules directly rather than extrapolating
"form" or "points per game".

Every term of FPL's scoring system is estimated separately from
``features.parquet`` (Step 5) and ``scoring_rules.parquet`` (Step 3), then
summed:

    xP = P_play * pts_appearance + P_60 * (pts_60mins - pts_appearance)
       + E[xG] * pts_goal + E[xA] * pts_assist
       + P_60 * P_CS * pts_clean_sheet
       - P_play * E[floor(goals_conceded / 2)] * pts_per_2_conceded
       + P_DC * pts_defcon
       + E[saves] / SAVES_PER_POINT * pts_save_group
       + E[bonus]
       - E[cards]

Two things keep this honest, per the spec:

1. **Minutes dominate.** Nothing else matters if a player doesn't play, so
   ``P_play`` (``p_start`` plus a small chance of a sub appearance) and the
   60-minutes-or-not split are computed first and gate almost every other
   term.
2. **Don't overfit early.** Before about gameweek 6, this season's rates are
   unreliable (a hot streak over 2 matches looks like a superstar), so they
   are blended with last season's rates for the same player (matched by
   name via ``rapidfuzz``, since player ids aren't stable across seasons).
   The blend fades out linearly and is fully this-season's numbers by GW6.

Run with `uv run python -m fpl.xp`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
EXTERNAL_DIR = ROOT / "data" / "external" / "fantasy-premier-league" / "data"
PLAYER_MAP_PATH = Path(__file__).resolve().parent / "prior_season_player_map.csv"

# --- known FPL rules not expressed in bootstrap-static's game_config -------
SAVES_PER_POINT = 3  # 1 point per 3 saves
BLEND_CUTOFF_GW = 6  # this-season rates are fully trusted from GW6 onward
NAME_MATCH_MIN_SCORE = 88  # rapidfuzz score below which a prior-season name match is dropped

# Heuristics the spec explicitly flags as optional/approximate, since the
# project has no direct possession or shots-faced table to calibrate them
# against. Kept small and documented rather than omitted, per the "saves90
# scaled up when the opponent's lambda is high" / "optionally raise P_DC
# against teams that dominate the ball" instructions.
SUB_APPEARANCE_RATE = 0.22  # among fit non-starters, the share who still get some bench minutes
SAVE_LAMBDA_ELASTICITY = 1.0  # saves scale ~linearly with shots faced, proxied by opponent's lambda
DC_OPPONENT_ELASTICITY = 0.3  # defensive actions rise more gently with opponent threat

# Features blended toward last season's player-level rate before GW6.
# p_start and minutes_factor are deliberately excluded: a new signing or a
# player who has changed role under a new manager makes last season's
# selection/minutes pattern actively misleading, whereas his underlying
# output rates (xG share, discipline, bonus magnetism, keeper workload)
# carry over reasonably well.
BLEND_COLUMNS = [
    "share_xg", "share_xa", "bonus_rate", "saves90", "defcon_rate",
    "p_60", "yellow_rate90", "red_rate90",
]

POSITION_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": None}


# ---------------------------------------------------------------------------
# prior-season rates (for the "don't overfit early" blend)
# ---------------------------------------------------------------------------


def find_prior_season_dir(before: str | None = None) -> Path | None:
    """The most recent *complete* season folder in the cloned vaastav repo --
    i.e. the one before whichever season folder is currently being updated
    live. Sorting season-folder names (``"2025-26"`` etc.) chronologically
    and taking the second-to-last one avoids hardcoding a season string that
    goes stale every August.

    ``before``, if given, instead returns the season immediately preceding
    that specific season string -- used by ``fpl.backtest`` to ask for
    "whatever counted as last season" relative to a past season being
    replayed, not relative to today."""
    if not EXTERNAL_DIR.exists():
        return None
    seasons = sorted(p.name for p in EXTERNAL_DIR.iterdir() if re.fullmatch(r"\d{4}-\d{2}", p.name))
    if before is not None:
        earlier = [s for s in seasons if s < before]
        return EXTERNAL_DIR / earlier[-1] if earlier else None
    if len(seasons) < 2:
        return None
    return EXTERNAL_DIR / seasons[-2]


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def build_prior_season_player_map(
    full_names: pd.Series, prior_names: pd.Series, cache_path: Path = PLAYER_MAP_PATH
) -> pd.Series:
    """Fuzzy-match each of this season's player names (``full_names``,
    indexed by player id) to a unique name in last season's merged gameweek
    file, caching the result (and its match score) to ``cache_path`` so the
    leftovers can be fixed by hand, same pattern as Step 4b's Understat name
    mapping.

    Returns a Series of prior-season names indexed by current player id,
    with no entry for players below the match-score threshold (new signings,
    promoted-division players, youth debutants -- anyone with no sensible
    prior-season rate)."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path).set_index("player_id")["prior_season_name"]
        return cached.dropna()

    choices = prior_names.unique().tolist()
    normalized_choices = {_normalize_name(c): c for c in choices}
    normalized_keys = list(normalized_choices.keys())
    normalized_full_names = full_names.map(_normalize_name)

    rows = []
    for player_id, full_name in normalized_full_names.items():
        if full_name in normalized_choices:
            rows.append((player_id, normalized_choices[full_name], 100.0))
            continue
        match = process.extractOne(full_name, normalized_keys, scorer=fuzz.WRatio)
        if match is not None and match[1] >= NAME_MATCH_MIN_SCORE:
            rows.append((player_id, normalized_choices[match[0]], match[1]))
        else:
            rows.append((player_id, None, match[1] if match else 0.0))

    mapping = pd.DataFrame(rows, columns=["player_id", "prior_season_name", "match_score"])
    mapping.to_csv(cache_path, index=False)
    return mapping.set_index("player_id")["prior_season_name"].dropna()


def build_prior_season_rates(
    full_names: pd.Series, season_dir: Path | None = None, cache_path: Path = PLAYER_MAP_PATH
) -> pd.DataFrame:
    """Season-long (no recency weighting -- there's no "recent" within a
    finished season) per-player rates from last season's merged gameweek
    file, re-indexed onto this season's player id (``full_names``'s index).
    Players with no match get no row and simply fall back to this season's
    own (position-shrunk) numbers in ``blend_with_prior_season``.

    ``season_dir``/``cache_path`` default to "whatever's one season before
    today" and the shared live cache file; ``fpl.backtest`` passes both
    explicitly so it can ask for an arbitrary past season's "prior season"
    without touching the live cache."""
    if season_dir is None:
        season_dir = find_prior_season_dir()
    empty = pd.DataFrame(columns=["player_id", *BLEND_COLUMNS]).set_index("player_id")
    if season_dir is None:
        return empty

    path = season_dir / "gws" / "merged_gw.csv"
    if not path.exists():
        return empty

    gw = pd.read_csv(path)
    gw["position"] = gw["position"].map(POSITION_MAP)
    gw = gw.dropna(subset=["position"])

    name_map = build_prior_season_player_map(full_names, gw["name"], cache_path)
    if name_map.empty:
        return empty

    gw["name_norm"] = gw["name"].map(_normalize_name)
    name_to_player = pd.Series(name_map.index, index=name_map.map(_normalize_name).to_numpy())

    team_match_xg = gw.groupby(["fixture", "team"])["expected_goals"].transform("sum")
    gw["match_share_xg"] = np.where(team_match_xg > 0, gw["expected_goals"] / team_match_xg, np.nan)
    team_match_xa = gw.groupby(["fixture", "team"])["expected_assists"].transform("sum")
    gw["match_share_xa"] = np.where(team_match_xa > 0, gw["expected_assists"] / team_match_xa, np.nan)
    threshold = gw["position"].map(DC_THRESHOLD)
    if "defensive_contribution" in gw.columns:
        gw["hit_defcon"] = (gw["defensive_contribution"] >= threshold).astype(float)
    else:
        # defensive contribution points didn't exist before 2025/26; a
        # season folder from before that rule simply contributes no
        # evidence for this rate (callers fall back to this-season-only).
        gw["hit_defcon"] = 0.0
    gw["played_60"] = (gw["minutes"] >= 60).astype(float)

    played = gw[gw["minutes"] > 0]
    minutes90 = played.groupby("name_norm")["minutes"].sum() / 90.0

    def per90(col: str) -> pd.Series:
        return played.groupby("name_norm")[col].sum() / minutes90

    def share(col: str) -> pd.Series:
        sub = played.dropna(subset=[col])
        return (sub[col] * sub["minutes"]).groupby(sub["name_norm"]).sum() / sub.groupby("name_norm")["minutes"].sum()

    started = played[played["starts"] == 1]

    rates = pd.DataFrame(
        {
            "share_xg": share("match_share_xg"),
            "share_xa": share("match_share_xa"),
            "bonus_rate": per90("bonus"),
            "saves90": per90("saves"),
            "defcon_rate": played.groupby("name_norm")["hit_defcon"].mean(),
            "p_60": started.groupby("name_norm")["played_60"].mean(),
            "yellow_rate90": per90("yellow_cards"),
            "red_rate90": per90("red_cards"),
        }
    )
    position_by_name = gw.groupby("name_norm")["position"].first().reindex(rates.index)
    rates.loc[position_by_name == "GKP", "defcon_rate"] = 0.0
    rates.loc[position_by_name != "GKP", "saves90"] = 0.0

    rates = rates.reindex(name_to_player.index)
    rates.index = name_to_player.to_numpy()
    rates.index.name = "player_id"
    return rates[~rates.index.duplicated()]


def blend_with_prior_season(features: pd.DataFrame, prior: pd.DataFrame, gw_so_far: int) -> pd.DataFrame:
    """Blend each of ``BLEND_COLUMNS`` toward the matching prior-season rate,
    linearly fading the prior out as this season's own sample grows: full
    weight on last season at GW0 (preseason), none from GW6 onward. Players
    absent from ``prior`` (no good name match) keep this season's own
    (already position-shrunk) value untouched."""
    features = features.copy()
    this_season_weight = min(1.0, gw_so_far / BLEND_CUTOFF_GW)
    prior_aligned = prior.reindex(features["player_id"])
    prior_aligned.index = features.index
    has_prior = prior_aligned.notna()

    for col in BLEND_COLUMNS:
        prior_col = prior_aligned[col]
        blended = this_season_weight * features[col] + (1 - this_season_weight) * prior_col
        features[col] = np.where(has_prior[col], blended, features[col])
    return features


# ---------------------------------------------------------------------------
# the scoring-rule terms
# ---------------------------------------------------------------------------


def expected_floor_half(lam: np.ndarray, k_max: int = 14) -> np.ndarray:
    """E[floor(X/2)] for X ~ Poisson(lam), computed from the actual
    distribution (not lam/2, which is a biased shortcut) -- the spec is
    explicit that goals-conceded points should be modeled this way since
    the floor makes the naive halving wrong on average."""
    lam = np.asarray(lam, dtype=float)
    k = np.arange(0, k_max + 1)
    pmf = poisson.pmf(k[None, :], lam[:, None])
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)  # fold negligible tail mass into k_max
    return (pmf * (k // 2)[None, :]).sum(axis=1)


def build_xp(
    features: pd.DataFrame,
    scoring: pd.DataFrame,
    players: pd.DataFrame,
    gw_so_far: int,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``prior`` defaults to last complete season's rates for today's
    players (the live behaviour); ``fpl.backtest`` passes its own
    pre-computed prior so a replayed gameweek blends toward *its* prior
    season, not toward whatever's one season before today."""
    if prior is None:
        full_names = players.set_index("id")["first_name"] + " " + players.set_index("id")["second_name"]
        prior = build_prior_season_rates(full_names)
    features = blend_with_prior_season(features, prior, gw_so_far)

    fitness_by_player = (players.set_index("id")["chance_of_playing_next_round"].fillna(100.0) / 100.0)
    fitness = features["player_id"].map(fitness_by_player).fillna(1.0)

    scoring = scoring.set_index("position")
    pts_appearance = features["position"].map(scoring["points_per_appearance"])
    pts_60 = features["position"].map(scoring["points_per_60_mins"])
    pts_goal = features["position"].map(scoring["points_per_goal"])
    pts_assist = features["position"].map(scoring["points_per_assist"])
    pts_cs = features["position"].map(scoring["points_per_clean_sheet"])
    pts_per_2_conceded = features["position"].map(scoring["points_per_2_goals_conceded"]).abs()
    pts_defcon = features["position"].map(scoring["points_per_defensive_contribution"])
    pts_save_group = features["position"].map(scoring["points_per_save_group"])
    pts_yellow = features["position"].map(scoring["points_per_yellow_card"]).abs()
    pts_red = features["position"].map(scoring["points_per_red_card"]).abs()

    # --- minutes: the term the spec says to spend the most care on -------
    # fit non-starters still get a small chance of a bench cameo; an unfit
    # one (fitness near 0) realistically doesn't even make the squad.
    p_play = features["p_start"] + (1 - features["p_start"]) * SUB_APPEARANCE_RATE * fitness
    p_60plus = features["p_start"] * features["p_60"]
    expected_minutes_frac = p_play * features["minutes_factor"]

    league_avg_lambda = features["lambda_for"].mean()

    # --- goal involvements: a player's usual share of his team's chances,
    # scaled by this fixture's threat and by how much of it he'll play ------
    expected_xg = features["share_xg"] * features["lambda_for"] * expected_minutes_frac
    expected_xa = features["share_xa"] * features["lambda_for"] * expected_minutes_frac

    # --- clean sheets / goals conceded, both driven by the Poisson(lambda_against)
    # the opponent is expected to score -----------------------------------
    clean_sheet_component = p_60plus * features["clean_sheet_prob"] * pts_cs
    conceded_penalty = p_play * expected_floor_half(features["lambda_against"].to_numpy()) * pts_per_2_conceded

    # --- defensive contribution, nudged up against a tougher (more
    # possession-dominant) opponent -- an explicitly optional, undercalibrated
    # heuristic per the spec -----------------------------------------------
    dc_adjustment = np.clip(
        1 + DC_OPPONENT_ELASTICITY * (features["lambda_against"] / league_avg_lambda - 1), 0.8, 1.3
    )
    defcon_component = features["defcon_rate"] * expected_minutes_frac * dc_adjustment * pts_defcon

    # --- saves, scaled up when the opponent creates more than an average
    # amount of threat (their lambda_for against this team, i.e. our
    # lambda_against) -------------------------------------------------------
    save_adjustment = np.clip(
        1 + SAVE_LAMBDA_ELASTICITY * (features["lambda_against"] / league_avg_lambda - 1), 0.5, 2.0
    )
    expected_saves = features["saves90"] * expected_minutes_frac * save_adjustment
    saves_component = (expected_saves / SAVES_PER_POINT) * pts_save_group

    # --- bonus: start with the historical rate (already points/90) -------
    expected_bonus = features["bonus_rate"] * expected_minutes_frac

    # --- cards: a rate (per 90) times the points lost, scaled by minutes --
    expected_cards = (
        features["yellow_rate90"] * pts_yellow + features["red_rate90"] * pts_red
    ) * expected_minutes_frac

    appearance_component = p_play * pts_appearance + p_60plus * (pts_60 - pts_appearance)

    xp = (
        appearance_component
        + expected_xg * pts_goal
        + expected_xa * pts_assist
        + clean_sheet_component
        - conceded_penalty
        + defcon_component
        + saves_component
        + expected_bonus
        - expected_cards
    )

    out = features[
        ["fixture_id", "event", "kickoff_time", "is_home", "player_id", "web_name",
         "position", "now_cost", "team_short", "opponent_short", "source"]
    ].copy()
    out["p_play"] = p_play
    out["p_60"] = p_60plus
    out["expected_minutes"] = expected_minutes_frac * 90
    out["expected_xg"] = expected_xg
    out["expected_xa"] = expected_xa
    out["expected_goal_pts"] = expected_xg * pts_goal
    out["expected_assist_pts"] = expected_xa * pts_assist
    out["clean_sheet_pts"] = clean_sheet_component
    out["conceded_pts"] = -conceded_penalty
    out["defcon_pts"] = defcon_component
    out["saves_pts"] = saves_component
    out["bonus_pts"] = expected_bonus
    out["card_pts"] = -expected_cards
    out["appearance_pts"] = appearance_component
    out["xp"] = xp
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def current_gameweek(fixtures: pd.DataFrame) -> int:
    finished = fixtures.loc[fixtures["finished"], "event"]
    return int(finished.max()) if not finished.empty else 0


def build_xp_table() -> pd.DataFrame:
    features = pd.read_parquet(TABLES_DIR / "features.parquet")
    scoring = pd.read_parquet(TABLES_DIR / "scoring_rules.parquet")
    fixtures = pd.read_parquet(TABLES_DIR / "fixtures.parquet")
    players = pd.read_parquet(TABLES_DIR / "players.parquet")

    gw_so_far = current_gameweek(fixtures)
    xp = build_xp(features, scoring, players, gw_so_far)
    return xp.sort_values(["event", "xp"], ascending=[True, False]).reset_index(drop=True)


def build_xp_by_gameweek(xp: pd.DataFrame) -> pd.DataFrame:
    """One row per player per gameweek, summing doubles and leaving blanks
    at 0 -- the horizon's worth of gameweeks the spec asks to store."""
    horizon_events = sorted(xp["event"].unique())
    players = xp[["player_id", "web_name", "position", "team_short", "now_cost"]].drop_duplicates("player_id")
    totals = xp.groupby(["player_id", "event"])["xp"].sum().unstack(fill_value=0.0)
    totals = totals.reindex(columns=horizon_events, fill_value=0.0)
    totals.columns = [f"gw{e}_xp" for e in totals.columns]
    totals["xp_total"] = totals.sum(axis=1)
    return players.merge(totals, on="player_id", how="left")


PREDICTIONS_DIR = ROOT / "data" / "predictions"


def log_live_predictions(xp: pd.DataFrame) -> Path:
    """Snapshot this run's predictions with a timestamp (Step 7's "from this
    season onward, log every live prediction... After each gameweek, score
    it automatically" requirement). Every run gets its own file -- nothing
    is overwritten -- since scoring a prediction later (Step 11) needs to
    know exactly what was predicted and when, not just the latest one."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    logged_at = pd.Timestamp.now("UTC")
    snapshot = xp[
        ["fixture_id", "event", "player_id", "web_name", "position", "team_short", "opponent_short", "xp"]
    ].copy()
    snapshot.insert(0, "logged_at", logged_at)
    path = PREDICTIONS_DIR / f"{logged_at.strftime('%Y-%m-%dT%H%M%S')}.parquet"
    snapshot.to_parquet(path, index=False)
    return path


def main() -> None:
    print("Building xp.parquet...")
    xp = build_xp_table()
    path = TABLES_DIR / "xp.parquet"
    xp.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(xp)} rows, {xp['player_id'].nunique()} players)")

    by_gw = build_xp_by_gameweek(xp)
    gw_path = TABLES_DIR / "xp_by_gameweek.parquet"
    by_gw.to_parquet(gw_path, index=False)
    print(f"  wrote {gw_path} ({len(by_gw)} players x {len(by_gw.columns) - 5} gameweeks)")

    log_path = log_live_predictions(xp)
    print(f"  logged predictions snapshot to {log_path}")


if __name__ == "__main__":
    main()
