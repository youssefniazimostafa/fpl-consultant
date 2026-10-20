"""Feature engineering -- builds ``data/tables/features.parquet``, one row
per player per upcoming fixture (two rows for a double gameweek, none for a
blank), with every number the model needs.

Two kinds of feature, joined together:

**Player rates** (from ``player_matches``, this season only): exponentially
decay-weighted so recent matches count more (half-life ~6 matches), and
shrunk toward the player's position average so a 30-minute cameo doesn't
look like a superstar (an empirical-Bayes blend -- see
``_weighted_rate_per90`` / ``_weighted_rate``).

**Fixture features** (from Step 4): each team's expected goals for/against
and clean-sheet probability for its next match(es). Odds-implied numbers
(``odds_goals.parquet``) are used when they cover a fixture; anything
further out falls back to a blend of Elo and rolling Understat xG, since
odds markets only open a week or two ahead.

The key link between the two: a player's expected goals in a match is
roughly ``share_xg * lambda_for * p_start * minutes_factor`` -- his usual
slice of his team's chances, scaled by how threatening this fixture is for
his team, by whether he plays at all, and by how much of the match he
usually gets through once selected.

Run with `uv run python -m fpl.features`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fpl.fetch import load_config

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"

HALF_LIFE_MATCHES = 6  # recency decay: a match 6 back carries half the weight of the most recent one
SHRINKAGE_MATCHES = 3  # pseudo-matches of "position average" blended in for players with little evidence
P_START_WINDOW = 6  # "last 6" team matches, per the spec
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": None}  # 2025/26 defensive-contribution rule
HOME_ADV_ELO = 100  # standard rule-of-thumb home-field Elo bonus (clubelo-style Elo systems)
DEFAULT_HORIZON = 5


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["players", "teams", "fixtures", "player_matches"]:
        tables[name] = pd.read_parquet(TABLES_DIR / f"{name}.parquet")
    for name in ["elo", "team_xg", "odds_goals"]:
        path = TABLES_DIR / f"{name}.parquet"
        tables[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    tables["aliases"] = pd.read_csv(ALIASES_PATH)

    # expected_* columns come back from the FPL API as strings; every other
    # table reader in this project casts what it needs, so do the same here.
    pm = tables["player_matches"]
    for col in ["expected_goals", "expected_assists"]:
        pm[col] = pd.to_numeric(pm[col])
    pm["kickoff_time"] = pd.to_datetime(pm["kickoff_time"])

    f = tables["fixtures"]
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"])
    return tables


def attach_match_team(player_matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Each match's actual team (not the player's *current* team), so a
    mid-season transfer's history still lines up with the right club."""
    pm = player_matches.merge(fixtures[["id", "team_h", "team_a"]], left_on="fixture", right_on="id", how="left")
    pm["team_id"] = np.where(pm["was_home"], pm["team_h"], pm["team_a"])
    return pm.drop(columns=["id", "team_h", "team_a"])


# ---------------------------------------------------------------------------
# recency weighting + empirical-Bayes shrinkage toward the position average
# ---------------------------------------------------------------------------


def add_recency_weight(pm: pd.DataFrame) -> pd.DataFrame:
    """weight = 0.5 ** (matches_ago / HALF_LIFE_MATCHES), most recent match
    per player first."""
    pm = pm.sort_values(["player_id", "kickoff_time"], ascending=[True, False]).copy()
    matches_ago = pm.groupby("player_id").cumcount()
    pm["weight"] = 0.5 ** (matches_ago / HALF_LIFE_MATCHES)
    return pm


def _weighted_rate_per90(pm: pd.DataFrame, value_col: str, position_by_player: pd.Series) -> pd.Series:
    """Shrunk, recency-weighted `value_col` per 90 minutes, blended toward
    the player's position average using `SHRINKAGE_MATCHES` pseudo-matches
    of evidence (so a handful of cameo minutes can't look elite).

    `position_by_player` must be indexed by player_id (not by match row).
    Every player in `position_by_player` is included in the result, even one
    with zero matches in `pm` -- they get exactly their position's average
    (zero evidence + the shrinkage prior)."""
    w = pm["weight"]
    w90 = w * pm["minutes"] / 90.0
    num = (w * pm[value_col]).groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)
    den = w90.groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)

    pos_num = num.groupby(position_by_player).transform("sum")
    pos_den = den.groupby(position_by_player).transform("sum")
    prior = (pos_num / pos_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return (num + prior * SHRINKAGE_MATCHES) / (den + SHRINKAGE_MATCHES)


def _weighted_rate(pm: pd.DataFrame, flag_col: str, position_by_player: pd.Series) -> pd.Series:
    """Shrunk, recency-weighted share of matches where `flag_col` is true
    (e.g. hit the defensive-contribution threshold), not a per-90 rate.

    `position_by_player` must be indexed by player_id (not by match row).
    Every player in `position_by_player` is included in the result, even one
    with zero matches in `pm` -- they get exactly their position's average
    (zero evidence + the shrinkage prior)."""
    w = pm["weight"]
    num = (w * pm[flag_col]).groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)
    den = w.groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)

    pos_num = num.groupby(position_by_player).transform("sum")
    pos_den = den.groupby(position_by_player).transform("sum")
    prior = (pos_num / pos_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return (num + prior * SHRINKAGE_MATCHES) / (den + SHRINKAGE_MATCHES)


# ---------------------------------------------------------------------------
# player rate features
# ---------------------------------------------------------------------------


def build_player_rates(pm: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    pm = add_recency_weight(pm)
    position_by_player = players.set_index("id")["position"]

    # team xG per match = the sum of every one of that match's player rows'
    # own expected_goals, i.e. built entirely from FPL's own numbers so it
    # stays keyed on the same fixture id as the player rows (no need to
    # reconcile Understat's separate match ids here).
    team_match_xg = pm.groupby(["fixture", "team_id"])["expected_goals"].transform("sum")
    pm["match_share_xg"] = np.where(team_match_xg > 0, pm["expected_goals"] / team_match_xg, np.nan)

    threshold = pm["player_id"].map(position_by_player).map(DC_THRESHOLD)
    pm["hit_defcon"] = (pm["defensive_contribution"] >= threshold).astype(float)
    pm["played_60"] = (pm["minutes"] >= 60).astype(float)

    rates = pd.DataFrame(index=position_by_player.index)
    rates.index.name = "player_id"
    rates["xg90"] = _weighted_rate_per90(pm, "expected_goals", position_by_player)
    rates["xa90"] = _weighted_rate_per90(pm, "expected_assists", position_by_player)
    rates["saves90"] = _weighted_rate_per90(pm, "saves", position_by_player)
    rates["bonus_rate"] = _weighted_rate_per90(pm, "bonus", position_by_player)

    # share_xg and p_60 are match-level ratios, not per-90 counts: weight
    # each match's evidence by (recency weight * minutes/90) instead.
    share_pm = pm.dropna(subset=["match_share_xg"]).copy()
    share_pm["weight"] = share_pm["weight"] * share_pm["minutes"] / 90.0
    rates["share_xg"] = (
        _weighted_rate(share_pm, "match_share_xg", position_by_player) if not share_pm.empty else np.nan
    )

    started = pm[pm["starts"] == 1]
    rates["p_60"] = _weighted_rate(started, "played_60", position_by_player) if not started.empty else np.nan

    rates["defcon_rate"] = _weighted_rate(pm, "hit_defcon", position_by_player)
    rates.loc[position_by_player == "GKP", "defcon_rate"] = 0.0  # keepers don't earn defensive-contribution points
    rates.loc[position_by_player != "GKP", "saves90"] = 0.0  # saves90 only makes sense for keepers

    return rates.reset_index()


def build_p_start(pm: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.Series:
    """Starts over the player's *current* team's last `P_START_WINDOW`
    matches, scaled down by `chance_of_playing_next_round` (which is 0 for
    injured/suspended/unavailable players and 100 or blank -- i.e. fully fit
    -- for everyone else healthy)."""
    finished = fixtures[fixtures["finished"]]
    recent_fixture_ids: dict[int, set[int]] = {}
    for team_id in players["team"].unique():
        team_fixtures = finished[(finished["team_h"] == team_id) | (finished["team_a"] == team_id)]
        recent = team_fixtures.sort_values("kickoff_time", ascending=False).head(P_START_WINDOW)
        recent_fixture_ids[team_id] = set(recent["id"])

    pm_team = pm.merge(players[["id", "team"]], left_on="player_id", right_on="id", suffixes=("", "_current"))
    pm_team = pm_team[pm_team.apply(lambda r: r["fixture"] in recent_fixture_ids.get(r["team"], set()), axis=1)]
    recent_starts = pm_team[pm_team["starts"] == 1].groupby("player_id").size()

    team_match_counts = players["team"].map(lambda t: len(recent_fixture_ids.get(t, set())))
    denom = team_match_counts.replace(0, np.nan)
    p_start_base = (recent_starts.reindex(players["id"]).fillna(0).to_numpy()) / denom.to_numpy()

    fitness = players["chance_of_playing_next_round"].fillna(100.0) / 100.0
    p_start = pd.Series(p_start_base, index=players["id"]) * fitness.to_numpy()
    return p_start.clip(0, 1).fillna(0.0)


def build_minutes_factor(pm: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    """Shrunk, recency-weighted average of minutes/90 across matches the
    player actually appeared in -- how big a chunk of the game he eats up
    once selected."""
    pm = add_recency_weight(pm)
    position_by_player = players.set_index("id")["position"]
    w = pm["weight"]
    num = (w * pm["minutes"] / 90.0).groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)
    den = w.groupby(pm["player_id"]).sum().reindex(position_by_player.index, fill_value=0.0)
    pos_num = num.groupby(position_by_player).transform("sum")
    pos_den = den.groupby(position_by_player).transform("sum")
    prior = (pos_num / pos_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (num + prior * SHRINKAGE_MATCHES) / (den + SHRINKAGE_MATCHES)


def build_set_piece_features(players: pd.DataFrame) -> pd.DataFrame:
    sp = players[["id", "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"]].copy()
    sp = sp.rename(
        columns={
            "penalties_order": "penalty_order",
            "direct_freekicks_order": "direct_fk_order",
            "corners_and_indirect_freekicks_order": "corner_fk_order",
        }
    )
    sp["takes_penalties"] = sp["penalty_order"] == 1
    return sp.rename(columns={"id": "player_id"})


# ---------------------------------------------------------------------------
# fixture features: lambda_for / lambda_against / clean_sheet_prob per team
# ---------------------------------------------------------------------------


def build_fixture_team_table(fixtures: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per team per unplayed fixture within the next `horizon`
    gameweeks -- a team with a double gameweek gets two rows, one with a
    blank gets none, exactly matching how those fixtures appear upstream."""
    upcoming = fixtures[~fixtures["finished"]].copy()
    next_events = sorted(upcoming["event"].dropna().unique())[:horizon]
    upcoming = upcoming[upcoming["event"].isin(next_events)]

    home = upcoming.rename(columns={"team_h": "team_id", "team_a": "opponent_id"}).assign(is_home=True)
    away = upcoming.rename(columns={"team_a": "team_id", "team_h": "opponent_id"}).assign(is_home=False)
    cols = ["id", "event", "kickoff_time", "team_id", "opponent_id", "is_home"]
    return pd.concat([home[cols], away[cols]], ignore_index=True).rename(columns={"id": "fixture_id"})


def _elo_xg_fallback_lambdas(fixture_team: pd.DataFrame, elo: pd.DataFrame, team_xg: pd.DataFrame) -> pd.DataFrame:
    latest_elo = elo.sort_values("date").groupby("team_id")["elo"].last() if not elo.empty else pd.Series(dtype=float)
    if not team_xg.empty:
        latest_xg = team_xg.sort_values("date").groupby("team_id").last()
        attack_12 = latest_xg["attack_xg_12"]
        defence_12 = latest_xg["defence_xg_12"]
        league_avg = team_xg["xg_for"].mean()
        league_avg_home = team_xg.loc[team_xg["is_home"], "xg_for"].mean()
        league_avg_away = team_xg.loc[~team_xg["is_home"], "xg_for"].mean()
    else:
        attack_12 = defence_12 = pd.Series(dtype=float)
        league_avg = league_avg_home = league_avg_away = np.nan

    def lambda_for_row(team_id: int, opp_id: int, is_home: bool) -> float:
        league_base = league_avg_home if is_home else league_avg_away
        estimates = []

        if team_id in attack_12.index and opp_id in defence_12.index and pd.notna(league_avg):
            attack_index = attack_12[team_id] / league_avg
            defence_index = defence_12[opp_id] / league_avg
            estimates.append(league_base * attack_index * defence_index)

        if team_id in latest_elo.index and opp_id in latest_elo.index and pd.notna(league_base):
            elo_diff = latest_elo[team_id] - latest_elo[opp_id] + (HOME_ADV_ELO if is_home else -HOME_ADV_ELO)
            p_this_side = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
            estimates.append(league_base * 2 * p_this_side)

        if estimates:
            return float(np.mean(estimates))
        return float(league_base) if pd.notna(league_base) else np.nan

    return fixture_team.apply(
        lambda r: lambda_for_row(r["team_id"], r["opponent_id"], r["is_home"]), axis=1
    )


def attach_fixture_lambdas(fixture_team: pd.DataFrame, odds_goals: pd.DataFrame, elo: pd.DataFrame, team_xg: pd.DataFrame) -> pd.DataFrame:
    fixture_team = fixture_team.copy()
    fixture_team["lambda_for"] = np.nan
    fixture_team["source"] = "elo_xg_blend"

    if not odds_goals.empty:
        home_rows = fixture_team["is_home"]
        home_match = fixture_team.loc[home_rows].merge(
            odds_goals[["home_team_id", "away_team_id", "lambda_home", "home_clean_sheet_prob"]],
            left_on=["team_id", "opponent_id"],
            right_on=["home_team_id", "away_team_id"],
            how="left",
        )
        away_match = fixture_team.loc[~home_rows].merge(
            odds_goals[["home_team_id", "away_team_id", "lambda_away", "away_clean_sheet_prob"]],
            left_on=["opponent_id", "team_id"],
            right_on=["home_team_id", "away_team_id"],
            how="left",
        )
        fixture_team.loc[home_rows, "lambda_for"] = home_match["lambda_home"].to_numpy()
        fixture_team.loc[home_rows, "clean_sheet_prob"] = home_match["home_clean_sheet_prob"].to_numpy()
        fixture_team.loc[~home_rows, "lambda_for"] = away_match["lambda_away"].to_numpy()
        fixture_team.loc[~home_rows, "clean_sheet_prob"] = away_match["away_clean_sheet_prob"].to_numpy()
        fixture_team.loc[fixture_team["lambda_for"].notna(), "source"] = "odds"

    missing = fixture_team["lambda_for"].isna()
    if missing.any():
        fixture_team.loc[missing, "lambda_for"] = _elo_xg_fallback_lambdas(fixture_team.loc[missing], elo, team_xg)

    # lambda_against = the opponent's lambda_for in the same fixture, so
    # clean_sheet_prob (odds-sourced or not) always lines up with lambda_against.
    by_fixture_team = fixture_team.set_index(["fixture_id", "team_id"])["lambda_for"]
    fixture_team["lambda_against"] = fixture_team.apply(
        lambda r: by_fixture_team.get((r["fixture_id"], r["opponent_id"]), np.nan), axis=1
    )
    if "clean_sheet_prob" not in fixture_team.columns:
        fixture_team["clean_sheet_prob"] = np.nan
    fallback_cs = fixture_team["clean_sheet_prob"].isna() & fixture_team["lambda_against"].notna()
    fixture_team.loc[fallback_cs, "clean_sheet_prob"] = np.exp(-fixture_team.loc[fallback_cs, "lambda_against"])

    return fixture_team


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_features() -> pd.DataFrame:
    tables = load_tables()
    players, teams, fixtures = tables["players"], tables["teams"], tables["fixtures"]
    pm = attach_match_team(tables["player_matches"], fixtures)

    rates = build_player_rates(pm, players)
    p_start = build_p_start(pm, players, fixtures).rename("p_start")
    minutes_factor = build_minutes_factor(pm, players).rename("minutes_factor")
    set_pieces = build_set_piece_features(players)

    player_features = (
        players[["id", "web_name", "team", "position", "now_cost"]]
        .rename(columns={"id": "player_id", "team": "team_id"})
        .merge(rates, on="player_id", how="left")
        .merge(p_start, left_on="player_id", right_index=True, how="left")
        .merge(minutes_factor, left_on="player_id", right_index=True, how="left")
        .merge(set_pieces, on="player_id", how="left")
    )

    horizon = load_config().get("horizon") or DEFAULT_HORIZON
    fixture_team = build_fixture_team_table(fixtures, horizon)
    fixture_team = attach_fixture_lambdas(fixture_team, tables["odds_goals"], tables["elo"], tables["team_xg"])

    aliases = tables["aliases"].set_index("fpl_team_id")["fpl_short_name"]
    fixture_team["team_short"] = fixture_team["team_id"].map(aliases)
    fixture_team["opponent_short"] = fixture_team["opponent_id"].map(aliases)

    features = fixture_team.merge(player_features, on="team_id", how="left")
    features["expected_player_goals"] = (
        features["share_xg"] * features["lambda_for"] * features["p_start"] * features["minutes_factor"]
    )

    columns = [
        "fixture_id", "event", "kickoff_time", "is_home",
        "player_id", "web_name", "position", "now_cost",
        "team_id", "team_short", "opponent_id", "opponent_short",
        "xg90", "xa90", "share_xg", "defcon_rate", "saves90", "bonus_rate",
        "p_start", "p_60", "minutes_factor",
        "penalty_order", "direct_fk_order", "corner_fk_order", "takes_penalties",
        "lambda_for", "lambda_against", "clean_sheet_prob", "source",
        "expected_player_goals",
    ]
    return features[columns].sort_values(["event", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)


def main() -> None:
    print("Building features.parquet...")
    features = build_features()
    path = TABLES_DIR / "features.parquet"
    features.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(features)} rows, {features['player_id'].nunique()} players)")


if __name__ == "__main__":
    main()
