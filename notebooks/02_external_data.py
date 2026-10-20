import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    TABLES = Path(__file__).resolve().parent.parent / "data" / "tables"
    return TABLES, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # Step 4 checkpoint: FPL difficulty, Elo, rolling xG, and odds-implied goals

        For every fixture in the next unplayed gameweek, line up the four signals
        side by side:

        - **FPL difficulty** (`fixtures.parquet`) -- barely moves all season.
        - **Elo gap** (`elo.parquet`) -- reacts after every match.
        - **Rolling xG** (`team_xg.parquet`) -- attack/defence form over the last 6/12 games.
        - **Odds-implied expected goals** (`odds_goals.parquet`) -- the market's view, when available.

        Where they disagree is where the edge lives.
        """
    )
    return


@app.cell
def _(TABLES, pd):
    teams = pd.read_parquet(TABLES / "teams.parquet")
    fixtures = pd.read_parquet(TABLES / "fixtures.parquet")
    aliases = pd.read_csv(TABLES.parent.parent / "fpl" / "team_aliases.csv")

    next_event = fixtures.loc[~fixtures["finished"], "event"].min()
    upcoming = fixtures[fixtures["event"] == next_event].copy()
    return aliases, fixtures, next_event, teams, upcoming


@app.cell
def _(mo, next_event):
    mo.md(f"## Gameweek {next_event}")
    return


@app.cell
def _(TABLES, aliases, pd, upcoming):
    def latest_elo() -> pd.DataFrame:
        path = TABLES / "elo.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["team_id", "elo"])
        elo = pd.read_parquet(path)
        latest = elo.sort_values("date").groupby("team_id", as_index=False).tail(1)
        return latest[["team_id", "elo"]]

    def latest_rolling_xg() -> pd.DataFrame:
        path = TABLES / "team_xg.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["team_id", "attack_xg_6", "defence_xg_6", "attack_xg_12", "defence_xg_12"])
        team_xg = pd.read_parquet(path)
        latest = team_xg.sort_values("date").groupby("team_id", as_index=False).tail(1)
        return latest[["team_id", "attack_xg_6", "defence_xg_6", "attack_xg_12", "defence_xg_12"]]

    def odds_by_event() -> pd.DataFrame:
        path = TABLES / "odds_goals.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["home_team_id", "away_team_id", "lambda_home", "lambda_away"])
        return pd.read_parquet(path)[["home_team_id", "away_team_id", "lambda_home", "lambda_away"]]

    elo_latest = latest_elo()
    xg_latest = latest_rolling_xg()
    odds = odds_by_event()

    board = upcoming[["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "kickoff_time"]].copy()
    short_name = aliases.set_index("fpl_team_id")["fpl_short_name"]
    board["home"] = board["team_h"].map(short_name)
    board["away"] = board["team_a"].map(short_name)

    board = board.merge(elo_latest.rename(columns={"team_id": "team_h", "elo": "home_elo"}), on="team_h", how="left")
    board = board.merge(elo_latest.rename(columns={"team_id": "team_a", "elo": "away_elo"}), on="team_a", how="left")
    board["elo_gap"] = board["home_elo"] - board["away_elo"]

    board = board.merge(
        xg_latest.rename(columns={"team_id": "team_h"}).add_prefix("home_").rename(columns={"home_team_h": "team_h"}),
        on="team_h",
        how="left",
    )
    board = board.merge(
        xg_latest.rename(columns={"team_id": "team_a"}).add_prefix("away_").rename(columns={"away_team_a": "team_a"}),
        on="team_a",
        how="left",
    )

    board = board.merge(odds, left_on=["team_h", "team_a"], right_on=["home_team_id", "away_team_id"], how="left")

    cols = [
        "kickoff_time", "home", "away",
        "team_h_difficulty", "team_a_difficulty",
        "home_elo", "away_elo", "elo_gap",
        "home_attack_xg_6", "home_defence_xg_6", "away_attack_xg_6", "away_defence_xg_6",
        "lambda_home", "lambda_away",
    ]
    checkpoint_table = board[cols].sort_values("kickoff_time").reset_index(drop=True)
    return checkpoint_table, elo_latest, odds, xg_latest


@app.cell
def _(checkpoint_table, mo):
    mo.ui.table(checkpoint_table)
    return


@app.cell
def _(mo, odds):
    mo.md(
        "_`lambda_home`/`lambda_away` are blank above until `ODDS_API_KEY` is set and "
        "`uv run python -m fpl.odds` has been run; every other column works offline._"
        if odds.empty
        else "_All four signals are populated for this gameweek._"
    )
    return


if __name__ == "__main__":
    app.run()
