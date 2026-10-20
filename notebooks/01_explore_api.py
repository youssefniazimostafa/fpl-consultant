import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import requests
    import marimo as mo

    return mo, pd, requests


@app.cell
def _(requests):
    BASE_URL = "https://fantasy.premierleague.com/api"
    HEADERS = {"User-Agent": "fpl-consultant"}

    def get_json(path, params=None):
        response = requests.get(
            f"{BASE_URL}/{path}",
            headers=HEADERS,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    return (get_json,)


@app.cell
def _(get_json):
    bootstrap = get_json("bootstrap-static/")
    players = bootstrap["elements"]
    teams = bootstrap["teams"]
    events = bootstrap["events"]
    return bootstrap, events, players, teams


@app.cell
def _(bootstrap, mo, pd, players, teams):
    players_df = pd.DataFrame(players)
    teams_df = pd.DataFrame(teams)

    mo.vstack(
        [
            mo.md("## Bootstrap: players, teams, gameweeks"),
            mo.md(f"**Bootstrap keys:** `{list(bootstrap.keys())}`"),
            mo.md("**Player columns (complete):**"),
            mo.md(f"`{players_df.columns.tolist()}`"),
            mo.ui.table(teams_df),
        ]
    )
    return players_df, teams_df


@app.cell
def _(bootstrap, mo, pd):
    mo.md("## Scoring configuration")

    if "game_config" in bootstrap:
        game_config = bootstrap["game_config"]
        config_view = mo.vstack(
            [
                mo.md("`game_config` is present in this season's bootstrap response."),
                mo.ui.table(pd.json_normalize(game_config)),
            ]
        )
    else:
        game_config = None
        config_view = mo.md("`game_config` is not present in this response.")

    if "game_settings" in bootstrap:
        game_settings = bootstrap["game_settings"]
        scoring = game_settings.get("scoring")
        if scoring is not None:
            settings_view = mo.vstack(
                [
                    mo.md("`game_settings` is also included; inspect its scoring fields below."),
                    mo.ui.table(pd.json_normalize(scoring)),
                ]
            )
        else:
            settings_view = mo.vstack(
                [
                    mo.md("`game_settings` is also included; inspect its scoring fields below."),
                    mo.ui.table(pd.json_normalize(game_settings)),
                ]
            )
    else:
        game_settings = None
        settings_view = mo.md("`game_settings` is not present in this response.")

    mo.vstack([config_view, settings_view])
    return game_config, game_settings


@app.cell
def _(get_json, mo, pd):
    fixtures = get_json("fixtures/")
    fixtures_df = pd.DataFrame(fixtures)

    mo.vstack(
        [
            mo.md("## Fixtures"),
            mo.md(f"**Fixture columns:** `{fixtures_df.columns.tolist()}`"),
            mo.ui.table(fixtures_df),
        ]
    )
    return fixtures, fixtures_df


@app.cell
def _(events, get_json, mo, pd):
    current_events = [event for event in events if event.get("is_current")]
    next_events = [event for event in events if event.get("is_next")]
    live_event = (current_events or next_events or [None])[0]

    if live_event is None:
        live_event_id = None
        live_data = None
        live_view = mo.md(
            "No current or next gameweek is listed in bootstrap; live data was not requested."
        )
    else:
        live_event_id = live_event["id"]
        live_data = get_json(f"event/{live_event_id}/live/")
        live_elements = live_data["elements"]
        live_df = pd.json_normalize(live_elements)
        live_view = mo.vstack(
            [
                mo.md(f"## Live points — gameweek {live_event_id}"),
                mo.md(f"**Live columns:** `{live_df.columns.tolist()}`"),
                mo.ui.table(live_df),
            ]
        )

    live_view
    return live_data, live_event_id


@app.cell
def _(get_json, mo, pd, players):
    PLAYER_ID = players[0]["id"]
    element_summary = get_json(f"element-summary/{PLAYER_ID}/")
    history_df = pd.DataFrame(element_summary["history"])
    upcoming_df = pd.DataFrame(element_summary["fixtures"])

    mo.vstack(
        [
            mo.md(f"## Player match history and upcoming fixtures — player {PLAYER_ID}"),
            mo.md(f"**History columns:** `{history_df.columns.tolist()}`"),
            mo.ui.table(history_df),
            mo.md(f"**Upcoming fixture columns:** `{upcoming_df.columns.tolist()}`"),
            mo.ui.table(upcoming_df),
            mo.md("Change `PLAYER_ID` above to explore a different player's record."),
        ]
    )
    return element_summary, history_df, upcoming_df


@app.cell
def _(get_json, mo, pd):
    set_piece_notes = get_json("team/set-piece-notes/")
    notes_df = pd.json_normalize(set_piece_notes)

    mo.vstack(
        [
            mo.md("## Official set-piece notes"),
            mo.md(f"**Response columns:** `{notes_df.columns.tolist()}`"),
            mo.ui.table(notes_df),
        ]
    )
    return notes_df, set_piece_notes


@app.cell
def _(mo):
    MANAGER_ID = None
    LEAGUE_ID = None
    PICK_EVENT_ID = None

    mo.md(
        """
        ## Manager and mini-league endpoints

        Set `MANAGER_ID` and `LEAGUE_ID` in this cell using the IDs in the
        FPL site's manager and mini-league URLs. Set `PICK_EVENT_ID` to the
        gameweek whose picks you want to inspect. These endpoints are skipped
        until their IDs are supplied.
        """
    )
    return LEAGUE_ID, MANAGER_ID, PICK_EVENT_ID


@app.cell
def _(LEAGUE_ID, get_json, mo, pd):
    if LEAGUE_ID is None:
        standings_page_1 = None
        standings_page_2 = None
        standings_view = mo.md("Enter a `LEAGUE_ID` above to load standings.")
    else:
        standings_page_1 = get_json(
            f"leagues-classic/{LEAGUE_ID}/standings/",
            params={"page_standings": 1},
        )
        standings_page_2 = get_json(
            f"leagues-classic/{LEAGUE_ID}/standings/",
            params={"page_standings": 2},
        )
        page_1_df = pd.DataFrame(standings_page_1["standings"]["results"])
        page_2_df = pd.DataFrame(standings_page_2["standings"]["results"])
        standings_view = mo.vstack(
            [
                mo.md(f"## Mini-league standings — league {LEAGUE_ID}"),
                mo.md(f"**Standings columns:** `{page_1_df.columns.tolist()}`"),
                mo.ui.table(page_1_df),
                mo.md("Page 2 (`?page_standings=2`):"),
                mo.ui.table(page_2_df),
            ]
        )
    standings_view
    return standings_page_1, standings_page_2


@app.cell
def _(MANAGER_ID, PICK_EVENT_ID, get_json, mo, pd):
    if MANAGER_ID is None or PICK_EVENT_ID is None:
        picks_data = None
        picks_view = mo.md(
            "Enter `MANAGER_ID` and `PICK_EVENT_ID` above to load a manager's picks."
        )
    else:
        picks_data = get_json(
            f"entry/{MANAGER_ID}/event/{PICK_EVENT_ID}/picks/"
        )
        picks_df = pd.DataFrame(picks_data["picks"])
        picks_view = mo.vstack(
            [
                mo.md(f"## Manager picks — manager {MANAGER_ID}, gameweek {PICK_EVENT_ID}"),
                mo.md(f"**Pick columns:** `{picks_df.columns.tolist()}`"),
                mo.ui.table(picks_df),
                mo.md("**Entry history:**"),
                mo.ui.table(pd.json_normalize(picks_data.get("entry_history", {}))),
            ]
        )
    picks_view
    return (picks_data,)


@app.cell
def _(MANAGER_ID, get_json, mo, pd):
    if MANAGER_ID is None:
        manager_history = None
        transfers = None
        manager_view = mo.md(
            "Enter a `MANAGER_ID` above to load season history and past transfers."
        )
    else:
        manager_history = get_json(f"entry/{MANAGER_ID}/history/")
        transfers = get_json(f"entry/{MANAGER_ID}/transfers/")
        current_history_df = pd.DataFrame(manager_history["current"])
        past_history_df = pd.DataFrame(manager_history["past"])
        chips_df = pd.DataFrame(manager_history["chips"])
        transfers_df = pd.DataFrame(transfers)
        manager_view = mo.vstack(
            [
                mo.md(f"## Manager season history — manager {MANAGER_ID}"),
                mo.md(f"**Gameweek history columns:** `{current_history_df.columns.tolist()}`"),
                mo.ui.table(current_history_df),
                mo.md("**Past seasons:**"),
                mo.ui.table(past_history_df),
                mo.md("**Chips used:**"),
                mo.ui.table(chips_df),
                mo.md(f"**Transfer columns:** `{transfers_df.columns.tolist()}`"),
                mo.ui.table(transfers_df),
            ]
        )
    manager_view
    return manager_history, transfers


if __name__ == "__main__":
    app.run()
