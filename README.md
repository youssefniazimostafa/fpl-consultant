# FPL Consultant

A personal Fantasy Premier League advisor: a Python pipeline models every
player's expected points, works out the transfers/captain/chip plan that
actually respects your free-transfer count and your mini-league standing,
and publishes the result as a static site (GitHub Pages) that refreshes
automatically every 3 hours.

It's built around one rule: **recommend what's actually correct given your
real situation**, not a generic "best XI". That means using your real free
transfer count (not just assuming you have one), only taking a -4 points
hit when it's genuinely justified, and being honest on the site when an
external data source (odds, Elo, xG) has gone stale or failed.

## What it does

- **Models expected points (xP)** per player per gameweek from recent form,
  minutes/starts, fixture difficulty (bookmaker odds blended with Elo and
  xG), set-piece duties, and the FPL scoring rules themselves -- not just
  FPL's own `ep_next`.
- **Recommends transfers** that respect your actual free-transfer count
  (rolled-over transfers included), only recommending a points hit when an
  important starter is flagged injured/suspended *and* the replacement's
  projected gain clearly beats the -4.
- **Builds a full squad plan**: starting XI, bench order, captain and
  vice-captain, for the next gameweek, after the recommended transfer(s).
- **Tracks your mini-league**: ownership/EO, rival swing threats, a
  chips-remaining grid, and a win-probability simulation for this
  gameweek and the rest of the season.
- **Grades itself**: every gameweek's predictions are logged before the
  deadline and scored against what actually happened, so the site's
  "Methods & track record" page shows real, live accuracy -- not just a
  one-off backtest.
- **Flags its own blind spots**: if ClubElo, Understat, or The Odds API
  fails or returns something implausible, the pipeline falls back to the
  last known-good data and the site shows a visible warning naming which
  source and how stale the fallback is.

## Pipeline stages

The scheduled job (`.github/workflows/update.yml`) runs `fpl.pipeline`,
which runs every stage below in order. Each stage's `main()` degrades
gracefully (prints a message and returns) if something it needs isn't
configured yet, so a partially-filled `config.json` still produces
whatever downstream panels it can.

| # | Stage (`python -m fpl.<name>`) | What it builds |
|---|---|---|
| 1 | `fetch` | `data/tables/*.parquet` from the public FPL API: players, teams, fixtures, gameweeks, your transfer history, your mini-league's managers/picks/history. |
| 2 | `odds`, `elo`, `understat` | External fixture-difficulty inputs -- bookmaker odds (The Odds API), club strength (ClubElo scrape), team/player xG (Understat). Each is independent; a failure in one doesn't stop the others. Each records its own health status and, on failure, falls back to its last known-good table (see [Source health](#source-health-and-fallbacks) below). |
| 3 | `features` | `features.parquet`: per-player, per-fixture inputs the model needs (scoring rates, start probability, fixture lambda/clean-sheet probability, set-piece duties), with a truthful `source` label per fixture (`odds`, `elo+xg`, `elo_only`, `xg_only`, or `league_avg`). |
| 4 | `xp` | Expected points per player per gameweek over the lookahead horizon, plus a frozen pre-deadline snapshot per gameweek for later scoring. |
| 5 | `league` | Your mini-league's ownership/EO, ownership swings, chips-remaining grid, strategy mode (chasing/protecting/balanced), and ("manual mode") converts a hand-transcribed `data/manual/league_snapshot.json` into the same tables if `league_id` isn't usable. |
| 6 | `rival_transfers` | A model of when each rival is likely to transfer, to anticipate incoming threats. |
| 7 | `scoring` | Post-gameweek feedback: once a gameweek's deadline has passed and its scores lock, compares the frozen pre-deadline xP snapshot against what actually happened and appends to `data/accuracy/accuracy_log.csv`. Safe to run every 3 hours -- it's a no-op except right after a gameweek finishes. |
| 8 | `optimise` | The actual recommendation: best XI, transfer plan (respecting free transfers and the hit-justification policy), captain/vice-captain, chip plan. |
| 9 | `simulation` | Monte Carlo win-probability for this gameweek and the rest of the season, comparing your recommended plan against rolling the transfer. |
| 10 | `publish` | Writes `site/data/latest.json`, the one file the static site fetches. Every panel is `null` with a short `reason` if its config/table isn't available yet. |

Two more modules are **not** part of the scheduled pipeline (one-off model
validation, not something that needs to re-run every 3 hours):

- `fpl.historical` / `fpl.backtest` -- clones reference repos of past
  seasons and backtests the model against simple baselines (PPG, last-5
  average, FPL's own `ep_next`), logging results to
  `fpl/backtest_results_log.csv`.
- `fpl.check_transfer_policy` -- a standing regression check (not a
  pytest suite) that the transfer-hit policy behaves correctly against a
  known scenario; prints `PASS`/`FAIL`.
- `fpl.migrate_prediction_snapshots` -- a one-off migration script, already
  run, that collapsed the old per-run timestamped prediction logs down to
  one frozen file per gameweek.

## Running it locally

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.14+ (see
`.python-version`); `uv` installs the right Python and every dependency in
`pyproject.toml`/`uv.lock` automatically.

```bash
# Run the whole pipeline once (fetch through publish):
uv run python -m fpl.pipeline

# Or run an individual stage while developing, e.g.:
uv run python -m fpl.optimise

# Regression-check the transfer-hit policy:
uv run python -m fpl.check_transfer_policy

# Serve the site locally to view the result in a browser:
cd site && python3 -m http.server 8000
# then open http://localhost:8000
```

`fpl.odds` needs `ODDS_API_KEY` set in your environment (see below); every
other stage works with just the public FPL API.

## How to run the app

`app/` is a macOS desktop app (Tauri v2 + React/TypeScript) for personal use,
not signed or distributed. For now it's a shell: a status bar (next-deadline
countdown, when the data was generated, data source, Refresh button) and five
placeholder tabs showing raw `latest.json` fields.

Needs Node, the Rust toolchain (`rustup`), `uv` and `git` on your PATH.

```sh
cd app
npm install
npm run tauri dev                          # development, with hot reload
npm run tauri build -- --bundles app       # builds the .app
```

The built app is `app/src-tauri/target/release/bundle/macos/FPL Consultant.app`.
Drag it into Applications.

- **Data:** on launch it loads the latest local `latest.json` (from your last
  Refresh) if there is one, otherwise the published site's. The status bar
  shows which one is in use.
- **Refresh:** runs the full pipeline locally and reloads the data when it
  finishes. It runs in its own copy of the repo,
  `~/Library/Application Support/FPL Consultant/engine/` (cloned on first
  use, reset to GitHub's `main` before every run), never in this dev repo,
  and it never commits or pushes from there. `ODDS_API_KEY` is passed through
  only if it's set in your shell; without it the pipeline skips the odds feed.
  The first Refresh is slower because it clones the repo and installs the
  Python environment.
- **Diagnostics:** `~/Library/Application Support/FPL Consultant/app.log`
  (restarted on each launch) records what was loaded and each refresh's
  stages and result. Launching with `FPL_CONSULTANT_AUTOREFRESH=1` (e.g.
  `open --env FPL_CONSULTANT_AUTOREFRESH=1 "FPL Consultant.app"`) starts one
  Refresh automatically, which is useful for testing.

## `config.json` fields

All fields are optional -- an empty or missing `config.json` still runs
the pipeline, just with the panels that need it unavailable.

| Field | Type | Meaning |
|---|---|---|
| `my_entry_id` | number | Your FPL entry/team ID, from the URL of your team's "Points" page (`team_id` is accepted too, as a backward-compatible alias). Needed for every "your team" panel: transfer plan, XI, captain, free transfers. |
| `league_id` | number | Your mini-league's ID, for the standing/ownership/rival panels. If your league's API endpoint isn't usable, a hand-transcribed `data/manual/league_snapshot.json` can be used instead -- see `fpl/league_snapshot.py`. |
| `free_transfers` | number or `null` | Overrides the free-transfer count the pipeline would otherwise calculate from your entry's transfer history. Leave `null` to use the calculated value (shown on the site either way); only set this if you've found a real discrepancy with the FPL app. |
| `strategy` | `"chasing"` / `"protecting"` / `"balanced"` | Overrides the automatically-derived strategy mode (based on your mini-league gap to the leader), which trades off expected points against variance. |
| `bank_override` | number | Overrides your in-the-bank money, if it can't be read correctly from the API. |
| `horizon` | number | How many gameweeks ahead the optimiser/simulation look (default 5). |
| `avoid` | list of player IDs | Players the optimiser should never recommend (e.g. a club you don't trust, or someone you already know you're not touching). |
| `rival_window_points` | number | Point gap that counts as "in rival-catching-up window" for the standing panels. |
| `differential_exclude_managers` | list of manager names | Rivals to exclude from the "differential" (low-ownership pick) calculation. |
| `p_start_overrides` | list of `{player, value, reason, date}` | Manual press-conference-news overrides for a player's start probability (replaces, not scales, the modelled value) for news the data can't see yet. |
| `override_reasons` | object | Free-form notes documenting *why* an override above was set -- not read by any code, just a changelog for your own future reference. |

## `ODDS_API_KEY` secret

`fpl.odds` fetches bookmaker odds from [The Odds API](https://the-odds-api.com)
(free tier is enough). Get a free key, then:

- **Locally**: `export ODDS_API_KEY=...` before running the pipeline.
- **GitHub Actions**: add it as a repository secret named `ODDS_API_KEY`
  (Settings → Secrets and variables → Actions) -- the workflow reads it
  from `secrets.ODDS_API_KEY`. Never commit it to any file.

If it's missing, `fpl.odds` records a "failed" source-health status with a
clear "ODDS_API_KEY not set" message, falls back to the last good odds
table if one exists, and the pipeline otherwise continues unaffected --
fixture difficulty just falls back further to Elo/xG/league-average.

## Source health and fallbacks

ClubElo and Understat are both scraped from endpoints that aren't an
official, documented API, and have each broken silently before. Every
external source (`fpl_api`, `odds`, `elo`, `understat`) now goes through
`fpl/source_health.py`:

- A **sanity check** (team coverage, plausible value ranges) runs on every
  fetch; a failed check is treated the same as an empty fetch.
- On failure, the pipeline **falls back to the last known-good table**
  (committed at `data/last_good/<source>.parquet`) instead of silently
  losing that input, and records a `"stale"` status.
- If there's no fallback either, it records `"failed"`.
- The result is published in `latest.json`'s `source_health` panel, and
  the site shows a warning banner naming which source is degraded and how
  old the fallback data is.

## How the site is deployed

`.github/workflows/update.yml` runs on a schedule (every 3 hours), on
manual dispatch, and on any push touching `fpl/`, `site/`, or
`config.json`. Each run:

1. Runs `uv run python -m fpl.pipeline` (writes `site/data/latest.json`
   and the small data files the pipeline needs to persist: prediction
   snapshots, the accuracy log, and the source-health fallback tables).
2. Commits those data files back to `main` (`fpl-bot`).
3. Deploys the `site/` folder straight to GitHub Pages via
   `actions/upload-pages-artifact` + `actions/deploy-pages` -- no build
   step; `site/index.html` is plain HTML/CSS/JS that fetches
   `data/latest.json` at load time.
