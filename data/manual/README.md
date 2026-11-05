# Manual league snapshot

`league_snapshot.json` is a hand-transcribed copy of your real FPL mini-league
(the one the live API can't see into, since FPL only exposes a `league_id`'s
picks for leagues whose standings page you can actually load with your own
login). It stands in for Step 8/9's `league_picks`/`league_history`/
`league_chips` tables when `config.json`'s `league_id` is **not** set --
`fpl.league` reads this file instead and converts it into the exact same
table shapes, so every downstream step (`fpl.optimise`, `fpl.publish`, the
site) works identically either way and never needs to know which source it
came from.

**If you ever get a real `league_id`** (e.g. the league's admin shares a
classic-league ID you can query), set it in `config.json` and this file is
ignored entirely -- `fpl.league` always prefers the live API when
`league_id` is set, and only falls back to this snapshot otherwise.

## Updating it each week

After your league's gameweek deadline has passed and scores are final,
re-open the league standings on the FPL site/app and update three things:

1. **`standings`** -- one entry per manager: `rank`, `team` (entry name),
   `manager` (real name), `gw` (points that gameweek), `total` (cumulative
   points). Keep the `"is_me": true` flag on your own entry.
2. **`squads`** -- keyed by team name (must match `standings[].team`
   exactly). Each squad has `chip` (`"wildcard"`/`"freehit"`/`"bboost"`/
   `"3xc"`/`null`), `formation`, and `xi`/`bench` lists of
   `{name, club, pos, pts, role}` (`role` is `"C"` for captain, `"V"` for
   vice, omitted otherwise). Use each player's **current** club code and
   position -- `fpl.league_snapshot` matches against the live API by
   `(name, club, position)` and will raise a clear error (not fail
   silently) if a transfer/position change means the club you wrote down
   no longer matches, so you'll know to fix it.
   - **Your own squad** (the entry with `"is_me": true` in `standings`) is
     the one exception: write down your **current** 15 for the *upcoming*
     gameweek, with `opp` (next fixture) instead of `pts` on each player --
     see the `"note"` field already in your squad entry. Everyone else's
     squad should be their picks from the gameweek that just finished.
3. **`chips_used`** -- append one `{"team": ..., "chip": ..., "gameweek":
   null}` entry per chip you see a rival use for the first time this week
   (leave `gameweek` as `null`; it's resolved automatically against
   whichever gameweek is currently live). This is the only way the chips
   grid (`fpl.league.build_chips_grid`) can track rivals' first-half chips
   at all, since FPL doesn't expose chip history for leagues you can't
   query live -- a chip nobody has reported yet will correctly show as
   `"unknown"` rather than a guessed `"available"`. **Log your own chips
   here too** -- add a `{"team": "JP Morgan", "chip": ..., "gameweek":
   null}` entry the week you play one. `build_chips_grid` already treats
   every `manager_id` (including yours) the same way, so this is the only
   step needed to keep your own chip history visible on the site.

Leave `gameweek` (the top-level field) as `null` -- `fpl.league_snapshot`
always resolves it from the live API's current gameweek at load time and
never writes back to this file, so your manual edits are never clobbered.

## Validating a new snapshot

Before running the full pipeline, sanity-check the transcription on its
own:

```
uv run python -m fpl.league_snapshot
```

This matches every squad against the live player list and re-derives each
manager's reported gameweek total from their XI + (if they played Bench
Boost) bench points -- if those don't add up, or a name/club/position
triple doesn't resolve to exactly one live player, it fails loudly with
the specific mismatch rather than silently producing wrong tables.

## `JP Morgan`

There's no real numeric FPL team ID behind "JP Morgan" in this setup --
`fpl.league_snapshot` uses your league rank as a stand-in `manager_id`
(whatever `standings` entry has `"is_me": true`), and that same ID flows
through to `fpl.optimise`/`fpl.publish` via an `is_me` column on the
persisted `league_managers` table. You never need to set `team_id` in
`config.json` in this mode.

## Step 11 -- retrospective archives (`data/manual_snapshots/<gw>.json`)

`league_snapshot.json` above is always a *live, forward-looking* snapshot
(rivals show last week's `pts`, you show next week's `opp`) -- it gets
overwritten each time you update it, so there's no record of exactly what
every squad scored on a given finished gameweek. `data/manual_snapshots/
<gw>.json` is a separate, append-only archive: one file per finished
gameweek, same `standings`/`squads` shape, but **every** squad (including
yours) uses `pts`, not `opp`, because by the time you archive a gameweek
it's over for everyone. Validate a new one with:

```
uv run python3 -c "
import json, pandas as pd
from fpl.league_snapshot import validate_gw_totals, match_players
snap = json.load(open('data/manual_snapshots/<gw>.json'))
players = pd.read_parquet('data/tables/players.parquet')
validate_gw_totals(snap, skip_me=False)   # skip_me=False: your squad has pts too here
match_players(snap, players)
"
```

If 2+ squads tie on `gw` points and the screenshot doesn't say whose is
whose ("Squad X/Y/Z"), `fpl.league_snapshot.resolve_anonymous_squads`
resolves them by maximum 15-man roster overlap against each candidate's
*previous* archived/live squad -- see the `"note"` field inside the
affected archive for the specific overlap counts used that week. `fpl.
scoring` (Step 11) reads these archives to log each rival's actual GW
score, captain, hits and chips alongside your own prediction accuracy.

## `override_log.json` (Part B.3 -- "was going against the site right?")

When you make a captain/transfer/bench call that goes against the site's
`verdict` panel, append an entry to `data/manual/override_log.json`:

```json
{
  "gameweek": 10,
  "site_recommendation": "N.Jackson",
  "what_you_did": "Haaland",
  "reason": "Haaland's underlying numbers against a weak defence felt safer than the model's EO-adjusted pick",
  "logged_at": "2026-11-07T08:00:00Z",
  "outcome": null
}
```

Leave `outcome` as `null` -- the next scoring run for that `gameweek`
(`fpl.scoring.score_override_log`) resolves it automatically once actual
points are known: it looks up both named players' actual points for that
gameweek and fills in `site_points`, `chosen_points`, and `outcome`
(`"override was right"` / `"wrong"` / `"tied"`), rewriting the file in
place so it becomes a permanent, growing track record. `site_recommendation`/
`what_you_did` just need to be a player's `web_name` (matched the same
accent/case-insensitive way as everywhere else in this project).

## `config.json`'s `p_start_overrides` (mid-week manual input)

Press-conference news the model can't see yet -- "ruled out", "expected
to start despite the knock", a new signing's expected involvement, etc.
Each entry is `{"player": "web_name", "value": 0.0-1.0, "reason": "...",
"date": "YYYY-MM-DD"}`; `fpl.features.apply_p_start_overrides` matches
`player` to a live player by name (accent/case-insensitive) and
*replaces* the modelled `p_start` for every one of his upcoming fixtures
with `value` -- it isn't scaled, since the news itself already is the
probability you want used. What was matched/applied each run is logged
to `data/tables/p_start_overrides_applied.json` (previous vs. new
`p_start`) so the site can flag anything dated after the last run.
