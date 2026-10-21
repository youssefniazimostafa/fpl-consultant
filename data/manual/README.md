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
   `"unknown"` rather than a guessed `"available"`.

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
