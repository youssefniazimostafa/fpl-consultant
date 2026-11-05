"""Step 11 Part B.4 extension 3 -- rival transfer model (SCAFFOLD, not yet
implemented).

**Why this is a scaffold, not a working model.** The spec asks to predict
each rival's likely transfer from "their habits (how often they hit,
whether they chase form, whether they sell flagged players)". Learning a
habit needs *repeated observations* of that rival's decisions -- but this
private league has no ``league_id`` (no ``entry/{id}/event/{gw}/picks``
access, see ``fpl.league_snapshot``'s module docstring), so the only
source of each rival's squad is a manually-transcribed ``data/
manual_snapshots/<gw>.json`` archive, and there is exactly **one** of
those so far (GW9, built this Step). One data point isn't a habit, it's
a single observation -- any "chases form" / "sells flags" classifier
built on it now would just be noise dressed up as a model. Price-change
data (``cost_change_event`` in ``bootstrap-static``) is available every
run and *could* support an ownership-swing proxy for rival transfers in
general, but without a per-rival history to calibrate against, there's
nothing to validate a per-rival model's predictions.

**The plan for when enough archives exist** (roughly 3-4 gameweeks of
``data/manual_snapshots/<gw>.json``, so each rival has shown a few
transfer decisions):

1. ``build_rival_transfer_history()`` -- diff consecutive archives'
   ``squads[team]["xi"] + ["bench"]`` (by player id, via
   ``fpl.league_snapshot.match_players``) to get each rival's actual
   transfers in/out per gameweek, same shape as ``fpl.optimise``'s own
   ``my_transfers.parquet``.
2. Per-rival habit features from that history: hit rate (mean
   ``max(0, transfers - 1)`` per week), "chases form" (correlation
   between a bought player's *recent* form and the transfer), "sells
   flags" (did they ever sell a player who was ``⚠``-flagged the week
   before, like this GW's João Pedro).
3. ``forecast_rival_transfers(next_gw)`` -- for each rival, rank
   plausible buys among players with rising ``cost_change_event`` /
   falling ownership among *that rival's historical price-chase
   tendency*, weighted by their hit-rate (a rival who never hits only
   gets forecast transfers up to their free-transfer count).
4. Feed the forecast squads into next gameweek's league EO
   (``fpl.league``'s ``league_eo``) *before* the deadline, instead of
   EO staying frozen at last gameweek's actual ownership -- this is the
   actual payoff the spec wants ("forecast next GW's league EO before
   the deadline").

Until then, ``main()`` below just explains the gap rather than
pretending to forecast anything; call it to get the same message that
would otherwise be silently missing from the site.

Run with `uv run python -m fpl.rival_transfers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANUAL_SNAPSHOTS_DIR = ROOT / "data" / "manual_snapshots"


def available_archives() -> list[int]:
    return sorted(int(p.stem) for p in MANUAL_SNAPSHOTS_DIR.glob("*.json") if p.stem.isdigit())


def build_rival_transfer_history() -> dict[str, Any]:
    """TODO (needs >= 2 archived gameweeks, see module docstring): diff
    consecutive ``data/manual_snapshots/<gw>.json`` squads to get each
    rival's actual transfers in/out, the foundation every later step here
    depends on."""
    raise NotImplementedError(
        "needs at least two data/manual_snapshots/<gw>.json archives to diff -- "
        f"currently have {available_archives()}. Log another gameweek "
        "(see data/manual/README.md) and this becomes possible."
    )


def forecast_rival_transfers(next_gw: int) -> dict[str, Any]:
    """TODO: per-rival buy/sell forecast for ``next_gw``, from habits
    learned by ``build_rival_transfer_history`` plus ``bootstrap-static``'s
    ``cost_change_event`` price-change signal. Feeds into a forecast
    league EO for the captain board / differentials panels, ahead of the
    deadline rather than a gameweek behind it."""
    raise NotImplementedError("see build_rival_transfer_history's TODO -- same blocker.")


def main() -> None:
    archives = available_archives()
    print(
        "Rival transfer model: scaffold only (see fpl/rival_transfers.py's module "
        f"docstring). Have {len(archives)} archived gameweek(s) ({archives}); need >= 2 "
        "to diff a single transfer, and several more to learn a habit reliably. "
        "Nothing forecast this run."
    )


if __name__ == "__main__":
    main()
