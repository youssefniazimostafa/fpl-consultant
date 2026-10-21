"""Pipeline -- the one entry point the scheduled GitHub Actions run calls
(``.github/workflows/update.yml``). Runs every step in dependency order:

    fetch -> external data -> features -> model -> league -> optimise -> publish

Each stage's own ``main()`` already guards itself against missing
config/credentials -- no ``team_id``, no ``ODDS_API_KEY``, ClubElo being
down -- by printing a message and returning rather than raising, so a
config that's only partially filled in still produces whatever downstream
tables it can (see ``fpl/publish.py``'s module docstring for how that
missing-table/missing-config degradation shows up in the published JSON).

The three **external data** sources (``fpl.odds``, ``fpl.elo``,
``fpl.understat``) are mutually independent third-party API calls, not a
single step -- so an unexpected failure in one (a genuine bug or outage,
not just "no data today", which each module already handles itself) is
caught and logged here rather than aborting the other two, or the rest of
the pipeline. ``fpl.fetch``, ``fpl.features``, ``fpl.xp``, ``fpl.league``,
``fpl.optimise`` and ``fpl.publish`` are *not* wrapped this way: each
depends on the previous stage's tables, so letting a real failure there
raise and fail the Actions run loudly is safer than silently publishing a
stale or wrong site.

``fpl.historical`` and ``fpl.backtest`` are deliberately not part of this
chain: cloning two reference repos and backtesting a full past season is
one-off model validation (Step 7), not something that needs to re-run on
every scheduled update.

Run with `uv run python -m fpl.pipeline`.
"""

from __future__ import annotations

import traceback
from types import ModuleType

import fpl.elo as elo
import fpl.features as features
import fpl.fetch as fetch
import fpl.league as league
import fpl.odds as odds
import fpl.optimise as optimise
import fpl.publish as publish
import fpl.understat as understat
import fpl.xp as xp

EXTERNAL_DATA_STAGES: list[tuple[str, ModuleType]] = [
    ("odds", odds),
    ("elo", elo),
    ("understat", understat),
]


def run_stage(label: str, module: ModuleType) -> None:
    print(f"\n=== {label} ===")
    module.main()


def run_external_data() -> None:
    for name, module in EXTERNAL_DATA_STAGES:
        print(f"\n=== external data: {name} ===")
        try:
            module.main()
        except Exception:
            print(f"  {name} failed unexpectedly -- continuing without it:")
            traceback.print_exc()


def main() -> None:
    run_stage("fetch", fetch)
    run_external_data()
    run_stage("features", features)
    run_stage("model (xp)", xp)
    run_stage("league", league)
    run_stage("optimise", optimise)
    run_stage("publish", publish)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
