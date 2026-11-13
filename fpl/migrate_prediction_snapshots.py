"""One-off migration: collapse the old one-timestamped-file-per-run
prediction log (every 3 hours, whole horizon, never pruned -- 96 files /
6.5MB after 3 weeks) down to the new ``data/predictions/gw{N}.parquet``
scheme (one frozen file per gameweek) that ``fpl.xp.log_live_predictions``
now writes going forward.

For every gameweek that has at least one legacy snapshot, this picks
*exactly* the file ``fpl.scoring.find_prediction_snapshot`` would have
picked under the old scheme (the latest one logged strictly before that
gameweek's deadline), writes that gameweek's slice of it to
``data/predictions/gw{N}.parquet`` with the original ``logged_at``
preserved, and only then deletes every old timestamped file. Nothing is
rewritten in git history -- this just stops the directory from growing
further, as a normal commit.

Run with `uv run python -m fpl.migrate_prediction_snapshots` (add
``--dry-run`` to preview without writing/deleting anything).
"""

from __future__ import annotations

import sys

import pandas as pd

from fpl.fetch import TABLES_DIR, get
from fpl.scoring import (
    PREDICTIONS_DIR,
    _LEGACY_TIMESTAMP_RE,
    _find_legacy_prediction_snapshot,
    deadline_time,
)


def _day_dir():
    from datetime import date

    from fpl.fetch import RAW_DIR

    d = RAW_DIR / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def legacy_files() -> list:
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(
        p for p in PREDICTIONS_DIR.glob("*.parquet") if _LEGACY_TIMESTAMP_RE.fullmatch(p.stem)
    )


def gameweeks_with_legacy_data(files: list) -> list[int]:
    events: set[int] = set()
    for path in files:
        df = pd.read_parquet(path, columns=["event"])
        events.update(int(e) for e in df["event"].unique())
    return sorted(events)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    files = legacy_files()
    if not files:
        print("No legacy data/predictions/<timestamp>.parquet files found -- nothing to migrate.")
        return
    print(f"Found {len(files)} legacy snapshot file(s).")

    bootstrap = get("bootstrap-static/", _day_dir())
    if bootstrap is None:
        print("bootstrap-static/ returned nothing -- can't look up deadlines, aborting.")
        return

    gws = gameweeks_with_legacy_data(files)
    print(f"Legacy files cover gameweek(s): {gws}")

    migrated: dict[int, str] = {}
    skipped: list[int] = []
    for gw in gws:
        try:
            deadline = deadline_time(bootstrap, gw)
        except RuntimeError as exc:
            print(f"  GW{gw}: {exc} -- skipping")
            skipped.append(gw)
            continue
        source = _find_legacy_prediction_snapshot(gw, deadline)
        if source is None:
            # every legacy snapshot that mentioned this gw was logged
            # at-or-after its deadline (e.g. it only appears as a stray
            # post-deadline row) -- nothing valid to migrate for it.
            skipped.append(gw)
            continue
        dest = PREDICTIONS_DIR / f"gw{gw}.parquet"
        if dest.exists():
            # the new scheme has already produced a (more current) file
            # for this gameweek -- don't clobber it with older data.
            skipped.append(gw)
            continue
        df = pd.read_parquet(source)
        slice_df = df[df["event"] == gw].copy()
        migrated[gw] = source.name
        print(f"  GW{gw}: picking {source.name} -> {dest.name} ({len(slice_df)} rows)")
        if not dry_run:
            slice_df.to_parquet(dest, index=False)

    print(f"\nMigrated {len(migrated)} gameweek(s), skipped {len(skipped)} (no valid pre-deadline data).")

    if dry_run:
        print("--dry-run: not deleting any legacy files.")
        return

    for path in files:
        path.unlink()
    print(f"Deleted {len(files)} legacy snapshot file(s).")


if __name__ == "__main__":
    main()
