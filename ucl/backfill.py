"""Build the UCL history tables from UEFA's public feeds.

* data/ucl/history/matches_{tour}.parquet  - every match: matchday, match day,
  kick-off (UTC), clubs, score, stage.
* data/ucl/history/players_{tour}.parquet  - every player (id, name, club,
  position, price, ownership, status) as of the latest fetch.
* data/ucl/history/appearances_{tour}.parquet - one row per player per match
  he played: minutes and every scoring stat, the points UEFA gave, and his
  price and ownership for that matchday.

The per-player stats come from popupstats (one file per player, fetched
politely). The players feed only has season totals (the snapshots for
earlier matchdays show today's totals too), so it can't be differenced.

Incremental: a player's file is fetched again only when his season totals
in the players feed (points, minutes) differ from what the stored rows add
up to, so a normal run touches only the players who played since the last
run. Run with ``uv run python -m ucl.backfill [--tour 80]``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ucl import feed

HIST = feed.DATA / "history"
# sS = shots saved (UEFA's "saves" field is always 0); popupstats also lists
# not-yet-played fixtures as 0-minute rows, which model.load drops
STAT_COLS = ["oF", "gS", "gA", "cS", "gC", "sS", "bR", "gOB", "mOM", "yC", "rC", "pS", "pM", "pE", "pC", "oG"]


def matches_table(fixtures: list[dict], tour: int) -> pd.DataFrame:
    rows = []
    for md in fixtures:
        for m in md["match"]:
            done = str(m.get("matchStatus")) == "2"
            rows.append({
                "tour": tour, "md": int(md["mdId"]), "gd": int(m.get("gdId") or 0), "mid": int(m["mId"]),
                "kickoff": feed.cet_to_utc(m["dateTime"]),
                "home_id": int(m["htId"]), "home": m["htName"], "home_code": m.get("htCCode"),
                "away_id": int(m["atId"]), "away": m["atName"], "away_code": m.get("atCCode"),
                "home_pot": m.get("htPtName"), "away_pot": m.get("atPtName"),
                "home_goals": int(m["htScore"]) if done and str(m.get("htScore", "")).strip() != "" else None,
                "away_goals": int(m["atScore"]) if done and str(m.get("atScore", "")).strip() != "" else None,
                "finished": done, "stage": m.get("groupName") or "", "lineups": int(m.get("lineupAnnounced") or 0),
                "deadline": feed.cet_to_utc(md["deadline"]),
            })
    return pd.DataFrame(rows)


def players_table(plist: list[dict], tour: int) -> pd.DataFrame:
    keep = {"id": "pid", "pDName": "name", "pFName": "full_name", "tId": "team_id", "tName": "team",
            "cCode": "team_code", "skill": "pos", "value": "price", "selPer": "sel", "isActive": "active",
            "pStatus": "status", "qStatus": "qstatus", "trained": "news", "totPts": "tot_pts",
            "minsPlyd": "tot_mins", "curGDPts": "cur_md_pts", "isPlayed": "is_played", "avgPlayerPts": "ppg"}
    df = pd.DataFrame([{v: p.get(k) for k, v in keep.items()} for p in plist])
    df["tour"] = tour
    for c in ("pid", "team_id", "pos", "active", "tot_pts", "tot_mins", "cur_md_pts", "is_played"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ("price", "sel", "ppg"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def appearance_rows(pid: int, team_id: int, pos: int, pop: dict, tour: int) -> list[dict]:
    pts = {int(p["mId"]): p for p in pop.get("points") or []}
    mdinfo = {int(s["mdId"]): s for s in pop.get("matchdayStats") or []}
    out = []
    for s in pop.get("stats") or []:
        mid = int(s["mId"])
        p = pts.get(mid, {})
        info = mdinfo.get(int(s["mdId"]), {})
        row = {"tour": tour, "pid": pid, "team_id": team_id, "pos": pos, "md": int(s["mdId"]), "mid": mid,
               "points": int(p.get("tPoints") or 0),
               "sel": pd.to_numeric(info.get("selPer"), errors="coerce"),
               "price": pd.to_numeric(info.get("pValue"), errors="coerce")}
        for c in STAT_COLS:
            row[c] = int(s.get(c) or 0)
        out.append(row)
    return out


def run(tour: int, md: int | None = None, limit: int | None = None, verbose: bool = True) -> dict:
    HIST.mkdir(parents=True, exist_ok=True)
    fx = feed.fixtures(tour)
    matches = matches_table(fx, tour)
    matches.to_parquet(HIST / f"matches_{tour}.parquet", index=False)
    if md is None:
        md = int(feed.constraints(tour)["matchdayId"]) if tour == feed.TOUR else int(max(m["mdId"] for m in fx))
    try:
        players = players_table(feed.players(tour, md), tour)
        players.to_parquet(HIST / f"players_{tour}.parquet", index=False)
    except RuntimeError:
        if not (HIST / f"players_{tour}.parquet").exists():
            raise
        players = pd.read_parquet(HIST / f"players_{tour}.parquet")  # last good
        feed.note_health(f"players_{tour}", "stale", "players feed failed; using the last stored table")

    app_path = HIST / f"appearances_{tour}.parquet"
    old = pd.read_parquet(app_path) if app_path.exists() else pd.DataFrame(columns=["pid", "points", "oF"])
    sums = old.groupby("pid").agg(points=("points", "sum"), mins=("oF", "sum")) if len(old) else pd.DataFrame()
    todo = []
    for p in players.itertuples():
        if p.tot_mins <= 0:
            continue
        have = sums.loc[p.pid] if p.pid in sums.index else None
        if have is None or int(have["points"]) != p.tot_pts or int(have["mins"]) != p.tot_mins:
            todo.append(p)
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"tour {tour}: {len(players)} players, {len(todo)} to (re)fetch")
    new_rows, failed = [], 0
    for i, p in enumerate(todo, 1):
        pop = feed.popupstats(tour, p.pid, refresh=p.pid in sums.index if len(sums) else False)
        if pop is None:
            failed += 1
            continue
        new_rows += appearance_rows(p.pid, p.team_id, p.pos, pop, tour)
        if verbose and i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    done = {p.pid for p in todo}
    keep = old[~old["pid"].isin(done)] if len(old) else old
    apps = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else keep
    if len(apps):
        apps = apps.drop_duplicates(["pid", "mid"], keep="last").sort_values(["md", "pid"]).reset_index(drop=True)
        for c in ["tour", "pid", "team_id", "pos", "md", "mid", "points", *STAT_COLS]:
            apps[c] = apps[c].astype(int)
        apps.to_parquet(app_path, index=False)
    feed.note_health(f"popupstats_{tour}", "ok" if failed == 0 else "partial",
                     None if failed == 0 else f"{failed} player file(s) failed", fetched=len(todo) - failed)
    return {"tour": tour, "players": len(players), "fetched": len(todo) - failed, "failed": failed,
            "appearances": len(apps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", type=int, default=feed.TOUR)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(run(a.tour, limit=a.limit))
    feed.write_health()
    sys.stdout.flush()


if __name__ == "__main__":
    main()


def from_cache(tour: int) -> pd.DataFrame:
    """Rebuild the appearances table from the raw cache alone (no requests)."""
    players = pd.read_parquet(HIST / f"players_{tour}.parquet").set_index("pid")
    rows = []
    for f in sorted((feed.RAW / str(tour)).glob("popup_*.json")):
        pid = int(f.stem.split("_")[1])
        if pid in players.index:
            import json as _json
            p = players.loc[pid]
            rows += appearance_rows(pid, int(p.team_id), int(p.pos), _json.loads(f.read_text()), tour)
    return pd.DataFrame(rows)
