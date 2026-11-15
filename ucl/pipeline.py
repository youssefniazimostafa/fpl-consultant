"""The UCL pipeline: fetch -> back-fill -> odds -> model -> knockout simulation
-> optimiser and chips -> between days -> last matchday and analytics ->
backtest -> schedule -> publish (data/ucl/latest.json) -> consistency check.

Run with ``uv run python -m ucl.pipeline``. The workflow .github/workflows/ucl.yml
runs it on a schedule derived from the fixture list (ucl.schedule) and
commits data/ucl/; the app reads data/ucl/latest.json from GitHub.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ucl import backfill, backtest, between_days, feed, knockout, model, odds, optimise, rules, schedule, sim

OUT = feed.DATA / "latest.json"
RECS = feed.DATA / "recommendations"
REPLAYS = feed.DATA / "replays"
FREEZE_WINDOW = timedelta(hours=2)
CAPTAIN_CANDIDATES = 5


def stage(name: str) -> None:
    print(f"=== {name} ===", flush=True)


def day_label(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts).tz_convert("Europe/Paris")
    return t.strftime("%a %-d %b")


# ------------------------------------------------------------ team helpers

def score_team(team: dict, md: int, apps: pd.DataFrame, pos: dict[int, int]) -> dict:
    """Real points of a recommended team on matchday md, with automatic
    substitutions and the vice-captain stepping in if the captain didn't play."""
    a = apps[apps["md"] == md].groupby("pid").agg(points=("points", "sum"), mins=("oF", "sum"))
    pts = a["points"].to_dict()
    played = {p: a["mins"].get(p, 0) > 0 for p in team["xi"] + team["bench"]}
    final = rules.auto_subs([(p, pos[p], played[p]) for p in team["xi"]],
                            [(b, pos[b], played[b]) for b in team["bench"]])
    cap = team["captain"] if played.get(team["captain"]) else team.get("vice")
    if cap not in final:
        cap = None
    total = sum(pts.get(p, 0) for p in final) + (pts.get(cap, 0) if cap else 0)
    return {"points": int(total), "final_xi": final, "captain_used": cap,
            "by_player": {int(p): int(pts.get(p, 0)) for p in team["xi"] + team["bench"]}}


def team_from_plan(step: dict, xpm: dict[int, float], pos: dict[int, int]) -> dict:
    xi = sorted(step["xi"], key=lambda p: (pos[p], -xpm.get(p, 0)))
    bench = optimise.bench_order(step["squad"], step["xi"], xpm, pos)
    vice = max([p for p in xi if p != step["captain"]], key=lambda p: xpm.get(p, 0))
    return {"squad": step["squad"], "xi": xi, "bench": bench, "captain": step["captain"], "vice": vice}


def choose_captain(team: dict, xp_md: pd.DataFrame, pos: dict[int, int]) -> dict:
    """Captain with the between-days option counted: a captain who plays on
    the first day can be switched to a later-day starter if he scores little,
    so his expected captain points are E[max-ish] rather than his xP."""
    rows = xp_md.set_index("pid").reindex(team["xi"]).fillna(0)
    rows["pos"] = [pos[p] for p in rows.index]
    draws = sim.sample(rows.reset_index(), n=sim.N, seed=3)
    col = {p: j for j, p in enumerate(rows.index)}
    gd = rows["gd"].astype(int).to_dict()
    xpv = rows["xp"].to_dict()
    last_day = max(gd.values())
    allowed = rules.between_days_allowed(int(xp_md["md"].iloc[0])) if len(xp_md) else False
    out = []
    for p in sorted(team["xi"], key=lambda q: -xpv[q])[:CAPTAIN_CANDIDATES]:
        d = draws[:, col[p]]
        later = [q for q in team["xi"] if gd[q] > gd[p]]
        if allowed and later and gd[p] < last_day:
            best_later = max(later, key=lambda q: xpv[q])
            alt = xpv[best_later]
            switch = d + between_days.MIN_SWAP_GAIN <= alt
            bonus = np.where(switch, draws[:, col[best_later]], d)
            out.append({"pid": int(p), "xp": round(xpv[p], 2), "with_switch": round(float(bonus.mean()), 2),
                        "switch_share": round(float(switch.mean()), 2), "switch_to": int(best_later)})
        else:
            out.append({"pid": int(p), "xp": round(xpv[p], 2), "with_switch": round(float(d.mean()), 2),
                        "switch_share": 0.0, "switch_to": None})
    out.sort(key=lambda r: -r["with_switch"])
    return {"options": out, "captain": out[0]["pid"], "vice": out[1]["pid"] if len(out) > 1 else None}


def reason(p, r: pd.Series | None, day: str) -> str:
    if r is None or r.get("xp", 0) <= 0:
        return "Cheap bench filler to keep money for the starters."
    parts = {"goals": "goal threat", "assists": "chances created", "clean_sheet": "clean-sheet chance",
             "balls_rec": "balls recovered", "potm": "Player of the Match chances", "saves": "saves"}
    top = sorted(((k, float(r.get(k, 0) or 0)) for k in parts), key=lambda x: -x[1])
    lead = [parts[k] for k, v in top[:2] if v >= 0.4]
    where = f"{'home to' if r['home'] else 'away at'} {r['opp']}"
    play = "a likely starter" if r["p60"] >= 0.7 else "should play" if r["p_app"] >= 0.6 else "rotation risk"
    txt = f"{r['xp']:.1f} xP {where} on {day}: {play}"
    if lead:
        txt += ", " + " and ".join(lead)
    if p.status in ("I", "D"):
        txt += " (flagged: " + ("injured" if p.status == "I" else "doubtful") + ")"
    return txt + "."


def player_view(pid: int, pl: pd.DataFrame, xrow: pd.Series | None, matches_md: pd.DataFrame) -> dict:
    p = pl.loc[pid]
    out = {"id": int(pid), "name": p["name"], "club": p["team"], "club_code": p["team_code"],
           "team_id": int(p["team_id"]), "pos": rules.POS_NAME[int(p["pos"])], "price": float(p["price"]),
           "sel": float(p["sel"]), "status": p["status"] or None, "news": p["news"] or None}
    if xrow is not None:
        out.update({"xp": round(float(xrow["xp"]), 2), "opp": xrow["opp"], "home": bool(xrow["home"]),
                    "day": day_label(xrow["kickoff"]), "gd": int(xrow["gd"]),
                    "kickoff": pd.Timestamp(xrow["kickoff"]).isoformat(), "p_play": round(float(xrow["p_app"]), 2),
                    "p60": round(float(xrow["p60"]), 2)})
    else:
        out.update({"xp": 0.0, "opp": None, "day": None})
    return out


# --------------------------------------------------- hindsight / averages

def best_possible(md: int, apps: pd.DataFrame, players: pd.DataFrame) -> dict | None:
    a = apps[apps["md"] == md]
    if a.empty:
        return None
    rows = a.groupby("pid").agg(xp=("points", "sum"), price=("price", "last")).reset_index()
    rows["md"] = md
    rows["gd"] = 0
    pl = players.set_index("pid").copy()
    pr = rows.set_index("pid")["price"].dropna()
    pl.loc[pr.index.intersection(pl.index), "price"] = pr
    pl["active"] = 1
    res = optimise.solve(rows, pl.reset_index(), md, mds=[md])
    step = res["plan"][0]
    pts = rows.set_index("pid")["xp"].to_dict()
    return {"points": int(sum(pts.get(p, 0) for p in step["xi"]) + pts.get(step["captain"], 0)),
            "captain": int(step["captain"]), "xi": [int(p) for p in step["xi"]]}


def estimated_average(md: int, apps: pd.DataFrame) -> float | None:
    """Rough average manager score: the ownership-weighted points of every
    player, for an XI (11 of 15), plus a captain bonus from the most-owned
    players. UEFA's real average sits behind a login."""
    a = apps[(apps["md"] == md) & apps["sel"].notna()]
    if a.empty:
        return None
    w = a["sel"] / 100.0
    xi_part = float((w * a["points"]).sum()) * rules.XI_SIZE / rules.SQUAD_SIZE
    top = a.nlargest(5, "sel")
    cap = float((top["points"] * top["sel"]).sum() / max(top["sel"].sum(), 1e-9))
    return round(xi_part + cap, 1)


def replay(md: int, players: pd.DataFrame) -> dict:
    """The team XPert would have picked for md from data before it (fresh team,
    no lookahead), cached in data/ucl/replays/."""
    f = REPLAYS / f"md{md}.json"
    if f.exists():
        return json.loads(f.read_text())
    xp, _ = model.predict(feed.TOUR, as_of_md=md, mds=[md], use_status=False)
    pos = dict(zip(players["pid"].astype(int), players["pos"].astype(int)))
    res = optimise.solve(xp.assign(pos=xp["pid"].map(pos)), players, md, mds=[md])
    xpm = xp.groupby("pid")["xp"].sum().to_dict()
    t = team_from_plan(res["plan"][0], xpm, pos)
    t = {k: ([int(x) for x in v] if isinstance(v, list) else int(v)) for k, v in t.items()}
    t["xp"] = round(sum(xpm.get(p, 0) for p in t["xi"]) + xpm.get(t["captain"], 0), 2)
    t["kind"] = "replay"
    REPLAYS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(t))
    return t


# ------------------------------------------------------------------ main

def main() -> None:
    now = datetime.now(timezone.utc)
    if os.environ.get("UCL_NOW"):
        now = datetime.fromisoformat(os.environ["UCL_NOW"]).astimezone(timezone.utc)
    stage("fetch")
    cons = feed.constraints()
    md = int(cons["matchdayId"])
    deadline = feed.cet_to_utc(cons["deadline"])
    stage("back-fill")
    if os.environ.get("UCL_SKIP_BACKFILL") != "1":  # test aid: use the stored history as it is
        print(backfill.run(feed.TOUR, md))
        if not (backfill.HIST / f"appearances_{feed.PREV_TOUR}.parquet").exists():
            print(backfill.run(feed.PREV_TOUR))
    matches, players, apps = model.load(feed.TOUR)
    pl = players.set_index("pid")
    pos = dict(zip(players["pid"].astype(int), players["pos"].astype(int)))

    stage("odds")
    ol = odds.lambdas(matches, now)
    print(f"odds for {len(ol)} match(es)")

    stage("model")
    xp, team_model = model.predict(feed.TOUR, odds_lambdas=ol)
    xp["pos"] = xp["pid"].map(pos)
    xp_md = xp[xp["md"] == md]
    xpm = xp_md.groupby("pid")["xp"].sum().to_dict()

    stage("knockout simulation")
    ko = knockout.simulate(matches, team_model)

    stage("optimise")
    base = optimise.solve(xp, players, md)
    mds = base["mds"]
    chip_values = {}
    for m in mds:
        if m != md and rules.chip_allowed(m, "limitless"):  # MD5 builds a new team anyway
            r = optimise.solve(xp, players, md, limitless_md=m)
            chip_values[f"limitless_md{m}"] = round(r["objective"] - base["objective"], 2)
        if m != md and rules.chip_allowed(m, "wildcard"):
            r = optimise.solve(xp, players, md, wildcard_md=m)
            chip_values[f"wildcard_md{m}"] = round(r["objective"] - base["objective"], 2)
    print("chip values over", mds, chip_values)
    step0 = base["plan"][0]
    team = team_from_plan(step0, xpm, pos)
    cap = choose_captain(team, xp_md, pos)
    team["captain"], team["vice"] = cap["captain"], cap["vice"]
    flex = between_days.flexibility_value(team, xp_md, pos)
    print(f"captain {pl.loc[team['captain'], 'name']}, vice {pl.loc[team['vice'], 'name']}, flexibility {flex}")

    stage("between days")
    actual_now = {int(p): float(v) for p, v in zip(players["pid"], players["cur_md_pts"])}
    frozen_path = RECS / f"md{md}.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    bd_team = frozen["team"] if frozen else team
    bd = between_days.advise(bd_team, md, xp_md, players, matches, actual_now, pd.Timestamp(now))

    # freeze the recommendation in the final 2 hours before the deadline
    if timedelta(0) < deadline - now <= FREEZE_WINDOW and not (frozen and frozen.get("final")):
        RECS.mkdir(parents=True, exist_ok=True)
        frozen = {"md": md, "frozen_at": now.isoformat(timespec="seconds"), "final": True,
                  "team": {k: ([int(x) for x in v] if isinstance(v, list) else int(v)) for k, v in team.items()},
                  "xp": step0["xp"]}
        frozen_path.write_text(json.dumps(frozen))
        print(f"froze the MD{md} recommendation")

    stage("last matchday and analytics")
    per_md = []
    for m in range(1, md):
        f = RECS / f"md{m}.json"
        t = json.loads(f.read_text())["team"] if f.exists() else replay(m, players)
        kind = "recommended" if f.exists() else "replay"
        sc = score_team(t, m, apps, pos)
        best = best_possible(m, apps, players)
        per_md.append({"md": m, "kind": kind, "points": sc["points"], "xp": t.get("xp"),
                       "average_est": estimated_average(m, apps), "best": best["points"] if best else None,
                       "team": t, "score": sc, "best_team": best})
    last = per_md[-1] if per_md else None

    stage("backtest")
    bt_path = backtest.OUT
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else None
    done_mds = sorted(set(apps["md"])) if len(apps) else []
    if not bt or max((r["md"] for r in bt["rows"] if r["tour"] == feed.TOUR), default=0) < max(done_mds or [0]):
        bt = backtest.run(verbose=False)
    print(backtest.plain_words(bt)[1] if bt else "no backtest")

    stage("schedule")
    sched = schedule.build(matches)
    schedule.write(sched)

    stage("publish")
    xp.to_parquet(feed.RAW / "xp_latest.parquet", index=False)  # for ucl.my_plan (git-ignored)
    payload = build_payload(now, cons, md, deadline, matches, players, xp, xp_md, team, cap, step0, base, chip_values,
                            flex, bd, frozen, per_md, last, ko, bt, sched, apps, team_model)
    payload = feed.clean(payload)
    OUT.write_text(json.dumps(payload, indent=None, separators=(",", ":"), default=str, allow_nan=False))
    feed.write_health()
    print(f"wrote {OUT.relative_to(feed.ROOT)} ({OUT.stat().st_size // 1024} KB)")

    stage("consistency check")
    from ucl import check_consistency

    problems = check_consistency.check(payload)
    if problems:
        for p in problems:
            print("FAIL:", p)
        raise SystemExit(1)
    print("PASS: UCL consistency")
    if os.environ.get("UCL_SLOT"):
        schedule.mark_done(os.environ["UCL_SLOT"], now)
    print("Pipeline complete.")


def planner_view(md: int, matches: pd.DataFrame, base: dict, xp: pd.DataFrame, pl: pd.DataFrame, team_model, ko) -> dict:
    """The UCL Planner tab: the matchday calendar with each stage's rules,
    every club's coming fixtures (day and difficulty), and each planned move's
    gain against the bar the optimiser applies."""
    plan_ft = {st["md"]: st["ft"] for st in base["plan"]}
    calendar = []
    for m in range(1, rules.N_MATCHDAYS + 1):
        g = matches[matches["md"] == m]
        days = []
        for gd, gg in g.sort_values("kickoff").groupby("gd"):
            days.append(day_label(gg["kickoff"].min()))
        ft = rules.FREE_TRANSFERS[m]
        played = bool(len(g)) and bool(g["finished"].all())
        calendar.append({
            "md": m, "stage": rules.STAGE_LABEL[rules.STAGE[m]], "days": days, "matches": int(len(g)),
            "free_transfers": "unlimited" if ft is None else ft,
            "available": plan_ft.get(m) if m in plan_ft and m != md else None,
            "carry": ("unused: up to 1 carries over" if rules.STAGE[m] == "league" else "no carry-over") if ft is not None else "",
            "club_limit": rules.club_limit(m), "budget": rules.budget(m),
            "chips_allowed": m not in rules.CHIP_BANNED_MDS, "between_days": rules.between_days_allowed(m),
            "status": "played" if played else "current" if m == md else "upcoming",
        })
    # difficulty 1-5 from the opponent's strength in the team model (attack + defence), by quintile
    strength = {}
    if team_model is not None:
        for t in team_model.pot_of:
            a, d = team_model.rating(t)
            strength[t] = a + d
    order = sorted(strength, key=strength.get)
    diff = {t: 1 + min(4, int(5 * i / max(len(order), 1))) for i, t in enumerate(order)}
    fixtures = {}
    names = {}
    upcoming = matches[(matches["md"] >= md) & ~matches["finished"]].sort_values("kickoff")
    for r in upcoming.itertuples():
        for tid, code, name, opp, opp_code, home in ((r.home_id, r.home_code, r.home, r.away_id, r.away_code, True),
                                                     (r.away_id, r.away_code, r.away, r.home_id, r.home_code, False)):
            names[code] = name
            fixtures.setdefault(code, []).append({"md": int(r.md), "opp": opp_code, "home": home, "day": day_label(r.kickoff),
                                                  "diff": diff.get(int(opp), 3)})
    club_rows = [{"code": c, "club": names[c], "fixtures": f,
                  "ease": round(sum(6 - x["diff"] for x in f) / max(len(f), 1), 2),
                  "p_top8": float(ko.loc[ko["code"] == c, "p_top8"].iloc[0]) if (ko["code"] == c).any() else None}
                 for c, f in fixtures.items()]
    club_rows.sort(key=lambda r: -r["ease"])
    # each planned move's gain over the rest of the horizon, against the bar
    xpm = xp.groupby(["pid", "md"])["xp"].sum()
    mds = base["mds"]
    moves = []
    for st in base["plan"][1:]:
        rest = [m for m in mds if m >= st["md"]]
        outs = sorted(st["out"], key=lambda p: int(pl.loc[p, "pos"]))
        ins = sorted(st["in"], key=lambda p: int(pl.loc[p, "pos"]))
        for i, (o, n) in enumerate(zip(outs, ins)):
            gain = sum(float(xpm.get((n, m), 0.0)) - float(xpm.get((o, m), 0.0)) for m in rest)
            hit = i >= (st["ft"] or 99)
            bar = optimise.TRANSFER_FRICTION + (rules.HIT_COST if hit else 0)
            moves.append({"md": st["md"], "out": pl.loc[o, "name"], "in": pl.loc[n, "name"],
                          "out_code": pl.loc[o, "team_code"], "in_code": pl.loc[n, "team_code"],
                          "pos": rules.POS_NAME[int(pl.loc[n, "pos"])], "gain": round(gain, 2), "bar": bar, "hit": hit,
                          "over": f"MD{rest[0]}-{rest[-1]}" if len(rest) > 1 else f"MD{rest[0]}"})
    return {"calendar": calendar, "clubs": club_rows, "moves": moves,
            "bar_note": (f"The optimiser makes a move only when it gains more than {optimise.TRANSFER_FRICTION} expected "
                         f"points over the rest of the plan ({optimise.TRANSFER_FRICTION + rules.HIT_COST:g} if it costs a "
                         "-4 hit), so it doesn't churn on small differences."),
            "chip_banned": sorted(rules.CHIP_BANNED_MDS)}


def chip_plan(md: int, chip_values: dict) -> dict:
    """Wildcard and Limitless: the league-phase values from the optimiser
    against when they're usually worth most (the knockout rounds)."""
    best_now = max(chip_values.items(), key=lambda kv: kv[1]) if chip_values else (None, 0.0)
    use_now = best_now[0] and best_now[1] >= 12.0
    plan = {
        "values_now": chip_values,
        "wildcard": {"md": 13, "when": "Quarter-finals, first leg (MD13)",
                     "why": "After the round of 16, half the clubs are out. MD11 already has unlimited transfers, "
                            "so the Wildcard is most useful at a quarter-final or semi-final first leg (MD13 or MD15), "
                            "if more of your players are out than your 5 free transfers can replace."},
        "limitless": {"md": 17, "when": "The final (MD17)",
                      "why": "Only two clubs play the final and the club limit rises to 8, so a one-day team with no "
                             "budget limit can be filled with the best players of both finalists."},
        "now": None,
    }
    if use_now:
        chip, m = best_now[0].split("_md")
        plan["now"] = {"chip": chip, "md": int(m), "value": best_now[1]}
    plan["summary"] = (f"Keep both chips for the knockouts: the best use before the play-offs "
                       f"({best_now[0].replace('_md', ' on MD') if best_now[0] else 'none'}) is worth only "
                       f"{best_now[1]:.1f} points, against what they can do at the quarter-finals and the final."
                       if not use_now else
                       f"Play the {best_now[0].split('_md')[0].title()} on MD{best_now[0].split('_md')[1]}: "
                       f"worth {best_now[1]:.1f} points over the league phase.")
    return plan


def build_payload(now, cons, md, deadline, matches, players, xp, xp_md, team, cap, step0, base, chip_values, flex, bd,
                  frozen, per_md, last, ko, bt, sched, apps, team_model=None) -> dict:
    pl = players.set_index("pid")
    rows_md = xp_md.groupby("pid").agg({"xp": "sum", "opp": "first", "home": "first", "kickoff": "first", "gd": "first",
                                        "p_app": "first", "p60": "first", "goals": "sum", "assists": "sum",
                                        "clean_sheet": "sum", "balls_rec": "sum", "potm": "sum",
                                        "saves": "sum"})
    mm = matches[matches["md"] == md]

    def view(pid: int, **extra) -> dict:
        r = rows_md.loc[pid] if pid in rows_md.index else None
        v = player_view(pid, pl, r, mm)
        v["reason"] = reason(pl.loc[pid], r, v.get("day") or "")
        v.update(extra)
        return v

    bench_set = team["bench"]
    squad_view = [view(p, role="start", captain=p == team["captain"], vice=p == team["vice"]) for p in team["xi"]]
    squad_view += [view(p, role="bench", bench_order=i + 1) for i, p in enumerate(bench_set)]
    days = []
    for gd, g in mm.sort_values("kickoff").groupby("gd"):
        days.append({"gd": int(gd), "label": day_label(g["kickoff"].min()), "first_kickoff": g["kickoff"].min().isoformat(),
                     "matches": [{"home": r.home, "away": r.away, "home_code": r.home_code, "away_code": r.away_code,
                                  "kickoff": r.kickoff.isoformat(), "finished": bool(r.finished),
                                  "score": f"{int(r.home_goals)}-{int(r.away_goals)}" if r.finished else None}
                                 for r in g.itertuples()]})
    form = apps[apps["md"] >= md - 2].groupby("pid")["points"].sum()

    def top(series: pd.Series, n=10, **fmt) -> list[dict]:
        return [view(int(p), value=round(float(v), 2)) for p, v in series.nlargest(n).items() if p in pl.index]

    xps = rows_md["xp"]
    active = pl[(pl["active"] == 1)].index
    scout = {
        "top_xp": top(xps),
        "captains": top(xps * 2),
        "value": top((xps / pl["price"].reindex(xps.index).clip(lower=3.5)).dropna()),
        "form": top(form.reindex(active).dropna()),
        "differentials": top(xps[pl["sel"].reindex(xps.index) < 5.0]),
        "most_selected": top(pl.loc[active, "sel"]),
        "news": [view(int(p)) for p in pl[pl["status"].isin(["I", "D", "S"])].sort_values("sel", ascending=False).index[:10]],
    }
    plan_rows = []
    for st in base["plan"]:
        plan_rows.append({"md": st["md"], "xp": st["xp"], "ft": st["ft"], "hits": st["hits"], "chip": st["chip"],
                          # sorted by position so in[k] replaces out[k]
                          "in": [view(p) | {"md_note": f"MD{st['md']}"} for p in sorted(st["in"], key=lambda q: int(pl.loc[q, "pos"]))],
                          "out": [view(p) for p in sorted(st["out"], key=lambda q: int(pl.loc[q, "pos"]))], "cost": st["cost"],
                          "captain": pl.loc[st["captain"], "name"]})
    last_view = None
    if last:
        t = last["team"]
        lv_players = []
        for p in t["xi"] + t["bench"]:
            v = player_view(p, pl, None, mm)
            v.update({"points": last["score"]["by_player"].get(p, 0), "role": "start" if p in t["xi"] else "bench",
                      "captain": p == t["captain"], "counted": p in last["score"]["final_xi"]})
            lv_players.append(v)
        last_view = {"md": last["md"], "kind": last["kind"], "points": last["points"], "xp": last.get("xp"),
                     "average_est": last["average_est"], "best": last["best"], "players": lv_players,
                     "best_team": [player_view(p, pl, None, mm) | {"captain": p == last["best_team"]["captain"]}
                                   for p in (last["best_team"] or {}).get("xi", [])]}
    ko_rows = [{"team": r.team, "code": r.code, "rank": int(r.rank), "played": int(r.played), "pts": int(r.pts),
                "gd": int(r.gd), "p_top8": round(r.p_top8, 3), "p_playoff": round(r.p_po, 3), "p_out": round(r.p_out, 3),
                "p_r16": round(r.p_r16, 3), "p_qf": round(r.p_qf, 3), "p_final": round(r.p_final, 3),
                "p_win": round(r.p_win, 3), "exp_matches": round(r.exp_matches, 2)} for r in ko.itertuples()]
    bd_view = dict(bd)
    for k in ("subs",):
        bd_view[k] = [s | {"out_name": pl.loc[s["out"], "name"], "in_name": pl.loc[s["in"], "name"],
                           "out_code": pl.loc[s["out"], "team_code"], "in_code": pl.loc[s["in"], "team_code"],
                           "out_gk": int(pl.loc[s["out"], "pos"]) == 1, "in_gk": int(pl.loc[s["in"], "pos"]) == 1}
                      for s in bd.get(k, [])]
    if bd.get("captain_switch"):
        cs = bd["captain_switch"]
        bd_view["captain_switch"] = cs | {"from_name": pl.loc[cs["from"], "name"], "to_name": pl.loc[cs["to"], "name"],
                                          "from_code": pl.loc[cs["from"], "team_code"], "to_code": pl.loc[cs["to"], "team_code"]}
    cost = round(sum(float(pl.loc[p, "price"]) for p in team["squad"]), 1)
    return {
        "game": "ucl", "season": "2026/27", "tour": feed.TOUR,
        "generated_at": now.isoformat(timespec="seconds"),
        "md": md, "stage": rules.STAGE_LABEL[rules.STAGE[md]], "deadline": deadline.isoformat(),
        "budget": rules.budget(md), "club_limit": rules.club_limit(md),
        "matchday": {"md": md, "days": days, "between_days_allowed": rules.between_days_allowed(md)},
        "team": {
            "players": squad_view, "cost": cost, "bank": round(rules.budget(md) - cost, 1),
            "xp": round(sum(float(xps.get(p, 0)) for p in team["xi"]) + float(xps.get(team["captain"], 0)), 2),
            "captain": int(team["captain"]), "vice": int(team["vice"]),
            "formation": "-".join(str(sum(1 for p in team["xi"] if pl.loc[p, "pos"] == q)) for q in (2, 3, 4)),
            "captain_options": [o | {"name": pl.loc[o["pid"], "name"],
                                     "switch_to_name": pl.loc[o["switch_to"], "name"] if o["switch_to"] else None}
                                for o in cap["options"]],
            "flexibility": flex,
            "transfers": {"new_team": True, "note": "No UCL team saved: this is the squad to create (new teams are free, within €100m)."},
            "frozen": frozen and {"frozen_at": frozen["frozen_at"]},
        },
        "plan": plan_rows,
        "planner": planner_view(md, matches, base, xp, pl, team_model, ko),
        "chips": chip_plan(md, chip_values),
        "between_days": bd_view,
        "last_md": last_view,
        "analytics": {"per_md": [{k: r[k] for k in ("md", "kind", "points", "xp", "average_est", "best")} for r in per_md],
                      "backtest": {"summary": (bt or {}).get("summary"), "plain": backtest.plain_words(bt or {}),
                                   "method": (bt or {}).get("method"),
                                   "rows": [{"tour": r["tour"], "md": r["md"], "model": r["model"], "ppg": r["ppg"]}
                                            for r in (bt or {}).get("rows", [])]}},
        "knockout": ko_rows,
        "scout": scout,
        "schedule": sched,
        "sources": feed.health(),
        "notes": {
            "average": "The game average is an estimate from ownership: UEFA shows the real one only after you log in.",
            "my_team": "Reading your own UCL team needs a UEFA login, so XPert recommends the squad to create and "
                       "gives between-days advice for that squad.",
            "hindsight": "Best possible uses the real results, which nobody knows in advance; it shows the ceiling, not a fair target.",
            "prices": "Prices change from MD3; the plan uses today's prices.",
        },
    }


if __name__ == "__main__":
    main()
