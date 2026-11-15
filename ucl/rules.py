"""UCL Fantasy 2026/27 rules, in one place, with a self-test.

Everything else in ucl/ reads the rules from here. Run the self-test with
``uv run python -m ucl.rules`` (it also checks the scoring function against
every stored appearance, where UEFA's own points are known).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ------------------------------------------------------------------ squad

GK, DEF, MID, FWD = 1, 2, 3, 4
POS_NAME = {GK: "GK", DEF: "DEF", MID: "MID", FWD: "FWD"}
SQUAD_SIZE = 15
SQUAD_SHAPE = {GK: 2, DEF: 5, MID: 5, FWD: 3}
XI_SIZE = 11
# a valid starting XI: 1 goalkeeper, 3-5 defenders, 3-5 midfielders, 1-3 forwards
XI_MIN = {GK: 1, DEF: 3, MID: 3, FWD: 1}
XI_MAX = {GK: 1, DEF: 5, MID: 5, FWD: 3}
CAPTAIN_MULT = 2

# ------------------------------------------------------------- the season

N_MATCHDAYS = 17
LEAGUE_PHASE = range(1, 9)  # MD1-8
STAGE = {**{m: "league" for m in LEAGUE_PHASE}, 9: "playoff", 10: "playoff", 11: "r16", 12: "r16",
         13: "qf", 14: "qf", 15: "sf", 16: "sf", 17: "final"}
STAGE_LABEL = {"league": "League phase", "playoff": "Knockout play-offs", "r16": "Round of 16",
               "qf": "Quarter-finals", "sf": "Semi-finals", "final": "Final"}
CLUB_LIMIT = {"league": 3, "playoff": 4, "r16": 4, "qf": 5, "sf": 6, "final": 8}
BUDGET_LEAGUE = 100.0
BUDGET_KNOCKOUT = 105.0  # from the knockout play-offs on

# free transfers *before* each matchday (None = unlimited)
UNLIMITED = None
FREE_TRANSFERS = {1: UNLIMITED, **{m: 2 for m in range(2, 9)}, 9: UNLIMITED, 10: 2, 11: UNLIMITED,
                  12: 3, 13: 5, 14: 3, 15: 5, 16: 3, 17: 5}
LEAGUE_MAX_CARRY = 1  # league phase: at most 1 unused free transfer carries over
KNOCKOUT_CARRY = 0  # knockouts: none
HIT_COST = 4  # per transfer above the free ones

CHIPS = ("wildcard", "limitless")
CHIP_BANNED_MDS = {1, 9, 11}  # the unlimited-transfer windows

# between match days within a matchday
BETWEEN_DAYS_MAX_SUBS = 4
BETWEEN_DAYS_BANNED_MDS = {8, 17}  # MD8 is played on one day; so is the final


def budget(md: int) -> float:
    return BUDGET_LEAGUE if md <= 8 else BUDGET_KNOCKOUT


def club_limit(md: int) -> int:
    return CLUB_LIMIT[STAGE[md]]


def is_unlimited(md: int) -> bool:
    return FREE_TRANSFERS[md] is UNLIMITED


def free_transfers_next(md: int, available: int | None, used: int) -> int | None:
    """Free transfers available before md+1, given `available` before md and
    `used` made before md (None = that matchday was unlimited)."""
    nxt = md + 1
    if nxt > N_MATCHDAYS:
        return 0
    base = FREE_TRANSFERS[nxt]
    if base is UNLIMITED:
        return UNLIMITED
    if STAGE[nxt] == "league" and available is not UNLIMITED:
        carry = min(LEAGUE_MAX_CARRY, max(0, available - used))
        return base + carry
    return base  # first league matchday after MD1, or any knockout matchday: no carry


def transfer_cost(md: int, available: int | None, used: int, chip: str | None = None) -> int:
    """Points deducted for `used` transfers before md."""
    if available is UNLIMITED or chip == "wildcard" or chip == "limitless":
        return 0
    return HIT_COST * max(0, used - available)


def chip_allowed(md: int, chip: str, used_chips: Iterable[str] = ()) -> bool:
    return chip in CHIPS and chip not in set(used_chips) and md not in CHIP_BANNED_MDS


def between_days_allowed(md: int) -> bool:
    return md not in BETWEEN_DAYS_BANNED_MDS


# -------------------------------------------------------------- formation

def valid_xi(positions: list[int]) -> bool:
    if len(positions) != XI_SIZE:
        return False
    return all(XI_MIN[p] <= positions.count(p) <= XI_MAX[p] for p in POS_NAME)


def valid_squad(positions: list[int], clubs: list[int], prices: list[float], md: int,
                chip: str | None = None) -> list[str]:
    """Problems with a 15-man squad (an empty list means it's valid)."""
    out = []
    if len(positions) != SQUAD_SIZE:
        out.append(f"{len(positions)} players, not {SQUAD_SIZE}")
    for p, n in SQUAD_SHAPE.items():
        if positions.count(p) != n:
            out.append(f"{positions.count(p)} {POS_NAME[p]}, not {n}")
    if chip != "limitless":
        if sum(prices) > budget(md) + 1e-6:
            out.append(f"costs €{sum(prices):.1f}m, over €{budget(md):.0f}m")
        lim = club_limit(md)
        for c in set(clubs):
            if clubs.count(c) > lim:
                out.append(f"{clubs.count(c)} players from one club, limit {lim}")
    return out


# ---------------------------------------------------------------- scoring

# per-position points: goal, clean sheet, every 2 conceded (GK/DEF)
GOAL = {GK: 6, DEF: 6, MID: 5, FWD: 4}
CLEAN_SHEET = {GK: 4, DEF: 4, MID: 1, FWD: 0}
CONCEDED_PER_2 = {GK: -1, DEF: -1, MID: 0, FWD: 0}
APPEARANCE, SIXTY_PLUS = 1, 1
ASSIST = 3
BALLS_RECOVERED_PER_3 = 1
PLAYER_OF_MATCH = 3
GOAL_OUTSIDE_BOX = 1
PEN_WON, PEN_CONCEDED, PEN_MISSED = 2, -1, -2
YELLOW, RED, OWN_GOAL = -1, -3, -2
GK_PEN_SAVE = 5
GK_SAVES_PER_3 = 1
CLEAN_SHEET_MIN_MINUTES = 60


@dataclass
class Stats:
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheet: int = 0  # as UEFA records it (already needs 60+ minutes)
    conceded: int = 0
    saves: int = 0
    balls_recovered: int = 0
    goals_outside_box: int = 0
    player_of_match: int = 0
    yellow: int = 0
    red: int = 0
    pen_saved: int = 0
    pen_missed: int = 0
    pen_won: int = 0
    pen_conceded: int = 0
    own_goals: int = 0

    @classmethod
    def from_feed(cls, r) -> "Stats":
        g = (lambda k: int(r[k] if isinstance(r, dict) else getattr(r, k)))
        return cls(g("oF"), g("gS"), g("gA"), g("cS"), g("gC"), g("sS"), g("bR"), g("gOB"), g("mOM"),
                   g("yC"), g("rC"), g("pS"), g("pM"), g("pE"), g("pC"), g("oG"))


def points(pos: int, s: Stats) -> int:
    """UCL Fantasy points for one match (extra time counts in the stats, but
    a player can't get the appearance or clean-sheet points twice)."""
    if s.minutes <= 0:
        return 0
    p = APPEARANCE + (SIXTY_PLUS if s.minutes >= 60 else 0)
    p += GOAL[pos] * s.goals + ASSIST * s.assists
    p += GOAL_OUTSIDE_BOX * s.goals_outside_box
    p += BALLS_RECOVERED_PER_3 * (s.balls_recovered // 3)
    p += PLAYER_OF_MATCH * s.player_of_match
    p += PEN_WON * s.pen_won + PEN_CONCEDED * s.pen_conceded + PEN_MISSED * s.pen_missed
    p += YELLOW * s.yellow + RED * s.red + OWN_GOAL * s.own_goals
    if s.clean_sheet and s.minutes >= CLEAN_SHEET_MIN_MINUTES:
        p += CLEAN_SHEET[pos]
    p += CONCEDED_PER_2[pos] * (s.conceded // 2)
    if pos == GK:
        p += GK_PEN_SAVE * s.pen_saved + GK_SAVES_PER_3 * (s.saves // 3)
    return p


# --------------------------------------------------------------- auto-subs

def auto_subs(xi: list[tuple[int, int, bool]], bench: list[tuple[int, int, bool]]) -> list[int]:
    """Final XI after automatic substitutions.

    `xi` and `bench` are (player id, position, played) with the bench in
    order (bench goalkeeper first). A starter who didn't play is replaced by
    the first bench player who played and keeps the XI valid; the bench
    goalkeeper can only replace the goalkeeper."""
    final = list(xi)
    used: set[int] = set()
    for i, (pid, pos, played) in enumerate(list(final)):
        if played:
            continue
        for b in bench:
            bid, bpos, bplayed = b
            if bid in used or not bplayed:
                continue
            if (pos == GK) != (bpos == GK):
                continue
            trial = [x for j, x in enumerate(final) if j != i] + [b]
            if valid_xi([x[1] for x in trial]):
                final[i] = b
                used.add(bid)
                break
    return [x[0] for x in final]


# -------------------------------------------------------------- self-test

def self_test(verbose: bool = True) -> list[str]:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # Haaland, MD1 2026/27 (Porto 0-2 Man City): 90 min, 2 goals, a yellow, Player of the Match -> 12
    check(points(FWD, Stats(minutes=90, goals=2, yellow=1, player_of_match=1)) == 12, "Haaland MD1 should be 12")
    check(points(MID, Stats(minutes=59, clean_sheet=1)) == 1, "clean sheet needs 60 minutes")
    check(points(DEF, Stats(minutes=90, clean_sheet=1, goals=1)) == 2 + 4 + 6, "defender goal + clean sheet")
    check(points(GK, Stats(minutes=90, saves=7, conceded=3, pen_saved=1)) == 2 + 2 - 1 + 5, "keeper saves/conceded")
    check(points(FWD, Stats(minutes=90, balls_recovered=8)) == 2 + 2, "every 3 balls recovered")
    check(points(MID, Stats(minutes=30, goals=1, goals_outside_box=1)) == 1 + 5 + 1, "goal from outside the box")
    check(points(FWD, Stats(minutes=90, pen_won=1, pen_missed=1, red=1, own_goals=1)) == 2 + 2 - 2 - 3 - 2,
          "penalties, red, own goal")
    check(points(FWD, Stats(minutes=0, goals=1)) == 0, "no minutes, no points")
    # transfers
    check(FREE_TRANSFERS[2] == 2 and is_unlimited(1) and is_unlimited(9) and is_unlimited(11), "unlimited windows")
    check([FREE_TRANSFERS[m] for m in (10, 12, 13, 14, 15, 16, 17)] == [2, 3, 5, 3, 5, 3, 5], "knockout transfers")
    check(free_transfers_next(4, 2, 0) == 3, "league: carry one")
    check(free_transfers_next(4, 3, 0) == 3, "league: carry at most one")
    check(free_transfers_next(4, 2, 2) == 2, "league: nothing to carry")
    check(free_transfers_next(1, None, 15) == 2, "after the first squad: 2")
    check(free_transfers_next(12, 3, 0) == 5, "knockouts: no carry")
    check(free_transfers_next(8, 3, 0) is UNLIMITED, "before the play-offs: unlimited")
    check(transfer_cost(5, 2, 4) == 8 and transfer_cost(5, 2, 2) == 0, "-4 per extra transfer")
    check(transfer_cost(5, 2, 6, "wildcard") == 0, "wildcard")
    # stages
    check([club_limit(m) for m in (1, 9, 11, 13, 15, 17)] == [3, 4, 4, 5, 6, 8], "club limits")
    check(budget(8) == 100 and budget(9) == 105, "budget rise")
    check(not chip_allowed(9, "wildcard") and chip_allowed(5, "limitless") and
          not chip_allowed(5, "wildcard", ["wildcard"]), "chip rules")
    check(not between_days_allowed(8) and not between_days_allowed(17) and between_days_allowed(5), "between days")
    # formations
    check(valid_xi([GK, DEF, DEF, DEF, MID, MID, MID, MID, MID, FWD, FWD]), "3-5-2 valid")
    check(not valid_xi([GK, DEF, DEF, MID, MID, MID, MID, MID, FWD, FWD, FWD]), "2 defenders invalid")
    check(not valid_xi([GK, GK, DEF, DEF, DEF, MID, MID, MID, MID, FWD, FWD]), "2 keepers invalid")
    shape = [GK] * 2 + [DEF] * 5 + [MID] * 5 + [FWD] * 3
    check(valid_squad(shape, list(range(15)), [6.6] * 15, 5) == [], "a valid squad")
    check(valid_squad(shape, [1] * 4 + list(range(2, 13)), [6.0] * 15, 5) != [], "4 from one club in the league")
    check(valid_squad(shape, [1] * 4 + list(range(2, 13)), [6.0] * 15, 9) == [], "4 from one club in the play-offs")
    check(valid_squad(shape, list(range(15)), [7.0] * 15, 5, chip="limitless") == [], "limitless ignores budget")
    # auto-subs: a defender who didn't play in a 3-4-3 needs a defender (or the XI turns invalid)
    xi = [(1, GK, True), (2, DEF, False), (3, DEF, True), (4, DEF, True), (5, MID, True), (6, MID, True),
          (7, MID, True), (8, MID, True), (9, FWD, True), (10, FWD, True), (11, FWD, True)]
    bench = [(12, GK, True), (13, FWD, True), (14, DEF, True), (15, MID, True)]
    check(auto_subs(xi, bench)[1] == 14, "auto-sub keeps 3 defenders")
    xi2 = [(1, GK, False)] + xi[1:]
    check(auto_subs(xi2, bench)[0] == 12, "bench keeper replaces keeper")

    # against UEFA's own points for every stored appearance
    try:
        import pandas as pd

        from ucl.feed import DATA

        frames = [pd.read_parquet(f) for f in sorted((DATA / "history").glob("appearances_*.parquet"))]
        if frames:
            a = pd.concat(frames, ignore_index=True)
            a = a[a["oF"] > 0]  # appearances (listed-but-unplayed rows score 0 trivially)
            mine = [points(int(r.pos), Stats.from_feed(r)) for r in a.itertuples()]
            agree = (pd.Series(mine) == a["points"].values).mean()
            if verbose:
                print(f"scoring vs UEFA's points: {agree:.1%} of {len(a)} appearances match exactly")
            check(agree >= 0.97, f"scoring matches UEFA's points on only {agree:.1%} of appearances")
    except Exception as e:  # noqa: BLE001
        fails.append(f"could not check against stored points: {e}")
    return fails


def main() -> None:
    fails = self_test()
    for f in fails:
        print("FAIL:", f)
    print("PASS: UCL rules self-test" if not fails else f"FAIL: {len(fails)} rule check(s)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
