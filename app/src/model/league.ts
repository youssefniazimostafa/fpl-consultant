// Reads latest.json's `league_page` section (fpl/league_page.py) for the
// League page. Display only -- every number is computed in the pipeline.

import type { Latest } from "../data";
import type { PlayerView } from "./team";

type Raw = Record<string, any>;

export interface Standing {
  rank: number;
  managerId: number;
  team: string;
  manager: string;
  total: number;
  gwPoints: number;
  isYou: boolean;
  inYourArea: boolean;
}

export interface RivalTeam {
  gameweek: number;
  chip: string | null;
  captain: string | null;
  xi: PlayerView[];
  bench: PlayerView[];
  freeTransfers: number | null;
  likelyIn: string | null;
  likelyOut: string | null;
}

export interface RivalInArea {
  managerId: number;
  team: string;
  manager: string;
  rank: number;
  pointsVsYou: number; // + = ahead of you
  theyHave: string[];
  youHave: string[];
  freeTransfers: number | null;
  likelyIn: string | null;
  likelyOut: string | null;
  chipsUsed: string[];
  chipsLeft: string[];
}

export interface OwnedPlayer {
  id: number;
  name: string;
  teamCode: number | null;
  teamShort: string | null;
  position: string;
  rivalsOwning: number;
  rivalsCaptaining: number;
  rivalsBenching: number;
  youOwn: boolean;
  youBenched: boolean;
  projected: number | null;
}

export interface LeaguePageData {
  gameweek: number;
  picksGameweek: number;
  leagueId: number | null;
  leagueName: string | null;
  leagues: { leagueId: number; name: string; supported: boolean }[];
  you: { rank: number; rankText: string; managers: number; total: number; behindFirst: number; lead: number | null };
  chances: { thisGameweek: number | null; gameweek: number | null; topBy: number | null; topByGameweek: number | null };
  strategy: { mode: string; text: string };
  standings: Standing[];
  teams: Record<string, RivalTeam>;
  rivalsInArea: RivalInArea[];
  rivalsCount: number;
  topOwned: OwnedPlayer[];
  popularNotOwned: OwnedPlayer[];
  differentials: OwnedPlayer[];
}

function card(p: Raw): PlayerView {
  return {
    id: p.player_id,
    name: p.web_name,
    teamCode: p.team_code ?? null,
    teamShort: p.team_short ?? null,
    position: p.position ?? "",
    opponent: p.team_short ?? null,
    projected: null,
    points: p.points ?? null,
    price: null,
    isCaptain: !!p.is_captain,
    isVice: !!p.is_vice_captain,
    isNew: false,
    flagged: false,
    news: null,
    benchLabel: null,
  };
}

const owned = (p: Raw): OwnedPlayer => ({
  id: p.player_id,
  name: p.web_name,
  teamCode: p.team_code ?? null,
  teamShort: p.team_short ?? null,
  position: p.position ?? "",
  rivalsOwning: p.rivals_owning ?? 0,
  rivalsCaptaining: p.rivals_captaining ?? 0,
  rivalsBenching: p.rivals_benching ?? 0,
  youOwn: !!p.you_own,
  youBenched: !!p.you_benched,
  projected: p.projected_points ?? null,
});

export function readLeague(data: Latest): LeaguePageData | null {
  const l: Raw | undefined = data.league_page;
  if (!l?.available) return null;
  const teams: Record<string, RivalTeam> = {};
  for (const [id, t] of Object.entries<Raw>(l.teams ?? {})) {
    teams[id] = {
      gameweek: t.gameweek,
      chip: t.chip ?? null,
      captain: t.captain ?? null,
      xi: (t.starting_xi ?? []).map(card),
      bench: (t.bench ?? []).map(card),
      freeTransfers: t.free_transfers ?? null,
      likelyIn: t.likely_in ?? null,
      likelyOut: t.likely_out ?? null,
    };
  }
  return {
    gameweek: l.gameweek,
    picksGameweek: l.picks_gameweek,
    leagueId: l.league_id ?? null,
    leagueName: l.league_name ?? null,
    leagues: (l.leagues ?? []).map((x: Raw) => ({ leagueId: x.league_id, name: x.name, supported: !!x.supported })),
    you: {
      rank: l.you.rank,
      rankText: l.you.rank_text,
      managers: l.you.managers,
      total: l.you.total_points,
      behindFirst: l.you.points_behind_first,
      lead: l.you.lead ?? null,
    },
    chances: {
      thisGameweek: l.chances?.this_gameweek ?? null,
      gameweek: l.chances?.gameweek ?? null,
      topBy: l.chances?.top_by ?? null,
      topByGameweek: l.chances?.top_by_gameweek ?? null,
    },
    strategy: { mode: l.strategy?.mode ?? "balanced", text: l.strategy?.text ?? "" },
    standings: (l.standings ?? []).map((s: Raw) => ({
      rank: s.rank,
      managerId: s.manager_id,
      team: s.entry_name,
      manager: s.player_name,
      total: s.total_points,
      gwPoints: s.gw_points,
      isYou: !!s.is_you,
      inYourArea: !!s.in_your_area,
    })),
    teams,
    rivalsInArea: (l.rivals_in_area ?? []).map((r: Raw) => ({
      managerId: r.manager_id,
      team: r.entry_name,
      manager: r.player_name,
      rank: r.rank,
      pointsVsYou: r.points_vs_you,
      theyHave: r.they_have ?? [],
      youHave: r.you_have ?? [],
      freeTransfers: r.free_transfers ?? null,
      likelyIn: r.likely_in ?? null,
      likelyOut: r.likely_out ?? null,
      chipsUsed: r.chips_used ?? [],
      chipsLeft: r.chips_left ?? [],
    })),
    rivalsCount: l.rivals_count ?? 0,
    topOwned: (l.top_owned ?? []).map(owned),
    popularNotOwned: (l.popular_not_owned ?? []).map(owned),
    differentials: (l.differentials ?? []).map(owned),
  };
}

/** The rival shown by default: the nearest one above you (or just below if you're 1st). */
export function defaultRival(l: LeaguePageData): number | null {
  const me = l.standings.find((s) => s.isYou);
  if (!me) return l.standings[0]?.managerId ?? null;
  const above = l.standings.find((s) => s.rank === me.rank - 1);
  const below = l.standings.find((s) => s.rank === me.rank + 1);
  return (above ?? below)?.managerId ?? null;
}
