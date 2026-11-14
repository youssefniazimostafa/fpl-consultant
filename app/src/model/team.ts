// Reads the ONE recommendation for the upcoming gameweek -- latest.json's
// `decision` object, built by fpl/decision.py -- into typed views for the
// "Recommended team" page. Nothing here decides anything: every panel (pitch,
// bench, transfers, captain, chips, "Why this team") renders these fields
// as published, and fpl/check_consistency.py has already verified they
// agree with each other.

import type { Latest } from "../data";

export interface PlayerView {
  id: number;
  name: string;
  teamCode: number | null;
  teamShort: string | null;
  position: string;
  opponent: string | null; // "NEW (A)"
  projected: number | null;
  points?: number | null; // actual points (rival teams on the League page)
  marker?: "beat" | "missed" | null; // Last gameweek: beat / missed expected points
  price: number | null;
  isCaptain: boolean;
  isVice: boolean;
  isNew: boolean;
  flagged: boolean;
  news: string | null;
  benchLabel: string | null;
}

export interface ChipStatus {
  chip: string;
  label: string;
  available: boolean;
  note: string;
}

export interface Decision {
  gameweek: number;
  action: "transfer" | "roll";
  transfers: { out: PlayerView; in: PlayerView }[];
  transferNews: { name: string; news: string; urgency: string }[];
  hits: number;
  hitCost: number;
  hitReason: string | null;
  freeTransfers: { available: number | null; used: number | null; left: number | null; nextWeek: number | null };
  xi: PlayerView[];
  bench: PlayerView[];
  captain: {
    id: number;
    name: string;
    projected: number;
    viceName: string | null;
    riskText: string;
  } | null;
  chip: {
    play: { label: string; gain: number } | null;
    upcoming: { label: string; gameweek: number; gain: number } | null;
    urgent: string[];
    useBy: number | null;
    statuses: ChipStatus[];
    nextHalf: ChipStatus[]; // second set, unlocks after GW19
  };
  horizonGain: number | null;
  horizonWeeks: number;
  why: string[];
}

type Raw = Record<string, any>;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function toView(p: Raw): PlayerView {
  const fx = p.fixture;
  return {
    id: Number(p.player_id ?? -1),
    name: p.web_name ?? "?",
    teamCode: num(p.team_code),
    teamShort: p.team_short ?? null,
    position: p.position ?? "",
    opponent: fx?.opponent_short ? `${fx.opponent_short} (${fx.is_home ? "H" : "A"})` : null,
    projected: num(p.projected_points),
    price: num(p.price),
    isCaptain: !!p.is_captain,
    isVice: !!p.is_vice_captain,
    isNew: !!p.transferred_in,
    flagged: !!p.flagged,
    news: p.news ?? null,
    benchLabel: p.bench_label ?? null,
  };
}

/** The optimiser's hit reason in plainer words (wording only, not a decision). */
export function plainHitReason(reason: string | null): string | null {
  if (!reason) return null;
  return reason
    .replace(/\(\s*[a-z]\s*,\s*/g, "(")
    .replace(/\(\s*[a-z]\s*\)/g, "")
    .replace(/\s+--\s+/g, "; ")
    .replace(/\bpts\b/g, "points")
    .replace(/\bGWs\b/g, "gameweeks")
    .replace(/(\d+)-point hit cost/g, "$1-point hit");
}

/** latest.json's `decision`, or null if this data file predates it. */
export function readDecision(data: Latest): Decision | null {
  const d: Raw | undefined = data.decision;
  if (!d?.available) return null;
  const ft = d.free_transfers ?? {};
  const cap = d.captain;
  return {
    gameweek: d.gameweek,
    action: d.action,
    transfers: (d.transfers ?? []).map((t: Raw) => ({ out: toView(t.player_out), in: toView(t.player_in) })),
    transferNews: (d.transfer_news ?? []).map((n: Raw) => ({ name: n.web_name, news: n.news, urgency: n.urgency })),
    hits: d.hits ?? 0,
    hitCost: d.hit_cost ?? 0,
    hitReason: plainHitReason(d.hit_reason ?? null),
    freeTransfers: { available: num(ft.available), used: num(ft.used), left: num(ft.left), nextWeek: num(ft.next_week) },
    xi: (d.starting_xi ?? []).map(toView),
    bench: (d.bench ?? []).map(toView),
    captain: cap
      ? {
          id: cap.player_id,
          name: cap.web_name,
          projected: cap.projected_points,
          viceName: cap.vice_captain_name ?? null,
          riskText: cap.risk?.text ?? "",
        }
      : null,
    chip: {
      play: d.chip?.play ? { label: d.chip.play.label, gain: d.chip.play.gain } : null,
      upcoming: d.chip?.upcoming
        ? { label: d.chip.upcoming.label, gameweek: d.chip.upcoming.gameweek, gain: d.chip.upcoming.gain }
        : null,
      urgent: d.chip?.urgent ?? [],
      useBy: d.chip?.use_by_gameweek ?? null,
      statuses: d.chip?.statuses ?? [],
      nextHalf: d.chip?.next_half ?? [],
    },
    horizonGain: num(d.headline?.horizon_gain),
    horizonWeeks: d.headline?.horizon_weeks ?? 5,
    why: d.why ?? [],
  };
}

// ---------------------------------------------------------------------------
// "You vs the consultant" (fpl/consultant.py, latest.json consultant_vs_you)
// ---------------------------------------------------------------------------

export interface Tracker {
  started: boolean;
  firstGameweek: number;
  lastWeek: { gameweek: number; consultant: number; you: number } | null;
  totals: { consultant: number; you: number; difference: number } | null;
}

export function readTracker(data: Latest): Tracker {
  const t: Raw | undefined = data.consultant_vs_you;
  if (!t?.available || !t.started) {
    return { started: false, firstGameweek: t?.first_gameweek ?? 11, lastWeek: null, totals: null };
  }
  return {
    started: true,
    firstGameweek: t.first_gameweek,
    lastWeek: { gameweek: t.last_week.gameweek, consultant: t.last_week.consultant_points, you: t.last_week.my_points },
    totals: { consultant: t.totals.consultant, you: t.totals.you, difference: t.totals.difference },
  };
}
