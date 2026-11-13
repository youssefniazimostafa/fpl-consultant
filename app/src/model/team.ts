// Turns latest.json into what the "Your team" page shows, in plain
// football language. Pure functions, no React -- the page just renders
// these. Reads the same panels the old site's "Recommended Team",
// "My team", "Transfer plan" and "Chips" sections use:
//   recommended_team, verdict, transfer_plan, captain_board, chips,
//   status, strategy, top_xp, config.
// Fields added for the app (price, projected_points, team_short,
// recommended_team.transfers) are optional: older data files without them
// fall back to captain_board / top_xp / transfer_plan.

import type { Latest } from "../data";

export interface PlayerView {
  id: number;
  name: string;
  teamCode: number | null;
  teamShort: string | null;
  position: "GKP" | "DEF" | "MID" | "FWD" | string;
  opponent: string | null; // "NEW (A)"
  projected: number | null; // projected points next gameweek
  price: number | null; // £m
  isCaptain: boolean;
  isVice: boolean;
  isNew: boolean;
  flagged: boolean;
  news: string | null;
  benchLabel: string | null;
}

export interface TransferView {
  out: PlayerView;
  in: PlayerView;
}

export interface TeamView {
  gameweek: number;
  xi: PlayerView[];
  bench: PlayerView[];
  captain: PlayerView | null;
  vice: PlayerView | null;
  transfers: TransferView[];
  roll: boolean;
  freeTransfers: number | null;
  hitsTaken: number;
  hitReason: string | null;
  horizonGain: number | null; // projected points gained over the horizon
  horizonWeeks: number;
  outNews: { name: string; news: string; urgency: string }[];
}

type Raw = Record<string, any>;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function projectionLookup(data: Latest): (id: number) => number | null {
  const m = new Map<number, number>();
  for (const p of data.top_xp?.players ?? []) if (num(p.next_gw_xp) != null) m.set(p.player_id, p.next_gw_xp);
  for (const p of data.captain_board?.players ?? []) if (num(p.xp) != null) m.set(p.player_id, p.xp);
  return (id) => m.get(id) ?? null;
}

function priceLookup(data: Latest): (id: number) => number | null {
  const m = new Map<number, number>();
  for (const p of data.top_xp?.players ?? []) if (num(p.now_cost) != null) m.set(p.player_id, p.now_cost / 10);
  return (id) => m.get(id) ?? null;
}

/** The optimiser's hit reason, e.g. "Souza (u, 0% chance of playing) --
 *  replacing him gains 5.0 pts over the next 3 GWs, beating the 4-point hit
 *  cost", in plainer words. */
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

function opponentLabel(fx: Raw | null | undefined): string | null {
  if (!fx?.opponent_short) return null;
  return `${fx.opponent_short} (${fx.is_home ? "H" : "A"})`;
}

function toView(p: Raw, data: Latest): PlayerView {
  const proj = projectionLookup(data);
  const price = priceLookup(data);
  const id = Number(p.player_id ?? -1);
  return {
    id,
    name: p.web_name ?? "?",
    teamCode: num(p.team_code),
    teamShort: p.team_short ?? null,
    position: p.position ?? "",
    opponent: opponentLabel(p.fixture),
    projected: num(p.projected_points) ?? proj(id),
    price: num(p.price) ?? price(id),
    isCaptain: !!p.is_captain,
    isVice: !!p.is_vice_captain,
    isNew: !!p.transferred_in,
    flagged: !!p.flagged,
    news: p.news ?? null,
    benchLabel: p.bench_label ?? null,
  };
}

export function teamView(data: Latest): TeamView | null {
  const rt = data.recommended_team;
  if (!rt?.available) return null;
  const gw: number = rt.gameweek;
  const xi = (rt.starting_xi as Raw[]).map((p) => toView(p, data));
  const bench = (rt.bench as Raw[]).map((p) => toView(p, data));

  const weeks: Raw[] = data.transfer_plan?.available ? data.transfer_plan.weeks : [];
  const week0 = weeks.filter((w) => w.gameweek === gw && (w.player_out || w.player_in));
  const recommended = data.verdict?.available ? !!data.verdict.transfer_recommended : week0.some((w) => w.recommended);

  let transfers: TransferView[] = [];
  if (recommended) {
    if (Array.isArray(rt.transfers) && rt.transfers.length) {
      transfers = rt.transfers.map((t: Raw) => ({ out: toView(t.player_out, data), in: toView(t.player_in, data) }));
    } else {
      // older data: names only for the outgoing player
      transfers = week0.map((w) => {
        const inP = [...xi, ...bench].find((p) => p.name === w.player_in && p.isNew);
        return {
          out: toView({ web_name: w.player_out, position: inP?.position }, data),
          in: inP ?? toView({ web_name: w.player_in }, data),
        };
      });
    }
  }

  const horizonWeeks: number = data.top_xp?.horizon_weeks?.length || 5;
  return {
    gameweek: gw,
    xi,
    bench,
    captain: xi.find((p) => p.isCaptain) ?? null,
    vice: xi.find((p) => p.isVice) ?? null,
    transfers,
    roll: !recommended || transfers.length === 0,
    freeTransfers: num(rt.free_transfers_available) ?? num(data.status?.your_free_transfers),
    hitsTaken: num(rt.hits_taken) ?? 0,
    hitReason: plainHitReason(rt.hit_reason ?? null),
    horizonGain: num(data.verdict?.horizon_gain_xp) ?? num(week0[0]?.horizon_gain_xp),
    horizonWeeks,
    outNews: week0
      .filter((w) => w.news && w.player_out)
      .map((w) => ({ name: w.player_out, news: w.news, urgency: w.urgency })),
  };
}

// ---------------------------------------------------------------------------
// captain
// ---------------------------------------------------------------------------

export interface CaptainView {
  captain: PlayerView;
  vice: PlayerView | null;
  /** The pick most of your rivals are likely to captain, if not yours. */
  rivalPick: { name: string; projected: number } | null;
  /** Points you'd lose if rivals captain rivalPick and both score as projected. */
  possibleLoss: number | null;
  rivalsBackYours: boolean;
}

export function captainView(data: Latest, team: TeamView): CaptainView | null {
  const cap = team.captain;
  if (!cap) return null;
  const board: Raw[] = data.captain_board?.available ? data.captain_board.players : [];
  // league_eo above 1 means rivals on average get more than 1x his points,
  // i.e. many of them captain him -- the highest is their likely armband.
  const favourite = [...board].sort((a, b) => (b.league_eo ?? 0) - (a.league_eo ?? 0))[0];
  if (!favourite || (favourite.league_eo ?? 0) <= 0) {
    return { captain: cap, vice: team.vice, rivalPick: null, possibleLoss: null, rivalsBackYours: false };
  }
  if (favourite.player_id === cap.id) {
    return { captain: cap, vice: team.vice, rivalPick: null, possibleLoss: null, rivalsBackYours: true };
  }
  const capXp = cap.projected ?? 0;
  return {
    captain: cap,
    vice: team.vice,
    rivalPick: { name: favourite.web_name, projected: favourite.xp },
    possibleLoss: Math.max(0, favourite.xp - capXp),
    rivalsBackYours: false,
  };
}

// ---------------------------------------------------------------------------
// chips
// ---------------------------------------------------------------------------

export const CHIP_LABELS: Record<string, string> = {
  wildcard: "Wildcard",
  freehit: "Free Hit",
  bboost: "Bench Boost",
  "3xc": "Triple Captain",
};
const CHIP_ORDER = ["wildcard", "freehit", "bboost", "3xc"];

export interface ChipStatus {
  key: string;
  label: string;
  available: boolean;
  note: string; // "Available" / "Used GW4"
}

export interface ChipsView {
  chips: ChipStatus[];
  playNow: { label: string; gain: number } | null;
  next: { label: string; gameweek: number; gain: number } | null;
  urgent: string[]; // chips that must be used before the half-season cut-off
  deadlineGw: number | null;
}

export function chipsView(data: Latest, gw: number): ChipsView {
  const half = gw <= 19 ? "h1" : "h2";
  const teamId = data.config?.team_id;
  const grid: Raw[] = data.chips?.rivals?.available ? data.chips.rivals.grid : [];
  const mine = grid.find((r) => r.manager_id === teamId);
  const plan: Raw[] = data.chips?.yours?.available ? data.chips.yours.plan : [];

  const chips: ChipStatus[] = CHIP_ORDER.map((key) => {
    const label = CHIP_LABELS[key];
    const cell: string | undefined = mine?.[`${key}_${half}`];
    if (typeof cell === "string") {
      const available = cell === "available";
      return { key, label, available, note: available ? "Available" : cell.replace(/^used/, "Used") };
    }
    const rows = plan.filter((r) => r.chip === key);
    const available = rows.some((r) => r.available);
    return { key, label, available, note: rows.length ? (available ? "Available" : "Used") : "Unknown" };
  });

  // best week for each available chip, over the planning window
  let playNow: ChipsView["playNow"] = null;
  let next: ChipsView["next"] = null;
  const urgent: string[] = [];
  for (const key of CHIP_ORDER) {
    const rows = plan.filter((r) => r.chip === key && r.available);
    if (!rows.length) continue;
    const best = [...rows].sort((a, b) => b.estimated_gain - a.estimated_gain)[0];
    if (rows.some((r) => r.urgent)) urgent.push(CHIP_LABELS[key]);
    if (best.gameweek === gw && (!playNow || best.estimated_gain > playNow.gain)) {
      playNow = { label: CHIP_LABELS[key], gain: best.estimated_gain };
    } else if (best.gameweek !== gw && (!next || best.estimated_gain > next.gain)) {
      next = { label: CHIP_LABELS[key], gameweek: best.gameweek, gain: best.estimated_gain };
    }
  }
  return { chips, playNow, next, urgent, deadlineGw: half === "h1" ? 19 : null };
}

// ---------------------------------------------------------------------------
// "Why this team" -- template sentences, no AI
// ---------------------------------------------------------------------------

/** News text quoted mid-sentence: lower-case first letter, no full stop. */
function quiet(news: string): string {
  const t = news.trim().replace(/\.$/, "");
  return /^[A-Z][a-z]/.test(t) ? t[0].toLowerCase() + t.slice(1) : t;
}

function pts(n: number): string {
  const r = Math.round(n * 10) / 10;
  return `${r} point${r === 1 ? "" : "s"}`;
}

export function whyThisTeam(data: Latest, team: TeamView, cap: CaptainView | null, chips: ChipsView): string[] {
  const out: string[] = [];

  // 1. transfers / roll
  if (team.roll) {
    const nextFt = team.freeTransfers != null ? Math.min(team.freeTransfers + 1, 5) : null;
    out.push(
      `No transfer is worth making this week, so save your free transfer` +
        (nextFt != null ? ` and you'll have ${nextFt} to use next week.` : "."),
    );
  } else {
    for (const t of team.transfers) {
      const news = team.outNews.find((n) => n.name === t.out.name);
      const why =
        news?.urgency === "forced"
          ? `${t.out.name} won't play for you (${quiet(news.news)})`
          : news?.urgency === "flagged"
            ? `${t.out.name} is a doubt to play (${quiet(news.news)})`
            : `${t.in.name} is expected to outscore ${t.out.name}`;
      const inPts = t.in.projected != null ? `, who is projected ${pts(t.in.projected)} this week` : "";
      out.push(`${why}, so bring in ${t.in.name}${inPts}.`);
    }
    const gain =
      team.horizonGain != null
        ? `Over the next ${team.horizonWeeks} gameweeks the move${team.transfers.length > 1 ? "s are" : " is"} worth about ${pts(team.horizonGain)}`
        : "";
    if (team.hitsTaken > 0) {
      out.push(
        team.hitReason
          ? `It's worth taking a points hit: ${team.hitReason.replace(/\.$/, "")}.`
          : `${gain ? `${gain}, enough` : "It's worth it"} to justify a points hit.`,
      );
    } else if (gain) {
      out.push(`${gain}, and it doesn't cost a points hit.`);
    }
  }

  // 2. captain
  if (cap) {
    const c = cap.captain;
    const vs = c.opponent ? ` against ${c.opponent.replace(" (H)", " (home)").replace(" (A)", " (away)")}` : "";
    if (cap.rivalsBackYours) {
      out.push(`${c.name} gets the armband${vs}: he's one of your best scorers and your rivals are backing him too, so it's the safe choice.`);
    } else if (cap.rivalPick && cap.rivalPick.projected > (c.projected ?? 0)) {
      out.push(
        `${c.name} gets the armband${vs}. ${cap.rivalPick.name} is projected slightly more, but most of your rivals will captain him, ` +
          `so ${c.name} gives you a chance to gain ground on them.`,
      );
    } else {
      out.push(`${c.name} gets the armband${vs} as your highest projected scorer${c.projected != null ? ` (${pts(c.projected)})` : ""}.`);
    }
  }

  // 3. chips
  if (chips.playNow) {
    out.push(`Play your ${chips.playNow.label} now: this is its best week, worth about ${pts(chips.playNow.gain)}.`);
  } else if (chips.next) {
    out.push(`Keep your chips for now: your ${chips.next.label} is worth more in GW${chips.next.gameweek} (about ${pts(chips.next.gain)}).`);
  }

  // 4. league position + your chances
  const s = data.strategy;
  const sc = data.chips?.season_context;
  if (s?.available && typeof s.gap_to_first === "number") {
    const where =
      s.gap_to_first > 0
        ? `You're ${s.gap_to_first} point${s.gap_to_first === 1 ? "" : "s"} off first place`
        : "You're top of your league";
    const style =
      s.mode === "chasing"
        ? "so the plan takes a few more risks to catch up"
        : s.mode === "protecting"
          ? "so the plan plays it safe to protect your lead"
          : "so the plan simply picks the strongest team rather than gambling";
    const chance =
      sc && typeof sc.p_finish_1st_by_target === "number"
        ? `; your chances of being top by GW${sc.target_gameweek} are about ${Math.round(sc.p_finish_1st_by_target * 100)}%`
        : "";
    out.push(`${where} with ${s.gameweeks_remaining} gameweeks left, ${style}${chance}.`);
  }
  return out.slice(0, 5);
}
