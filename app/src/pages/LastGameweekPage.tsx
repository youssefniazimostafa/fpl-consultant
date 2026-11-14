import type { Latest } from "../data";
import { Card } from "../components/Card";
import { Bench, Pitch } from "../components/Pitch";
import { Shirt } from "../components/Shirt";
import type { PlayerView } from "../model/team";

// Last gameweek: where your points came from and how each player did
// against what was expected. Everything comes from latest.json's
// `last_gameweek` section (fpl/last_gameweek.py), which only publishes
// numbers that reconcile with your official FPL points.

type Raw = Record<string, any>;

export const ordinal = (n: number) =>
  `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;

// Green if the player scored at least what was expected, pink if not; none
// when no expectation was saved (or for bench players who weren't due to play).
function markerOf(c: Raw, bench: boolean): "beat" | "missed" | null {
  if (c.expected == null || (bench && !c.subbed_off)) return null;
  return c.points >= c.expected ? "beat" : "missed";
}

export function toView(c: Raw, bench = false): PlayerView {
  return {
    id: c.player_id,
    name: c.web_name,
    teamCode: c.team_code ?? null,
    teamShort: c.team_short ?? null,
    position: c.position,
    opponent: c.team_short ?? null,
    projected: null,
    points: bench ? c.points : c.counted_points,
    price: null,
    isCaptain: !bench && !!c.had_armband,
    isVice: !bench && !!c.is_vice_captain && !c.had_armband,
    isNew: false,
    flagged: false,
    news: null,
    benchLabel: bench ? (c.subbed_off ? "Subbed off" : null) : null,
    marker: markerOf(c, bench),
  };
}

export function Stat({ label, value, sub, dark }: { label: string; value: string; sub?: string; dark?: boolean }) {
  return (
    <div className={`lg-stat${dark ? " dark" : ""}`}>
      <div className="lg-stat-label">{label}</div>
      <div className="lg-stat-value">{value}</div>
      {sub && <div className="lg-stat-sub">{sub}</div>}
    </div>
  );
}

// The points breakdown: FPL's own per-item points (fpl/last_gameweek.py,
// already reconciled with FPL's official totals) shown as "bubbles":
// points in large bold text, the count underneath. Laptop: a table with a
// column per kind of points. Phone: the same bubbles in a wrapping row.

const COLUMNS: [string, string][] = [
  ["minutes", "Minutes"],
  ["goals", "Goals"],
  ["assists", "Assists"],
  ["clean_sheet", "Clean sheet"],
  ["bonus", "Bonus"],
  ["defensive", "Defensive"],
  ["saves", "Saves"],
  ["other", "Other"],
];

function Bubble({ it }: { it: Raw }) {
  return (
    <span className={`bub ${it.points < 0 ? "neg" : "pos"}`} title={it.label}>
      <b>
        {it.points > 0 ? "+" : "−"}
        {Math.abs(it.points)}
      </b>
      <small>{it.short}</small>
    </span>
  );
}

function CaptainBubble({ c }: { c: Raw }) {
  return c.multiplier > 1 ? (
    <span className="bub cap" title={`Captain: points ×${c.multiplier}`}>
      <b>×{c.multiplier}</b>
      <small>captain</small>
    </span>
  ) : null;
}

function PlayerCell({ c }: { c: Raw }) {
  const s = c.surprise;
  return (
    <div className="bt-player">
      <Shirt teamCode={c.team_code} teamShort={c.team_short} goalkeeper={c.position === "GKP"} size={32} title={c.web_name} />
      <div className="bt-name">
        <b>{c.web_name}</b>
        <small>
          {c.expected != null ? `expected ${c.expected.toFixed(1)}` : "no expectation saved"}
          {s != null && <span className={s >= 0 ? "beat" : "missed"}> ({s >= 0 ? "+" : "−"}{Math.abs(s).toFixed(1)})</span>}
          {c.subbed_on ? " · came off the bench" : ""}
          {c.subbed_off ? " · replaced" : ""}
        </small>
      </div>
    </div>
  );
}

const Total = ({ c }: { c: Raw }) => <span className="bt-total" title="Points that counted">{c.counted_points}</span>;

/** Laptop: one row per player (one per match in a double gameweek). */
export function BreakdownTable({ players }: { players: Raw[] }) {
  return (
    <div className="bt" role="table">
      <div className="bt-row bt-head" role="row">
        <span>Player</span>
        {COLUMNS.map(([, h]) => (
          <span key={h}>{h}</span>
        ))}
        <span>Captain</span>
        <span>Total</span>
      </div>
      {players.map((c) => {
        const matches: Raw[] = c.matches?.length ? c.matches : [{ items: [], opponent: null }];
        return matches.map((m, i) => (
          <div key={`${c.player_id}-${i}`} className={`bt-row${i > 0 ? " bt-cont" : ""}`} role="row">
            {i === 0 ? <PlayerCell c={c} /> : <span />}
            {COLUMNS.map(([key]) => (
              <span key={key} className="bt-cell">
                {key === "minutes" && !m.items.length ? <span className="bt-dnp">Didn't play</span> : null}
                {m.items.filter((it: Raw) => it.column === key).map((it: Raw, j: number) => (
                  <Bubble key={j} it={it} />
                ))}
              </span>
            ))}
            <span className="bt-cell">{i === 0 && <CaptainBubble c={c} />}</span>
            <span className="bt-cell">
              {matches.length > 1 && <span className="bt-vs">vs {m.opponent}</span>}
              {i === 0 && <Total c={c} />}
            </span>
          </div>
        ));
      })}
    </div>
  );
}

/** Phone: name line with the total on the right, bubbles wrapping below. */
export function BreakdownCards({ players }: { players: Raw[] }) {
  return (
    <div className="bc">
      {players.map((c) => (
        <div key={c.player_id} className="bc-card">
          <div className="bc-top">
            <PlayerCell c={c} />
            <Total c={c} />
          </div>
          {(c.matches?.length ? c.matches : [{ items: [] }]).map((m: Raw, i: number, all: Raw[]) => (
            <div key={i} className="bc-bubbles">
              {all.length > 1 && <span className="bt-vs">vs {m.opponent}</span>}
              {m.items.length ? m.items.map((it: Raw, j: number) => <Bubble key={j} it={it} />) : <span className="bt-dnp">Didn't play</span>}
              {i === all.length - 1 && <CaptainBubble c={c} />}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// "Best you could have done": hindsight, but only with the moves you had
// that week (fpl/hindsight.py): your squad and bank at the deadline, your
// free transfers, no hits, no chips. The gap is split in a fixed order.
function Mini({ p, out }: { p: Raw; out?: boolean }) {
  return (
    <span className={`pl-mini${out ? " out" : ""}`}>
      <Shirt teamCode={p.team_code} teamShort={p.team_short} goalkeeper={p.position === "GKP"} size={26} dimmed={out} title={p.web_name} />
      <span className="pl-mini-name">{p.web_name}</span>
      <small className="muted"> {p.points}</small>
    </span>
  );
}

export const HINDSIGHT_NOTE = "Hindsight only: the best moves for this one week, not necessarily good long-term picks.";

function BestPossibleCard({ b }: { b: Raw | undefined }) {
  if (!b) return null;
  if (!b.available) {
    return (
      <Card title="Best you could have done" className="best-card">
        <p className="muted">{b.reason}</p>
      </Card>
    );
  }
  const p = b.parts;
  const sign = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v)}`;
  const split = [
    p.chip_hit ? `${sign(p.chip_hit)} from ${b.chip ? `not playing your ${b.chip}` : "not taking your points hit"}` : null,
    `${sign(p.lineup)} from your lineup`,
    `${sign(p.captain)} from the captain`,
    `${sign(p.transfer)} from the transfer${b.transfers.length === 1 ? "" : "s"}`,
  ].filter(Boolean);
  return (
    <Card title="Best you could have done" aside={`${b.free_transfers} free transfer${b.free_transfers === 1 ? "" : "s"} · no hits · no chips`} className="best-card">
      {(b.chip || b.hits > 0) && (
        <p className="note warn">
          You {b.chip ? `played ${b.chip}` : ""}{b.chip && b.hits ? " and " : ""}{b.hits ? `took a ${4 * b.hits}-point hit` : ""} that week; this is the best with free transfers only.
        </p>
      )}
      <div className="best-rows">
        <div className="best-row">
          <span className="best-label">You did</span>
          {b.your_transfers?.length ? (
            b.your_transfers.map((t: Raw, i: number) => (
              <span key={i} className="pl-move">
                <Mini p={t.out} out /> <span className="pl-arrow">→</span> <Mini p={t.in} />
              </span>
            ))
          ) : (
            <span className="small">No transfers</span>
          )}
        </div>
        <div className="best-row">
          <span className="best-label">Best was</span>
          {b.transfers.length ? (
            b.transfers.map((t: Raw, i: number) => (
              <span key={i} className="pl-move">
                <Mini p={t.out} out /> <span className="pl-arrow">→</span> <Mini p={t.in} />
              </span>
            ))
          ) : (
            <span>Best was to keep your squad</span>
          )}
        </div>
        <div className="best-row">
          <span className="best-label">Captain</span>
          {b.captain && <Mini p={b.captain} />}
          {b.your_captain && b.captain && b.your_captain.player_id !== b.captain.player_id && (
            <small className="muted">(you had {b.your_captain.web_name})</small>
          )}
        </div>
        <div className="best-row">
          <span className="best-label">Lineup</span>
          {b.lineup_in.length || b.lineup_out.length ? (
            <span className="small">
              {b.lineup_in.length ? `Also start ${b.lineup_in.map((x: Raw) => x.web_name).join(", ")}` : ""}
              {b.lineup_in.length && b.lineup_out.length ? ", " : ""}
              {b.lineup_out.length ? `bench ${b.lineup_out.map((x: Raw) => x.web_name).join(", ")}` : ""}
              <span className="muted"> (besides the transfers)</span>
            </span>
          ) : (
            <span className="small">No other changes to your starters</span>
          )}
        </div>
      </div>
      <div className="best-nums">
        <div><small>You scored</small><b>{b.you}</b></div>
        <div><small>The consultant's team</small><b>{b.consultant ?? "–"}</b>{b.consultant == null && <small>not tracked yet</small>}</div>
        <div className="hl"><small>Best possible</small><b>{b.best}</b></div>
      </div>
      <p className="best-split">{split.join(", ")} = {sign(b.best - b.you)}</p>
      <p className="note muted">{HINDSIGHT_NOTE}</p>
    </Card>
  );
}

export function LastGameweekPage({ data }: { data: Latest }) {
  const l: Raw | undefined = data.last_gameweek;
  if (!l?.available) {
    return (
      <Card title="Last gameweek">
        <p className="muted">{l?.reason ?? "No finished gameweek in this update yet. Press Refresh to run a fresh update."}</p>
      </Card>
    );
  }
  const xi: PlayerView[] = l.starting_xi.map((c: Raw) => toView(c));
  const bench: PlayerView[] = l.bench.map((c: Raw) => toView(c, true));
  const diff = l.fpl_average != null ? l.total - l.fpl_average : null;

  return (
    <div className="lastgw-page">
      <div className="page-title">
        <h1>Last gameweek</h1>
        <span className="muted">
          GW{l.gameweek} as it was played{l.chip ? ` · ${l.chip} played` : ""}
        </span>
      </div>

      <div className="lg-strip">
        <Stat
          dark
          label="Your points"
          value={String(l.total)}
          sub={l.hit_cost ? `after a ${l.hit_cost}-point hit` : "no points hit"}
        />
        <Stat
          label="FPL average"
          value={l.fpl_average != null ? String(l.fpl_average) : "–"}
          sub={diff == null ? undefined : diff === 0 ? "level with average" : `you were ${Math.abs(diff)} ${diff > 0 ? "above" : "below"}`}
        />
        <Stat
          label="Your league rank that week"
          value={l.league_rank ? ordinal(l.league_rank) : "–"}
          sub={l.league_size ? `of ${l.league_size}` : undefined}
        />
        <Stat
          label="Captain points"
          value={String(l.captain_points)}
          sub={l.captain_name ? `${l.captain_name}${l.captain_stood_in ? " (vice stood in)" : ""}` : "no captain played"}
        />
      </div>

      <div className="lg-grid">
        <div className="lg-left">
          <Pitch players={xi} legend="Points under each name · dot: beat (green) or missed (pink) expected" />
          <Bench players={bench} />
          {l.autosubs.length > 0 && (
            <p className="note muted">
              Automatic subs: {l.autosubs.map((s: Raw) => `${s.on} came in for ${s.off}`).join(", ")}.
            </p>
          )}
        </div>
        <div className="lg-right">
          <Card title="In short" className="lg-summary">
            <p className="why">{l.summary.join(" ")}</p>
          </Card>
          <BestPossibleCard b={l.best_possible} />
        </div>
      </div>

      <Card title="Where every point came from" aside="biggest surprise first · FPL's own points" className="lg-breakdown">
        {!l.has_expected && (
          <p className="note muted">No saved pre-deadline expectations for this gameweek, so only actual points are shown.</p>
        )}
        <div className="show-laptop">
          <BreakdownTable players={l.players} />
        </div>
        <div className="show-phone">
          <BreakdownCards players={l.players} />
        </div>
      </Card>
    </div>
  );
}
