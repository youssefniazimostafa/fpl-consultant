import { Fragment } from "react";
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

const ordinal = (n: number) =>
  `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;

// Green if the player scored at least what was expected, pink if not; none
// when no expectation was saved (or for bench players who weren't due to play).
function markerOf(c: Raw, bench: boolean): "beat" | "missed" | null {
  if (c.expected == null || (bench && !c.subbed_off)) return null;
  return c.points >= c.expected ? "beat" : "missed";
}

function toView(c: Raw, bench = false): PlayerView {
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

function Stat({ label, value, sub, dark }: { label: string; value: string; sub?: string; dark?: boolean }) {
  return (
    <div className={`lg-stat${dark ? " dark" : ""}`}>
      <div className="lg-stat-label">{label}</div>
      <div className="lg-stat-value">{value}</div>
      {sub && <div className="lg-stat-sub">{sub}</div>}
    </div>
  );
}

// FPL's own scoring items for one player, e.g.
// "85 min +2 · 1 goal +4 · 1 assist +3 · 2 bonus +2 = 11", one line per
// match (double gameweeks), then the captain step if he had the armband.
function Breakdown({ c }: { c: Raw }) {
  const matches: Raw[] = c.matches ?? [];
  const captainStep =
    c.multiplier > 1 ? (
      <>
        , ×{c.multiplier} as captain = <b>{c.counted_points}</b>
      </>
    ) : null;
  if (!matches.length) {
    return (
      <div className="lg-break">
        <span className="lg-item">Didn't play</span> <b>= 0</b>
        {captainStep}
      </div>
    );
  }
  const many = matches.length > 1;
  return (
    <div className="lg-break">
      {matches.map((m, i) => (
        <div key={i} className="lg-match">
          {many && <span className="lg-match-no">Match {i + 1}: </span>}
          {m.items.map((it: Raw, j: number) => (
            // the separator sits outside the no-wrap item, so a long line
            // wraps between items, never inside one
            <Fragment key={j}>
              {j > 0 && <span className="lg-sep"> · </span>}
              <span className={`lg-item${it.points < 0 ? " neg" : ""}`}>
                {it.label} {it.points > 0 ? "+" : "−"}
                {Math.abs(it.points)}
              </span>
            </Fragment>
          ))}{" "}
          <b>= {m.total}</b>
          {!many && i === matches.length - 1 && captainStep}
        </div>
      ))}
      {many && (
        <div className="lg-match">
          Total <b>= {c.points}</b>
          {captainStep}
        </div>
      )}
    </div>
  );
}

function Row({ c }: { c: Raw }) {
  const surprise = c.surprise;
  return (
    <div className="lg-row">
      <Shirt teamCode={c.team_code} teamShort={c.team_short} goalkeeper={c.position === "GKP"} size={34} title={c.web_name} />
      <div className="lg-row-text">
        <div className="lg-row-name">
          <b>{c.web_name}</b>
          <span className="muted small">
            {" "}
            {c.expected != null ? `expected ${c.expected.toFixed(1)}` : ""}
            {c.subbed_on ? " · came in from the bench" : ""}
            {c.subbed_off ? " · replaced from the bench" : ""}
          </span>
        </div>
        <Breakdown c={c} />
      </div>
      {surprise != null && (
        <span className={`lg-diff ${surprise >= 0 ? "beat" : "missed"}`}>
          {surprise >= 0 ? "+" : "−"}
          {Math.abs(surprise).toFixed(1)}
        </span>
      )}
    </div>
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
          <Card title="Player by player" aside="biggest surprise first">
            {!l.has_expected && (
              <p className="note muted">No saved pre-deadline expectations for this gameweek, so only actual points are shown.</p>
            )}
            <div className="lg-list">
              {l.players.map((c: Raw) => (
                <Row key={c.player_id} c={c} />
              ))}
            </div>
          </Card>
          <Card title="In short" className="lg-summary">
            <p className="why">{l.summary.join(" ")}</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
