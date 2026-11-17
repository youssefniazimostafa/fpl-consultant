import { useState, type ReactNode } from "react";
import type { Latest } from "../data";
import { HelpButton, Pill } from "../components/Card";
import { helpFor } from "../help";
import { Shirt } from "../components/Shirt";
import { chanceClass } from "../theme/theme";
import { PlayerLookup } from "../components/PlayerLookup";
import { CrashCard, ErrorBoundary } from "../components/ErrorBoundary";

// Scout (the old FPL overview): FPL as a whole, not your league. Everything comes from
// latest.json's `overview` section (fpl/overview.py); this page only
// filters and lays it out.

type Raw = Record<string, any>;

const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;
const money = (p: number | null | undefined) => (p == null ? "" : `£${p.toFixed(1)}m`);

function ago(iso: string): string {
  const h = Math.max(0, (Date.now() - new Date(iso).getTime()) / 3600000);
  if (h < 1) return "just now";
  if (h < 24) return `${Math.round(h)} h ago`;
  const d = Math.round(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

const TOP_N = 10;

type BubbleProps = { title: string; explain: string; controls?: ReactNode; children: ReactNode };

function Bubble(props: BubbleProps) {
  return (
    <ErrorBoundary where={`Scout card ${props.title}`} fallback={<CrashCard className="sc-card" />}>
      <BubbleInner {...props} />
    </ErrorBoundary>
  );
}

function BubbleInner({ title, explain, controls, children }: BubbleProps) {
  return (
    <section className="sc-card">
      <div className="sc-head">
        <h2>
          {title}
          {helpFor(title) && <HelpButton text={helpFor(title)!} />}
        </h2>
        {controls && <div className="sc-controls">{controls}</div>}
      </div>
      <p className="sc-explain">{explain}</p>
      {children}
    </section>
  );
}

function Tabs<T extends string>({ options, value, onChange }: { options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="ov-tabs sc-tabs" role="tablist">
      {options.map(([v, label]) => (
        <button key={v} role="tab" aria-selected={v === value} className={v === value ? "active" : ""} onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

/** One ranked row: rank, shirt, name, club · position, key number, "You own". */
const row = (i: number, p: Raw, keyNumber: ReactNode, sub?: ReactNode, extra?: ReactNode, className = "") => (
  <li key={p.player_id} className={`sc-row ${className}`}>
    <span className="sc-rank">{i + 1}</span>
    <Shirt teamCode={p.team_code ?? null} teamShort={p.team_short ?? null} goalkeeper={p.position === "GKP"} size={24} title={p.web_name} />
    <span className="sc-name">
      <b>{p.web_name}</b>
      <small>
        {[p.team_short, p.position].filter(Boolean).join(" · ")}
        {sub ? <> · {sub}</> : null}
      </small>
      {extra ? <span className="sc-extra">{extra}</span> : null}
    </span>
    <span className="sc-key">
      {keyNumber}
      {p.you_own && <span className="sc-own">You own</span>}
    </span>
  </li>
);

type Pos = "ALL" | "GKP" | "DEF" | "MID" | "FWD";
type ValueWindow = "4" | "1";
type Cap = "5" | "6" | "7" | "any";

export function OverviewPage({ data }: { data: Latest }) {
  const o: Raw | undefined = data.overview;
  const [pos, setPos] = useState<Pos>("ALL");
  const [win, setWin] = useState<ValueWindow>("4");
  const [cap, setCap] = useState<Cap>("6");

  if (!o?.available) {
    return <p className="muted">No Scout data yet. Press Refresh to load the latest data.</p>;
  }
  const gw = o.next_gameweek;
  const last = o.last_gameweek;
  const top: Raw[] = o.top_projected?.[pos] ?? [];
  const capLimit = cap === "any" ? Infinity : Number(cap);
  const value: Raw[] = [...(o.value ?? [])]
    .filter((p) => p.price != null && p.price < capLimit)
    .sort((a, b) => (win === "4" ? b.per_m_last_4 - a.per_m_last_4 : b.per_m_last_gw - a.per_m_last_gw))
    .slice(0, TOP_N);

  return (
    <div className="overview-page">
      <div className="page-title">
        <h1>Scout</h1>
        <span className="muted">All of FPL, not just your league · GW{gw}</span>
      </div>

      <div className="sc-grid">
        <Bubble
          title={`Top expected points, GW${gw}`}
          explain="The players expected to score the most next gameweek."
          controls={
            <Tabs<Pos>
              options={[["ALL", "All"], ["GKP", "GK"], ["DEF", "DEF"], ["MID", "MID"], ["FWD", "FWD"]]}
              value={pos}
              onChange={setPos}
            />
          }
        >
          <ol className="sc-list">
            {top.slice(0, TOP_N).map((p, i) => row(i, p, `${p.projected_points.toFixed(1)} pts`, [money(p.price), p.opponent].filter(Boolean).join(" · ")))}
          </ol>
        </Bubble>

        <Bubble title="Best captain options" explain="Who would score the most with the armband next gameweek (points are doubled).">
          <ol className="sc-list">
            {(o.captain_options ?? []).slice(0, TOP_N).map((p: Raw, i: number) =>
              row(i, p, `${p.as_captain.toFixed(1)} as C`, [money(p.price), p.opponent].filter(Boolean).join(" · ")),
            )}
          </ol>
        </Bubble>

        <Bubble
          title={`Slept-on players, GW${last}`}
          explain="Players who scored far more than expected. Players beating expectations week after week may be finding form."
        >
          {(o.slept_on ?? []).length ? (
            <ol className="sc-list">
              {o.slept_on.slice(0, TOP_N).map((p: Raw, i: number) =>
                row(
                  i,
                  p,
                  <>
                    {p.points} <small>vs {Math.round(p.expected)}</small>
                  </>,
                  money(p.price),
                  p.weeks_running >= 2 ? <Pill tone="pink">{ordinal(p.weeks_running)} week running</Pill> : undefined,
                ),
              )}
            </ol>
          ) : (
            <p className="muted">Nobody beat expectations by much last gameweek.</p>
          )}
        </Bubble>

        <Bubble
          title="In form lately"
          explain={`Players scoring clearly more per game over the last ${o.form_weeks} gameweeks than earlier in the season.`}
        >
          <ol className="sc-list">
            {(o.in_form ?? []).slice(0, TOP_N).map((p: Raw, i: number) =>
              row(i, p, `${p.recent_ppg.toFixed(1)} a game`, `GW${p.recent_from}-${last} · before ${p.before_ppg.toFixed(1)}`),
            )}
          </ol>
        </Bubble>

        <Bubble
          title="Cheap high scorers"
          explain="Best value: points scored for every £1m of price."
          controls={
            <>
              <Tabs<ValueWindow> options={[["4", `Last ${o.form_weeks} GWs`], ["1", "Last GW"]]} value={win} onChange={setWin} />
              <Tabs<Cap> options={[["5", "Under £5m"], ["6", "Under £6m"], ["7", "Under £7m"], ["any", "Any price"]]} value={cap} onChange={setCap} />
            </>
          }
        >
          {value.length ? (
            <ol className="sc-list">
              {value.map((p, i) =>
                row(
                  i,
                  p,
                  `${(win === "4" ? p.per_m_last_4 : p.per_m_last_gw).toFixed(1)} per £1m`,
                  `${win === "4" ? p.points_last_4 : p.points_last_gw} pts · ${money(p.price)}`,
                ),
              )}
            </ol>
          ) : (
            <p className="muted">No players under that price scored this period.</p>
          )}
        </Bubble>

        <Bubble title="Injury and team news" explain="Players whose availability changed in the last week, newest first, in FPL's own words.">
          {(o.news ?? []).length ? (
            <ol className="sc-list">
              {o.news.slice(0, TOP_N).map((p: Raw, i: number) =>
                row(
                  i,
                  p,
                  <span className={`cop-badge ${chanceClass(p.chance ?? 50)}`}>{p.chance == null ? "Doubtful" : `${p.chance}%`}</span>,
                  ago(p.added),
                  <span className={`sc-news ${p.chance === 0 ? "news-out" : ""}`}>{p.news}</span>,
                  "sc-row-news",
                ),
              )}
            </ol>
          ) : (
            <p className="muted">No new injury or team news in the last week.</p>
          )}
        </Bubble>

        {o.lookup?.available && (
          <ErrorBoundary where="Player lookup" fallback={<CrashCard className="sc-card sc-lookup" />}>
            <PlayerLookup lookup={o.lookup} />
          </ErrorBoundary>
        )}
      </div>
    </div>
  );
}
