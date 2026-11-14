import { useState, type ReactNode } from "react";
import type { Latest } from "../data";
import { Pill } from "../components/Card";
import { PlayerCard } from "../components/PlayerCard";

// FPL overview: FPL as a whole, not your league. Everything comes from
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

function Section({ title, explain, controls, children }: { title: string; explain: string; controls?: ReactNode; children: ReactNode }) {
  return (
    <section className="ov-section">
      <div className="ov-head">
        <h2>{title}</h2>
        {controls && <div className="ov-controls">{controls}</div>}
      </div>
      <p className="ov-explain">{explain}</p>
      {children}
    </section>
  );
}

function Tabs<T extends string>({ options, value, onChange }: { options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="ov-tabs" role="tablist">
      {options.map(([v, label]) => (
        <button key={v} role="tab" aria-selected={v === value} className={v === value ? "active" : ""} onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

const card = (p: Raw, keyNumber: ReactNode, sub?: ReactNode, extra?: ReactNode) => (
  <PlayerCard
    key={p.player_id}
    name={p.web_name}
    teamCode={p.team_code ?? null}
    teamShort={p.team_short ?? null}
    position={p.position ?? null}
    youOwn={!!p.you_own}
    keyNumber={keyNumber}
    sub={sub}
    extra={extra}
  />
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
    return <p className="muted">No FPL overview in this update yet. Press Refresh to run a fresh update.</p>;
  }
  const gw = o.next_gameweek;
  const last = o.last_gameweek;
  const top: Raw[] = o.top_projected?.[pos] ?? [];
  const capLimit = cap === "any" ? Infinity : Number(cap);
  const value: Raw[] = [...(o.value ?? [])]
    .filter((p) => p.price != null && p.price < capLimit)
    .sort((a, b) => (win === "4" ? b.per_m_last_4 - a.per_m_last_4 : b.per_m_last_gw - a.per_m_last_gw))
    .slice(0, 12);

  return (
    <div className="overview-page">
      <div className="page-title">
        <h1>FPL overview</h1>
        <span className="muted">All of FPL, not just your league · GW{gw}</span>
      </div>

      <Section
        title={`Top projected points, GW${gw}`}
        explain="The players expected to score the most next gameweek."
        controls={
          <Tabs<Pos>
            options={[["ALL", "All"], ["GKP", "GK"], ["DEF", "DEF"], ["MID", "MID"], ["FWD", "FWD"]]}
            value={pos}
            onChange={setPos}
          />
        }
      >
        <div className="pcard-grid">
          {top.map((p) => card(p, `${p.projected_points.toFixed(1)} pts`, [money(p.price), p.opponent].filter(Boolean).join(" · ")))}
        </div>
      </Section>

      <Section title="Best captain options" explain="Who would score the most with the armband next gameweek (points are doubled).">
        <div className="pcard-grid">
          {(o.captain_options ?? []).map((p: Raw) =>
            card(p, `${p.as_captain.toFixed(1)} as captain`, [money(p.price), p.opponent].filter(Boolean).join(" · ")),
          )}
        </div>
      </Section>

      <Section
        title={`Slept-on players, GW${last}`}
        explain="Players who scored far more than expected. Players beating expectations week after week may be finding form."
      >
        {(o.slept_on ?? []).length ? (
          <div className="pcard-grid">
            {o.slept_on.map((p: Raw) =>
              card(
                p,
                `scored ${p.points}, expected ${Math.round(p.expected)}`,
                money(p.price),
                p.weeks_running >= 2 ? <Pill tone="pink">{ordinal(p.weeks_running)} week running</Pill> : undefined,
              ),
            )}
          </div>
        ) : (
          <p className="muted">Nobody beat expectations by much last gameweek.</p>
        )}
      </Section>

      <Section
        title="In form lately"
        explain={`Players scoring clearly more per game over the last ${o.form_weeks} gameweeks than earlier in the season.`}
      >
        <div className="pcard-grid">
          {(o.in_form ?? []).map((p: Raw) =>
            card(
              p,
              `${p.recent_ppg.toFixed(1)} a game`,
              `GW${p.recent_from}-${last}: ${p.recent_ppg.toFixed(1)} · before: ${p.before_ppg.toFixed(1)}`,
            ),
          )}
        </div>
      </Section>

      <Section
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
          <div className="pcard-grid">
            {value.map((p) =>
              card(
                p,
                `${(win === "4" ? p.per_m_last_4 : p.per_m_last_gw).toFixed(1)} pts per £1m`,
                `${win === "4" ? p.points_last_4 : p.points_last_gw} pts · ${money(p.price)}`,
              ),
            )}
          </div>
        ) : (
          <p className="muted">No players under that price scored this period.</p>
        )}
      </Section>

      <Section title="Injury and team news" explain="Players whose availability changed in the last week, newest first, in FPL's own words.">
        {(o.news ?? []).length ? (
          <div className="pcard-grid">
            {o.news.map((p: Raw) =>
              card(
                p,
                p.chance == null ? "Doubtful" : `${p.chance}% chance of playing`,
                ago(p.added),
                <span className={p.chance === 0 ? "news-out" : ""}>{p.news}</span>,
              ),
            )}
          </div>
        ) : (
          <p className="muted">No new injury or team news in the last week.</p>
        )}
      </Section>
    </div>
  );
}
