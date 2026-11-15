// The five UCL pages. They read data/ucl/latest.json (ucl/pipeline.py's
// build_payload) loosely; every field is optional so an older file can't
// crash a page.
import type { ReactNode } from "react";
import type { ChartConfiguration } from "chart.js";
import { Card } from "../components/Card";
import { Chart } from "../components/Chart";
import { uclHelp } from "./help";
import { UclShirt } from "./UclShirt";
import type { Ucl } from "./data";

type P = Record<string, any>;

const CYAN = "#2be4ff";
const ROYAL = "#6f8cff";
const WHITE = "#ffffff";

export function UCard({ title, aside, children, className = "" }: { title: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Card title={title} aside={aside} className={`ucard ${className}`} help={uclHelp(title) ?? ""}>
      {children}
    </Card>
  );
}

const dayShort = (d?: string | null) => (d ? d.split(" ")[0] : "–");

function Tile({ p, label, points }: { p: P; label?: string; points?: number }) {
  return (
    <div className={`utile${p.status === "I" || p.status === "D" ? " flagged" : ""}`} title={p.news ?? p.name}>
      {label && <div className="utile-label">{label}</div>}
      <div className="utile-badge">
        <UclShirt code={p.club_code} goalkeeper={p.pos === "GK"} size={46} title={`${p.club} · ${p.name}`} />
        {p.captain && <span className="ubadge ubadge-c">C</span>}
        {p.vice && <span className="ubadge ubadge-v">V</span>}
        {(p.status === "I" || p.status === "D") && <span className="ubadge ubadge-flag">!</span>}
      </div>
      <div className="utile-name">{p.name}</div>
      <div className="utile-meta">
        <span>{points != null ? (p.counted === false ? "bench" : p.pos) : `${p.opp ?? "–"} · ${dayShort(p.day)}`}</span>
        <b>{points != null ? points : p.xp != null ? Number(p.xp).toFixed(1) : "–"}</b>
      </div>
    </div>
  );
}

function UPitch({ players, legend, points }: { players: P[]; legend: string; points?: boolean }) {
  return (
    <div className="upitch">
      {["GK", "DEF", "MID", "FWD"].map((pos) => {
        const row = players.filter((p) => p.pos === pos);
        return row.length ? (
          <div key={pos} className="upitch-row">
            {row.map((p) => (
              <Tile key={p.id} p={p} points={points ? p.points : undefined} />
            ))}
          </div>
        ) : null;
      })}
      <div className="upitch-legend">{legend}</div>
    </div>
  );
}

const fmtDeadline = (iso?: string) =>
  iso
    ? new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
    : "–";

function Strip({ d }: { d: Ucl }) {
  return (
    <div className="ustrip">
      <span className="ustrip-md">MD{d.md}</span>
      <span>{d.stage}</span>
      <span>Deadline {fmtDeadline(d.deadline)}</span>
      <span>
        €{d.budget}m · max {d.club_limit} per club
      </span>
    </div>
  );
}

// ------------------------------------------------------------- Recommended team

export function UclTeamPage({ d, teamSaved }: { d: Ucl; teamSaved: boolean }) {
  const t = d.team ?? {};
  const ps: P[] = t.players ?? [];
  const xi = ps.filter((p) => p.role === "start");
  const bench = ps.filter((p) => p.role === "bench").sort((a, b) => a.bench_order - b.bench_order);
  const cap = ps.find((p) => p.captain);
  const vice = ps.find((p) => p.vice);
  const days: P[] = d.matchday?.days ?? [];
  return (
    <div className="upage">
      <Strip d={d} />
      <div className="unote">
        {teamSaved ? (
          <>
            <b>Your UCL team ID is saved,</b> but UEFA only shows teams after you log in, so XPert can't read your players. The advice is for this
            squad.
          </>
        ) : (
          <>
            <b>No UCL team saved.</b> Create this squad in UCL Fantasy: a new team is free, within €{d.budget}m. Reading your own team would need a
            UEFA login, so XPert gives its advice for this squad.
          </>
        )}
      </div>
      <div className="ugrid">
        <UCard title="Recommended squad" aside={`${t.formation} · €${t.cost}m · €${t.bank}m left · ${t.xp} xP`} className="ucard-pitch">
          <UPitch players={xi} legend="Club · opponent · day · expected points" />
          <div className="ubench">
            {bench.map((p) => (
              <Tile key={p.id} p={p} label={p.bench_order === 1 ? "GK" : `Sub ${p.bench_order - 1}`} />
            ))}
          </div>
        </UCard>
        <div className="ucol">
          <UCard title="Captain">
            {cap && (
              <p className="ubig">
                {cap.name} <span className="umuted">({cap.club}, {cap.day})</span>
              </p>
            )}
            {vice && <p className="umuted">Vice-captain: {vice.name}</p>}
            <table className="utable">
              <thead>
                <tr>
                  <th>Option</th>
                  <th>xP</th>
                  <th title="Expected captain points counting the between-days switch">With switch</th>
                </tr>
              </thead>
              <tbody>
                {(t.captain_options ?? []).slice(0, 4).map((o: P) => (
                  <tr key={o.pid} className={o.pid === t.captain ? "on" : ""}>
                    <td>{o.name}</td>
                    <td>{o.xp.toFixed(1)}</td>
                    <td>
                      {o.with_switch.toFixed(1)}
                      {o.switch_to_name && <span className="umuted"> (→ {o.switch_to_name} in {Math.round(o.switch_share * 100)}% of weeks)</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </UCard>
          <UCard title="Match days">
            {days.map((day) => (
              <div key={day.gd} className="uday">
                <b>{day.label}</b>
                <span className="umuted"> · {ps.filter((p) => p.gd === day.gd).length} of your 15 play</span>
                <div className="uday-names">
                  {ps
                    .filter((p) => p.gd === day.gd)
                    .map((p) => p.name)
                    .join(", ") || "none"}
                </div>
              </div>
            ))}
          </UCard>
          <UCard title="Value of flexibility">
            <p>
              Changing players between days should add about <b>{t.flexibility?.value ?? 0} points</b> on average to this squad
              {t.flexibility?.captain_switch_value ? ` (${t.flexibility.captain_switch_value} of it from switching the captain)` : ""}
              {t.flexibility?.share_of_weeks != null ? `, and is worth doing in about ${Math.round(t.flexibility.share_of_weeks * 10)} weeks in 10.` : "."}
            </p>
          </UCard>
        </div>
      </div>
      <UCard title="Why these players">
        <ul className="ureasons">
          {ps.map((p) => (
            <li key={p.id}>
              <UclShirt code={p.club_code} goalkeeper={p.pos === "GK"} size={26} /> <b>{p.name}</b> <span className="umuted">{p.pos} · €{p.price}m · {p.day ?? "no match"}</span>
              <div>{p.reason}</div>
            </li>
          ))}
        </ul>
      </UCard>
      <div className="ugrid2">
        <PlanCard d={d} />
        <ChipsCard d={d} />
      </div>
    </div>
  );
}

function PlanCard({ d }: { d: Ucl }) {
  return (
    <UCard title="Transfers and plan" aside="up to the play-offs (unlimited transfers before MD9)">
      <table className="utable">
        <thead>
          <tr>
            <th>MD</th>
            <th>Transfers</th>
            <th>Free</th>
            <th>xP</th>
          </tr>
        </thead>
        <tbody>
          {(d.plan ?? []).map((s: P, i: number) => (
            <tr key={s.md}>
              <td>MD{s.md}</td>
              <td>
                {i === 0 && d.team?.transfers?.new_team
                  ? "New team (free)"
                  : s.in.length
                    ? s.in.map((p: P, k: number) => `${s.out[k]?.name ?? "?"} → ${p.name}`).join(", ")
                    : "Keep, save the transfer"}
                {s.hits ? ` (−${4 * s.hits})` : ""}
                {s.chip ? ` · ${s.chip}` : ""}
              </td>
              <td>{s.ft ?? "∞"}</td>
              <td>{s.xp != null ? Number(s.xp).toFixed(2) : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="umuted small">Only this matchday's team is final. Later moves are re-checked every run.</p>
    </UCard>
  );
}

function ChipsCard({ d }: { d: Ucl }) {
  const c = d.chips ?? {};
  return (
    <UCard title="Chips">
      <p>{c.summary}</p>
      <ul className="ulist">
        {c.wildcard && (
          <li>
            <b>Wildcard: {c.wildcard.when}.</b> {c.wildcard.why}
          </li>
        )}
        {c.limitless && (
          <li>
            <b>Limitless: {c.limitless.when}.</b> {c.limitless.why}
          </li>
        )}
      </ul>
      {c.values_now && (
        <p className="umuted small">
          Worth now (extra points over MD{d.md}–8):{" "}
          {Object.entries(c.values_now)
            .map(([k, v]) => `${k.replace("_md", " MD")} ${Number(v).toFixed(1)}`)
            .join(" · ")}
        </p>
      )}
    </UCard>
  );
}

// ------------------------------------------------------------------ Between days

export function UclBetweenPage({ d }: { d: Ucl }) {
  const b = d.between_days ?? {};
  const sched: P[] = (d.schedule ?? []).filter((s: P) => s.kind === "between" && new Date(s.at).getTime() > Date.now() - 3600e3);
  return (
    <div className="upage">
      <Strip d={d} />
      <UCard title="Between days" aside={`MD${b.md ?? d.md}`}>
        <p className="ubig">{b.note}</p>
        {b.status === "advice" && (
          <>
            {b.recommend ? (
              <ul className="ulist">
                {b.subs.map((s: P) => (
                  <li key={s.out} className="ushirt-line">
                    <UclShirt code={s.out_code} goalkeeper={s.out_gk} size={26} />
                    <b>
                      {s.out_name} → {s.in_name}
                    </b>{" "}
                    <UclShirt code={s.in_code} goalkeeper={s.in_gk} size={26} />{" "}
                    <span className="umuted">
                      ({s.out_played ? `${s.out_name} scored ${s.out_value}` : `${s.out_value} xP`}, {s.in_name} {s.in_value} xP: +{s.gain})
                    </span>
                  </li>
                ))}
                {b.captain_switch && (
                  <li className="ushirt-line">
                    <UclShirt code={b.captain_switch.from_code} size={26} />
                    <b>
                      Captain: {b.captain_switch.from_name} → {b.captain_switch.to_name}
                    </b>{" "}
                    <span className="umuted">
                      ({b.captain_switch.from_name} scored {b.captain_switch.from_points}; {b.captain_switch.to_name} {b.captain_switch.to_xp} xP)
                    </span>
                  </li>
                )}
              </ul>
            ) : null}
            <table className="utable">
              <tbody>
                <tr>
                  <td>Keep the team (automatic substitutions on)</td>
                  <td>{b.expected_keep}</td>
                </tr>
                <tr>
                  <td>Make the changes (automatic substitutions off)</td>
                  <td>{b.expected_change}</td>
                </tr>
                <tr>
                  <td>Automatic substitutions are worth</td>
                  <td>{b.autosub_value}</td>
                </tr>
              </tbody>
            </table>
          </>
        )}
        <p className="umuted small">
          Up to 4 substitutions and a captain switch between the days of a matchday; any change turns off the automatic substitutions. None on
          MD8 or the final. The advice is for the recommended squad (your own team needs a UEFA login).
        </p>
      </UCard>
      {sched.length > 0 && (
        <UCard title="When XPert updates">
          <ul className="ulist">
            {sched.slice(0, 6).map((s) => (
              <li key={s.id}>
                MD{s.md}: {s.id.includes("morning") ? "the next morning" : "after the match day"} ·{" "}
                {new Date(s.at).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
              </li>
            ))}
          </ul>
        </UCard>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Last matchday

export function UclLastPage({ d }: { d: Ucl }) {
  const l = d.last_md;
  if (!l) return <div className="upage"><div className="unote">No matchday played yet.</div></div>;
  const ps: P[] = l.players ?? [];
  return (
    <div className="upage">
      <div className="ustats">
        <div>
          <span>{l.kind === "replay" ? "XPert's replay team" : "Recommended team"}</span>
          <b>{l.points}</b>
        </div>
        <div>
          <span>Game average (estimate)</span>
          <b>{l.average_est ?? "–"}</b>
        </div>
        <div>
          <span>Best possible</span>
          <b>{l.best ?? "–"}</b>
        </div>
      </div>
      <div className="ugrid">
        <UCard title="Last matchday" aside={`MD${l.md} · expected ${l.xp ?? "–"}`} className="ucard-pitch">
          <UPitch players={ps.filter((p) => p.role === "start")} legend="Real points (captain doubled in the total)" points />
          <div className="ubench">
            {ps
              .filter((p) => p.role === "bench")
              .map((p) => (
                <Tile key={p.id} p={p} points={p.points} label={p.counted ? "came on" : "bench"} />
              ))}
          </div>
          {l.kind === "replay" && (
            <p className="umuted small">
              XPert started live recommendations at MD5. This is the team it would have picked fresh for MD{l.md} using only the matches before it.
            </p>
          )}
        </UCard>
        <UCard title="Best possible">
          <p>
            Knowing the results, the best €{d.budget}m team scored <b>{l.best}</b>.
          </p>
          <ul className="ulist">
            {(l.best_team ?? []).map((p: P) => (
              <li key={p.id} className="ushirt-line">
                <UclShirt code={p.club_code} goalkeeper={p.pos === "GK"} size={26} /> {p.name}
                {p.captain ? " (C)" : ""} <span className="umuted">{p.club}</span>
              </li>
            ))}
          </ul>
          <p className="umuted small">{d.notes?.hindsight}</p>
          <p className="umuted small">{d.notes?.average}</p>
        </UCard>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------ Scout

const LISTS: [string, string, (p: P) => string][] = [
  ["top_xp", "Top expected points", (p) => `${p.xp.toFixed(1)} xP`],
  ["captains", "Best captains", (p) => `${(2 * p.xp).toFixed(1)}`],
  ["value", "Best value", (p) => `${p.value.toFixed(2)} xP/€m`],
  ["form", "In form", (p) => `${p.value} pts`],
  ["differentials", "Differentials", (p) => `${p.xp.toFixed(1)} xP · ${p.sel}%`],
  ["most_selected", "Most selected", (p) => `${p.sel}%`],
  ["news", "Injury and team news", (p) => (p.status === "I" ? "injured" : p.status === "D" ? "doubtful" : "suspended")],
];

export function UclScoutPage({ d }: { d: Ucl }) {
  return (
    <div className="upage">
      <Strip d={d} />
      <div className="uscout">
        {LISTS.map(([key, title, val]) => (
          <UCard key={key} title={title}>
            <ol className="utop">
              {(d.scout?.[key] ?? []).map((p: P) => (
                <li key={p.id}>
                  <UclShirt code={p.club_code} goalkeeper={p.pos === "GK"} size={26} />
                  <span className="utop-name">
                    {p.name} <span className="umuted">{p.pos} · €{p.price}m{p.day ? ` · ${dayShort(p.day)}` : ""}</span>
                  </span>
                  <b>{val(p)}</b>
                </li>
              ))}
            </ol>
          </UCard>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------------- Analytics

export function UclAnalyticsPage({ d }: { d: Ucl }) {
  const rows: P[] = d.analytics?.per_md ?? [];
  const bt = d.analytics?.backtest ?? {};
  const s = bt.summary?.all;
  const axis = { ticks: { color: "#b5b4d6" }, grid: { color: "rgba(255,255,255,0.08)" } };
  const config: ChartConfiguration = {
    type: "bar",
    data: {
      labels: rows.map((r) => `MD${r.md}${r.kind === "replay" ? "*" : ""}`),
      datasets: [
        { type: "bar", label: "XPert team", data: rows.map((r) => r.points), backgroundColor: CYAN },
        { type: "line", label: "Game average (estimate)", data: rows.map((r) => r.average_est), borderColor: WHITE, backgroundColor: WHITE },
        { type: "line", label: "Best possible", data: rows.map((r) => r.best), borderColor: ROYAL, backgroundColor: ROYAL, borderDash: [5, 4] },
      ] as any,
    },
    options: { scales: { x: axis, y: { ...axis, beginAtZero: true } }, plugins: { legend: { labels: { color: "#ffffff" } } } } as any,
  };
  const ko: P[] = d.knockout ?? [];
  return (
    <div className="upage">
      <UCard title="Points vs the average" aside="* replay: picked from the data before that matchday">
        <Chart config={config} height={240} />
        <p className="umuted small">{d.notes?.average}</p>
      </UCard>
      <div className="ugrid2">
        <UCard title="Backtest">
          <ul className="ulist">
            {(bt.plain ?? []).map((line: string) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          {s && (
            <table className="utable">
              <thead>
                <tr>
                  <th />
                  <th>XPert</th>
                  <th>Points per game</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Ranking of starters</td>
                  <td>{s.rank_starters_model != null ? Number(s.rank_starters_model).toFixed(2) : "–"}</td>
                  <td>{s.rank_starters_ppg != null ? Number(s.rank_starters_ppg).toFixed(2) : "–"}</td>
                </tr>
                <tr>
                  <td>Top-10 hits (of 10)</td>
                  <td>{s.top10_hits_model != null ? Number(s.top10_hits_model).toFixed(2) : "–"}</td>
                  <td>{s.top10_hits_ppg != null ? Number(s.top10_hits_ppg).toFixed(2) : "–"}</td>
                </tr>
                <tr>
                  <td>Top-10 average points</td>
                  <td>{s.top10_points_model != null ? Number(s.top10_points_model).toFixed(2) : "–"}</td>
                  <td>{s.top10_points_ppg != null ? Number(s.top10_points_ppg).toFixed(2) : "–"}</td>
                </tr>
                <tr>
                  <td>Captain points</td>
                  <td>{s.captain_points_model != null ? Number(s.captain_points_model).toFixed(2) : "–"}</td>
                  <td>{s.captain_points_ppg != null ? Number(s.captain_points_ppg).toFixed(2) : "–"}</td>
                </tr>
              </tbody>
            </table>
          )}
          <p className="umuted small">{bt.method}</p>
        </UCard>
        <UCard title="Knockout chances" aside="top 8 / 9th–24th / out">
          <table className="utable ko">
            <thead>
              <tr>
                <th>#</th>
                <th>Club</th>
                <th>Pts</th>
                <th>Top 8</th>
                <th>9–24</th>
                <th>Out</th>
                <th title="Expected UCL matches left">Left</th>
              </tr>
            </thead>
            <tbody>
              {ko.map((r) => (
                <tr key={r.code}>
                  <td>{r.rank}</td>
                  <td className="ko-club">
                    <UclShirt code={r.code} size={20} label={false} /> {r.team}
                  </td>
                  <td>{r.pts}</td>
                  <td>{Math.round(r.p_top8 * 100)}%</td>
                  <td>{Math.round(r.p_playoff * 100)}%</td>
                  <td>{Math.round(r.p_out * 100)}%</td>
                  <td>{r.exp_matches.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </UCard>
      </div>
    </div>
  );
}
