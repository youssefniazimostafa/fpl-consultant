import { useMemo, useState, type ReactNode } from "react";
import type { ChartConfiguration } from "chart.js";
import type { Latest } from "../data";
import { Card } from "../components/Card";
import { Chart } from "../components/Chart";
import { colors } from "../theme/theme";

// Analytics: the one page for technical detail. A segmented control
// (Descriptive / Predictive / Prescriptive, as in the mock-ups), then
// chart cards -- each with a one-line plain explanation. Data comes from
// latest.json: the `analytics` section (fpl/analytics.py) plus existing
// sections (ownership_heatmap, consultant_vs_you, live_accuracy,
// season_simulation, captain_board, decision).

type Raw = Record<string, any>;
type Tab = "descriptive" | "predictive" | "prescriptive";

const pct = (p: number | null | undefined) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const fmt = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

function ChartCard({ title, explain, empty, children }: { title: string; explain: string; empty?: string | false; children?: ReactNode }) {
  return (
    <Card title={title} className="an-card">
      <p className="an-explain">{explain}</p>
      {empty ? <p className="an-empty">{empty}</p> : children}
    </Card>
  );
}

const lineOpts = (yTitle: string, reverse = false): ChartConfiguration["options"] => ({
  scales: {
    y: { reverse, title: { display: true, text: yTitle }, ticks: { precision: 0 } },
    x: { grid: { display: false } },
  },
});

function Descriptive({ d }: { d: Latest }) {
  const a = d.analytics ?? {};
  const s: Raw[] = a.league_series ?? [];
  const points = useMemo<ChartConfiguration>(() => ({
    type: "line",
    data: {
      labels: s.map((r) => `GW${r.gameweek}`),
      datasets: [
        { label: "You", data: s.map((r) => r.you), borderColor: colors.purple, backgroundColor: colors.purple, borderWidth: 3, tension: 0.25 },
        { label: "League average", data: s.map((r) => r.league_average), borderColor: colors.cyan, backgroundColor: colors.cyan, borderWidth: 2, tension: 0.25 },
        { label: `Leader (${a.leader_name ?? "1st"})`, data: s.map((r) => r.leader), borderColor: colors.pink, backgroundColor: colors.pink, borderWidth: 2, borderDash: [5, 4], tension: 0.25 },
      ],
    },
    options: lineOpts("Points that week"),
  }), [s, a.leader_name]);
  const position = useMemo<ChartConfiguration>(() => ({
    type: "line",
    data: {
      labels: s.map((r) => `GW${r.gameweek}`),
      datasets: [{ label: "Your league position", data: s.map((r) => r.position), borderColor: colors.purple, backgroundColor: colors.neon, pointRadius: 5, borderWidth: 3, stepped: false }],
    },
    options: { ...lineOpts("Position (1 = top)", true), plugins: { legend: { display: false } } },
  }), [s]);
  const cvy: Raw = d.consultant_vs_you ?? {};
  const heat: Raw = d.ownership_heatmap ?? {};
  const me = d.config?.team_id;

  return (
    <div className="an-grid">
      <ChartCard title="Points per gameweek" explain="Your points each week against the league average and the current leader, to see where you gained or lost ground." empty={!s.length && "No league history yet."}>
        <Chart config={points} />
      </ChartCard>
      <ChartCard title="Your league position" explain="Where you stood in your league after each gameweek (top of the chart is 1st)." empty={!s.length && "No league history yet."}>
        <Chart config={position} />
      </ChartCard>
      <ChartCard
        title="You vs the consultant"
        explain="Your real points each week against the consultant's frozen pre-deadline team, scored with FPL's rules."
        empty={!cvy.started && `Tracking starts with Gameweek ${cvy.first_gameweek ?? 11}. First result after it finishes.`}
      >
        <table className="an-table">
          <thead>
            <tr><th>GW</th><th>Consultant</th><th>You</th><th>Difference</th><th>Same transfers</th><th>Same captain</th><th>Running total</th></tr>
          </thead>
          <tbody>
            {(cvy.history ?? []).map((r: Raw) => (
              <tr key={r.gameweek}>
                <td>{r.gameweek}</td><td>{r.consultant_points}</td><td>{r.my_points}</td><td>{r.difference > 0 ? "+" : ""}{r.difference}</td>
                <td>{r.followed_transfers}</td><td>{r.followed_captain}</td><td>{r.total_difference > 0 ? "+" : ""}{r.total_difference}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ChartCard>
      <ChartCard
        title="Who owns what in your league"
        explain="The most-owned players in your league and how each manager used them last gameweek: starting, captain, benched, or not owned."
        empty={!heat.available && "No league ownership data yet."}
      >
        <div className="heat-scroll">
          <table className="heat">
            <thead>
              <tr>
                <th />
                {(heat.managers ?? []).map((m: Raw) => (
                  <th key={m.manager_id} className={m.manager_id === me ? "me" : ""} title={m.entry_name}>
                    {m.manager_id === me ? "You" : m.entry_name.slice(0, 9)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(heat.players ?? []).map((p: Raw) => (
                <tr key={p.player_id}>
                  <th>{p.web_name}</th>
                  {(heat.managers ?? []).map((m: Raw) => {
                    const v = p[`m_${m.manager_id}`];
                    const owns = (p.owned_by ?? []).includes(m.manager_id);
                    const cls = !owns ? "hn" : `h${v}`;
                    return <td key={m.manager_id} className={cls}>{v === 2 ? "C" : v === 3 ? "TC" : v === 1 ? "✓" : owns ? "B" : ""}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="heat-legend"><span className="h1">✓</span> starting <span className="h2">C</span> captain <span className="h0">B</span> owns him, benched <span className="hn" /> doesn't own him</div>
      </ChartCard>
    </div>
  );
}

function Predictive({ d }: { d: Latest }) {
  const a = d.analytics ?? {};
  const proj: Raw[] = a.team_projection ?? [];
  const cum: Raw[] = d.live_accuracy?.cumulative ?? [];
  const projChart = useMemo<ChartConfiguration>(() => ({
    type: "bar",
    data: {
      labels: proj.map((r) => `GW${r.gameweek}`),
      datasets: [
        { label: "Projected before the deadline", data: proj.map((r) => r.projected), backgroundColor: colors.lilacDeep, borderColor: colors.purple, borderWidth: 1, borderRadius: 6 },
        { label: "Actually scored", data: proj.map((r) => r.actual), backgroundColor: colors.neon, borderRadius: 6 },
      ],
    },
    options: lineOpts("Your team's points"),
  }), [proj]);
  const errChart = useMemo<ChartConfiguration>(() => ({
    type: "line",
    data: {
      labels: cum.map((r) => `GW${r.gameweek}`),
      datasets: [
        { label: "Average miss per player (points)", data: cum.map((r) => r.mae), borderColor: colors.purple, backgroundColor: colors.purple, borderWidth: 3 },
        { label: "Average over/under (points)", data: cum.map((r) => r.bias), borderColor: colors.pink, backgroundColor: colors.pink, borderWidth: 2, borderDash: [5, 4] },
      ],
    },
    options: lineOpts("Points"),
  }), [cum]);
  const weeks: number[] = a.fixture_weeks ?? [];

  return (
    <div className="an-grid">
      <ChartCard title="Projected vs actual" explain="Your team's frozen pre-deadline projection against what it really scored, for every week a projection was saved." empty={!proj.length && "No saved projections for played gameweeks yet."}>
        <Chart config={projChart} />
      </ChartCard>
      <ChartCard title="How far off the projections were" explain="Across every player each week: the average size of the miss, and whether projections ran high (above 0) or low." empty={!cum.length && "No scored gameweeks yet."}>
        <Chart config={errChart} />
      </ChartCard>
      <ChartCard title="Biggest misses by gameweek" explain="The players the projections got most wrong, so you can see what the numbers can't predict (hauls, surprise starts)." empty={!(a.misses ?? []).length && "No scored gameweeks yet."}>
        <div className="miss-weeks">
          {(a.misses ?? []).slice(-2).reverse().map((m: Raw) => (
            <div key={m.gameweek} className="miss-week">
              <div className="miss-gw">GW{m.gameweek}</div>
              {m.players.map((p: Raw) => (
                <div key={p.web_name} className="miss-row">
                  <span className="miss-name"><b>{p.web_name}</b> <small>{p.team_short}</small></span>
                  <span className="miss-bar">
                    <span className="miss-exp" style={{ width: `${Math.min(100, p.expected * 6)}%` }} />
                    <span className={`miss-act ${p.difference >= 0 ? "up" : "down"}`} style={{ width: `${Math.min(100, p.actual * 6)}%` }} />
                  </span>
                  <span className="miss-num">{p.expected.toFixed(1)} → <b>{p.actual}</b></span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </ChartCard>
      <ChartCard title="Fixture difficulty, next 6 gameweeks" explain="Every club's upcoming games by FPL's own difficulty rating, easiest run at the top." empty={!(a.fixture_difficulty ?? []).length && "No fixtures yet."}>
        <div className="heat-scroll">
          <table className="heat fdr">
            <thead>
              <tr><th />{weeks.map((g) => <th key={g}>GW{g}</th>)}</tr>
            </thead>
            <tbody>
              {(a.fixture_difficulty ?? []).map((r: Raw) => (
                <tr key={r.team_short}>
                  <th>{r.team_short}</th>
                  {weeks.map((g) => {
                    const games: Raw[] = r.cells[String(g)] ?? [];
                    if (!games.length) return <td key={g} className="d0">–</td>;
                    return (
                      <td key={g} className={`d${games[0].difficulty}${games.length > 1 ? " dbl" : ""}`}>
                        {games.map((x) => `${x.opponent}${x.home ? "" : "*"}`).join(" ")}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="an-foot">* away game. Green easy → pink hard.</p>
      </ChartCard>
      <ChartCard
        title="Our projections vs FPL's own"
        explain="A live comparison of our projected points with FPL's own projection, gameweek by gameweek."
        empty={!(a.experiment ?? []).length && "First result after GW11."}
      >
        <table className="an-table">
          <thead><tr><th>GW</th>{Object.keys(a.experiment?.[0] ?? {}).filter((k) => k !== "gameweek").map((k) => <th key={k}>{k}</th>)}</tr></thead>
          <tbody>
            {(a.experiment ?? []).map((r: Raw) => (
              <tr key={r.gameweek}><td>{r.gameweek}</td>{Object.entries(r).filter(([k]) => k !== "gameweek").map(([k, v]) => <td key={k}>{fmt(v as number)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </ChartCard>
    </div>
  );
}

function Prescriptive({ d }: { d: Latest }) {
  const pc: Raw = d.decision?.plan_choice ?? {};
  const season: Raw[] = d.season_simulation?.standings ?? [];
  const me = d.config?.team_id;
  const board: Raw[] = d.captain_board?.players ?? [];
  const winChart = useMemo<ChartConfiguration>(() => ({
    type: "bar",
    data: {
      labels: ["Make the transfer", "Hold (save it)"],
      datasets: [{
        label: "Chance of winning this gameweek",
        data: [pc.win_chance_transfer, pc.win_chance_hold].map((v) => (v == null ? null : Math.round(v * 1000) / 10)),
        backgroundColor: [d.decision?.action === "transfer" ? colors.neon : colors.lilacDeep, d.decision?.action === "roll" ? colors.neon : colors.lilacDeep],
        borderRadius: 8,
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, title: { display: true, text: "% chance of top score in your league" } }, y: { grid: { display: false }, ticks: { autoSkip: false } } } },
  }), [pc, d.decision?.action]);
  const seasonChart = useMemo<ChartConfiguration>(() => ({
    type: "bar",
    data: {
      labels: season.map((s) => (s.manager_id === me ? "You" : s.entry_name)),
      datasets: [{
        label: `Chance of being top by GW${d.season_simulation?.target_gameweek ?? 19}`,
        data: season.map((s) => Math.round(s.p_finish_1st_by_gw19 * 1000) / 10),
        backgroundColor: season.map((s) => (s.manager_id === me ? colors.neon : colors.purple)),
        borderRadius: 6,
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, title: { display: true, text: "%" } }, y: { grid: { display: false }, ticks: { autoSkip: false } } } },
  }), [season, me, d.season_simulation?.target_gameweek]);
  const capChart = useMemo<ChartConfiguration>(() => ({
    type: "bar",
    data: {
      labels: board.map((p) => p.web_name),
      datasets: [{
        label: "Projected points next gameweek",
        data: board.map((p) => p.xp),
        backgroundColor: board.map((p) => (p.is_optimiser_captain ? colors.neon : colors.purple)),
        borderRadius: 6,
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, title: { display: true, text: "Projected points (before doubling)" } }, y: { grid: { display: false }, ticks: { autoSkip: false } } } },
  }), [board]);
  const back: Raw[] = d.analytics?.backtest ?? [];

  return (
    <div className="an-grid">
      <ChartCard title="Transfer or hold: chances this week" explain="How often each option tops your league this gameweek in thousands of simulated weeks; the green bar is the recommendation." empty={pc.win_chance_transfer == null && pc.win_chance_hold == null && "No simulation in this update."}>
        <Chart config={winChart} height={150} />
        <p className="an-foot">Recommendation follows {pc.rule === "win_chances" ? "win chances (you're chasing)" : "projected points over the next weeks"}.</p>
      </ChartCard>
      <ChartCard title="Chances of being top by GW19" explain="Each manager's chance of leading your league at the halfway point, from their form so far." empty={!season.length && "No season simulation yet."}>
        <Chart config={seasonChart} height={Math.max(220, season.length * 26)} />
      </ChartCard>
      <ChartCard title="Captain board" explain="Your starters' projected points next gameweek; green is the recommended captain (fewer rivals own him can tip a close call)." empty={!board.length && "No captain board yet."}>
        <Chart config={capChart} height={Math.max(220, board.length * 24)} />
      </ChartCard>
      <ChartCard title="Backtest: how the projections did on past seasons" explain="Average miss per player and how well each method ranked players, replayed over past gameweeks. Lower miss and higher ranking scores are better." empty={!back.length && "No backtest results yet."}>
        {back.map((b) => (
          <div key={b.gameweeks} className="bt-block">
            <div className="bt-caption">
              {b.gameweeks} gameweeks · {b.predictions?.toLocaleString()} player predictions
              {b.scope === "fpl_xp_valid_gws" ? " · only weeks where FPL's own projection exists" : " · all weeks"}
            </div>
            <table className="an-table">
              <thead><tr><th>Method</th><th>Average miss</th><th>Ranking (all)</th><th>Ranking (starters)</th><th>Top-10 hit</th><th>Top-20 hit</th><th>Captain pts</th></tr></thead>
              <tbody>
                {b.rows.map((r: Raw) => (
                  <tr key={r.model}><td>{r.model}</td><td>{fmt(r.mae)}</td><td>{fmt(r.rank_all)}</td><td>{fmt(r.rank_starters)}</td><td>{pct(r.top10)}</td><td>{pct(r.top20)}</td><td>{fmt(r.captain, 1)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
        <p className="an-foot">Caveat: FPL's own historical projections may be optimistic, so read their numbers with caution. They only exist for 10 of the 37 gameweeks, which is why that row is blank in the 37-gameweek table.</p>
      </ChartCard>
    </div>
  );
}

export function AnalyticsPage({ data }: { data: Latest }) {
  const [tab, setTab] = useState<Tab>("descriptive");
  return (
    <div className="analytics-page">
      <div className="page-title">
        <h1>Analytics</h1>
        <span className="muted">The detail behind the advice</span>
      </div>
      <div className="seg" role="tablist">
        {([["descriptive", "Descriptive", "what happened"], ["predictive", "Predictive", "what's expected"], ["prescriptive", "Prescriptive", "what to do"]] as [Tab, string, string][]).map(([id, label, sub]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            <b>{label}</b>
            <small>{sub}</small>
          </button>
        ))}
      </div>
      {tab === "descriptive" && <Descriptive d={data} />}
      {tab === "predictive" && <Predictive d={data} />}
      {tab === "prescriptive" && <Prescriptive d={data} />}
    </div>
  );
}

