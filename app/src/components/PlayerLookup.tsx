import { useMemo, useState } from "react";
import type { ChartConfiguration } from "chart.js";
import { HelpButton } from "./Card";
import { Chart } from "./Chart";
import { Shirt } from "./Shirt";
import { helpFor } from "../help";
import { chanceClass, colors } from "../theme/theme";

// Scout page: Player lookup. Every FPL player in a sortable list (left) and
// one player's full report (right; below on phone). Data: latest.json's
// overview.lookup (fpl/overview.py build_lookup) -- compact rows plus one
// report per player. Nothing here changes the plan.

type Raw = Record<string, any>;
interface Row {
  id: number;
  name: string;
  club: string;
  team_code: number | null;
  pos: string;
  price: number;
  xp: number | null;
  form: number | null;
  sel: number | null;
  own: boolean;
}
type SortKey = "name" | "price" | "xp" | "form" | "sel";

const fold = (s: string) =>
  s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/ø/g, "o")
    .replace(/ß/g, "ss")
    .toLowerCase();
const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const chanceWords = (x: number | null | undefined) =>
  x == null ? "–" : x >= 0.85 ? "very likely" : x >= 0.6 ? "likely" : x >= 0.35 ? "about even" : x >= 0.1 ? "unlikely" : "very unlikely";
const PRICE_CAPS: [string, number][] = [
  ["Any price", Infinity],
  ["Up to £4.5m", 4.5],
  ["Up to £5m", 5],
  ["Up to £5.5m", 5.5],
  ["Up to £6m", 6],
  ["Up to £7m", 7],
  ["Up to £8m", 8],
  ["Up to £10m", 10],
];

export function PlayerLookup({ lookup, title = "Player lookup" }: { lookup: Raw; title?: string }) {
  const rows: Row[] = useMemo(
    () =>
      (lookup.rows ?? []).map((r: any[]) => Object.fromEntries((lookup.columns as string[]).map((c, i) => [c, r[i]])) as unknown as Row),
    [lookup],
  );
  const clubs = useMemo(() => [...new Set(rows.map((r) => r.club))].sort(), [rows]);
  const [q, setQ] = useState("");
  const [pos, setPos] = useState("ALL");
  const [club, setClub] = useState("ALL");
  const [cap, setCap] = useState(0);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "xp", desc: true });
  const [picked, setPicked] = useState<number | null>(null);

  const shown = useMemo(() => {
    const needle = fold(q.trim());
    const max = PRICE_CAPS[cap][1];
    const out = rows.filter(
      (r) =>
        (pos === "ALL" || r.pos === pos) &&
        (club === "ALL" || r.club === club) &&
        r.price <= max &&
        (!needle || fold(r.name).includes(needle) || fold(lookup.reports?.[r.id]?.full ?? "").includes(needle)),
    );
    const k = sort.key;
    out.sort((a, b) => {
      const va = a[k] ?? -1;
      const vb = b[k] ?? -1;
      const c = typeof va === "string" ? (va as string).localeCompare(vb as string) : (va as number) - (vb as number);
      return sort.desc ? -c : c;
    });
    return out;
  }, [rows, q, pos, club, cap, sort, lookup]);

  const head = (key: SortKey, label: string, cls = "num") => (
    <button
      className={`lk-sort ${cls}${sort.key === key ? " on" : ""}`}
      onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "name" }))}
      aria-label={`Sort by ${label}`}
    >
      {label}
      {sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
    </button>
  );
  const sel = picked != null ? rows.find((r) => r.id === picked) : undefined;
  const help = helpFor(title);

  return (
    <section className="sc-card sc-lookup">
      <div className="sc-head">
        <h2>
          {title}
          {help && <HelpButton text={help} />}
        </h2>
        <div className="sc-controls lk-controls">
          <input className="lk-search" type="search" placeholder="Search a player" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search a player" />
          <select value={pos} onChange={(e) => setPos(e.target.value)} aria-label="Position">
            <option value="ALL">All positions</option>
            <option value="GKP">Goalkeepers</option>
            <option value="DEF">Defenders</option>
            <option value="MID">Midfielders</option>
            <option value="FWD">Forwards</option>
          </select>
          <select value={club} onChange={(e) => setClub(e.target.value)} aria-label="Club">
            <option value="ALL">All clubs</option>
            {clubs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select value={cap} onChange={(e) => setCap(Number(e.target.value))} aria-label="Price">
            {PRICE_CAPS.map(([l], i) => (
              <option key={l} value={i}>
                {l}
              </option>
            ))}
          </select>
        </div>
      </div>
      <p className="sc-explain">
        Every FPL player, by expected points for GW{lookup.next_gameweek}. Tap a column to sort, tap a player for his full report.
      </p>
      <div className="lk-body">
        <div className="lk-list">
          <div className="lk-row lk-headrow">
            {head("name", "Player", "lk-player")}
            {head("price", "Price")}
            {head("xp", "xP")}
            {head("form", "Form")}
            {head("sel", "Own")}
          </div>
          <div className="lk-scroll">
            {shown.map((r) => (
              <button key={r.id} className={`lk-row${r.own ? " you" : ""}${r.id === picked ? " selected" : ""}`} onClick={() => setPicked(r.id)}>
                <span className="lk-player">
                  <Shirt teamCode={r.team_code} teamShort={r.club} goalkeeper={r.pos === "GKP"} size={24} title={r.name} />
                  <span className="st-name">
                    <b>{r.name}</b>
                    <small>
                      {r.club} · {r.pos}
                      {r.own && <span className="sc-own lk-own">You own</span>}
                    </small>
                  </span>
                </span>
                <span className="num">£{r.price.toFixed(1)}</span>
                <span className="num total">{r.xp?.toFixed(1) ?? "–"}</span>
                <span className="num">{r.form?.toFixed(1) ?? "–"}</span>
                <span className="num">{r.sel != null ? `${r.sel.toFixed(1)}%` : "–"}</span>
              </button>
            ))}
            {!shown.length && <p className="muted lk-empty">No player matches.</p>}
          </div>
        </div>
        <div className="lk-report">
          {sel ? <Report row={sel} rep={lookup.reports?.[sel.id] ?? {}} lookup={lookup} /> : <p className="muted lk-empty">Pick a player to see his full report.</p>}
        </div>
      </div>
    </section>
  );
}

function Report({ row, rep, lookup }: { row: Row; rep: Raw; lookup: Raw }) {
  const d = rep.drv ?? {};
  const recent: number[][] = rep.recent ?? [];
  const chart = useMemo<ChartConfiguration>(
    () => ({
      type: "bar",
      data: {
        labels: recent.map((r) => `GW${r[0]}`),
        datasets: [{ label: "Points", data: recent.map((r) => r[1]), backgroundColor: colors.purple, borderRadius: 4 }],
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { afterLabel: (c: any) => { const r = recent[c.dataIndex]; return `${r[2]} min · ${r[3]} G · ${r[4]} A`; } } },
        },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      } as any,
    }),
    [recent],
  );
  const s = rep.season ?? {};
  const swap = rep.swap;
  return (
    <div className="lk-rep">
      <div className="lk-rep-head">
        <Shirt teamCode={row.team_code} teamShort={row.club} goalkeeper={row.pos === "GKP"} size={46} title={row.name} />
        <div>
          <h3>{rep.full || row.name}</h3>
          <p className="muted small">
            {row.club} · {row.pos} · £{row.price.toFixed(1)}m · {row.sel?.toFixed(1)}% of all managers · {rep.rivals} of your {lookup.rivals} rivals
          </p>
          {row.own ? <span className="sc-own">You own</span> : <span className="lk-notown">Not in your team</span>}
        </div>
      </div>

      <h4>Availability</h4>
      <p className="lk-avail">
        <span className={`cop-badge ${chanceClass(rep.chance ?? 100)}`}>{rep.chance == null ? "Fit" : `${rep.chance}%`}</span>{" "}
        <span className={rep.chance === 0 ? "news-out" : ""}>{rep.news || "No injury or team news."}</span>
      </p>

      <h4>Next {(rep.fx ?? []).length} gameweeks</h4>
      <table className="lk-fx">
        <tbody>
          <tr>
            {(rep.fx ?? []).map((f: any[], i: number) => (
              <td key={i} className={`d${f[3] ?? 0}`} title={`Difficulty ${f[3] ?? "?"}`}>
                <small>GW{f[0]}</small>
                <b>
                  {f[1]} ({f[2]})
                </b>
                <span>{f[4]?.toFixed(1)} xP</span>
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <p className="small">
        <b>{rep.xp5?.toFixed(1)} expected points</b> over these {(rep.fx ?? []).length} gameweeks. Colours are FPL's own difficulty: green easy, pink hard.
      </p>

      <h4>What drives his expected points (GW{lookup.next_gameweek}{d.opp ? ` v ${d.opp}` : ""})</h4>
      {rep.drv ? (
        <dl className="ra-facts lk-facts">
          <dt>Starting</dt>
          <dd>
            {pct(d.start)} chance ({chanceWords(d.start)}), and {pct(d.p60)} to play 60+ minutes (about {d.mins ?? "–"} minutes expected)
          </dd>
          <dt>Goal</dt>
          <dd>{pct(d.goal)} chance to score</dd>
          <dt>Assist</dt>
          <dd>{pct(d.assist)} chance to set one up</dd>
          {row.pos !== "FWD" && (
            <>
              <dt>Clean sheet</dt>
              <dd>{pct(d.cs)} chance his team keeps one</dd>
            </>
          )}
          <dt>Defending</dt>
          <dd>{d.defcon != null ? `${d.defcon.toFixed(1)} expected points from defensive contributions` : "–"}</dd>
          <dt>Bonus</dt>
          <dd>{d.bonus != null ? `${d.bonus.toFixed(1)} expected bonus points` : "–"}</dd>
        </dl>
      ) : (
        <p className="muted small">No fixture next gameweek.</p>
      )}

      <h4>Recent form</h4>
      {recent.length ? (
        <>
          <Chart config={chart} height={130} />
          <table className="lk-recent">
            <thead>
              <tr>
                <th />
                {recent.map((r) => (
                  <th key={r[0]}>GW{r[0]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(["Points", "Minutes", "Goals", "Assists"] as const).map((label, k) => (
                <tr key={label}>
                  <td>{label}</td>
                  {recent.map((r) => (
                    <td key={r[0]}>{r[k + 1]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="muted small">No minutes in the last six gameweeks.</p>
      )}
      <p className="small">
        Season: <b>{s.pts} points</b> in {s.mins} minutes ({s.starts} starts) · {s.g} goals · {s.a} assists · {s.cs} clean sheets · {s.bonus} bonus ·{" "}
        {s.ppg} points per game
      </p>

      <h4>Value</h4>
      <p className="small">
        <b>{rep.ppm} points per £1m</b> this season · price {rep.change > 0 ? `up £${rep.change.toFixed(1)}m` : rep.change < 0 ? `down £${Math.abs(rep.change).toFixed(1)}m` : "unchanged"} since the start
      </p>

      {!row.own && (
        <>
          <h4>Best swap from your team</h4>
          {swap ? (
            <>
              <p className="small lk-swap">
                <b>
                  {swap.out_name} → {row.name}: {swap.week_gain >= 0 ? "+" : ""}
                  {swap.week_gain.toFixed(1)} xP this week,{" "}
                  {swap.clears_bar ? <span className="lk-ok">clears the bar</span> : <span className="lk-no">below the bar</span>}
                </b>
              </p>
              <p className="small muted">
                Judged like the optimiser: making it now rather than next week is worth {swap.gain >= 0 ? "+" : ""}
                {swap.gain.toFixed(1)} against the {lookup.transfer_bar} bar ({swap.horizon_gain >= 0 ? "+" : ""}
                {swap.horizon_gain.toFixed(1)} over {(rep.fx ?? []).length} gameweeks, best XI re-picked each week)
                {swap.hit ? `, after a ${swap.hit}-point hit` : ""}. Sells {swap.out_name} for £{swap.out_sell.toFixed(1)}m, leaving £
                {swap.bank_after.toFixed(1)}m.
              </p>
            </>
          ) : (
            <p className="muted small">No {row.pos} in your team can be swapped for him within your budget and the 3-per-club limit.</p>
          )}
        </>
      )}
    </div>
  );
}
