// UCL Planner tab: the same layout as the PL Planner (title, note, Your plan
// on the Mac, the plan, then fixtures), with the UCL rules: stages, free
// transfers incl. the unlimited windows, club limit and budget per stage, each
// planned move against the bar, the chip plan, and every club's fixtures with
// the day it plays. Data: latest.json's `planner`, `plan` and `chips`
// (ucl/pipeline.py). "Your plan" runs ucl/my_plan.py through the backend.
import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ChatMarkdown } from "../components/ChatMarkdown";
import { DESKTOP } from "../platform";
import type { Ucl } from "./data";
import { UCard } from "./pages";
import { UclShirt } from "./UclShirt";

type P = Record<string, any>;
const POS_ORDER = ["GK", "DEF", "MID", "FWD"];

export function UclPlannerPage({ d }: { d: Ucl }) {
  const pl = d.planner ?? {};
  const cal: P[] = pl.calendar ?? [];
  const moves: P[] = pl.moves ?? [];
  const plan: P[] = d.plan ?? [];
  const c = d.chips ?? {};
  const fxMds = [...new Set((pl.clubs ?? []).flatMap((r: P) => r.fixtures.map((f: P) => f.md)))].sort((a: any, b: any) => a - b) as number[];
  return (
    <div className="upage">
      <div className="upage-title">
        <h1>Planner</h1>
        <span className="umuted">
          MD{d.md}–{plan.length ? plan[plan.length - 1].md : d.md}, up to the next unlimited-transfer window
        </span>
      </div>
      <p className="umuted small">
        Only this matchday's team is final; later moves are re-checked every run. Transfers are unlimited before MD1, the play-offs (MD9) and the round of 16 (MD11), so
        the plan stops there.
      </p>
      {DESKTOP && <UclYourPlan />}

      <UCard title="Matchday plan" aside={pl.bar_note ? "each move against the bar" : undefined}>
        <div className="uplan">
          {plan.map((st, i) => {
            const mv = moves.filter((m) => m.md === st.md);
            return (
              <div key={st.md} className={`uplan-md${i === 0 ? " now" : ""}`}>
                <div className="uplan-head">
                  <b>MD{st.md}</b>
                  <span className="umuted">{i === 0 && d.team?.transfers?.new_team ? "new team, free" : `${st.ft ?? "∞"} free`}</span>
                </div>
                {i === 0 && d.team?.transfers?.new_team ? (
                  <p className="small">Create the recommended squad (Recommended team).</p>
                ) : mv.length ? (
                  mv.map((m, k) => (
                    <div key={k} className="uplan-move">
                      <span className="ushirt-line">
                        <UclShirt code={m.out_code} goalkeeper={m.pos === "GK"} size={26} />
                        <span>{m.out}</span>
                        <span className="umuted">→</span>
                        <UclShirt code={m.in_code} goalkeeper={m.pos === "GK"} size={26} />
                        <b>{m.in}</b>
                      </span>
                      <span className={`uplan-gain ${m.gain >= m.bar ? "ok" : "no"}`}>
                        +{m.gain.toFixed(1)} over {m.over} <small>vs bar {m.bar}</small>
                        {m.hit && <small> · −4 hit</small>}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="small umuted">Keep: save the free transfer{st.md < 8 ? " (1 can carry over)" : ""}.</p>
                )}
                <div className="uplan-foot umuted small">
                  {st.xp} xP · captain {st.captain}
                  {st.hits ? ` · −${4 * st.hits}` : ""}
                  {st.chip ? ` · ${st.chip}` : ""}
                </div>
              </div>
            );
          })}
        </div>
        {pl.bar_note && <p className="umuted small">{pl.bar_note}</p>}
      </UCard>

      <div className="ugrid2">
        <UCard title="Matchday calendar">
          <div className="utable-wrap">
            <table className="utable ucal">
              <thead>
                <tr>
                  <th>MD</th>
                  <th>Stage</th>
                  <th>Days</th>
                  <th>Free transfers</th>
                  <th>Club limit</th>
                  <th>Budget</th>
                  <th>Chips</th>
                </tr>
              </thead>
              <tbody>
                {cal.map((r) => (
                  <tr key={r.md} className={`${r.status}${r.md === d.md ? " on" : ""}`}>
                    <td>MD{r.md}</td>
                    <td>{r.stage}</td>
                    <td>{r.days.length ? r.days.join(", ") : <span className="umuted">TBC</span>}</td>
                    <td>
                      {r.free_transfers === "unlimited" ? <b className="uok">Unlimited</b> : r.free_transfers}
                      {r.available != null && r.available !== r.free_transfers && <span className="umuted"> ({r.available} with carry)</span>}
                      {r.carry && <div className="umuted tiny">{r.carry}</div>}
                    </td>
                    <td>{r.club_limit}</td>
                    <td>€{r.budget}m</td>
                    <td>{r.chips_allowed ? "✓" : <span className="uno">not allowed</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="umuted small">−4 per transfer over the free ones. Between-days changes are allowed on every matchday but MD8 and the final.</p>
        </UCard>

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
          <p className="small">
            Each chip once, never on {(pl.chip_banned ?? [1, 9, 11]).map((m: number) => `MD${m}`).join(", ")} (the unlimited-transfer windows).
          </p>
          {c.values_now && (
            <p className="umuted small">
              Worth now (extra points to MD8):{" "}
              {Object.entries(c.values_now)
                .map(([k, v]) => `${k.replace("_md", " MD")} ${Number(v).toFixed(1)}`)
                .join(" · ")}
            </p>
          )}
        </UCard>
      </div>

      <UCard title="Club fixtures" aside="easiest run first · difficulty from our team ratings: green easy, red hard">
        <div className="utable-wrap">
          <table className="utable ufx">
            <thead>
              <tr>
                <th>Club</th>
                {fxMds.map((m) => (
                  <th key={m}>MD{m}</th>
                ))}
                <th title="Chance of finishing in the top 8">Top 8</th>
              </tr>
            </thead>
            <tbody>
              {(pl.clubs ?? []).map((r: P) => (
                <tr key={r.code}>
                  <td className="ko-club">
                    <UclShirt code={r.code} size={22} label={false} /> {r.club}
                  </td>
                  {fxMds.map((m) => {
                    const f = r.fixtures.find((x: P) => x.md === m);
                    return f ? (
                      <td key={m} className={`ufd d${f.diff}`}>
                        <b>
                          {f.opp} ({f.home ? "H" : "A"})
                        </b>
                        <small>{f.day.split(" ")[0]}</small>
                      </td>
                    ) : (
                      <td key={m} className="ufd d0">–</td>
                    );
                  })}
                  <td>{r.p_top8 != null ? `${Math.round(r.p_top8 * 100)}%` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </UCard>
    </div>
  );
}

// ------------------------------------------------------------------ Your plan

interface Move {
  out: number | null;
  in: number | null;
}
interface Plan {
  md: number;
  transfers: Move[];
  captain: number | null;
  vice_captain: number | null;
  chip: string | null;
}
interface Saved {
  plan: Plan;
  review?: P;
  stamp?: { plan_key: string; data_stamp: string | null };
  saved_at?: string;
}
const planKey = (p: Plan) => JSON.stringify({ ...p, transfers: p.transfers.filter((t) => t.out && t.in) });
const eur = (x: number) => `€${x.toFixed(1)}m`;

function fromXpert(ctx: P): Plan {
  const r = ctx.recommendation ?? {};
  return { md: ctx.md, transfers: [], captain: r.captain ?? null, vice_captain: r.vice_captain ?? null, chip: null };
}

/** The same rules ucl/my_plan.py validate() applies, live in the card. */
function check(plan: Plan, ctx: P, byId: Map<number, P>) {
  const moves = plan.transfers.filter((t) => t.out && t.in) as { out: number; in: number }[];
  const outs = moves.map((m) => m.out);
  const ins = moves.map((m) => m.in);
  const squad = new Set<number>((ctx.squad as number[]).filter((p) => !outs.includes(p)).concat(ins));
  const items: { ok: boolean; text: string }[] = [];
  const cost = [...squad].reduce((a, p) => a + (byId.get(p)?.price ?? 0), 0);
  const limitless = plan.chip === "limitless";
  const left = ctx.budget - cost;
  items.push({
    ok: limitless || left >= -1e-9,
    text: limitless ? "Budget: no limit this matchday (Limitless)" : left >= -1e-9 ? `Budget: ${eur(left)} left of ${eur(ctx.budget)}` : `Over budget by ${eur(-left)}`,
  });
  const clubs = new Map<string, string[]>();
  for (const p of squad) {
    const i = byId.get(p);
    if (i) clubs.set(i.club, [...(clubs.get(i.club) ?? []), i.name]);
  }
  const over = [...clubs.values()].filter((n) => n.length > ctx.club_limit);
  items.push({ ok: !over.length, text: over.length ? `More than ${ctx.club_limit} from one club: ${over[0].join(", ")}` : `Max ${ctx.club_limit} per club: OK` });
  const shapeOk = Object.entries<number>(ctx.rules.shape).every(([pos, n]) => [...squad].filter((p) => byId.get(p)?.pos === pos).length === n);
  const posOk = moves.every((m) => byId.get(m.out)?.pos === byId.get(m.in)?.pos);
  items.push({ ok: shapeOk && posOk, text: shapeOk && posOk ? "Squad 2-5-5-3: the best valid XI is picked for you" : "Each player out must be replaced in the same position" });
  if (new Set(ins).size !== ins.length || ins.some((i) => (ctx.squad as number[]).includes(i))) items.push({ ok: false, text: "The same player appears twice" });
  const capOk = plan.captain != null && squad.has(plan.captain);
  const viceOk = plan.vice_captain != null && squad.has(plan.vice_captain) && plan.vice_captain !== plan.captain;
  items.push({ ok: capOk && viceOk, text: capOk && viceOk ? "Captain and vice-captain: set" : "Pick a captain and a different vice-captain" });
  const chipOk = !plan.chip || (ctx.chips as P[]).some((c) => c.chip === plan.chip && c.available);
  if (!chipOk) items.push({ ok: false, text: `That chip can't be played on MD${ctx.md}` });
  const free = ctx.free_transfers;
  const hits = free == null || plan.chip ? 0 : Math.max(0, moves.length - free);
  items.push({
    ok: true,
    text: free == null ? `${moves.length} change${moves.length === 1 ? "" : "s"}: free (new team)` : `${moves.length} transfer${moves.length === 1 ? "" : "s"}, ${free} free${hits ? `: −${4 * hits}` : ": no hit"}`,
  });
  return { ok: items.every((i) => i.ok), items, squad };
}

function Search({ players, pos, exclude, value, onPick }: { players: P[]; pos: string | null; exclude: Set<number>; value: P | null; onPick: (id: number) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const norm = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const list = useMemo(() => {
    const n = norm(q.trim());
    return players
      .filter((p) => (!pos || p.pos === pos) && !exclude.has(p.id) && (!n || norm(p.name).includes(n) || norm(p.club).includes(n)))
      .sort((a, b) => b.xp_h - a.xp_h)
      .slice(0, 8);
  }, [q, players, pos, exclude]);
  return (
    <div className="yp-search">
      <input
        type="search"
        placeholder={value ? value.name : pos ? `Search ${pos}s…` : "Pick who goes out first"}
        value={q}
        disabled={!pos}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {value && !q && (
        <span className="yp-chosen">
          {value.name} · {value.club} · {eur(value.price)}
        </span>
      )}
      {open && pos && (
        <ul className="yp-results">
          {list.map((p) => (
            <li key={p.id}>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onPick(p.id);
                  setQ("");
                  setOpen(false);
                }}
              >
                <UclShirt code={p.club} goalkeeper={p.pos === "GK"} size={22} label={false} /> <b>{p.name}</b>{" "}
                <small>
                  {p.club} · {eur(p.price)}
                </small>
                <span className="yp-xp">{p.xp_next.toFixed(1)} xP</span>
              </button>
            </li>
          ))}
          {!list.length && <li className="muted small">No players match.</li>}
        </ul>
      )}
    </div>
  );
}

export function UclYourPlan() {
  const [ctx, setCtx] = useState<P | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<Saved | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [hasKey, setHasKey] = useState<boolean | null>(null);
  useEffect(() => {
    Promise.all([
      invoke<P>("plan_context", { game: "ucl" }),
      invoke<Saved | null>("plan_load", { game: "ucl" }),
      invoke<boolean>("anthropic_key_saved").catch(() => null),
    ])
      .then(([c, s, k]) => {
        if (!c.ok) throw new Error(c.error ?? "Couldn't load the squad.");
        setCtx(c);
        setSaved(s);
        setHasKey(k);
        setPlan(s?.plan && s.plan.md === c.md ? s.plan : fromXpert(c));
      })
      .catch((e) => setErr(String(e?.message ?? e)));
  }, []);
  const byId = useMemo(() => new Map<number, P>((ctx?.players ?? []).map((p: P) => [p.id, p])), [ctx]);
  if (err) return <UCard title="Your plan"><p className="umuted">{err}</p></UCard>;
  if (!ctx || !plan) return <UCard title="Your plan"><p className="umuted">Loading…</p></UCard>;

  const v = check(plan, ctx, byId);
  const squadIds: number[] = ctx.squad;
  const outsUsed = new Set(plan.transfers.map((t) => t.out).filter(Boolean) as number[]);
  const insUsed = new Set(plan.transfers.map((t) => t.in).filter(Boolean) as number[]);
  const mine = [...v.squad].map((id) => byId.get(id)).filter(Boolean).sort((a, b) => b!.xp_next - a!.xp_next) as P[];
  const set = (p: Partial<Plan>) => setPlan({ ...plan, ...p });
  const setMove = (i: number, m: Partial<Move>) => set({ transfers: plan.transfers.map((t, j) => (j === i ? { ...t, ...m } : t)) });
  const stale = !!saved?.review && (saved.stamp?.plan_key !== planKey(plan) || saved.stamp?.data_stamp !== ctx.data_stamp);
  const unsaved = !saved || planKey(saved.plan) !== planKey(plan);
  const dayOf = (p: P) => ctx.days.find((x: P) => x.gd === p.gd)?.label.split(" ")[0] ?? "";

  const save = async () => {
    setBusy(true);
    try {
      const base: Saved = { plan, saved_at: new Date().toISOString() };
      await invoke("plan_save", { saved: base, game: "ucl" });
      const review = await invoke<P>("plan_review", { plan, game: "ucl" });
      const full: Saved = { ...base, review, stamp: { plan_key: planKey(plan), data_stamp: review.data_stamp ?? ctx.data_stamp } };
      await invoke("plan_save", { saved: full, game: "ucl" });
      setSaved(full);
    } catch (e) {
      setSaved({ plan, review: { ok: false, error: String(e) }, stamp: { plan_key: planKey(plan), data_stamp: ctx.data_stamp } });
    } finally {
      setBusy(false);
    }
  };

  return (
    <UCard title={`Your plan for MD${ctx.md}`} aside="saved only on this Mac · XPert's advice doesn't change" className="yp-card uyp">
      <p className="umuted small">
        Your own UCL team can't be read (UEFA needs a login), so your plan starts from XPert's squad
        {ctx.new_team ? ": swap any players you'd pick differently (a new team's changes are free)." : "."}
      </p>
      <div className="yp-grid">
        <div className="yp-form">
          <div className="yp-label">
            Changes{" "}
            <small className="umuted">
              {ctx.free_transfers == null ? "free (new team)" : `${ctx.free_transfers} free`} · {eur(ctx.bank)} left
            </small>
          </div>
          {plan.transfers.map((t, i) => {
            const out = t.out ? byId.get(t.out) : null;
            return (
              <div key={i} className="yp-move">
                <select value={t.out ?? ""} onChange={(e) => setMove(i, { out: e.target.value ? Number(e.target.value) : null, in: null })}>
                  <option value="">Out…</option>
                  {squadIds
                    .filter((id) => id === t.out || !outsUsed.has(id))
                    .map((id) => byId.get(id))
                    .filter(Boolean)
                    .sort((a, b) => POS_ORDER.indexOf(a!.pos) - POS_ORDER.indexOf(b!.pos))
                    .map((p) => (
                      <option key={p!.id} value={p!.id}>
                        {p!.pos} · {p!.name} ({eur(p!.price)})
                      </option>
                    ))}
                </select>
                <span className="yp-arrow">→</span>
                <Search
                  players={ctx.players}
                  pos={out?.pos ?? null}
                  value={t.in ? byId.get(t.in) ?? null : null}
                  exclude={new Set([...squadIds, ...[...insUsed].filter((x) => x !== t.in)])}
                  onPick={(id) => setMove(i, { in: id })}
                />
                <button className="yp-x" title="Remove" onClick={() => set({ transfers: plan.transfers.filter((_, j) => j !== i) })}>
                  ×
                </button>
              </div>
            );
          })}
          <div className="yp-actions-row">
            <button className="yp-small" onClick={() => set({ transfers: [...plan.transfers, { out: null, in: null }] })}>
              + Add a change
            </button>
            {plan.transfers.length > 0 && (
              <button className="yp-small" onClick={() => set({ transfers: [] })}>
                No changes
              </button>
            )}
            <button className="yp-small" onClick={() => setPlan(fromXpert(ctx))}>
              Start from XPert's plan
            </button>
          </div>
          <div className="yp-pickers">
            <label>
              Captain
              <select value={plan.captain ?? ""} onChange={(e) => set({ captain: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Pick…</option>
                {mine.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.xp_next.toFixed(1)}, {dayOf(p)})
                  </option>
                ))}
              </select>
            </label>
            <label>
              Vice-captain
              <select value={plan.vice_captain ?? ""} onChange={(e) => set({ vice_captain: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Pick…</option>
                {mine.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.xp_next.toFixed(1)}, {dayOf(p)})
                  </option>
                ))}
              </select>
            </label>
            <label>
              Chip
              <select value={plan.chip ?? ""} onChange={(e) => set({ chip: e.target.value || null })}>
                <option value="">None</option>
                {(ctx.chips as P[]).map((c) => (
                  <option key={c.chip} value={c.chip} disabled={!c.available}>
                    {c.label}
                    {c.available ? "" : " (not this matchday)"}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
        <div className="yp-side">
          <ul className="yp-checks">
            {v.items.map((it) => (
              <li key={it.text} className={it.ok ? "ok" : "bad"}>
                {it.ok ? "✓" : "✕"} {it.text}
              </li>
            ))}
          </ul>
          <button className="yp-save" disabled={!v.ok || busy} onClick={save}>
            {busy ? "Checking your plan… (about 20 seconds)" : unsaved ? "Save plan" : "Save again"}
          </button>
          {!v.ok && <p className="umuted small">Fix the red items to save.</p>}
        </div>
      </div>
      {saved?.review && !busy && <Review r={saved.review} stale={stale} hasKey={hasKey} />}
    </UCard>
  );
}

function Review({ r, stale, hasKey }: { r: P; stale: boolean; hasKey: boolean | null }) {
  if (!r.ok) return <p className="note warn">{r.error ?? "The review failed."}</p>;
  if (!r.validation?.ok) return <p className="note warn">This plan breaks a rule: {(r.validation?.errors ?? []).join(" ")}</p>;
  const c = r.comparison;
  return (
    <div className={`yp-review${stale ? " stale" : ""}`}>
      {stale && <p className="note warn">Your plan or the data changed. Save again to refresh the review.</p>}
      <table className="utable">
        <thead>
          <tr>
            <th />
            <th>Your plan</th>
            <th>XPert</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Expected points this matchday</td>
            <td>{c.this_md.yours}</td>
            <td>{c.this_md.xpert}</td>
          </tr>
          <tr>
            <td>Expected points, {c.horizon.label}</td>
            <td>{c.horizon.yours}</td>
            <td>{c.horizon.xpert}</td>
          </tr>
        </tbody>
      </table>
      {r.review ? (
        <div className="yp-text">
          <ChatMarkdown text={r.review} />
          <p className="yp-cost umuted">Live consultant's review · cost about ${(r.cost_usd ?? 0).toFixed(3)}</p>
        </div>
      ) : (
        <p className="umuted small">
          {hasKey === false ? "Add your Anthropic API key in PL Settings (⚙) to get the written review." : "No written review this time."}
        </p>
      )}
    </div>
  );
}
