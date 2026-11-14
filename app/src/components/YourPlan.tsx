import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Card } from "./Card";
import { ChatMarkdown } from "./ChatMarkdown";
import { chanceClass } from "../theme/theme";

// "Your plan for GW<next>" (Planner, desktop app only): pick your own
// transfers, captain, vice-captain and chip; it's checked live (budget with
// selling prices, 3 per club, squad shape, free transfers and hits), saved
// only in the app's data folder on this Mac (never the repo, the engine copy
// or latest.json), and "Save plan" runs the consultant's review
// (fpl/my_plan.py: the Live consultant engine and its what_if comparison).
// The model's recommendation never reads it.

type Raw = Record<string, any>;
interface Move { out: number | null; in: number | null }
interface Plan { gameweek: number; transfers: Move[]; captain: number | null; vice_captain: number | null; chip: string | null }
interface Saved { plan: Plan; review?: Raw; stamp?: { plan_key: string; data_stamp: string | null }; saved_at?: string }

const money = (tenths: number) => `£${(tenths / 10).toFixed(1)}m`;
const planKey = (p: Plan) => JSON.stringify({ ...p, transfers: p.transfers.filter((t) => t.out && t.in) });
const CHIP_NAMES: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };

function fromRecommendation(ctx: Raw): Plan {
  const r = ctx.recommendation ?? {};
  return {
    gameweek: ctx.gameweek,
    transfers: (r.transfers ?? []).map((t: Raw) => ({ out: t.out, in: t.in })),
    captain: r.captain ?? null,
    vice_captain: r.vice_captain ?? null,
    chip: r.chip ?? null,
  };
}

/** The same checks fpl/my_plan.py's validate() runs, live in the card. */
function check(plan: Plan, ctx: Raw, byId: Map<number, Raw>) {
  const sell = new Map<number, number>(ctx.squad.map((s: Raw) => [s.id, s.sell_price]));
  const moves = plan.transfers.filter((t) => t.out && t.in) as { out: number; in: number }[];
  const outs = moves.map((m) => m.out);
  const ins = moves.map((m) => m.in);
  const squad = new Set<number>([...sell.keys()].filter((p) => !outs.includes(p)).concat(ins));
  const items: { ok: boolean; text: string }[] = [];
  const bank = ctx.bank + outs.reduce((a, o) => a + (sell.get(o) ?? 0), 0) - ins.reduce((a, i) => a + (byId.get(i)?.price ?? 0), 0);
  items.push({ ok: bank >= 0, text: bank >= 0 ? `Budget: ${money(bank)} left in the bank` : `Over budget by ${money(-bank)}` });
  const clubs = new Map<number, string[]>();
  for (const p of squad) {
    const i = byId.get(p);
    if (i) clubs.set(i.team, [...(clubs.get(i.team) ?? []), i.name]);
  }
  const over = [...clubs.values()].filter((n) => n.length > ctx.rules.max_per_club);
  items.push({ ok: !over.length, text: over.length ? `More than 3 from one club: ${over[0].join(", ")}` : "Max 3 per club: OK" });
  const shapeOk = Object.entries<number>(ctx.rules.quotas).every(([pos, n]) => [...squad].filter((p) => byId.get(p)?.position === pos).length === n);
  const posOk = moves.every((m) => byId.get(m.out)?.position === byId.get(m.in)?.position);
  items.push({ ok: shapeOk && posOk, text: shapeOk && posOk ? "Squad shape 2-5-5-3: a valid formation is picked for you" : "Each player out must be replaced in the same position" });
  const dupes = new Set(ins).size !== ins.length || new Set(outs).size !== outs.length || ins.some((i) => sell.has(i));
  if (dupes) items.push({ ok: false, text: "The same player appears twice, or you already own an incoming player" });
  const capOk = plan.captain != null && squad.has(plan.captain);
  const viceOk = plan.vice_captain != null && squad.has(plan.vice_captain) && plan.vice_captain !== plan.captain;
  items.push({ ok: capOk && viceOk, text: capOk && viceOk ? "Captain and vice-captain: set" : "Pick a captain and a different vice-captain from your squad" });
  const noHit = plan.chip === "wildcard" || plan.chip === "freehit";
  const hits = noHit ? 0 : Math.max(0, moves.length - ctx.free_transfers);
  items.push({
    ok: true,
    text: `${moves.length} transfer${moves.length === 1 ? "" : "s"}, ${ctx.free_transfers} free${hits ? `: a ${hits * ctx.rules.hit_cost}-point hit` : noHit && moves.length ? " (no hit with this chip)" : ": no hit"}`,
  });
  return { ok: items.every((i) => i.ok), items, squad, hits };
}

function PlayerSearch({ players, position, value, onPick, exclude }: { players: Raw[]; position: string | null; value: Raw | null; onPick: (id: number) => void; exclude: Set<number> }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const norm = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const matches = useMemo(() => {
    const n = norm(q.trim());
    return players
      .filter((p) => (!position || p.position === position) && !exclude.has(p.id) && (!n || norm(p.name).includes(n) || norm(p.team_short ?? "").includes(n)))
      .sort((a, b) => b.xp_5 - a.xp_5)
      .slice(0, 8);
  }, [q, players, position, exclude]);
  return (
    <div className="yp-search">
      <input
        type="search"
        placeholder={value ? value.name : position ? `Search ${position}s…` : "Pick who goes out first"}
        value={q}
        disabled={!position}
        onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {value && !q && <span className="yp-chosen">{value.name} · {value.team_short} · {money(value.price)}</span>}
      {open && position && (
        <ul className="yp-results">
          {matches.map((p) => (
            <li key={p.id}>
              <button onMouseDown={(e) => e.preventDefault()} onClick={() => { onPick(p.id); setQ(""); setOpen(false); }}>
                <b>{p.name}</b> <small>{p.team_short} · {money(p.price)}</small>
                <span className="yp-xp">{p.xp_5.toFixed(1)} pts / 5 GWs</span>
                {p.chance != null && p.chance < 100 && <span className={`cop-badge ${chanceClass(p.chance)}`}>{p.chance}%</span>}
              </button>
            </li>
          ))}
          {!matches.length && <li className="muted small">No players match.</li>}
        </ul>
      )}
    </div>
  );
}

function Review({ r, stale, hasKey }: { r: Raw; stale: boolean; hasKey: boolean | null }) {
  if (!r.ok) return <p className="note warn">{r.error ?? "The review failed."}</p>;
  const v = r.validation;
  if (!v.ok) return <p className="note warn">This plan breaks a rule: {v.errors.join(" ")}</p>;
  const c = r.comparison;
  const ch = r.chances;
  const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
  return (
    <div className={`yp-review${stale ? " stale" : ""}`}>
      {stale && <p className="note warn">Your plan or the data changed. Save again to refresh the review.</p>}
      <table className="an-table yp-compare">
        <thead><tr><th /><th>Your plan</th><th>Consultant</th></tr></thead>
        <tbody>
          <tr><td>Expected points this week</td><td>{c.projected_points_this_week.scenario}</td><td>{c.projected_points_this_week.recommendation}</td></tr>
          <tr><td>Expected points, next 5 weeks</td><td>{c.projected_points_next_5_weeks.scenario}</td><td>{c.projected_points_next_5_weeks.recommendation}</td></tr>
          {ch && <tr><td>Top score in your league this week</td><td>{pct(ch.yours.p_top_score)}</td><td>{pct(ch.consultant.p_top_score)}</td></tr>}
          {ch && <tr><td>Outscoring the leader this week</td><td>{pct(ch.yours.p_beat_leader)}</td><td>{pct(ch.consultant.p_beat_leader)}</td></tr>}
          {ch && <tr><td>A bad week / a good week</td><td>{ch.yours.bad_week} / {ch.yours.good_week}</td><td>{ch.consultant.bad_week} / {ch.consultant.good_week}</td></tr>}
        </tbody>
      </table>
      {r.review ? (
        <div className="yp-text">
          <ChatMarkdown text={r.review} />
          <p className="yp-cost muted">Consultant's review · cost about ${(r.cost_usd ?? 0).toFixed(3)}</p>
        </div>
      ) : (
        <p className="muted small">{hasKey === false ? "Add your Anthropic API key in Settings (the gear in the status bar) to get the consultant's written review." : "No written review this time."}</p>
      )}
    </div>
  );
}

export function YourPlanCard() {
  const [ctx, setCtx] = useState<Raw | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<Saved | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [hasKey, setHasKey] = useState<boolean | null>(null);

  useEffect(() => {
    Promise.all([invoke<Raw>("plan_context"), invoke<Saved | null>("plan_load"), invoke<boolean>("anthropic_key_saved").catch(() => null)])
      .then(([c, s, k]) => {
        if (!c.ok) throw new Error(c.error ?? "Couldn't load your squad.");
        setCtx(c);
        setSaved(s);
        setHasKey(k);
        setPlan(s?.plan && s.plan.gameweek === c.gameweek ? s.plan : fromRecommendation(c));
      })
      .catch((e) => setErr(String(e?.message ?? e)));
  }, []);

  const byId = useMemo(() => new Map<number, Raw>((ctx?.players ?? []).map((p: Raw) => [p.id, p])), [ctx]);
  if (err) return <Card title="Your plan" className="yp-card"><p className="muted">{err}</p></Card>;
  if (!ctx || !plan) return <Card title="Your plan" className="yp-card"><p className="muted">Loading your squad…</p></Card>;

  const v = check(plan, ctx, byId);
  const squadIds: number[] = ctx.squad.map((s: Raw) => s.id);
  const sell = new Map<number, number>(ctx.squad.map((s: Raw) => [s.id, s.sell_price]));
  const outsUsed = new Set(plan.transfers.map((t) => t.out).filter(Boolean) as number[]);
  const insUsed = new Set(plan.transfers.map((t) => t.in).filter(Boolean) as number[]);
  const newSquad = [...v.squad].map((id) => byId.get(id)).filter(Boolean).sort((a, b) => b!.xp_next - a!.xp_next) as Raw[];
  const set = (p: Partial<Plan>) => setPlan({ ...plan, ...p });
  const setMove = (i: number, m: Partial<Move>) => set({ transfers: plan.transfers.map((t, j) => (j === i ? { ...t, ...m } : t)) });
  const stale = !!saved?.review && (saved.stamp?.plan_key !== planKey(plan) || saved.stamp?.data_stamp !== ctx.data_stamp);
  const unsaved = !saved || planKey(saved.plan) !== planKey(plan);

  const save = async () => {
    setBusy(true);
    try {
      const base: Saved = { plan, saved_at: new Date().toISOString() };
      await invoke("plan_save", { saved: base });
      const review = await invoke<Raw>("plan_review", { plan });
      const full: Saved = { ...base, review, stamp: { plan_key: planKey(plan), data_stamp: review.data_stamp ?? ctx.data_stamp } };
      await invoke("plan_save", { saved: full });
      setSaved(full);
    } catch (e) {
      const full: Saved = { plan, review: { ok: false, error: String(e) }, stamp: { plan_key: planKey(plan), data_stamp: ctx.data_stamp } };
      setSaved(full);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title={`Your plan for GW${ctx.gameweek}`} aside="saved only on this Mac · the consultant's advice doesn't change" className="yp-card">
      <div className="yp-grid">
        <div className="yp-form">
          <div className="yp-label">Transfers <small className="muted">{ctx.free_transfers} free · bank {money(ctx.bank)}</small></div>
          {plan.transfers.map((t, i) => {
            const out = t.out ? byId.get(t.out) : null;
            return (
              <div key={i} className="yp-move">
                <select value={t.out ?? ""} onChange={(e) => setMove(i, { out: e.target.value ? Number(e.target.value) : null, in: null })}>
                  <option value="">Out…</option>
                  {squadIds
                    .filter((id) => id === t.out || !outsUsed.has(id))
                    .map((id) => byId.get(id))
                    .sort((a, b) => ["GKP", "DEF", "MID", "FWD"].indexOf(a!.position) - ["GKP", "DEF", "MID", "FWD"].indexOf(b!.position))
                    .map((p) => (
                      <option key={p!.id} value={p!.id}>{p!.position} · {p!.name} ({money(sell.get(p!.id) ?? 0)})</option>
                    ))}
                </select>
                <span className="yp-arrow">→</span>
                <PlayerSearch
                  players={ctx.players}
                  position={out?.position ?? null}
                  value={t.in ? byId.get(t.in) ?? null : null}
                  exclude={new Set([...squadIds, ...[...insUsed].filter((x) => x !== t.in)])}
                  onPick={(id) => setMove(i, { in: id })}
                />
                <button className="yp-x" title="Remove this transfer" onClick={() => set({ transfers: plan.transfers.filter((_, j) => j !== i) })}>×</button>
              </div>
            );
          })}
          <div className="yp-actions-row">
            <button className="yp-small" onClick={() => set({ transfers: [...plan.transfers, { out: null, in: null }] })}>+ Add a transfer</button>
            {plan.transfers.length > 0 && <button className="yp-small" onClick={() => set({ transfers: [] })}>No transfers</button>}
            <button className="yp-small" onClick={() => setPlan(fromRecommendation(ctx))}>Start from the consultant's plan</button>
          </div>

          <div className="yp-pickers">
            <label>Captain
              <select value={plan.captain ?? ""} onChange={(e) => set({ captain: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Pick…</option>
                {newSquad.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.xp_next.toFixed(1)})</option>)}
              </select>
            </label>
            <label>Vice-captain
              <select value={plan.vice_captain ?? ""} onChange={(e) => set({ vice_captain: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Pick…</option>
                {newSquad.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.xp_next.toFixed(1)})</option>)}
              </select>
            </label>
            <label>Chip
              <select value={plan.chip ?? ""} onChange={(e) => set({ chip: e.target.value || null })}>
                <option value="">None</option>
                {(ctx.chips as Raw[]).filter((c) => c.available).map((c) => <option key={c.chip} value={c.chip}>{CHIP_NAMES[c.chip] ?? c.label}</option>)}
              </select>
            </label>
          </div>
        </div>

        <div className="yp-side">
          <ul className="yp-checks">
            {v.items.map((it) => (
              <li key={it.text} className={it.ok ? "ok" : "bad"}>{it.ok ? "✓" : "✕"} {it.text}</li>
            ))}
          </ul>
          <button className="yp-save" disabled={!v.ok || busy} onClick={save}>
            {busy ? "Checking your plan… (about 20 seconds)" : unsaved ? "Save plan" : "Save again"}
          </button>
          {!v.ok && <p className="muted small">Fix the red items to save.</p>}
        </div>
      </div>
      {saved?.review && !busy && <Review r={saved.review} stale={stale} hasKey={hasKey} />}
    </Card>
  );
}
