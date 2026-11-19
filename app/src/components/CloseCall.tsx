import type { Latest } from "../data";

// Close calls scored on your league goals (fpl/goal_calls.py, latest.json's
// `goal_calls`): when captain or transfer options are close on expected
// points, each option's simulated chance of your goals and which one XPert
// picks. Display only; the pick itself is the decision's (checked by
// fpl/check_consistency.py rule 13).

type Raw = Record<string, any>;

const SHORT: Record<string, string> = { first: "to win", top2: "top 2", top3: "top 3" };
const SHOW = 3;

function chancesText(chances: Raw, goals: Raw[]): string {
  return [...goals]
    .sort((a, b) => a.weight - b.weight)
    .map((g) => `${(100 * (chances?.[g.kind] ?? 0)).toFixed(1)}% ${SHORT[g.kind] ?? g.kind}`)
    .join(", ");
}

function whenText(t: Raw): string {
  return t.target_gameweek === 38 ? "by the end of the season" : `by GW${t.target_gameweek}`;
}

/** The captain options to show: the expected-points pick, XPert's pick and the goal pick, then the next by expected points. */
function shownCaptains(options: Raw[]): Raw[] {
  const keep = options.filter((o) => o.is_xp_pick || o.is_xpert_pick || o.is_goal_pick);
  for (const o of [...options].sort((a, b) => b.xp - a.xp)) {
    if (keep.length >= SHOW) break;
    if (!keep.includes(o)) keep.push(o);
  }
  return keep.sort((a, b) => b.xp - a.xp);
}

export function CaptainCloseCall({ data }: { data: Latest }) {
  const gc: Raw | undefined = data.goal_calls;
  const c: Raw | undefined = gc?.captain;
  if (!gc?.available || !c?.close) return null;
  const goals: Raw[] = gc.goals ?? [];
  const extra = (c.options?.length ?? 0) - SHOW;
  return (
    <div className="close-call">
      <div className="close-call-title">Close call · your league goals {whenText(gc)}</div>
      <ul>
        {shownCaptains(c.options ?? []).map((o) => (
          <li key={o.player_id}>
            <span>
              <b>{o.web_name}</b>
              {o.is_xpert_pick && <span className="cc-tag">XPert's pick</span>}
            </span>
            <span className="cc-num">
              {o.xp.toFixed(2)} xP · {chancesText(o.chances, goals)}
            </span>
          </li>
        ))}
      </ul>
      <p className="note">
        {c.why}
        {extra > 0 ? ` (${extra} more players are within the close range; ask the Live consultant for all of them.)` : ""}
      </p>
      {gc.target_note && <p className="note muted">{gc.target_note}.</p>}
    </div>
  );
}

export function TransferCloseCall({ data }: { data: Latest }) {
  const gc: Raw | undefined = data.goal_calls;
  const t: Raw | undefined = gc?.transfers;
  if (!gc?.available || !t?.close) return null;
  const goals: Raw[] = gc.goals ?? [];
  const shown = (t.options ?? []).filter((o: Raw) => o.chances);
  return (
    <div className="close-call">
      <div className="close-call-title">Close transfer options · your league goals {whenText(gc)}</div>
      <ul>
        {shown.map((o: Raw) => (
          <li key={o.label}>
            <span>
              <b>{o.label}</b>
              {o.is_xpert_pick && <span className="cc-tag">XPert's pick</span>}
            </span>
            <span className="cc-num">
              {o.is_xpert_pick ? "" : `${o.gap > 0 ? "−" : "+"}${Math.abs(o.gap).toFixed(1)} xP vs XPert's pick · `}
              {chancesText(o.chances, goals)}
            </span>
          </li>
        ))}
      </ul>
      <p className="note">{t.why}</p>
    </div>
  );
}
