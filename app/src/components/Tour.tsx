import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { DESKTOP } from "../platform";

// First-launch guided tour: each step switches to its page if needed,
// spotlights the real element and explains it in a sentence or two.
// Completion is remembered in this browser/app (TOUR_KEY); Settings has
// "Take the tour" to run it again.

export const TOUR_KEY = "xpert:tour-done";
export const SETUP_KEY = "xpert:setup-done";

export function storageGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
export function storageSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: the tour just shows again next time */
  }
}

export interface Step {
  tab?: string;
  targets: string[]; // first visible match is highlighted (laptop, then phone)
  title: string;
  text: string;
  desktopOnly?: boolean;
}

const STEPS: Step[] = [
  {
    targets: [".xbar"],
    title: "Welcome to XPert",
    text: "The top bar shows the countdown to the next deadline and when your data was last updated. The PL | UCL switch flips between the Premier League game and the Champions League one.",
  },
  {
    tab: "team",
    targets: [".team-pitch", ".content"],
    title: "Recommended team",
    text: "Your best XI for this gameweek, with the captain (C), vice-captain (V) and any transfer. Copy it into the FPL app before the deadline.",
  },
  {
    tab: "lastgw",
    targets: [".content"],
    title: "Last gameweek and Team of the week",
    text: "Last gameweek shows where your points came from and what you left on the bench. Team of the week is the best XI across all of FPL.",
  },
  {
    tab: "league",
    targets: [".rivals-card", ".standings-card", ".content"],
    title: "Your leagues",
    text: "Your mini-league table, the rivals close to you and the transfer each is likely to make, and who owns which players.",
  },
  {
    tab: "overview",
    targets: [".sc-grid", ".content"],
    title: "Scout",
    text: "All of FPL in short lists: the most expected points, captain picks, players in form, cheap scorers and injury news. Player lookup finds any player and opens his full report.",
  },
  {
    tab: "planner",
    targets: [".yp-card", ".planner-card", ".pweeks", ".content"],
    title: "Planner",
    text: "The plan for the next few gameweeks and the best week for each chip. In Your plan you can make your own choices and the consultant compares them with its own.",
  },
  {
    tab: "analytics",
    targets: [".seg", ".content"],
    title: "Analytics",
    text: "The detail behind the advice: how your season has gone, how close our numbers have been, and your options this week. Tap any ? for a plain explanation.",
  },
  {
    targets: [".chat-btn"],
    title: "Live consultant",
    text: "Ask anything about this week's advice in plain words, like \"why Gabriel and not Haaland?\". It needs your Anthropic key in Settings (⚙).",
    desktopOnly: true,
  },
];

function visible(el: Element | null): el is HTMLElement {
  if (!el) return false;
  const r = (el as HTMLElement).getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

export function Tour({
  onTab,
  onClose,
  steps: custom,
  doneKey = TOUR_KEY,
}: {
  onTab: (tab: string) => void;
  onClose: () => void;
  steps?: Step[]; // UCL mode has its own tour
  doneKey?: string;
}) {
  const steps = (custom ?? STEPS).filter((s) => DESKTOP || !s.desktopOnly);
  const [i, setI] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const step = steps[i];

  const measure = useCallback(() => {
    const el = step.targets.map((t) => document.querySelector(t)).find(visible) ?? null;
    if (el) {
      el.scrollIntoView({ block: "nearest" });
      setRect(el.getBoundingClientRect());
    } else {
      setRect(null);
    }
  }, [step]);

  useEffect(() => {
    if (step.tab) onTab(step.tab);
    const t = setTimeout(measure, 250);
    window.addEventListener("resize", measure);
    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", measure);
    };
  }, [step, onTab, measure]);
  useLayoutEffect(measure, [measure]);

  const finish = () => {
    storageSet(doneKey, "1");
    onClose();
  };
  const pad = 6;
  const vh = window.innerHeight;
  const vw = window.innerWidth;
  // the explanation sits below the highlight if there's room, else above
  const below = rect ? rect.bottom + 190 < vh || rect.top < 200 : true;
  const boxTop = rect ? (below ? Math.min(rect.bottom + 14, vh - 190) : Math.max(rect.top - 190, 10)) : vh / 2 - 90;
  const boxLeft = rect ? Math.min(Math.max(rect.left, 12), vw - 352) : vw / 2 - 170;

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label="Guided tour">
      {rect ? (
        <div
          className="tour-spot"
          style={{ top: rect.top - pad, left: rect.left - pad, width: rect.width + 2 * pad, height: Math.min(rect.height + 2 * pad, vh - rect.top + pad) }}
        />
      ) : (
        <div className="tour-dim" />
      )}
      <div className="tour-box" style={{ top: boxTop, left: Math.max(boxLeft, 12) }}>
        <div className="tour-count">
          Step {i + 1} of {steps.length}
        </div>
        <h3>{step.title}</h3>
        <p>{step.text}</p>
        <div className="tour-actions">
          <button className="tour-skip" onClick={finish}>
            Skip
          </button>
          <span className="tour-grow" />
          {i > 0 && (
            <button className="tour-back" onClick={() => setI(i - 1)}>
              Back
            </button>
          )}
          <button className="tour-next" onClick={() => (i + 1 < steps.length ? setI(i + 1) : finish())}>
            {i + 1 < steps.length ? "Next" : "Done"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** First-run setup (Mac app): your FPL team and league IDs, plus optional keys. */
export function Setup({ defaults, onDone }: { defaults: { teamId: string; leagueId: string }; onDone: () => void }) {
  const [teamId, setTeamId] = useState(defaults.teamId);
  const [leagueId, setLeagueId] = useState(defaults.leagueId);
  const [odds, setOdds] = useState("");
  const [anthropic, setAnthropic] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      await invoke("set_setup", { teamId, leagueId });
      if (odds.trim()) await invoke("set_odds_key", { key: odds.trim() });
      if (anthropic.trim()) await invoke("set_anthropic_key", { key: anthropic.trim() });
      storageSet(SETUP_KEY, "1");
      onDone();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };
  const skip = () => {
    storageSet(SETUP_KEY, "1");
    onDone();
  };

  return (
    <div className="setup-backdrop" role="dialog" aria-modal="true" aria-label="Set up XPert">
      <div className="setup-card">
        <h2>Set up XPert</h2>
        <p className="muted">Tell XPert which team and mini-league are yours. You can change this later in Settings (⚙).</p>
        <label>
          Your FPL team ID
          <input inputMode="numeric" value={teamId} onChange={(e) => setTeamId(e.target.value)} placeholder="e.g. 1234567" />
        </label>
        <p className="setup-note">
          On the FPL website, open <b>Points</b>: your team ID is the number after <code>/entry/</code> in the address. In the
          FPL app: Pick Team → Team details.
        </p>
        <label>
          Your mini-league ID
          <input inputMode="numeric" value={leagueId} onChange={(e) => setLeagueId(e.target.value)} placeholder="e.g. 123456" />
        </label>
        <p className="setup-note">
          Open your league under <b>Leagues</b>: the league ID is the number after <code>/leagues/</code> in the address.
        </p>
        <details className="setup-keys">
          <summary>Optional: keys for odds and the Live consultant</summary>
          <label>
            Odds API key (better fixture predictions)
            <input type="password" value={odds} onChange={(e) => setOdds(e.target.value)} autoComplete="off" />
          </label>
          <label>
            Anthropic API key (Live consultant)
            <input type="password" value={anthropic} onChange={(e) => setAnthropic(e.target.value)} autoComplete="off" />
          </label>
          <p className="setup-note">Keys are stored only in your Mac's Keychain.</p>
        </details>
        {msg && <p className="note warn">{msg}</p>}
        <div className="tour-actions">
          <button className="tour-skip" onClick={skip}>
            Skip for now
          </button>
          <span className="tour-grow" />
          <button className="tour-next" disabled={busy || !teamId.trim()} onClick={save}>
            {busy ? "Saving…" : "Save and take the tour"}
          </button>
        </div>
        <p className="setup-note">After saving, press Refresh to load your own team.</p>
      </div>
    </div>
  );
}
