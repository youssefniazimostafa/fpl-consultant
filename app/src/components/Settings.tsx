import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Card } from "./Card";

// Settings: the betting-odds key for refreshes from this Mac. Stored only
// in the macOS Keychain (by the app's backend); the app never shows it,
// logs it or writes it to a file.
function KeyRow({ label, note, savedCmd, setCmd, clearCmd }: { label: string; note: string; savedCmd: string; setCmd: string; clearCmd: string }) {
  const [saved, setSaved] = useState<boolean | null>(null);
  const [key, setKey] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const refresh = () => invoke<boolean>(savedCmd).then(setSaved).catch(() => setSaved(false));
  useEffect(() => {
    refresh();
  }, []);
  const save = async () => {
    try {
      await invoke(setCmd, { key });
      setKey("");
      setMsg("Saved in your Keychain.");
      refresh();
    } catch (e) {
      setMsg(String(e));
    }
  };
  const remove = async () => {
    try {
      await invoke(clearCmd);
      setMsg("Removed from the Keychain.");
      refresh();
    } catch (e) {
      setMsg(String(e));
    }
  };
  return (
    <>
      <h3 className="set-h">{label}</h3>
      <p className="small muted">{note}</p>
      <p className="small">
        Status: <b>{saved == null ? "checking…" : saved ? "a key is saved in your Keychain" : "no key saved"}</b>
      </p>
      <div className="set-row">
        <input type="password" autoComplete="off" spellCheck={false} placeholder="Paste your key" value={key} onChange={(e) => setKey(e.target.value)} />
        <button className="refresh" disabled={!key.trim()} onClick={save}>
          Save
        </button>
        {saved && (
          <button className="refresh ghost" onClick={remove}>
            Remove
          </button>
        )}
      </div>
      {msg && <p className="small">{msg}</p>}
    </>
  );
}

const MODELS: [string, string][] = [
  ["claude-opus-5-5", "Claude Opus 5.5 (default, best answers)"],
  ["claude-sonnet-5-5", "Claude Sonnet 5.5 (cheaper)"],
  ["claude-haiku-5-5", "Claude Haiku 5.5 (cheapest)"],
];

function ModelRow() {
  const [model, setModel] = useState<string>("");
  useEffect(() => {
    invoke<string>("get_chat_model").then(setModel).catch(() => {});
  }, []);
  return (
    <div className="set-row">
      <label className="small" htmlFor="chat-model">Live consultant model</label>
      <select
        id="chat-model"
        value={model}
        onChange={(e) => {
          setModel(e.target.value);
          invoke("set_chat_model", { model: e.target.value }).catch(() => {});
        }}
      >
        {MODELS.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}

const STRATEGIES: [string, string][] = [
  ["auto", "Auto (from your league position)"],
  ["protect", "Protect (stick with players your rivals own)"],
  ["balanced", "Balanced (strongest team)"],
  ["chase", "Chase (lean toward differentials)"],
];

function StrategyRow() {
  const [strategy, setStrategy] = useState<string>("auto");
  useEffect(() => {
    invoke<string>("get_strategy").then(setStrategy).catch(() => {});
  }, []);
  return (
    <div className="set-row">
      <label className="small" htmlFor="strategy">Strategy</label>
      <select
        id="strategy"
        value={strategy}
        onChange={(e) => {
          setStrategy(e.target.value);
          invoke("set_strategy", { strategy: e.target.value }).catch(() => {});
        }}
      >
        {STRATEGIES.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <p className="small muted">Takes effect at your next Refresh. The published website always uses Auto.</p>
    </div>
  );
}

// League goals: used when two captain or transfer options are close on
// expected points (fpl/goals.py). Saved in settings.json and passed to the
// pipeline, chat and plan review as FPL_GOALS; the published site uses the
// defaults below.
type Goal = { kind: string; weight: number };
type Goals = { goals: Goal[]; by: "end" | number };
const GOAL_KINDS: [string, string][] = [
  ["first", "Finish 1st"],
  ["top2", "Stay in the top 2"],
  ["top3", "Stay in the top 3"],
];
const DEFAULT_GOALS: Goals = { goals: [{ kind: "first", weight: 2 }, { kind: "top2", weight: 1 }], by: "end" };

function GoalsRow() {
  const [g, setG] = useState<Goals>(DEFAULT_GOALS);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    invoke<Goals>("get_goals").then(setG).catch(() => {});
  }, []);
  const update = (next: Goals) => {
    setG(next);
    setMsg(null);
  };
  const save = async (next: Goals) => {
    try {
      await invoke("set_goals", { goals: next });
      setG(next);
      setMsg("Saved. Takes effect at your next Refresh.");
    } catch (e) {
      setMsg(String(e));
    }
  };
  const used = new Set(g.goals.map((x) => x.kind));
  const byChoice = g.by === "end" ? "end" : g.by === 19 ? "19" : "other";
  return (
    <>
      <h3 className="set-h">Your league goals</h3>
      <p className="small muted">
        When two captains or transfers are close on expected points, XPert compares their chances of reaching these goals.
        A bigger weight counts for more.
      </p>
      {g.goals.map((x, i) => (
        <div className="set-row goal-row" key={i}>
          <select
            aria-label={`Goal ${i + 1}`}
            value={x.kind}
            onChange={(e) => update({ ...g, goals: g.goals.map((y, j) => (j === i ? { ...y, kind: e.target.value } : y)) })}
          >
            {GOAL_KINDS.filter(([k]) => k === x.kind || !used.has(k)).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
          <label className="small goal-weight">
            weight
            <input
              type="number"
              min={1}
              max={10}
              step={1}
              value={x.weight}
              onChange={(e) =>
                update({ ...g, goals: g.goals.map((y, j) => (j === i ? { ...y, weight: Math.min(10, Math.max(1, Number(e.target.value) || 1)) } : y)) })
              }
            />
          </label>
          {g.goals.length > 1 && (
            <button className="link" onClick={() => update({ ...g, goals: g.goals.filter((_, j) => j !== i) })}>
              Remove
            </button>
          )}
        </div>
      ))}
      <div className="set-row">
        {g.goals.length < GOAL_KINDS.length && (
          <button
            className="refresh ghost"
            onClick={() => {
              const kind = GOAL_KINDS.map(([k]) => k).find((k) => !used.has(k))!;
              update({ ...g, goals: [...g.goals, { kind, weight: 1 }] });
            }}
          >
            Add a goal
          </button>
        )}
        <label className="small" htmlFor="goal-by">
          By
        </label>
        <select
          id="goal-by"
          value={byChoice}
          onChange={(e) => update({ ...g, by: e.target.value === "end" ? "end" : e.target.value === "19" ? 19 : 30 })}
        >
          <option value="end">the end of the season</option>
          <option value="19">GW19 (halfway)</option>
          <option value="other">another gameweek…</option>
        </select>
        {byChoice === "other" && (
          <input
            className="goal-gw"
            type="number"
            min={1}
            max={38}
            aria-label="Gameweek"
            value={g.by as number}
            onChange={(e) => update({ ...g, by: Math.min(38, Math.max(1, Number(e.target.value) || 38)) })}
          />
        )}
      </div>
      <div className="set-row">
        <button className="refresh" onClick={() => save(g)}>
          Save goals
        </button>
        <button className="refresh ghost" onClick={() => save(DEFAULT_GOALS)}>
          Reset to defaults
        </button>
      </div>
      <p className="small muted">Defaults: finish 1st (weight 2) and stay in the top 2 (weight 1), by the end of the season. The published website always uses the defaults.</p>
      {msg && <p className="small">{msg}</p>}
    </>
  );
}

export function Settings({ onClose, onTour }: { onClose: () => void; onTour?: () => void }) {
  const [saved, setSaved] = useState<boolean | null>(null);
  const [key, setKey] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = () => invoke<boolean>("odds_key_saved").then(setSaved).catch(() => setSaved(false));
  useEffect(() => {
    refresh();
  }, []);

  const save = async () => {
    try {
      await invoke("set_odds_key", { key });
      setKey("");
      setMsg("Saved. The next Refresh will include the betting odds.");
      refresh();
    } catch (e) {
      setMsg(String(e));
    }
  };
  const remove = async () => {
    try {
      await invoke("clear_odds_key");
      setMsg("Removed from the Keychain.");
      refresh();
    } catch (e) {
      setMsg(String(e));
    }
  };

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <Card title="Settings" aside={<button className="link" onClick={onClose}>Close</button>}>
          <h3 className="set-h">Betting-odds key</h3>
          <p className="small muted">
            Refreshes use bookmaker odds for fixture difficulty when you add your The Odds API key. It's stored only in your
            Mac's Keychain, never in a file, the logs or the app's data.
          </p>
          <p className="small">
            Status: <b>{saved == null ? "checking…" : saved ? "a key is saved in your Keychain" : "no key saved"}</b>
          </p>
          <div className="set-row">
            <input
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder="Paste your odds key"
              value={key}
              onChange={(e) => setKey(e.target.value)}
            />
            <button className="refresh" disabled={!key.trim()} onClick={save}>
              Save
            </button>
            {saved && (
              <button className="refresh ghost" onClick={remove}>
                Remove
              </button>
            )}
          </div>
          {msg && <p className="small">{msg}</p>}
          <hr className="set-sep" />
          <KeyRow
            label="Anthropic API key (Live consultant)"
            note="Used only by the app's backend to answer your Live consultant questions. Stored only in your Mac's Keychain, never in a file, the logs or the app's data."
            savedCmd="anthropic_key_saved"
            setCmd="set_anthropic_key"
            clearCmd="clear_anthropic_key"
          />
          <ModelRow />
          <hr className="set-sep" />
          <StrategyRow />
          <hr className="set-sep" />
          <GoalsRow />
          {onTour && (
            <>
              <hr className="set-sep" />
              <div className="set-row">
                <button className="refresh ghost" onClick={onTour}>
                  Take the tour
                </button>
                <p className="small muted">A quick walk through every page.</p>
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
