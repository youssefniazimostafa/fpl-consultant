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
