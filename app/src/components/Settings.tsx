import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Card } from "./Card";

// Settings: the betting-odds key for refreshes from this Mac. Stored only
// in the macOS Keychain (by the app's backend); the app never shows it,
// logs it or writes it to a file.
export function Settings({ onClose }: { onClose: () => void }) {
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
        </Card>
      </div>
    </div>
  );
}
