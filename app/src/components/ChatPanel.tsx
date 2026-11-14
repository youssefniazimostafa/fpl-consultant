import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ChatMarkdown } from "./ChatMarkdown";

// Live consultant: quick questions about the recommendation, answered by
// the backend (fpl/chat.py) from the app's own data. The conversation
// lives only in this component's memory: Clear wipes it, and closing the
// panel unmounts it, so reopening starts empty. Nothing is saved.

interface Msg {
  role: "user" | "assistant";
  content: string;
  cost?: number;
  error?: boolean;
}

const STARTERS = ["Why this transfer?", "Why this captain?", "Compare two players: Van Hecke or Vuskovic?", "What if I don't make the transfer?"];

const fmtCost = (c: number) => (c < 0.001 ? "under $0.001" : `about $${c.toFixed(c < 0.01 ? 3 : 2)}`);

export function ChatPanel({ onClose, initialQuestion = null }: { onClose: () => void; initialQuestion?: string | null }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [msgs, busy]);
  const asked = useRef(false);

  const send = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    const history = [...msgs.filter((m) => !m.error), { role: "user" as const, content: question }];
    setMsgs((m) => [...m, { role: "user", content: question }]);
    setText("");
    setBusy(true);
    try {
      const r = await invoke<Record<string, any>>("chat_send", { messages: history.map(({ role, content }) => ({ role, content })) });
      setMsgs((m) => [
        ...m,
        r.ok ? { role: "assistant", content: r.answer, cost: r.cost_usd } : { role: "assistant", content: r.error, error: true },
      ]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: String(e), error: true }]);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (initialQuestion && !asked.current) {
      asked.current = true;
      send(initialQuestion);
    }
  }, [initialQuestion]);

  return (
    <aside className="chat-panel" aria-label="Live consultant">
      <header className="chat-head">
        <b>Live consultant</b>
        <span className="chat-actions">
          <button className="link" onClick={() => setMsgs([])} disabled={busy || !msgs.length}>
            Clear
          </button>
          <button className="chat-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </span>
      </header>
      <div className="chat-list" ref={listRef}>
        {msgs.length === 0 && (
          <div className="chat-empty">
            <p className="muted small">Ask about this week's recommendation. Answers use only the app's own data.</p>
            <div className="chat-starters">
              {STARTERS.map((s) => (
                <button key={s} className="chat-starter" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}${m.error ? " error" : ""}`}>
            <div className="chat-bubble">{m.role === "assistant" && !m.error ? <ChatMarkdown text={m.content} /> : m.content}</div>
            {m.cost != null && <div className="chat-cost">{fmtCost(m.cost)}</div>}
          </div>
        ))}
        {busy && <div className="chat-msg assistant"><div className="chat-bubble muted">Looking at the data…</div></div>}
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          send(text);
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask a question…" disabled={busy} />
        <button className="refresh" type="submit" disabled={busy || !text.trim()}>
          Send
        </button>
      </form>
    </aside>
  );
}
