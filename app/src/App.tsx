import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getPath, loadInitial, loadLocal, type LoadedData } from "./data";
import { TABS } from "./tabs";

interface Progress {
  stage: string;
  line: string | null;
}
interface Done {
  ok: boolean;
  error: string | null;
  seconds: number;
}

// Module-level so React StrictMode's double effect run (dev only) can't
// start the opt-in auto-refresh twice.
let autorefreshStarted = false;

/** Best-effort line in ~/Library/Application Support/FPL Consultant/app.log. */
function log(message: string) {
  invoke("ui_log", { message }).catch(() => {});
}

function fmtTime(iso: string | null): string {
  if (!iso) return "unknown";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function countdown(deadlineIso: string | undefined, now: number): string {
  if (!deadlineIso) return "no upcoming deadline";
  const ms = new Date(deadlineIso).getTime() - now;
  if (Number.isNaN(ms)) return deadlineIso;
  if (ms <= 0) return "deadline passed";
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${d > 0 ? `${d}d ` : ""}${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function renderValue(value: unknown): string {
  if (value === undefined) return "(missing)";
  if (typeof value === "string") return value;
  const text = JSON.stringify(value);
  return text.length > 400 ? `${text.slice(0, 400)}…` : text;
}

export default function App() {
  const [loaded, setLoaded] = useState<LoadedData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tab, setTab] = useState(TABS[0].id);
  const [now, setNow] = useState(Date.now());

  const [refreshing, setRefreshing] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [lastLine, setLastLine] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);
  const startedAt = useRef<number | null>(null);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    loadInitial()
      .then(async (d) => {
        setLoaded(d);
        log(`loaded ${d.source} data, generated_at=${d.generatedAt} from ${d.location}`);
        if (!autorefreshStarted && (await invoke<boolean>("autorefresh_requested"))) {
          autorefreshStarted = true;
          log("autorefresh requested (FPL_CONSULTANT_AUTOREFRESH=1): starting refresh");
          refreshRef.current();
        }
      })
      .catch((e) => {
        setLoadError(String(e));
        log(`initial load failed: ${e}`);
      });
    // a refresh may already be running (e.g. after a dev-mode reload)
    invoke<boolean>("is_refreshing").then((r) => r && setRefreshing(true));
  }, []);

  useEffect(() => {
    const unlisten = [
      listen<Progress>("refresh-progress", (e) => {
        setStage(e.payload.stage);
        if (e.payload.line) setLastLine(e.payload.line);
      }),
      listen<Done>("refresh-done", async (e) => {
        setRefreshing(false);
        setStage(null);
        startedAt.current = null;
        if (e.payload.ok) {
          try {
            const local = await loadLocal();
            if (local) setLoaded(local);
            log(`reloaded after refresh: ${local ? `${local.source} data, generated_at=${local.generatedAt}` : "no local data found"}`);
            setRefreshError(null);
            setLastResult(`Refresh finished in ${Math.round(e.payload.seconds)}s`);
          } catch (err) {
            setRefreshError(`Refresh finished but the new data couldn't be loaded: ${err}`);
          }
        } else {
          // keep showing the previous data
          log("refresh failed; still showing the previous data");
          setRefreshError(e.payload.error ?? "Refresh failed.");
          setLastResult(`Refresh failed after ${Math.round(e.payload.seconds)}s`);
        }
      }),
    ];
    return () => {
      unlisten.forEach((p) => p.then((f) => f()));
    };
  }, []);

  const onRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshError(null);
    setLastResult(null);
    setLastLine(null);
    setStage("starting");
    setRefreshing(true);
    startedAt.current = Date.now();
    try {
      await invoke("start_refresh");
    } catch (e) {
      if (String(e).includes("already running")) {
        // the backend's lock: one is genuinely running, so stay in that state
        // and let its refresh-done event finish things off
        return;
      }
      setRefreshing(false);
      setStage(null);
      setRefreshError(String(e));
    }
  }, [refreshing]);
  const refreshRef = useRef(onRefresh);
  refreshRef.current = onRefresh;

  const data = loaded?.data;
  const deadline = data ? (getPath(data, "status.next_deadline_utc") as string | undefined) : undefined;
  const nextGw = data ? getPath(data, "status.next_gameweek") : undefined;
  const elapsed = startedAt.current ? Math.floor((now - startedAt.current) / 1000) : null;
  const current = TABS.find((t) => t.id === tab) ?? TABS[0];

  return (
    <div className="app">
      <header className="statusbar">
        <div className="stat">
          <span className="label">Next deadline{nextGw !== undefined ? ` (GW${nextGw})` : ""}</span>
          <span className="value mono">{data ? countdown(deadline, now) : "…"}</span>
        </div>
        <div className="stat">
          <span className="label">Last refreshed</span>
          <span className="value">{loaded ? fmtTime(loaded.generatedAt) : "…"}</span>
        </div>
        <div className="stat grow">
          <span className="label">Data source</span>
          <span className="value small" title={loaded?.location}>
            {loaded ? (loaded.source === "local" ? "Local refresh" : "Published site") : "loading…"}
          </span>
        </div>
        <button className="refresh" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {refreshing && (
        <div className="banner info">
          <strong>Running pipeline — {stage ?? "starting"}</strong>
          {elapsed !== null && <span> · {elapsed}s</span>}
          {lastLine && <div className="mono small faded">{lastLine}</div>}
        </div>
      )}
      {refreshError && (
        <div className="banner error">
          <strong>Refresh failed — still showing the previous data.</strong>
          <pre>{refreshError}</pre>
        </div>
      )}
      {!refreshing && !refreshError && lastResult && <div className="banner ok">{lastResult}</div>}

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={t.id === tab ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.name}
          </button>
        ))}
      </nav>

      <main className="content">
        <h1>{current.name}</h1>
        {loadError && !data && <div className="banner error">Couldn't load any data: {loadError}</div>}
        {data && (
          <dl className="fields">
            {current.fields.map((f) => (
              <div key={f.path} className="field">
                <dt>
                  {f.label} <code>{f.path}</code>
                </dt>
                <dd className="mono">{renderValue(getPath(data, f.path))}</dd>
              </div>
            ))}
          </dl>
        )}
        {loaded && (
          <p className="footnote">
            Source: {loaded.source} · <code>{loaded.location}</code> · generated_at {loaded.generatedAt ?? "unknown"}
          </p>
        )}
      </main>
    </div>
  );
}
