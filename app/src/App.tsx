import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getPath, loadInitial, loadLocal, sourceLabel, type LoadedData } from "./data";
import { TABS } from "./tabs";
import { AppHeader, Banner, BottomNav, StatusBar } from "./components/Chrome";
import { Settings } from "./components/Settings";
import { TeamPage } from "./pages/TeamPage";
import { PlaceholderPage } from "./pages/PlaceholderPage";
import { LeaguePage } from "./pages/LeaguePage";
import { OverviewPage } from "./pages/OverviewPage";
import { LastGameweekPage } from "./pages/LastGameweekPage";
import { PlannerPage } from "./pages/PlannerPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { TeamOfTheWeekPage } from "./pages/TeamOfTheWeekPage";

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
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
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

export default function App() {
  const [loaded, setLoaded] = useState<LoadedData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tab, setTab] = useState(TABS[0].id);
  const [startRival, setStartRival] = useState<number | null>(null);
  const [showSettings, setShowSettings] = useState(false);
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
        const auto = await invoke<number>("autorefresh_requested");
        if (!autorefreshStarted && auto > 0) {
          autorefreshStarted = true;
          log(`autorefresh requested (FPL_CONSULTANT_AUTOREFRESH=${auto}): starting refresh`);
          refreshRef.current();
          if (auto === 2) {
            log("autorefresh: starting a second refresh while the first runs (guard test)");
            refreshRef.current();
          }
        }
      })
      .catch((e) => {
        setLoadError(String(e));
        log(`initial load failed: ${e}`);
      });
    invoke<[string | null, number | null, boolean]>("start_view")
      .then(([t, r, scroll]) => {
        if (t && TABS.some((x) => x.id === t)) setTab(t);
        if (r != null) setStartRival(r);
        if (scroll) {
          setTimeout(() => {
            const c = document.querySelector(".content");
            if (c) c.scrollTop = c.scrollHeight;
          }, 2500);
        }
      })
      .catch(() => {});
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
      log(`refresh not started: ${e}`);
      if (String(e).includes("A refresh is already running.")) {
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
      <AppHeader tabs={TABS} active={tab} onSelect={setTab} />
      <div className="rule" />
      <StatusBar
        deadlineLabel={`GW${nextGw ?? "?"} deadline`}
        countdown={data ? countdown(deadline, now) : "…"}
        refreshed={loaded ? fmtTime(loaded.generatedAt) : "…"}
        source={loaded ? sourceLabel(loaded.source) : "loading…"}
        sourceTitle={loaded?.location}
        refreshing={refreshing}
        onRefresh={onRefresh}
        onSettings={() => setShowSettings(true)}
      />
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}

      {refreshing && (
        <Banner tone="info">
          <strong>Updating your data: {stage ?? "starting"}</strong>
          {elapsed !== null && <span> · {elapsed}s</span>}
          {lastLine && <div className="small faded">{lastLine}</div>}
        </Banner>
      )}
      {refreshError && (
        <Banner tone="error">
          <strong>Refresh failed. Still showing the previous data.</strong>
          <pre>{refreshError}</pre>
        </Banner>
      )}
      {!refreshing && !refreshError && lastResult && <Banner tone="ok">{lastResult}</Banner>}

      <main className="content">
        {loadError && !data && <Banner tone="error">Couldn't load any data: {loadError}</Banner>}
        {data &&
          (current.id === "team" ? (
            <TeamPage data={data} />
          ) : current.id === "league" ? (
            <LeaguePage key={startRival ?? "league"} data={data} startRival={startRival} />
          ) : current.id === "lastgw" ? (
            <LastGameweekPage data={data} />
          ) : current.id === "totw" ? (
            <TeamOfTheWeekPage data={data} />
          ) : current.id === "planner" ? (
            <PlannerPage data={data} />
          ) : current.id === "analytics" ? (
            <AnalyticsPage data={data} />
          ) : current.id === "overview" ? (
            <OverviewPage data={data} />
          ) : (
            <PlaceholderPage tab={current} data={data} />
          ))}
        {loaded && (
          <p className="footnote">
            Data: {sourceLabel(loaded.source)} · generated {fmtTime(loaded.generatedAt)}
          </p>
        )}
      </main>

      <BottomNav tabs={TABS} active={tab} onSelect={setTab} />
    </div>
  );
}
