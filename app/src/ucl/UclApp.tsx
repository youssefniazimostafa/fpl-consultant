// UCL mode: everything the Champions League game shows, kept apart from the
// PL pages. Its own data (data/ucl/latest.json), tabs, tour, settings note
// and Live consultant tools (ucl/chat.py); the PL state in App is untouched.
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { DESKTOP } from "../platform";
import { Banner, BottomNav, TabBar, TopBar, type Game } from "../components/Chrome";
import { ChatPanel } from "../components/ChatPanel";
import { Tour, storageGet, storageSet, type Step } from "../components/Tour";
import type { TabDef } from "../tabs";
import { loadUcl, type UclLoaded } from "./data";
import { UclAnalyticsPage, UclBetweenPage, UclLastPage, UclScoutPage, UclTeamPage } from "./pages";
import { UclPlannerPage } from "./planner";
import "./ucl.css";

export const UCL_TABS: TabDef[] = [
  { id: "team", name: "Recommended team", short: "Team", fields: [] },
  { id: "between", name: "Between days", short: "Days", fields: [] },
  { id: "last", name: "Last matchday", short: "Last", fields: [] },
  { id: "scout", name: "Scout", short: "Scout", fields: [] },
  { id: "planner", name: "Planner", short: "Plan", fields: [] },
  { id: "analytics", name: "Analytics", short: "Stats", fields: [] },
];

const UCL_TOUR_KEY = "xpert:ucl-tour-done";
const UCL_TAB_KEY = "xpert:ucl-tab";
const STARTERS = ["Why this captain?", "Why these defenders?", "Should I play a chip now?", "Who plays on the second day?"];

const STEPS: Step[] = [
  {
    targets: [".xbar"],
    title: "Champions League mode",
    text: "XPert for UCL Fantasy. The bar shows the countdown to this matchday's deadline (the first kick-off). Switch back to PL at any time.",
  },
  {
    tab: "team",
    targets: [".ucard-pitch", ".content"],
    title: "Recommended team",
    text: "The 15 to pick within the budget and club limit, with the day each player plays, the captain (C) and vice (V), the plan and the chips.",
  },
  {
    tab: "between",
    targets: [".content"],
    title: "Between days",
    text: "After the first match day, XPert says whether to make up to 4 substitutions or switch the captain, and only when it clearly beats keeping the automatic substitutions.",
  },
  {
    tab: "last",
    targets: [".ustats", ".content"],
    title: "Last matchday",
    text: "How the team did, against an estimate of the game average and the best possible team.",
  },
  {
    tab: "scout",
    targets: [".uscout", ".content"],
    title: "Scout",
    text: "Top-10 lists: expected points, captains, value, form, differentials, most selected and injury news.",
  },
  {
    tab: "planner",
    targets: [".uplan", ".content"],
    title: "Planner",
    text: "The plan up to the next unlimited-transfer window: transfers each matchday against the bar, free transfers, club limit and budget per stage, the chips and every club's fixtures. On the Mac, Your plan lets you enter your own and get it reviewed.",
  },
  {
    tab: "analytics",
    targets: [".content"],
    title: "Analytics",
    text: "Points against the average each matchday, how the model did when we replayed past matchdays, and every club's knockout chances. Tap any ? for a plain explanation.",
  },
  {
    targets: [".chat-btn"],
    title: "Live consultant",
    text: "Ask about the Champions League advice in plain words. It uses your Anthropic key from Settings in PL mode.",
    desktopOnly: true,
  },
];

function countdown(iso: string | undefined, now: number): string {
  if (!iso) return "–";
  const ms = new Date(iso).getTime() - now;
  if (Number.isNaN(ms)) return iso;
  if (ms <= 0) return "deadline passed";
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${d > 0 ? `${d}d ` : ""}${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

const fmt = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "unknown";

function UclSettings({ onClose, onTour, teamId, onSaved }: { onClose: () => void; onTour: () => void; teamId: string; onSaved: (t: string) => void }) {
  const [v, setV] = useState(teamId);
  const [msg, setMsg] = useState<string | null>(null);
  return (
    <div className="setup-backdrop" role="dialog" aria-modal="true" aria-label="UCL settings">
      <div className="setup-card ucl-settings">
        <h2>Champions League settings</h2>
        <label>
          Your UCL Fantasy team ID
          <input value={v} onChange={(e) => setV(e.target.value)} placeholder="optional" />
        </label>
        <p className="setup-note">
          UEFA only shows your team after you log in, so XPert can't read it yet. Until a team is saved, XPert shows the squad to create and
          gives its between-days advice for that squad. Keys (odds, Anthropic) are shared with PL mode: set them in PL Settings.
        </p>
        {msg && <p className="note warn">{msg}</p>}
        <div className="tour-actions">
          <button className="tour-skip" onClick={onTour}>
            Take the tour
          </button>
          <span className="tour-grow" />
          <button className="tour-skip" onClick={onClose}>
            Close
          </button>
          <button
            className="tour-next"
            onClick={() =>
              invoke("set_ucl_team", { teamId: v })
                .then(() => {
                  onSaved(v.trim());
                  onClose();
                })
                .catch((e) => setMsg(String(e)))
            }
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

export function UclApp({ game, onGame, now }: { game: Game; onGame: (g: Game) => void; now: number }) {
  const [loaded, setLoaded] = useState<UclLoaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTabState] = useState<string>(() => {
    const q = new URLSearchParams(location.search).get("tab");
    const saved = q ?? storageGet(UCL_TAB_KEY);
    return UCL_TABS.some((t) => t.id === saved) ? (saved as string) : "team";
  });
  const setTab = useCallback((t: string) => {
    setTabState(t);
    storageSet(UCL_TAB_KEY, t);
  }, []);
  const [showTour, setShowTour] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [teamId, setTeamId] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const reload = useCallback(() => loadUcl().then(setLoaded, (e) => setError(String(e))), []);
  useEffect(() => {
    reload();
    if (DESKTOP) invoke<Record<string, any>>("get_setup").then((s) => setTeamId(String(s?.ucl_team_id ?? ""))).catch(() => {});
  }, [reload]);
  useEffect(() => {
    if (loaded && !storageGet(UCL_TOUR_KEY) && !new URLSearchParams(location.search).get("notour")) setShowTour(true);
  }, [loaded]);
  useEffect(() => {
    if (!DESKTOP) return;
    const un = [
      listen<{ stage: string }>("refresh-progress", (e) => setStage(e.payload.stage)),
      listen<{ ok: boolean; error: string | null }>("refresh-done", (e) => {
        setRefreshing(false);
        setStage(null);
        if (e.payload.ok) reload();
        else setRefreshError(e.payload.error ?? "Refresh failed.");
      }),
    ];
    return () => un.forEach((p) => p.then((f) => f()));
  }, [reload]);

  const onRefresh = async () => {
    if (refreshing) return;
    setRefreshError(null);
    setRefreshing(true);
    try {
      await invoke("start_refresh", { game: "ucl" });
    } catch (e) {
      setRefreshing(false);
      setRefreshError(String(e));
    }
  };

  const d = loaded?.data;
  return (
    <div className="app ucl">
      <TopBar
        game={game}
        onGame={onGame}
        deadlineLabel={`MD${d?.md ?? "?"} deadline`}
        countdown={d ? countdown(d.deadline, now) : "…"}
        refreshedLabel={DESKTOP ? "Last refreshed" : "Updated"}
        refreshed={loaded ? fmt(loaded.generatedAt) : "…"}
        source={loaded ? (loaded.source === "local" ? "Your last refresh" : loaded.source === "published" ? "Published" : "Saved copy (offline)") : "loading…"}
        refreshing={refreshing}
        onRefresh={DESKTOP ? onRefresh : undefined}
        onSettings={DESKTOP ? () => setShowSettings(true) : undefined}
        onChat={DESKTOP ? () => setShowChat(true) : undefined}
      />
      <TabBar tabs={UCL_TABS} active={tab} onSelect={setTab} />
      {refreshing && (
        <Banner tone="info">
          <strong>Updating the Champions League data: {stage ?? "starting"}</strong>
        </Banner>
      )}
      {refreshError && (
        <Banner tone="error">
          <strong>Refresh failed. Still showing the previous data.</strong>
          <pre>{refreshError}</pre>
        </Banner>
      )}
      {loaded?.source === "saved" && <Banner tone="info">Offline: showing the saved Champions League data from {fmt(loaded.generatedAt)}.</Banner>}
      {DESKTOP && showChat && <ChatPanel game="ucl" starters={STARTERS} onClose={() => setShowChat(false)} />}
      {DESKTOP && showSettings && (
        <UclSettings
          teamId={teamId}
          onSaved={setTeamId}
          onClose={() => setShowSettings(false)}
          onTour={() => {
            setShowSettings(false);
            setShowTour(true);
          }}
        />
      )}
      {showTour && <Tour steps={STEPS} doneKey={UCL_TOUR_KEY} onTab={setTab} onClose={() => setShowTour(false)} />}
      <main className="content ucl-content">
        {error && !d && <Banner tone="error">{error}</Banner>}
        {!d && !error && <p className="umuted upad">Loading the Champions League data…</p>}
        {d &&
          (tab === "team" ? (
            <UclTeamPage d={d} teamSaved={!!teamId} />
          ) : tab === "between" ? (
            <UclBetweenPage d={d} />
          ) : tab === "last" ? (
            <UclLastPage d={d} />
          ) : tab === "scout" ? (
            <UclScoutPage d={d} />
          ) : tab === "planner" ? (
            <UclPlannerPage d={d} />
          ) : (
            <UclAnalyticsPage d={d} />
          ))}
        {d && (
          <p className="footnote ufoot">
            UCL data updated {fmt(loaded!.generatedAt)} ·{" "}
            <button className="link" onClick={() => setShowTour(true)}>
              Take the tour
            </button>
          </p>
        )}
      </main>
      <BottomNav tabs={UCL_TABS} active={tab} onSelect={setTab} />
    </div>
  );
}
