// App chrome shared by every page: a slim neutral top bar (wordmark, the
// PL | UCL switch, deadline, last refreshed and the tools), the game-coloured
// tab bar under it, and the phone-width bottom nav.
import type { ReactNode } from "react";
import type { TabDef } from "../tabs";
import { Wordmark } from "./Brand";

export type Game = "pl" | "ucl";

export function GameSwitch({ game, onChange }: { game: Game; onChange: (g: Game) => void }) {
  return (
    <div className="game-switch" role="tablist" aria-label="Game">
      {(["pl", "ucl"] as Game[]).map((g) => (
        <button key={g} role="tab" aria-selected={game === g} className={game === g ? "active" : ""} onClick={() => onChange(g)}>
          {g.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

export function TopBar({
  game,
  onGame,
  deadlineLabel,
  countdown,
  refreshedLabel = "Last refreshed",
  refreshed,
  source,
  sourceTitle,
  refreshing,
  onRefresh,
  onSettings,
  onChat,
}: {
  game: Game;
  onGame: (g: Game) => void;
  deadlineLabel: string;
  countdown: string;
  refreshedLabel?: string;
  refreshed: string;
  source: string;
  sourceTitle?: string;
  refreshing: boolean;
  onRefresh?: () => void; // absent in viewer mode (web / iPhone) and in UCL mode
  onSettings?: () => void;
  onChat?: () => void;
}) {
  const pl = game === "pl";
  return (
    <header className="xbar">
      <span className="show-laptop-inline"><Wordmark /></span>
      <span className="show-phone-inline"><Wordmark compact /></span>
      <GameSwitch game={game} onChange={onGame} />
      {pl && (
        <>
          <div className="stat">
            <span className="label">{deadlineLabel}</span>
            <span className="value tabular">{countdown}</span>
          </div>
          <div className="stat hide-phone">
            <span className="label">{refreshedLabel}</span>
            <span className="value">{refreshed}</span>
          </div>
          <div className={`stat grow hide-phone${onRefresh ? "" : " viewer-hide"}`}>
            <span className="label">Data</span>
            <span className="value small" title={sourceTitle}>
              {source}
            </span>
          </div>
        </>
      )}
      <span className="xbar-tools">
        {pl && onChat && (
          <button className="chat-btn" onClick={onChat} title="Live consultant">
            <span className="chat-btn-long">Live consultant</span>
            <span className="chat-btn-short">Ask</span>
          </button>
        )}
        {pl && onSettings && (
          <button className="settings-btn" onClick={onSettings} aria-label="Settings" title="Settings">
            ⚙
          </button>
        )}
        {pl && onRefresh && (
          <button className="refresh" onClick={onRefresh} disabled={refreshing} title="Refresh">
            <span className="refresh-long">{refreshing ? "Refreshing…" : "Refresh"}</span>
            <span className="refresh-short" aria-hidden>{refreshing ? "…" : "↻"}</span>
          </button>
        )}
      </span>
    </header>
  );
}

/** The game-coloured bar: only the page tabs, spread across the full width,
 * the active one underlined. */
export function TabBar({ tabs, active, onSelect }: { tabs: { id: string; name: string }[]; active: string; onSelect: (id: string) => void }) {
  return (
    <nav className="tabbar">
      {tabs.map((t) => (
        <button key={t.id} className={t.id === active ? "active" : ""} onClick={() => onSelect(t.id)}>
          {t.name}
        </button>
      ))}
    </nav>
  );
}

export function BottomNav({ tabs, active, onSelect }: { tabs: TabDef[]; active: string; onSelect: (id: string) => void }) {
  return (
    <nav className="bottom-nav">
      {tabs.map((t) => (
        <button key={t.id} className={t.id === active ? "active" : ""} onClick={() => onSelect(t.id)} aria-label={t.name}>
          <span className="dot" />
          <span className="bn-label">{t.short ?? t.name}</span>
        </button>
      ))}
    </nav>
  );
}

export function Banner({ tone, children }: { tone: "info" | "ok" | "error"; children: ReactNode }) {
  return <div className={`banner ${tone}`}>{children}</div>;
}
