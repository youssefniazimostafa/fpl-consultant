// App chrome shared by every page: purple header with the wordmark and
// page pills, the status bar, and the phone-width bottom nav.
import type { ReactNode } from "react";
import type { TabDef } from "../tabs";

export function AppHeader({ tabs, active, onSelect }: { tabs: TabDef[]; active: string; onSelect: (id: string) => void }) {
  return (
    <header className="topbar">
      <div className="wordmark">FPL Consultant</div>
      <nav className="top-tabs">
        {tabs.map((t) => (
          <button key={t.id} className={t.id === active ? "active" : ""} onClick={() => onSelect(t.id)}>
            {t.name}
          </button>
        ))}
      </nav>
    </header>
  );
}

export function StatusBar({
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
  deadlineLabel: string;
  countdown: string;
  refreshedLabel?: string;
  refreshed: string;
  source: string;
  sourceTitle?: string;
  refreshing: boolean;
  onRefresh?: () => void; // absent in viewer mode (web / iPhone): no refresh there
  onSettings?: () => void;
  onChat?: () => void;
}) {
  return (
    <div className="statusbar">
      <div className="stat">
        <span className="label">{deadlineLabel}</span>
        <span className="value tabular">{countdown}</span>
      </div>
      <div className="stat">
        <span className="label">{refreshedLabel}</span>
        <span className="value">{refreshed}</span>
      </div>
      <div className={`stat grow hide-phone${onRefresh ? "" : " viewer-hide"}`}>
        <span className="label">Data</span>
        <span className="value small" title={sourceTitle}>
          {source}
        </span>
      </div>
      {onChat && (
        <button className="chat-btn" onClick={onChat} title="Live consultant">
          <span className="chat-btn-long">Live consultant</span>
          <span className="chat-btn-short">Ask</span>
        </button>
      )}
      {onSettings && (
        <button className="settings-btn" onClick={onSettings} aria-label="Settings" title="Settings">
          ⚙
        </button>
      )}
      {onRefresh && (
        <button className="refresh" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      )}
    </div>
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
