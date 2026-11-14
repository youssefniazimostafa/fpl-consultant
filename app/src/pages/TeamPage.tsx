import type { Latest } from "../data";
import { Card, Pill } from "../components/Card";
import { Bench, Pitch } from "../components/Pitch";
import { Shirt } from "../components/Shirt";
import { readDecision, type Decision, type PlayerView } from "../model/team";

// Every panel below renders `decision` (fpl/decision.py) as published --
// none of them works out a recommendation of its own.

function money(p: number | null) {
  return p != null ? `£${p.toFixed(1)}m` : "";
}

function SwapArrow() {
  return (
    <svg className="swap-arrow" width="56" height="20" viewBox="0 0 56 20" aria-label="replaced by">
      <path d="M2 10 H48" stroke="var(--neon)" strokeWidth="4" strokeLinecap="round" />
      <path d="M40 2 L52 10 L40 18" fill="none" stroke="var(--neon)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SwapSide({ p, out }: { p: PlayerView; out?: boolean }) {
  return (
    <div className={`swap-side${out ? " out" : ""}`}>
      <Shirt teamCode={p.teamCode} teamShort={p.teamShort} goalkeeper={p.position === "GKP"} size={52} dimmed={out} title={p.name} />
      <div className="swap-name">{p.name}</div>
      <div className="swap-meta">{[p.teamShort, money(p.price)].filter(Boolean).join(" · ") || " "}</div>
    </div>
  );
}

function TransfersCard({ d }: { d: Decision }) {
  const ft = d.freeTransfers;
  const roll = d.action === "roll";
  return (
    <Card title="Transfers" aside={`GW${d.gameweek}`} className="transfers-card">
      {roll ? (
        <div className="roll">
          <div className="big">Save your free transfer this week</div>
          <p>
            Leave your team as it is
            {ft.nextWeek != null ? ` and you'll have ${ft.nextWeek} free transfer${ft.nextWeek === 1 ? "" : "s"} next week.` : "."}
          </p>
        </div>
      ) : (
        d.transfers.map((t, i) => (
          <div className="swap" key={i}>
            <SwapSide p={t.out} out />
            <SwapArrow />
            <SwapSide p={t.in} />
          </div>
        ))
      )}
      <div className="facts">
        {ft.available != null && (
          <Pill tone="purple">
            {roll
              ? `${ft.available} free transfer${ft.available === 1 ? "" : "s"}`
              : `Uses ${ft.used} of ${ft.available} free · ${ft.left} left`}
          </Pill>
        )}
        {!roll && (d.hits > 0 ? <Pill tone="pink">Take a {d.hitCost}-point hit</Pill> : <Pill tone="neon">No points hit needed</Pill>)}
      </div>
      {!roll && d.hits > 0 && (
        <p className="note warn">Take a points hit: {d.hitReason ?? "the extra transfer is worth more than the points it costs."}</p>
      )}
      {!roll &&
        d.transferNews.map((n) => (
          <p key={n.name} className={`note${n.urgency === "forced" ? " warn" : ""}`}>
            <b>{n.name}:</b> {n.news}
          </p>
        ))}
    </Card>
  );
}

function CaptainCard({ d }: { d: Decision }) {
  const c = d.captain!;
  const card = d.xi.find((p) => p.id === c.id);
  return (
    <Card title="Captain" className="captain-card">
      <div className="captain-row">
        <div className="tile-shirt">
          <Shirt teamCode={card?.teamCode} teamShort={card?.teamShort} goalkeeper={card?.position === "GKP"} size={52} title={c.name} />
          <span className="badge badge-c">C</span>
        </div>
        <div>
          <div className="big">Captain {c.name}</div>
          <div className="muted">
            {[card?.teamShort, card?.opponent].filter(Boolean).join(" · ")}
            {` · ${c.projected.toFixed(1)} projected points, about ${(c.projected * 2).toFixed(1)} as captain`}
          </div>
          {c.viceName && <div className="muted">Vice-captain: {c.viceName}</div>}
        </div>
      </div>
      {c.riskText && <p className="note">{c.riskText}</p>}
    </Card>
  );
}

function ChipsCard({ d }: { d: Decision }) {
  const chip = d.chip;
  return (
    <Card title="Chips" className="chips-card">
      <div className="big">{chip.play ? `Play your ${chip.play.label} this week` : "No chip this week"}</div>
      <p className="muted">
        {chip.play
          ? `This is its best week: about ${chip.play.gain.toFixed(0)} extra points.`
          : chip.upcoming
            ? `Best upcoming: ${chip.upcoming.label} in GW${chip.upcoming.gameweek} (about ${chip.upcoming.gain.toFixed(0)} extra points).`
            : "None of your chips is worth playing in the next few weeks."}
      </p>
      <div className="chip-row">
        {chip.statuses.map((c) => (
          <div key={c.chip} className={`chip ${c.available ? "available" : "used"}`} title={c.note}>
            <span className="chip-name">{c.label}</span>
            <span className="chip-note">{c.note}</span>
          </div>
        ))}
      </div>
      {chip.urgent.length > 0 && chip.useBy && (
        <p className="note warn">
          Use {chip.urgent.join(" and ")} by GW{chip.useBy} or lose {chip.urgent.length > 1 ? "them" : "it"}.
        </p>
      )}
    </Card>
  );
}

export function TeamPage({ data }: { data: Latest }) {
  const d = readDecision(data);
  if (!d) {
    return (
      <Card title="Your team">
        <p className="muted">
          This data was made by an older version of the update and doesn't include this week's recommendation yet.
          Press Refresh to run a fresh update.
        </p>
      </Card>
    );
  }
  return (
    <div className="team-page">
      <div className="team-title page-title">
        <h1>Your team</h1>
        <span className="muted">Gameweek {d.gameweek} · copy this team</span>
      </div>
      <div className="team-pitch">
        <Pitch players={d.xi} />
      </div>
      <div className="team-bench">
        <Bench players={d.bench} />
      </div>
      <aside className="team-side">
        <TransfersCard d={d} />
        {d.captain && <CaptainCard d={d} />}
        <ChipsCard d={d} />
      </aside>
      <Card title="Why this team" className="why-card">
        <p className="why">{d.why.join(" ")}</p>
      </Card>
    </div>
  );
}
