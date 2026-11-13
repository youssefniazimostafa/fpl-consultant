import type { Latest } from "../data";
import { Card, Pill } from "../components/Card";
import { Bench, Pitch } from "../components/Pitch";
import { Shirt } from "../components/Shirt";
import {
  captainView,
  chipsView,
  teamView,
  whyThisTeam,
  type CaptainView,
  type ChipsView,
  type PlayerView,
  type TeamView,
} from "../model/team";

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
      <div className="swap-meta">
        {[p.teamShort, money(p.price)].filter(Boolean).join(" · ") || " "}
      </div>
    </div>
  );
}

function TransfersCard({ team }: { team: TeamView }) {
  const ft = team.freeTransfers;
  return (
    <Card title="Transfers" aside={`GW${team.gameweek}`} className="transfers-card">
      {team.roll ? (
        <div className="roll">
          <div className="big">Save your free transfer this week</div>
          <p>
            No transfer is worth making. Leave your team as it is
            {ft != null ? ` and you'll have ${Math.min(ft + 1, 5)} free transfers next week.` : "."}
          </p>
        </div>
      ) : (
        team.transfers.map((t, i) => (
          <div className="swap" key={i}>
            <SwapSide p={t.out} out />
            <SwapArrow />
            <SwapSide p={t.in} />
          </div>
        ))
      )}
      <div className="facts">
        {ft != null && (
          <Pill tone="purple">
            {ft} free transfer{ft === 1 ? "" : "s"}
            {!team.roll ? ` · using ${Math.min(team.transfers.length, ft)}` : ""}
          </Pill>
        )}
        {!team.roll &&
          (team.hitsTaken > 0 ? (
            <Pill tone="pink">Take a {team.hitsTaken * 4}-point hit</Pill>
          ) : (
            <Pill tone="neon">No points hit needed</Pill>
          ))}
      </div>
      {!team.roll && team.hitsTaken > 0 && (
        <p className="note warn">Take a points hit: {team.hitReason ?? "the extra transfer is worth more than the 4 points it costs."}</p>
      )}
      {!team.roll &&
        team.outNews.map((n) => (
          <p key={n.name} className={`note${n.urgency === "forced" ? " warn" : ""}`}>
            <b>{n.name}:</b> {n.news}
          </p>
        ))}
    </Card>
  );
}

function CaptainCard({ cap }: { cap: CaptainView }) {
  const c = cap.captain;
  return (
    <Card title="Captain" className="captain-card">
      <div className="captain-row">
        <div className="tile-shirt">
          <Shirt teamCode={c.teamCode} teamShort={c.teamShort} goalkeeper={c.position === "GKP"} size={52} title={c.name} />
          <span className="badge badge-c">C</span>
        </div>
        <div>
          <div className="big">Captain {c.name}</div>
          <div className="muted">
            {[c.teamShort, c.opponent].filter(Boolean).join(" · ")}
            {c.projected != null && ` · ${c.projected.toFixed(1)} projected points, about ${(c.projected * 2).toFixed(1)} as captain`}
          </div>
          {cap.vice && <div className="muted">Vice-captain: {cap.vice.name}</div>}
        </div>
      </div>
      <p className="note">
        {cap.rivalsBackYours
          ? "Most of your rivals are likely to captain him too, so there's little risk of losing ground on the armband."
          : cap.rivalPick && cap.possibleLoss != null
            ? `If your rivals captain ${cap.rivalPick.name} instead, you could lose about ${cap.possibleLoss.toFixed(1)} point${
                cap.possibleLoss.toFixed(1) === "1.0" ? "" : "s"
              } on him${cap.possibleLoss < 1 ? ". They're projected almost the same, so it's a small risk, but it grows if he has a big game." : "."}`
            : "No clear favourite captain among your rivals this week."}
      </p>
    </Card>
  );
}

function ChipsCard({ chips }: { chips: ChipsView }) {
  return (
    <Card title="Chips" className="chips-card">
      <div className="big">
        {chips.playNow ? `Play your ${chips.playNow.label} this week` : "No chip this week"}
      </div>
      <p className="muted">
        {chips.playNow
          ? `This is its best week: worth about ${chips.playNow.gain.toFixed(1)} extra points.`
          : chips.next
            ? `Best upcoming: ${chips.next.label} in GW${chips.next.gameweek} (about ${chips.next.gain.toFixed(1)} extra points).`
            : "None of your chips is worth playing in the next few weeks."}
      </p>
      <div className="chip-row">
        {chips.chips.map((c) => (
          <div key={c.key} className={`chip ${c.available ? "available" : "used"}`} title={c.note}>
            <span className="chip-name">{c.label}</span>
            <span className="chip-note">{c.note}</span>
          </div>
        ))}
      </div>
      {chips.urgent.length > 0 && chips.deadlineGw && (
        <p className="note warn">
          Use {chips.urgent.join(" and ")} by GW{chips.deadlineGw} or lose {chips.urgent.length > 1 ? "them" : "it"}.
        </p>
      )}
    </Card>
  );
}

export function TeamPage({ data }: { data: Latest }) {
  const team = teamView(data);
  if (!team) {
    return (
      <Card title="Your team">
        <p className="muted">
          No recommended team in this data yet. Press Refresh to run the full update, or check that your team ID is set in
          config.json.
        </p>
      </Card>
    );
  }
  const cap = captainView(data, team);
  const chips = chipsView(data, team.gameweek);
  const why = whyThisTeam(data, team, cap, chips);

  return (
    <div className="team-page">
      <div className="team-title page-title">
        <h1>Your team</h1>
        <span className="muted">Gameweek {team.gameweek} · copy this team</span>
      </div>
      <div className="team-pitch">
        <Pitch players={team.xi} />
      </div>
      <div className="team-bench">
        <Bench players={team.bench} />
      </div>
      <aside className="team-side">
        <TransfersCard team={team} />
        {cap && <CaptainCard cap={cap} />}
        <ChipsCard chips={chips} />
      </aside>
      <Card title="Why this team" className="why-card">
        <p className="why">{why.join(" ")}</p>
      </Card>
    </div>
  );
}
