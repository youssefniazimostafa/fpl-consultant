import type { Latest } from "../data";
import { Card } from "../components/Card";
import { Shirt } from "../components/Shirt";

// Planner: the plan for the next few gameweeks at a glance. Everything
// comes from latest.json's `planner` section (fpl/planner.py); its first
// week is the ONE decision, checked by fpl/check_consistency.py (rule 10).
// Laptop: weeks across the top as pills, a row per item (as in the
// mock-up). Phone: one card per week (reads better at 390 px than six
// squeezed columns), and the fixture grid scrolls sideways.

type Raw = Record<string, any>;

function dateLabel(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

function MiniPlayer({ p, out }: { p: Raw; out?: boolean }) {
  return (
    <span className={`pl-mini${out ? " out" : ""}`}>
      <Shirt teamCode={p.team_code} teamShort={p.team_short} goalkeeper={p.position === "GKP"} size={26} dimmed={out} title={p.web_name} />
      <span className="pl-mini-name">{p.web_name}</span>
    </span>
  );
}

function TransferBlock({ t, now }: { t: Raw; now: boolean }) {
  if (!t.planned) return <div className="pblock empty">Not planned yet</div>;
  return (
    <div className={`pblock ${t.save ? "save" : "move"}${now ? " now" : ""}`}>
      {t.save ? (
        <span className="pblock-title">Save transfer</span>
      ) : (
        t.moves.map((m: Raw, i: number) => (
          <div key={i} className="pl-move">
            <MiniPlayer p={m.out} out />
            <span className="pl-arrow">→</span>
            <MiniPlayer p={m.in} />
          </div>
        ))
      )}
      <span className="pblock-sub">
        {t.free_transfers != null ? `${t.free_transfers} free transfer${t.free_transfers === 1 ? "" : "s"}` : ""}
        {t.hits ? ` · take a ${t.hit_cost}-point hit` : ""}
      </span>
      {t.hits > 0 && t.hit_reason && <span className="pblock-warn">{t.hit_reason}</span>}
    </div>
  );
}

function ChipBlock({ c }: { c: Raw }) {
  return (
    <div className="pblock chip">
      <span className="pblock-title">{c.label}</span>
      <span className="pblock-sub">about {Math.round(c.gain)} extra points</span>
    </div>
  );
}

function FixtureCell({ cell }: { cell: Raw }) {
  if (!cell || cell.blank) return <div className="fx blank">No game</div>;
  return (
    <div className={`fx-wrap${cell.double ? " double" : ""}`}>
      {cell.games.map((g: Raw, i: number) => (
        <div key={i} className={`fx d${g.difficulty}`} title={`FPL difficulty ${g.difficulty} of 5`}>
          {g.opponent} ({g.home ? "H" : "A"})
        </div>
      ))}
      {cell.double && <span className="fx-tag">Double</span>}
    </div>
  );
}

export function PlannerPage({ data }: { data: Latest }) {
  const p: Raw | undefined = data.planner;
  if (!p?.available) {
    return (
      <Card title="Planner">
        <p className="muted">{p?.reason ?? "No plan in this update yet. Press Refresh to run a fresh update."}</p>
      </Card>
    );
  }
  const weeks: Raw[] = p.weeks;
  const chipsAt = (g: number) => (p.chips as Raw[]).filter((c) => c.gameweek === g);
  const used = (p.chips as Raw[]).filter((c) => c.status === "used");
  const later = (p.chips as Raw[]).filter((c) => c.status === "later");
  const tFor = (g: number) => (p.transfers as Raw[]).find((t) => t.gameweek === g);

  const chipNotes = (
    <>
      {used.length > 0 && <div className="pgrid-note">Used: {used.map((c) => `${c.label} (${c.note.replace(/^Used /, "")})`).join(", ")}</div>}
      {later.length > 0 && <div className="pgrid-note">Later: {later.map((c) => c.label).join(", ")}</div>}
    </>
  );

  return (
    <div className="planner-page">
      <div className="page-title">
        <h1>Planner</h1>
        <span className="muted">
          GW{weeks[0].gameweek} to GW{weeks[weeks.length - 1].gameweek}
        </span>
      </div>
      <p className="planner-note">{p.note}</p>

      {/* laptop: weeks across, a row per item */}
      <div className="show-laptop">
        <Card className="planner-card">
          <div className="pgrid" style={{ ["--weeks" as string]: weeks.length }}>
            <div className="pgrid-label" />
            {weeks.map((w, i) => (
              <div key={w.gameweek} className={`pweek${i === 0 ? " now" : ""}`}>
                <b>GW{w.gameweek}</b>
                <small>{dateLabel(w.deadline)}</small>
              </div>
            ))}

            <div className="pgrid-label">
              <b>Transfers</b>
            </div>
            {weeks.map((w, i) => (
              <div key={w.gameweek} className="ptrack">
                {tFor(w.gameweek) && <TransferBlock t={tFor(w.gameweek)!} now={i === 0} />}
              </div>
            ))}

            <div className="pgrid-label">
              <b>Chips</b>
              {chipNotes}
            </div>
            {weeks.map((w) => (
              <div key={w.gameweek} className="ptrack">
                {chipsAt(w.gameweek).map((c) => (
                  <ChipBlock key={c.chip} c={c} />
                ))}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* phone: one card per week */}
      <div className="show-phone">
        <div className="pweeks">
          {weeks.map((w, i) => (
            <Card
              key={w.gameweek}
              title={`GW${w.gameweek}${i === 0 ? " · this week" : ""}`}
              aside={dateLabel(w.deadline)}
              className={`pweek-card${i === 0 ? " now" : ""}`}
            >
              {tFor(w.gameweek) && <TransferBlock t={tFor(w.gameweek)!} now={i === 0} />}
              {chipsAt(w.gameweek).map((c) => (
                <ChipBlock key={c.chip} c={c} />
              ))}
            </Card>
          ))}
          <div className="planner-chipnotes">{chipNotes}</div>
        </div>
      </div>

      <Card title="Your players' fixtures" aside="FPL's difficulty: green easy, pink hard" className="planner-fixtures">
        <div className="fxgrid-scroll">
          <div className="fxgrid" style={{ ["--weeks" as string]: weeks.length }}>
            <div className="fxgrid-club fxgrid-headcell" />
            {weeks.map((w) => (
              <div key={w.gameweek} className="fxgrid-head">GW{w.gameweek}</div>
            ))}
            {(p.fixtures as Raw[]).map((r) => (
              <Row key={r.team_short} r={r} weeks={weeks} />
            ))}
          </div>
        </div>
        <div className="fx-legend">
          <span className="fx d2">Easy</span>
          <span className="fx d3">Medium</span>
          <span className="fx d4">Hard</span>
          <span className="fx d5">Very hard</span>
          <span className="fx blank">No game</span>
        </div>
      </Card>
    </div>
  );
}

function Row({ r, weeks }: { r: Raw; weeks: Raw[] }) {
  return (
    <>
      <div className="fxgrid-club">
        <Shirt teamCode={r.team_code} teamShort={r.team_short} size={26} title={r.team_short} />
        <span>
          <b>{r.team_short}</b>
          <small>{r.players.join(", ")}</small>
        </span>
      </div>
      {weeks.map((w) => (
        <div key={w.gameweek} className="fxgrid-cell">
          <FixtureCell cell={r.cells[String(w.gameweek)]} />
        </div>
      ))}
    </>
  );
}

