import type { Latest } from "../data";
import { Card } from "../components/Card";
import { Bench, Pitch } from "../components/Pitch";
import type { PlayerView } from "../model/team";
import { BreakdownCards, BreakdownTable, HINDSIGHT_NOTE, Stat, toView } from "./LastGameweekPage";

// Team of the week: the best squad anyone could legally have picked for
// the latest finished gameweek (fpl/hindsight.py, the same MILP as the
// forward plan, scored with real points). Same layout and components as
// the Last gameweek page.

type Raw = Record<string, any>;

export function TeamOfTheWeekPage({ data }: { data: Latest }) {
  const t: Raw | undefined = data.team_of_the_week;
  if (!t?.available) {
    return (
      <Card title="Team of the week">
        <p className="muted">{t?.reason ?? "No finished gameweek in this update yet. Press Refresh to run a fresh update."}</p>
      </Card>
    );
  }
  const xi: PlayerView[] = t.starting_xi.map((c: Raw) => ({ ...toView(c), opponent: `£${c.price.toFixed(1)}m` }));
  const bench: PlayerView[] = t.bench.map((c: Raw) => ({ ...toView(c, true), opponent: `£${c.price.toFixed(1)}m`, benchLabel: null }));

  return (
    <div className="lastgw-page">
      <div className="page-title">
        <h1>Team of the week</h1>
        <span className="muted">
          GW{t.gameweek} · {t.formation} · £{t.cost.toFixed(1)}m
        </span>
      </div>
      <p className="planner-note">
        {t.note} {HINDSIGHT_NOTE}
      </p>

      <div className="lg-strip">
        <Stat dark label="Team of the week" value={String(t.total)} sub={t.captain_name ? `captain ${t.captain_name}` : undefined} />
        <Stat label="Your points" value={String(t.you)} sub={`${t.total - t.you} behind the best`} />
        <Stat label="The consultant's team" value={t.consultant != null ? String(t.consultant) : "–"} sub={t.consultant != null ? undefined : "not tracked yet"} />
        <Stat label="FPL average" value={t.fpl_average != null ? String(t.fpl_average) : "–"} />
      </div>

      <div className="lg-grid">
        <div className="lg-left">
          <Pitch players={xi} legend="Points under each name · price on the left" />
          <Bench players={bench} />
        </div>
        <div className="lg-right">
          <Card title="How predictable was it?" className="lg-summary">
            {t.has_expected ? (
              <p className="why">
                Before the deadline this team was expected to score about <b>{Math.round(t.expected_total)}</b> points; it scored{" "}
                <b>{t.total}</b>. The difference is the luck and surprises no projection could see.
              </p>
            ) : (
              <p className="why">No saved pre-deadline expectations for this gameweek, so only actual points are shown.</p>
            )}
            {t.dream_team && (
              <p className="note muted">
                FPL's own Team of the Week scored {t.dream_team.total} (no captain doubling), but it ignores the budget and the 3-per-club limit.
              </p>
            )}
          </Card>
        </div>
      </div>

      <Card title="Where every point came from" aside="FPL's own points" className="lg-breakdown">
        <div className="show-laptop">
          <BreakdownTable players={t.players} />
        </div>
        <div className="show-phone">
          <BreakdownCards players={t.players} />
        </div>
      </Card>
    </div>
  );
}
