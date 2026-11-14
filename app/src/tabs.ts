// The app's five pages. "Your team" and "League" are real pages
// (pages/TeamPage.tsx, pages/LeaguePage.tsx); the other three are still placeholders that show a few raw latest.json
// fields to prove the data flows. The real pages replace these one by one.

export interface TabDef {
  id: string;
  name: string;
  short?: string; // label under the dot in the phone bottom nav
  fields: { label: string; path: string }[];
}

export const TABS: TabDef[] = [
  { id: "team", name: "Your team", short: "Team", fields: [] },
  { id: "league", name: "League", fields: [] },
  {
    id: "overview",
    name: "FPL overview",
    short: "Overview",
    fields: [
      { label: "Current gameweek", path: "status.current_gameweek" },
      { label: "Next gameweek", path: "status.next_gameweek" },
      { label: "Next deadline (UTC)", path: "status.next_deadline_utc" },
      { label: "Top projected player", path: "top_xp.players.0" },
      { label: "News flags", path: "news_flags.flags" },
    ],
  },
  {
    id: "planner",
    name: "Planner",
    fields: [
      { label: "Next planned transfer", path: "transfer_plan.weeks.0" },
      { label: "Best simulated plan", path: "simulation.best" },
      { label: "Season projection (leader)", path: "season_simulation.standings.0" },
    ],
  },
  {
    id: "analytics",
    name: "Analytics",
    fields: [
      { label: "Backtest MAE (our model)", path: "model_accuracy.latest.mae_xp" },
      { label: "Backtest MAE (FPL ep_next)", path: "model_accuracy.latest.mae_fpl_xp" },
      { label: "Live accuracy, latest GW", path: "live_accuracy.latest_gameweek" },
      { label: "Data-source warnings", path: "source_health.warnings" },
    ],
  },
];
