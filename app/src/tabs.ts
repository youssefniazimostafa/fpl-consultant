// The app's six pages. "Recommended team", "Last gameweek", "League",
// "FPL overview" and "Planner" are real pages (pages/TeamPage.tsx,
// LastGameweekPage.tsx, LeaguePage.tsx, OverviewPage.tsx, PlannerPage.tsx);
// "Analytics" is still a placeholder that show a few raw latest.json
// fields to prove the data flows. The real pages replace these one by one.

export interface TabDef {
  id: string;
  name: string;
  short?: string; // label under the dot in the phone bottom nav
  fields: { label: string; path: string }[];
}

export const TABS: TabDef[] = [
  { id: "team", name: "Recommended team", short: "Team", fields: [] },
  { id: "lastgw", name: "Last gameweek", short: "Last GW", fields: [] },
  { id: "league", name: "League", fields: [] },
  { id: "overview", name: "FPL overview", short: "Overview", fields: [] },
  { id: "planner", name: "Planner", fields: [] },
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
