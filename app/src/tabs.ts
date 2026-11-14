// The app's six pages, all real: Recommended team, Last gameweek, League,
// FPL overview, Planner and Analytics (pages/*.tsx). PlaceholderPage remains
// as the fallback for a tab without a page that show a few raw latest.json
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
  { id: "analytics", name: "Analytics", fields: [] },
];
