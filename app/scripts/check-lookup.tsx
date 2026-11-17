// Render check: the Scout page's Player lookup report for every player in a
// latest.json, server-side, failing on the first exception.
//   node scripts/check-lookup.mjs [path/to/latest.json]
// (built by scripts/check-lookup.sh; defaults to ../site/data/latest.json)
import "./check-lookup-env"; // must come first
import { readFileSync } from "node:fs";
import { renderToString } from "react-dom/server";
import { PlayerLookup, Report, lookupRows } from "../src/components/PlayerLookup";

const path = process.argv[2] ?? "../site/data/latest.json";
const data = JSON.parse(readFileSync(path, "utf8"));
const lookup = data.overview?.lookup;
if (!lookup) {
  console.log(`${path}: no overview.lookup section (the page hides the lookup)`);
  process.exit(0);
}
renderToString(<PlayerLookup lookup={lookup} />);
const rows = lookupRows(lookup);
const failures: string[] = [];
for (const row of rows) {
  try {
    renderToString(<Report row={row} rep={lookup.reports?.[row.id] ?? {}} lookup={lookup} />);
  } catch (e) {
    failures.push(`${row.id} ${row.name} (${row.club} ${row.pos}): ${(e as Error).message}`);
  }
}
// Also a report with every field missing, and a row with nulls.
const empty = { id: -1, name: "Nobody", club: "", team_code: null, pos: "MID", price: null, xp: null, form: null, sel: null, own: false } as any;
for (const [label, rep] of [["empty report", {}], ["null fields", { fx: null, recent: null, season: null, drv: null, swap: {}, change: null }]] as const) {
  try {
    renderToString(<Report row={empty} rep={rep as any} lookup={{}} />);
  } catch (e) {
    failures.push(`${label}: ${(e as Error).message}`);
  }
}
console.log(`${path}: rendered ${rows.length} player reports, ${failures.length} failed`);
failures.slice(0, 20).forEach((f) => console.log("  " + f));
process.exit(failures.length ? 1 : 0);
