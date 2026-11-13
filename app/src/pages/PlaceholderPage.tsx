import { Card } from "../components/Card";
import { getPath, type Latest } from "../data";
import type { TabDef } from "../tabs";

function renderValue(value: unknown): string {
  if (value === undefined) return "(missing)";
  if (typeof value === "string") return value;
  const text = JSON.stringify(value);
  return text.length > 400 ? `${text.slice(0, 400)}…` : text;
}

/** Stand-in for a page that hasn't been built yet: a few raw fields. */
export function PlaceholderPage({ tab, data }: { tab: TabDef; data: Latest }) {
  return (
    <div className="placeholder-page">
      <div className="page-title">
        <h1>{tab.name}</h1>
        <span className="muted">Coming soon · raw data preview</span>
      </div>
      <div className="placeholder-grid">
        {tab.fields.map((f) => (
          <Card key={f.path} title={f.label} aside={<code>{f.path}</code>}>
            <div className="raw">{renderValue(getPath(data, f.path))}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}
