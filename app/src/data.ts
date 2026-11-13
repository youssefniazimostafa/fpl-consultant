// Loading latest.json: the local one (from the last local refresh, in the
// app's engine copy) if it exists, otherwise the published GitHub Pages one.
import { invoke } from "@tauri-apps/api/core";

export const PUBLISHED_URL =
  "https://youssefniazimostafa.github.io/fpl-consultant/data/latest.json";

// latest.json is large and its panels will each get a proper type when the
// real pages are built; for the placeholder shell a loose record is enough.
export type Latest = Record<string, any>;

export interface LoadedData {
  data: Latest;
  source: "local" | "published";
  location: string; // file path or URL
  generatedAt: string | null; // latest.json's own generated_at
}

interface LocalData {
  json: string;
  path: string;
  modified_ms: number;
}

export async function loadLocal(): Promise<LoadedData | null> {
  const local = await invoke<LocalData | null>("load_local_data");
  if (!local) return null;
  const data = JSON.parse(local.json) as Latest;
  return { data, source: "local", location: local.path, generatedAt: data.generated_at ?? null };
}

export async function loadPublished(): Promise<LoadedData> {
  const res = await fetch(`${PUBLISHED_URL}?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Published data: HTTP ${res.status}`);
  const data = (await res.json()) as Latest;
  return { data, source: "published", location: PUBLISHED_URL, generatedAt: data.generated_at ?? null };
}

/** Local if there is one, otherwise published. */
export async function loadInitial(): Promise<LoadedData> {
  return (await loadLocal()) ?? (await loadPublished());
}

/** Value at a dotted path, e.g. "status.next_deadline_utc" or "xi.players.0.web_name". */
export function getPath(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((cur, key) => {
    if (cur === null || cur === undefined) return undefined;
    return (cur as Record<string, unknown>)[key];
  }, obj);
}
