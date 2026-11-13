// Loading latest.json, in order of preference:
//   1. the local one from the last local refresh (in the app's engine copy)
//   2. the published GitHub Pages one (saved to disk on every successful fetch)
//   3. that saved copy of the published one -- so the app works offline
import { invoke } from "@tauri-apps/api/core";

export const PUBLISHED_URL =
  "https://youssefniazimostafa.github.io/fpl-consultant/data/latest.json";

// latest.json is large; each page reads its panels through its own typed
// view (see model/), so a loose record is enough here.
export type Latest = Record<string, any>;

export type Source = "local" | "published" | "saved-published";

export interface LoadedData {
  data: Latest;
  source: Source;
  location: string; // file path or URL
  generatedAt: string | null; // latest.json's own generated_at
}

interface FileData {
  json: string;
  path: string;
  modified_ms: number;
}

function fromFile(f: FileData, source: Source): LoadedData {
  const data = JSON.parse(f.json) as Latest;
  return { data, source, location: f.path, generatedAt: data.generated_at ?? null };
}

export async function loadLocal(): Promise<LoadedData | null> {
  const local = await invoke<FileData | null>("load_local_data");
  return local ? fromFile(local, "local") : null;
}

export async function loadPublished(): Promise<LoadedData> {
  if (await invoke<boolean>("offline_simulated").catch(() => false)) {
    throw new Error("offline (simulated with FPL_CONSULTANT_SIMULATE_OFFLINE=1)");
  }
  const res = await fetch(`${PUBLISHED_URL}?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Published data: HTTP ${res.status}`);
  const text = await res.text();
  const data = JSON.parse(text) as Latest;
  invoke("save_published_cache", { json: text }).catch(() => {});
  return { data, source: "published", location: PUBLISHED_URL, generatedAt: data.generated_at ?? null };
}

export async function loadSavedPublished(): Promise<LoadedData | null> {
  const saved = await invoke<FileData | null>("load_published_cache");
  return saved ? fromFile(saved, "saved-published") : null;
}

/** Local if there is one, otherwise published, otherwise the saved published copy. */
export async function loadInitial(): Promise<LoadedData> {
  const local = await loadLocal().catch(() => null);
  if (local) return local;
  try {
    return await loadPublished();
  } catch (e) {
    const saved = await loadSavedPublished().catch(() => null);
    if (saved) return saved;
    throw e;
  }
}

export function sourceLabel(s: Source): string {
  return s === "local" ? "Your last refresh" : s === "published" ? "Published site" : "Saved copy (offline)";
}

/** Value at a dotted path, e.g. "status.next_deadline_utc" or "xi.players.0.web_name". */
export function getPath(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((cur, key) => {
    if (cur === null || cur === undefined) return undefined;
    return (cur as Record<string, unknown>)[key];
  }, obj);
}
