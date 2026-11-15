// UCL mode data: data/ucl/latest.json, written by the UCL pipeline
// (ucl/pipeline.py) and committed by its workflow (.github/workflows/ucl.yml).
// It is read straight from the repo on GitHub, so the PL site and its Pages
// deploy are untouched. Desktop: the newer of the local copy (engine, last
// UCL refresh) and the published one. Both modes keep the last good copy in
// this device's storage for offline use.
import { invoke } from "@tauri-apps/api/core";
import { DESKTOP } from "../platform";

export const UCL_URL = "https://raw.githubusercontent.com/youssefniazimostafa/fpl-consultant/main/data/ucl/latest.json";
const CACHE_KEY = "xpert:ucl-latest";

export type Ucl = Record<string, any>;
export interface UclLoaded {
  data: Ucl;
  source: "local" | "published" | "saved";
  generatedAt: string | null;
}

async function published(): Promise<Ucl> {
  const res = await fetch(`${UCL_URL}?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  const data = JSON.parse(text) as Ucl;
  try {
    localStorage.setItem(CACHE_KEY, text);
  } catch {
    /* storage full or blocked: still show the data */
  }
  return data;
}

function saved(): Ucl | null {
  try {
    const t = localStorage.getItem(CACHE_KEY);
    return t ? (JSON.parse(t) as Ucl) : null;
  } catch {
    return null;
  }
}

export async function loadUcl(): Promise<UclLoaded> {
  const local = DESKTOP
    ? await invoke<{ json: string } | null>("load_local_ucl").then((f) => (f ? (JSON.parse(f.json) as Ucl) : null)).catch(() => null)
    : null;
  let pub: Ucl | null = null;
  let err: unknown = null;
  try {
    pub = await published();
  } catch (e) {
    err = e;
  }
  const t = (d: Ucl | null) => (d?.generated_at ? new Date(d.generated_at).getTime() : 0);
  if (local && t(local) >= t(pub)) return { data: local, source: "local", generatedAt: local.generated_at ?? null };
  if (pub) return { data: pub, source: "published", generatedAt: pub.generated_at ?? null };
  const s = saved();
  if (s) return { data: s, source: "saved", generatedAt: s.generated_at ?? null };
  throw new Error(`Couldn't load the Champions League data (${err}) and there's no saved copy yet.`);
}
