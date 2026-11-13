// The one place the app's colours, radii and type live. theme.css reads
// these as CSS variables (applyTheme() writes them onto :root at startup),
// so components use var(--purple) etc. and TS code can import the same
// values (e.g. for SVG fills).

export const colors = {
  purple: "#37003c", // header, bars, primary text
  purpleSoft: "#5a2a60", // inactive pills on the header
  neon: "#00ff85", // highlights, active tab, "good" signals
  neonSoft: "#c9ffe3", // pale green rule / tinted backgrounds
  cyan: "#04f5ff", // accent
  pink: "#e90052", // accent, warnings ("points hit", flagged players)
  pinkSoft: "#fde3ec",
  lilac: "#f4f0f5", // card backgrounds
  lilacDeep: "#e7dfe9", // borders, pill tracks on cards
  grey: "#b7a9ba", // used chips, dimmed shirts
  white: "#ffffff", // page
  text: "#37003c",
  textMuted: "#6f5a73",
  pitch: "#3b934d", // pitch stripes
  pitchAlt: "#52ab63",
  pitchLine: "rgba(255,255,255,0.65)",
} as const;

export const radius = {
  sm: "8px",
  md: "14px",
  lg: "20px",
  pill: "999px",
} as const;

export const font = {
  family: '"Inter Variable", Inter, -apple-system, BlinkMacSystemFont, sans-serif',
} as const;

/** Width (px) below which the app switches to the stacked phone layout. */
export const PHONE_BREAKPOINT = 720;

function kebab(s: string) {
  return s.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
}

export function applyTheme(root: HTMLElement = document.documentElement) {
  for (const [k, v] of Object.entries(colors)) root.style.setProperty(`--${kebab(k)}`, v);
  for (const [k, v] of Object.entries(radius)) root.style.setProperty(`--radius-${k}`, v);
  root.style.setProperty("--font", font.family);
}
