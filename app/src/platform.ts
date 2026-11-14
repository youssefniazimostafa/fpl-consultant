// One front end, two modes, fixed at build time (vite.config.ts):
//   "desktop" -- the Mac app: local refresh, Live consultant chat, Settings
//                (Keychain), shirts cached by the Rust backend.
//   "viewer"  -- the home-screen web app (GitHub Pages /app/) and the iPhone
//                app: read-only, reads the published latest.json, no keys,
//                no Python, no backend commands.
// Because the mode is a build-time constant, the desktop-only code (chat,
// Settings, refresh) is dropped from viewer builds entirely.

declare const __APP_MODE__: "desktop" | "viewer";

export const VIEWER = __APP_MODE__ === "viewer";
export const DESKTOP = !VIEWER;

/** Running inside a Tauri webview (Mac or iPhone app), not a browser. */
export const inTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
