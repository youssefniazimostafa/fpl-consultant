import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
// @ts-expect-error type error without @types/node package
import process from "node:process";
const host = process.env.TAURI_DEV_HOST;

// Build modes (see src/platform.ts):
//   default / development -> desktop (the Mac app, unchanged)
//   --mode web -> viewer, published on GitHub Pages at /fpl-consultant/app/
//                 (into ../site/app at deploy time; see the workflow)
//   --mode ios -> viewer, bundled into the iPhone app (dist-ios)
const PAGES_BASE = "/fpl-consultant/app/";

/** Home-screen install tags, web build only. */
function webAppTags(): Plugin {
  return {
    name: "web-app-tags",
    transformIndexHtml: () => [
      { tag: "link", attrs: { rel: "manifest", href: `${PAGES_BASE}manifest.webmanifest` }, injectTo: "head" },
      { tag: "link", attrs: { rel: "apple-touch-icon", href: `${PAGES_BASE}apple-touch-icon.png` }, injectTo: "head" },
      { tag: "link", attrs: { rel: "icon", type: "image/png", href: `${PAGES_BASE}icon-192.png` }, injectTo: "head" },
      { tag: "meta", attrs: { name: "apple-mobile-web-app-capable", content: "yes" }, injectTo: "head" },
      { tag: "meta", attrs: { name: "mobile-web-app-capable", content: "yes" }, injectTo: "head" },
      { tag: "meta", attrs: { name: "apple-mobile-web-app-title", content: "FPL Consultant" }, injectTo: "head" },
      { tag: "meta", attrs: { name: "apple-mobile-web-app-status-bar-style", content: "black-translucent" }, injectTo: "head" },
    ],
  };
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const viewer = mode === "web" || mode === "ios";
  return {
    plugins: [react(), ...(mode === "web" ? [webAppTags()] : [])],
    define: { __APP_MODE__: JSON.stringify(viewer ? "viewer" : "desktop") },
    base: mode === "web" ? PAGES_BASE : "/",
    publicDir: mode === "web" ? "public-web" : "public",
    build: { outDir: mode === "web" ? "dist-web" : mode === "ios" ? "dist-ios" : "dist", emptyOutDir: true },

    // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
    //
    // 1. prevent Vite from obscuring rust errors
    clearScreen: false,
    // 2. tauri expects a fixed port, fail if that port is not available
    server: {
      port: 1420,
      strictPort: true,
      host: host || false,
      hmr: host
        ? {
            protocol: "ws",
            host,
            port: 1421,
          }
        : undefined,
      watch: {
        // 3. tell Vite to ignore watching `src-tauri`
        ignored: ["**/src-tauri/**"],
      },
    },
  };
});
