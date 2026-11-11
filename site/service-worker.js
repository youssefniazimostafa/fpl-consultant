// Step 12 PWA support: makes the dashboard installable and usable with a
// flaky or absent connection. The static shell (markup/CSS/JS/icons) is
// cache-first since it only changes on deploy; `data/latest.json` is
// network-first with a cache fallback, so a connected visitor always sees
// the freshest pipeline output, while an offline one still sees the last
// data this device successfully fetched instead of a blank page.

const SHELL_CACHE = "fpl-consultant-shell-v1";
const DATA_CACHE = "fpl-consultant-data-v1";

const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./methods.html",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== SHELL_CACHE && key !== DATA_CACHE)
          .map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;

  if (url.pathname.endsWith("/data/latest.json")) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(DATA_CACHE).then((cache) => cache.put(event.request, copy));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
