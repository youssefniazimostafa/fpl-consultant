#!/bin/bash
# Regenerates every app icon from docs/brand/:
#   xpert-app-icon-mac.svg -> the Mac app icon (src-tauri/icons, via `tauri icon`)
#   xpert-app-icon.svg     -> the iPhone app icon (Xcode AppIcon set) and the
#                             web app's favicon / home-screen icons (public-web)
# The SVGs use Barlow Condensed (bundled via @fontsource), so they're
# rendered with headless Chrome and the local font file, not the system font.
#   cd app && ./scripts/make-icons.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CH" ] || { echo "Needs Google Chrome to render the SVGs."; exit 1; }
FONT="$PWD/node_modules/@fontsource/barlow-condensed/files/barlow-condensed-latin-700-normal.woff2"
[ -f "$FONT" ] || { echo "Run npm install first (Barlow Condensed)."; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

render() {  # render SVG OUT.png -- 1024x1024, transparent outside the artwork
  cat > "$TMP/page.html" <<HTML
<!doctype html><html><head><meta charset="utf-8"><style>
@font-face { font-family: "Barlow Condensed"; font-weight: 700; src: url("file://$FONT") format("woff2"); }
html, body { margin: 0; background: transparent; } svg { display: block; width: 1024px; height: 1024px; }
</style></head><body>$(cat "$1")</body></html>
HTML
  "$CH" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files --default-background-color=00000000 \
    --force-device-scale-factor=1 --window-size=1024,1024 --virtual-time-budget=3000 \
    --user-data-dir="$TMP/prof" --screenshot="$2" "file://$TMP/page.html" >/dev/null 2>&1
  [ -s "$2" ] || { echo "rendering $1 failed"; exit 1; }
}

render ../docs/brand/xpert-app-icon-mac.svg "$TMP/mac.png"
render ../docs/brand/xpert-app-icon.svg "$TMP/full.png"

# Mac (and the other desktop sizes) from the rounded tile
npx tauri icon "$TMP/mac.png" >/dev/null
rm -rf src-tauri/icons/android src-tauri/icons/64x64.png  # not used by this project
# iPhone: full-bleed square (iOS rounds the corners itself)
npx tauri icon "$TMP/full.png" -o "$TMP/out" >/dev/null
cp "$TMP/out/ios/"*.png src-tauri/gen/apple/Assets.xcassets/AppIcon.appiconset/
cp "$TMP/out/ios/"*.png src-tauri/icons/ios/ 2>/dev/null || true
# web app: favicon / home-screen icons
sips -z 180 180 "$TMP/full.png" --out public-web/apple-touch-icon.png >/dev/null
sips -z 192 192 "$TMP/full.png" --out public-web/icon-192.png >/dev/null
sips -z 512 512 "$TMP/full.png" --out public-web/icon-512.png >/dev/null
cp "$TMP/full.png" ../docs/brand/xpert-app-icon-1024.png
cp "$TMP/mac.png" ../docs/brand/xpert-app-icon-mac-1024.png
echo "Icons updated: src-tauri/icons, the Xcode AppIcon set, public-web icons, docs/brand/*-1024.png"
