#!/bin/bash
# Builds the FPL Consultant iPhone app (viewer mode) and installs it on the
# iPhone plugged into this Mac, signed with your free Apple ID (Personal
# Team). Free signing expires after 7 days: just run this again.
#
#   cd app && ./scripts/iphone-install.sh             # build + install
#   ./scripts/iphone-install.sh --simulator           # build + run in the iOS Simulator
#
# One-time setup (see README "Reinstalling the iPhone app every 7 days"):
# Xcode signed in with your Apple ID, iPhone in Developer Mode, trusted.
# Nothing personal is stored in the repo: the team ID is read from the
# "Apple Development" certificate Xcode created in your login keychain (or
# from APPLE_DEVELOPMENT_TEAM if you set it yourself).
set -euo pipefail
cd "$(dirname "$0")/.."

XCODE_DEV=/Applications/Xcode.app/Contents/Developer
[ -d "$XCODE_DEV" ] || { echo "Xcode isn't installed in /Applications. Install it from the App Store first."; exit 1; }
export DEVELOPER_DIR="$XCODE_DEV"

# Tauri's iOS build doesn't pass DEVELOPER_DIR on to xcodebuild, so when the
# Mac's active developer tools are the Command Line Tools (xcode-select -p),
# put small wrappers first on PATH that pin Xcode. (Alternative, once:
# sudo xcode-select -s /Applications/Xcode.app/Contents/Developer)
if [ "$(env -u DEVELOPER_DIR xcode-select -p)" != "$XCODE_DEV" ]; then
  SHIMS="$(mktemp -d)"
  trap 'rm -rf "$SHIMS"' EXIT
  for t in xcodebuild xcrun; do
    printf '#!/bin/sh\nexport DEVELOPER_DIR=%s\nexec /usr/bin/%s "$@"\n' "$XCODE_DEV" "$t" > "$SHIMS/$t"
    chmod +x "$SHIMS/$t"
  done
  export PATH="$SHIMS:$PATH"
fi

if [ "${1:-}" = "--simulator" ]; then
  npx tauri ios build --debug --target aarch64-sim
  APP="src-tauri/gen/apple/build/arm64-sim/FPL Consultant.app"
  DEV=$(xcrun simctl list devices booted | grep -m1 -o '[0-9A-F-]\{36\}' || true)
  if [ -z "$DEV" ]; then
    DEV=$(xcrun simctl list devices available | grep -m1 'iPhone' | grep -o '[0-9A-F-]\{36\}')
    xcrun simctl boot "$DEV"
  fi
  open -b com.apple.iphonesimulator 2>/dev/null || true  # the Simulator window, if this Xcode has one
  xcrun simctl install "$DEV" "$APP"
  xcrun simctl launch "$DEV" com.youssefmostafa.fplconsultant
  exit 0
fi

if [ -z "${APPLE_DEVELOPMENT_TEAM:-}" ]; then
  APPLE_DEVELOPMENT_TEAM=$(security find-certificate -c "Apple Development" -p 2>/dev/null \
    | openssl x509 -noout -subject -nameopt multiline 2>/dev/null \
    | awk -F' = ' '/organizationalUnitName/ {print $2; exit}' || true)
fi
if [ -z "${APPLE_DEVELOPMENT_TEAM:-}" ]; then
  cat <<'MSG'
No "Apple Development" signing certificate found.
Open Xcode > Settings > Accounts, sign in with your Apple ID, select your
"(Personal Team)" and click "Manage Certificates..." > "+" > "Apple Development".
Then run this script again.
MSG
  exit 1
fi
export APPLE_DEVELOPMENT_TEAM

echo "Building and installing on your iPhone (keep it unlocked and plugged in)..."
npx tauri ios run --release
