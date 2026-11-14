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
#
# How: `tauri ios build` (signed archive + export to an .ipa), then
# `xcrun devicectl` installs it on the phone and launches it. Not
# `tauri ios run`: in tauri-cli 2.12.4 it deletes its temporary
# ExportOptions.plist before the export step ("Couldn't load
# -exportOptionsPlist ... no such file", EXPORT FAILED, code 70); fixed
# upstream in tauri PR #14712 but not released yet.
# Pick a phone with IPHONE_DEVICE="<name or UDID>" if several are plugged in.
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
  for t in xcodebuild xcrun; do
    printf '#!/bin/sh\nexport DEVELOPER_DIR=%s\nexec /usr/bin/%s "$@"\n' "$XCODE_DEV" "$t" > "$SHIMS/$t"
    chmod +x "$SHIMS/$t"
  done
  export PATH="$SHIMS:$PATH"
fi

# Tauri writes your team ID into the generated Xcode project while building;
# take it out again afterwards so it never ends up in the repo.
PBXPROJ="src-tauri/gen/apple/fpl-consultant.xcodeproj/project.pbxproj"
cleanup() {
  [ -n "${SHIMS:-}" ] && rm -rf "$SHIMS"
  [ -f "$PBXPROJ" ] && sed -i '' '/DEVELOPMENT_TEAM = /d' "$PBXPROJ"
  return 0
}
trap cleanup EXIT

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

# The plugged-in iPhone (or IPHONE_DEVICE), from devicectl's device list.
DEVLIST="$(mktemp)"
xcrun devicectl list devices --quiet --json-output "$DEVLIST" >/dev/null
DEVICE=$(python3 - "$DEVLIST" "${IPHONE_DEVICE:-}" <<'PY'
import json, sys
devs = json.load(open(sys.argv[1]))["result"]["devices"]
want = sys.argv[2]
for d in devs:
    hw, props = d.get("hardwareProperties", {}), d.get("deviceProperties", {})
    if hw.get("reality") != "physical" or hw.get("platform") != "iOS":
        continue
    if want and want not in (props.get("name"), hw.get("udid"), d.get("identifier")):
        continue
    if d.get("connectionProperties", {}).get("tunnelState") == "unavailable":
        continue
    print(hw.get("udid") or d.get("identifier")); print(props.get("name", "iPhone")); break
PY
)
rm -f "$DEVLIST"
if [ -z "$DEVICE" ]; then
  echo "No iPhone found. Plug it in with a cable, unlock it, tap Trust if asked, and run this again."
  exit 1
fi
UDID=$(echo "$DEVICE" | sed -n 1p); NAME=$(echo "$DEVICE" | sed -n 2p)

IPA="src-tauri/gen/apple/build/arm64/FPL Consultant.ipa"
rm -f "$IPA"
echo "Building the signed app for $NAME (a few minutes)..."
npx tauri ios build
[ -f "$IPA" ] || { echo "The build didn't produce $IPA."; exit 1; }

echo "Installing on $NAME (keep it unlocked)..."
xcrun devicectl device install app --device "$UDID" "$IPA"

echo "Opening the app..."
if ! LAUNCH=$(xcrun devicectl device process launch --device "$UDID" com.youssefmostafa.fplconsultant 2>&1); then
  if echo "$LAUNCH" | grep -q "explicitly trusted\|invalid code signature"; then
    cat <<MSG

Installed on $NAME, but iOS won't open it until you trust your developer
certificate (first install only). On the iPhone:
  Settings > General > VPN & Device Management > under "Developer App" tap
  your Apple ID > Trust "Apple Development: ..." > Trust.
Then open FPL Consultant from the home screen.
MSG
    exit 0
  fi
  echo "$LAUNCH" | tail -5
  echo "Installed, but couldn't open it automatically: open FPL Consultant on the phone."
  exit 0
fi
echo "Done: FPL Consultant is installed and open on $NAME."
