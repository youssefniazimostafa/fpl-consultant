#!/bin/sh
# Builds and runs scripts/check-lookup.tsx (render every player's lookup report).
# Usage: ./scripts/check-lookup.sh [path/to/latest.json]
set -e
cd "$(dirname "$0")/.."
mkdir -p node_modules/.cache
out="node_modules/.cache/check-lookup.mjs"
npx rolldown scripts/check-lookup.tsx --platform node --format esm --file "$out" >/dev/null
node "$out" "${1:-../site/data/latest.json}"
