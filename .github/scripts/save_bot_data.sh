#!/usr/bin/env bash
# Commit and push the pipeline's data files from a GitHub Actions run.
#
# Why this isn't a plain `git push`: actions/checkout checks out the commit
# that *triggered* the run. Runs are serialized (concurrency group "pages"),
# so a run queued behind another one starts from a commit that's already out
# of date once the earlier run pushes its own data -- and a plain push is then
# rejected (non-fast-forward), failing the run and skipping the Pages deploy.
#
# So: commit (or exit 0 if there's nothing new), then pull --rebase and push,
# retrying up to MAX_ATTEMPTS times. On a rebase conflict, `-X theirs` keeps
# *this* run's version of the conflicting hunk/file: during a rebase "theirs"
# is the commit being replayed, i.e. this run's data, which is always the
# newer of the two because this run's pipeline finished later.
set -euo pipefail

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
BRANCH="${BRANCH:-main}"
COMMIT_MESSAGE="Prediction log + accuracy track record + GW snapshot archive"

git config user.name "fpl-bot"
git config user.email "fpl-bot@users.noreply.github.com"

mkdir -p data/recommendations  # frozen weekly recommendations (fpl/consultant.py)
git add data/predictions data/recommendations data/accuracy data/manual/override_log.json data/league data/last_good
if git diff --cached --quiet; then
  echo "No data changes to commit -- nothing to push."
  exit 0
fi
git commit -m "$COMMIT_MESSAGE"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if git pull --rebase -X theirs origin "$BRANCH" && git push origin "HEAD:$BRANCH"; then
    echo "Pushed data commit (attempt $attempt/$MAX_ATTEMPTS)."
    exit 0
  fi
  git rebase --abort >/dev/null 2>&1 || true
  echo "Push attempt $attempt/$MAX_ATTEMPTS failed."
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    sleep $((attempt * 5))
  fi
done

echo "::error::Could not push the data commit after $MAX_ATTEMPTS attempts."
exit 1
