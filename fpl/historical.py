"""Historical data for Step 7's model training -- sparsely clones just the
``data/`` directory of two reference repos into ``data/external/``:

- `vaastav/Fantasy-Premier-League`: past seasons' gameweek-by-gameweek
  player data.
- `olbauday/FPL-Core-Insights`: match-level stats joined with Elo.

Both repos are large (100-200MB of full history), so this uses a
blobless, sparse checkout restricted to their ``data/`` subdirectory rather
than a full clone. Safe to re-run: clones if missing, pulls if already
present.

Run with `uv run python -m fpl.historical`.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "external"

REPOS = {
    "fantasy-premier-league": "https://github.com/vaastav/Fantasy-Premier-League.git",
    "fpl-core-insights": "https://github.com/olbauday/FPL-Core-Insights.git",
}


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def sparse_clone(name: str, url: str) -> None:
    dest = EXTERNAL_DIR / name
    if dest.exists():
        print(f"  {name}: already cloned, pulling latest...")
        run("git", "pull", "--depth", "1", cwd=dest)
        return

    print(f"  {name}: cloning data/ only (blobless + sparse checkout)...")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    run(
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, name,
        cwd=EXTERNAL_DIR,
    )
    run("git", "sparse-checkout", "set", "data", cwd=dest)


def main() -> None:
    print("Fetching historical reference repos for Step 7 (data/ subtree only)...")
    for name, url in REPOS.items():
        try:
            sparse_clone(name, url)
        except subprocess.CalledProcessError as exc:
            print(f"  {name}: git failed ({exc}); skipping.")
    print(f"Done. See {EXTERNAL_DIR}")


if __name__ == "__main__":
    main()
