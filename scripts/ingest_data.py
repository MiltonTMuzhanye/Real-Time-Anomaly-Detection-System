"""Clone the NAB repository into data/raw/nab/."""
import subprocess
import sys
from pathlib import Path

NAB_REPO = "https://github.com/numenta/NAB.git"
DEST = Path("data/raw/nab")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"NAB already present at {DEST}")
        return 0
    try:
        subprocess.run(["git", "clone", "--depth", "1", NAB_REPO, str(DEST)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e}", file=sys.stderr)
        return 1
    print(f"NAB downloaded to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())