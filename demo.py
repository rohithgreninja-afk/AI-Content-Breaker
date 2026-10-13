"""
demo.py — AI Content Breaker demo runner

Edit demo_input.txt, then:

    python demo.py --text      run the full pipeline on demo_input.txt
    python demo.py --eval      evaluate both scorers on the held-out test split
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import main  # noqa: E402

DEMO_FILE = os.path.join(os.path.dirname(__file__), "demo_input.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("--text", "--eval"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--eval":
        main(["--eval"])
    else:
        if not os.path.exists(DEMO_FILE):
            print(f"demo_input.txt not found at {DEMO_FILE}")
            sys.exit(1)
        main([DEMO_FILE])
