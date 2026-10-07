"""Generate the OptoNet benchmark dataset.

Usage:
    python scripts/01_generate_data.py [--config configs/default.yaml] [--out data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config
from optonet.data import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = generate(cfg, Path(args.out) if args.out else None)
    print(f"dataset ready at {out}")


if __name__ == "__main__":
    main()
