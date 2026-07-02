#!/usr/bin/env python3
"""Convert gnb_metrics_ws_logger JSONL archive → ML training CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gnb_metrics_io import jsonl_to_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Convert gNB JSONL metrics log to CSV")
    p.add_argument("--input", type=Path, required=True, help="Source .jsonl from ws logger")
    p.add_argument("--output", type=Path, required=True, help="Destination .csv")
    p.add_argument("--rnti", default=None, help="Optional RNTI filter")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Missing input: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    n = jsonl_to_csv(args.input, args.output, rnti_filter=args.rnti)
    print(f"Wrote {n} rows to {args.output}")
    if n == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
