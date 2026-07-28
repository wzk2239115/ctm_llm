#!/usr/bin/env python3
"""Dedupe offline_compare CSV polluted by stale proc*.csv files.

BUG: paper/run_offline_compare_cluster.sh merges all `offline_compare_proc*.csv`
via glob WITHOUT clearing stale files first. If the script is run twice (e.g.
once with SEEDS="0 1 2 3 4" then once with SEEDS="0 1 2"), and some procs fail
on the second run, their proc*.csv from the FIRST run survives and gets merged
in again. Result: (env, backbone, seed) tuples for seeds 0-2 appear TWICE with
identical success_rate (deterministic) but different elapsed_s, biasing any
downstream mean.

FIX (data): this script drops duplicate (env, backbone, seed) rows, keeping the
first occurrence. Aborts with a report if duplicates have CONFLICTING
success_rate / expert_success / n_steps values (would indicate genuine
non-determinism or a config drift, not just a re-run).

FIX (script): run_offline_compare_cluster.sh now does
`rm -f "$OUT_DIR"/offline_compare_proc*.csv` before launching workers, so this
dedupe should only ever be needed for legacy CSVs.

Usage:
    python scripts/dedupe_offline_compare.py csv_data/offline_compare_full_0707.csv
    python scripts/dedupe_offline_compare.py csv_data/offline_compare_full_0707.csv --inplace
    python scripts/dedupe_offline_compare.py csv_data/offline_compare_full.csv -o csv_data/offline_compare_full_dedup.csv
"""

import argparse
import csv
import sys
from pathlib import Path

KEY_FIELDS = ["env", "backbone", "seed"]
# Fields that MUST agree across duplicates (deterministic); if they differ, we
# have a real conflict and should not silently drop.
CHECK_FIELDS = ["success_rate", "expert_success", "n_episodes", "n_steps"]


def dedupe(in_path: Path, out_path: Path) -> int:
    with in_path.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    seen: dict[tuple, dict] = {}
    kept: list[dict] = []
    conflicts: list[str] = []
    dropped = 0

    for r in rows:
        key = tuple(r[f] for f in KEY_FIELDS)
        if key not in seen:
            seen[key] = r
            kept.append(r)
            continue
        dropped += 1
        prev = seen[key]
        for fld in CHECK_FIELDS:
            if prev[fld] != r[fld]:
                conflicts.append(
                    f"  {key}: {fld} {prev[fld]!r} vs {r[fld]!r}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    print(f"[dedupe] {in_path}")
    print(f"  input  : {len(rows)} rows")
    print(f"  output : {len(kept)} rows ({out_path})")
    print(f"  dropped: {dropped} duplicate rows "
          f"({dropped / len(rows) * 100:.1f}% of input)")
    if conflicts:
        print(f"  [WARN] {len(conflicts)} CONFLICTS in deterministic fields "
              "(not silently dropped, investigate):")
        for c in conflicts[:20]:
            print(c)
        if len(conflicts) > 20:
            print(f"  ... +{len(conflicts) - 20} more")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="offline_compare_full*.csv to clean")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output path (default: <input>_dedup.csv)")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the input file (backup saved as .bak)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    if args.inplace:
        bak = args.input.with_suffix(args.input.suffix + ".bak")
        bak.write_text(args.input.read_text())
        out = args.input
        print(f"[dedupe] backup -> {bak}")
    else:
        out = args.output or args.input.with_name(
            args.input.stem + "_dedup" + args.input.suffix
        )

    sys.exit(dedupe(args.input, out))


if __name__ == "__main__":
    main()
