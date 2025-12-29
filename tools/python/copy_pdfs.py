#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys


def prefix_before_glob(pattern: str) -> str:
    specials = ["*", "?", "["]
    idx = min((pattern.find(c) for c in specials if c in pattern), default=-1)
    if idx == -1:
        return os.path.dirname(pattern) or "."
    # Walk back to the previous slash so we keep whole path segments
    cut = pattern.rfind("/", 0, idx)
    return pattern[:cut] if cut != -1 else "."


def collect_pdfs(patterns):
    results = []
    for pattern in patterns:
        base = prefix_before_glob(pattern)
        for src in glob.iglob(pattern, recursive=True):
            if src.lower().endswith(".pdf") and os.path.isfile(src):
                results.append((base, src))
    return results


def print_progress(done: int, total: int, width: int = 40) -> None:
    filled = int((done / total) * width) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {done}/{total}")
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Copy PDFs from glob patterns, preserving structure."
    )
    p.add_argument("dest", help="Destination directory (created if missing)")
    p.add_argument(
        "patterns",
        nargs="+",
        help="Glob patterns (use quotes to avoid shell expansion)",
    )
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    pdfs = collect_pdfs(args.patterns)
    total = len(pdfs)
    if total == 0:
        print("No PDFs matched.")
        return 1

    print(f"Copying {total} PDFs to {args.dest}")
    for idx, (base, src) in enumerate(pdfs, start=1):
        rel = os.path.relpath(src, base)
        dst = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print_progress(idx, total)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
