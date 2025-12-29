#!/usr/bin/env python3
"""
Move each file up one directory while prefixing it with its former folder name.

Example: A/B/C/d_file.pdf -> A/B/C_d_file.pdf
"""
import argparse
import logging
import os
import sys


def move_files(roots, dry_run: bool = False) -> int:
    exit_code = 0

    for root in roots:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            print(f"Root not found: {root}", file=sys.stderr)
            exit_code = 1
            continue

        for dirpath, _, filenames in os.walk(root_abs, topdown=False):
            if os.path.samefile(dirpath, root_abs):
                continue  # Do not move files out of the root itself

            parent = os.path.dirname(dirpath)
            dir_name = os.path.basename(dirpath)

            for filename in filenames:
                src = os.path.join(dirpath, filename)
                new_name = f"{dir_name}_{filename}"
                dst = os.path.join(parent, new_name)

                if os.path.exists(dst):
                    print(f"Skip (exists): {dst}")
                    continue

                print(f"{'DRY ' if dry_run else ''}Move {src} -> {dst}")
                if not dry_run:
                    os.rename(src, dst)
                    logging.info(filename)
                    logging.info(f"\t{new_name}")

    return exit_code


def main() -> int:
    p = argparse.ArgumentParser(
        description="Move files up one directory, prefixing with their former folder name."
    )
    p.add_argument(
        "roots",
        nargs="+",
        help="Root directories to process (e.g. one or more paths to A)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without changing anything",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return move_files(args.roots, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
