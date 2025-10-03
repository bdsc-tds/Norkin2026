import os
import argparse
from pathlib import Path


def rename_subpaths(root_dir: str, source_subpath: str, dest_name: str, dry_run: bool = True):
    """
    Recursively finds a specific subdirectory path and renames the final component.

    For example, find '.../CRC/hImmune_v1_mm/0YRI' and rename '0YRI' to 'OYRI'.

    Args:
        root_dir (str): The top-level directory to start the search from.
        source_subpath (str): The relative path to find (e.g., 'CRC/hImmune_v1_mm/0YRI').
        dest_name (str): The new name for the final directory in the path.
        dry_run (bool): If True, only print what would be changed without
                        actually renaming anything.
    """
    print(f"Starting search in: {root_dir}")
    print(f"Looking for sub-paths ending in: '{source_subpath}'")
    print(f"Will rename final component to: '{dest_name}'")
    if dry_run:
        print("--- DRY RUN MODE --- (No changes will be made)")
    print("-" * 20)

    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Root directory '{root_dir}' not found.")
        return

    # Use rglob to find all paths that match the source_subpath.
    # This is efficient as it uses the filesystem's own matching capabilities.
    found_count = 0
    renamed_count = 0

    for source_path in root_path.rglob(source_subpath):
        # rglob finds files and directories. Ensure we are only renaming directories.
        if not source_path.is_dir():
            continue

        found_count += 1
        # Use .with_name() to create the destination path by replacing the final component.
        dest_path = source_path.with_name(dest_name)

        print(f"Found: {source_path}")

        # Safety check: ensure the destination doesn't already exist
        if dest_path.exists():
            print(f"  [WARNING] Destination '{dest_path}' already exists. Skipping.")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would rename to: {dest_path}")
        else:
            try:
                print(f"  Renaming to: {dest_path}")
                source_path.rename(dest_path)
                renamed_count += 1
            except OSError as e:
                print(f"  [ERROR] Could not rename. Reason: {e}")

    print("-" * 20)
    if dry_run:
        print(f"--- DRY RUN COMPLETE ---")
        print(f"Found {found_count} paths that would be renamed.")
    else:
        print(f"--- RENAME COMPLETE ---")
        print(f"Successfully renamed {renamed_count} out of {found_count} found paths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to find a specific subdirectory path and rename its final component.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("root_directory", help="The top-level directory to start searching from.")
    parser.add_argument(
        "source_subpath",
        help="The full subdirectory path to find, relative to the root or other directories.\n"
        "Example: 'CRC/hImmune_v1_mm/0YRI'",
    )
    parser.add_argument(
        "dest_name", help="The new name for the final directory in the source_subpath.\nExample: 'OYRI'"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be renamed without actually making any changes."
    )
    parser.add_argument(
        "--execute",
        action="store_false",
        dest="dry_run",
        help="Execute the renaming. Use with caution. Always perform a dry run first.",
    )

    # By default, dry_run is True unless --execute is specified
    parser.set_defaults(dry_run=True)

    args = parser.parse_args()

    rename_subpaths(args.root_directory, args.source_subpath, args.dest_name, args.dry_run)
