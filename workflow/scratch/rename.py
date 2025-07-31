import os
import argparse
import pathlib


def rename_subfolders_glob(root_dir, target_parent, old_str, new_str, dry_run=True):
    """
    Finds all directories named 'target_parent' and then recursively renames
    subfolders within them containing old_str. This is much faster than
    walking the entire directory tree.

    Args:
        root_dir (str): The top-level directory to start from.
        target_parent (str): The name of the parent folder to constrain the search to.
        old_str (str): The string to search for in folder names.
        new_str (str): The string to replace old_str with.
        dry_run (bool): If True, only print changes without executing them.
    """
    root_path = pathlib.Path(root_dir).resolve()
    if not root_path.is_dir():
        print(f"Error: Directory not found at '{root_path}'")
        return

    print(f"Starting search in: {root_path}")
    if dry_run:
        print("--- DRY RUN MODE: No files will be renamed. ---\n")
    else:
        print("--- EXECUTE MODE: Renaming enabled. ---\n")

    # Step 1: Use glob to find all directories named 'target_parent' efficiently.
    # The '**' pattern makes the glob recursive.
    target_parent_paths = list(root_path.glob(f"**/{target_parent}"))

    if not target_parent_paths:
        print(f"No directories named '{target_parent}' were found.")
        print("\n--- Finished ---")
        return

    print(f"Found {len(target_parent_paths)} directories named '{target_parent}'. Processing each...")

    # Step 2: Iterate through only the found parent directories.
    for parent_path in target_parent_paths:
        print(f"\n--- Walking inside: {parent_path} ---")
        # Now, walk from the bottom up inside this specific directory.
        for dirpath, dirnames, _ in os.walk(parent_path, topdown=False):
            for dirname in dirnames:
                if old_str in dirname and not dirname.startswith("output-XET"):
                    old_path = os.path.join(dirpath, dirname)
                    new_name = dirname.replace(old_str, new_str)
                    new_path = os.path.join(dirpath, new_name)

                    if os.path.exists(new_path):
                        print(f"SKIPPING: Cannot rename '{old_path}' because '{new_path}' already exists.")
                        continue

                    print(f"Found match: '{old_path}'")
                    print(f"  -> Proposing new name: '{new_path}'")

                    if not dry_run:
                        try:
                            os.rename(old_path, new_path)
                            print("  -> SUCCESS: Renamed.")
                        except OSError as e:
                            print(f"  -> ERROR: Could not rename. Reason: {e}")

    print("\n--- Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Efficiently rename subfolders within specific parent directories using glob."
    )
    parser.add_argument("root_directory", help="The directory to start searching from.")
    parser.add_argument(
        "target_parent",
        default="CRC_PDO",
        help="The name of the parent folder to constrain the search to (e.g., 'CRC_PDO').",
    )
    parser.add_argument("old_string", help="The string to find in folder names (e.g., '1911').")
    parser.add_argument("new_string", help="The string to replace the old one with (e.g., '19II').")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the rename. The default is a dry run.",
    )
    args = parser.parse_args()

    rename_subfolders_glob(
        args.root_directory, args.target_parent, args.old_string, args.new_string, dry_run=not args.execute
    )
