#!/bin/bash

# --- Configuration ---
# The top-level directory to search in. Using "." for the current directory.
ROOT_DIR="/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/norkin_organoid/data/xenium/processed/" 

# The name of the fixed parent folder.
TARGET_GRANDPARENT="CRC_PDO"

# The name of the parent folder we are now searching *inside*.
# This should be the NEW name from the previous script.
TARGET_PARENT="19II"

# The strings for replacement inside the subfolders.
OLD_STR="19II"
NEW_STR="1911"

# The prefix we are NOW TARGETING for renaming.
INCLUDE_PREFIX="output-XET"

# Set to "true" to execute. Anything else is a dry run.
EXECUTE="true" 

# --- End of Configuration ---


# --- Script Logic ---
if [ "$EXECUTE" = "true" ]; then
  echo "--- EXECUTE MODE: Renaming enabled. ---"
else
  echo "--- DRY RUN MODE: No folders will be renamed. ---"
fi
echo "Searching in: $(realpath "$ROOT_DIR")"
# Define the glob pattern to find the PARENT '19II' folders.
# e.g., ./level1/level2/CRC_PDO/level4/19II/
PARENT_GLOB_PATTERN="$ROOT_DIR/*/*/$TARGET_GRANDPARENT/*/$TARGET_PARENT/"
echo "Using parent pattern: $PARENT_GLOB_PATTERN"
echo "------------------------------------------------"

found_match=false

# Outer loop: Find all the '19II' parent folders first.
for parent_path_with_slash in $PARENT_GLOB_PATTERN; do
    parent_path="${parent_path_with_slash%/}"

    if [ -d "$parent_path" ]; then
        echo ">>> Processing inside parent: $parent_path"
        
        # Inner loop: Find subfolders inside the parent that start with the INCLUDE_PREFIX.
        # The pattern is now relative to the parent_path.
        SUBFOLDER_GLOB_PATTERN="$parent_path/$INCLUDE_PREFIX*"
        
        for old_path in $SUBFOLDER_GLOB_PATTERN; do
            if [ -d "$old_path" ]; then
                found_match=true
                subfolder_name=$(basename "$old_path")

                # We only want to rename if OLD_STR is present
                if [[ "$subfolder_name" == *"$OLD_STR"* ]]; then
                    parent_of_subfolder=$(dirname "$old_path")
                    new_name="${subfolder_name//$OLD_STR/$NEW_STR}" # Global replace
                    new_path="$parent_of_subfolder/$new_name"

                    if [ -e "$new_path" ]; then
                        echo "[SKIP] Cannot rename '$old_path' because '$new_path' already exists."
                    else
                        echo "[MATCH] Proposing subfolder rename:"
                        echo "  FROM: $old_path"
                        echo "  TO:   $new_path"
                        
                        if [ "$EXECUTE" = "true" ]; then
                            mv "$old_path" "$new_path"
                            if [ $? -eq 0 ]; then
                                echo "  SUCCESS: Renamed."
                            else
                                echo "  ERROR: Failed to rename."
                            fi
                        fi
                    fi
                fi
            fi
        done
    fi
done

if [ "$found_match" = false ]; then
    echo "No matching subfolders found to rename."
fi

echo "------------------------------------------------"
echo "Script finished."