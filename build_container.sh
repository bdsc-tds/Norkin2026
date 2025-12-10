#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ================= CONFIGURATION =================
# 1. Auto-detect Root: Use argument if provided, else use current dir
if [ -n "$1" ]; then
    PROJECT_ROOT="$1"
    echo ">>> Using specified project root: $PROJECT_ROOT"
else
    PROJECT_ROOT="$(pwd)"
    echo ">>> Auto-detected project root: $PROJECT_ROOT"
fi

# 2. Define expected file locations
# Change these if your .def file is named differently
DEF_REL_PATH="containers/pixi.def" 
OUTPUT_REL_DIR="containers"
IMAGE_NAME="pixi.sif"

FULL_DEF_PATH="${PROJECT_ROOT}/${DEF_REL_PATH}"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_REL_DIR}"

# 3. Sanity Checks
if [[ ! -f "$PROJECT_ROOT/pixi.toml" ]]; then
    echo "ERROR: pixi.toml not found in $PROJECT_ROOT"
    exit 1
fi

if [[ ! -f "$FULL_DEF_PATH" ]]; then
    echo "ERROR: Definition file not found at $FULL_DEF_PATH"
    exit 1
fi

# 4. Setup Scratch
SCRATCH_WORK_DIR="/scratch/${USER}/build_pixi_$(date +%s)"
# =================================================

# Save starting location (though script execution creates a subshell, this is good practice)
ORIGINAL_DIR=$(pwd)

echo ">>> Setting up build environment in: $SCRATCH_WORK_DIR"
mkdir -p "$SCRATCH_WORK_DIR"

# -------------------------------------------------
# STAGE 1: Copy files to Scratch
# -------------------------------------------------
echo ">>> Staging files..."
cp "${PROJECT_ROOT}/pixi.toml" "$SCRATCH_WORK_DIR/"
cp "${PROJECT_ROOT}/pixi.lock" "$SCRATCH_WORK_DIR/"
cp "$FULL_DEF_PATH" "$SCRATCH_WORK_DIR/singularity.def"


# -------------------------------------------------
# STAGE 2: Build
# -------------------------------------------------
echo ">>> Changing directory to scratch..."
cd "$SCRATCH_WORK_DIR"

echo ">>> Starting Singularity build (this may take time)..."
# We use --fakeroot, which works on /scratch but not /work
singularity build --fakeroot --force "$IMAGE_NAME" singularity.def

# -------------------------------------------------
# STAGE 3: Finalize
# -------------------------------------------------
echo ">>> Moving image back to project: $OUTPUT_DIR/$IMAGE_NAME"
mkdir -p "$OUTPUT_DIR"
mv "$IMAGE_NAME" "$OUTPUT_DIR/"

echo ">>> Cleaning up scratch..."
cd "$ORIGINAL_DIR"
rm -rf "$SCRATCH_WORK_DIR"

echo ">>> Done. Your container is ready at: $OUTPUT_DIR/$IMAGE_NAME"