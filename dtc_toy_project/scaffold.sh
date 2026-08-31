#!/usr/bin/env bash
# Scaffolds the dtc_toy_project directory structure in one shot.
# Usage: bash scaffold.sh  (run from wherever you want dtc_toy_project/ created)

set -e

ROOT="dtc_toy_project"

mkdir -p "$ROOT"/{data,outputs,src}

touch "$ROOT/src/__init__.py"
touch "$ROOT/data/.gitkeep"
touch "$ROOT/outputs/.gitkeep"

# Placeholder files for each module -- fill these in from the guide, or copy
# the versions from the provided project zip if you already have them.
for f in config.py data_task1.py data_task2.py data_task3.py model.py metrics.py \
         baseline.py train_task1.py train_task2.py task3_allocator.py \
         simulate_task2_loop.py; do
  touch "$ROOT/src/$f"
done

touch "$ROOT/README.md"
touch "$ROOT/requirements.txt"

echo "Created:"
find "$ROOT" -type f | sort
