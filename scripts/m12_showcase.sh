#!/bin/bash
# Milestone 12 representative cities (spec §22): build, render (Blender), export (GLB) and check (Godot).
# Usage: bash scripts/m12_showcase.sh
cd "$(dirname "$0")/.."
P=.venv/bin/python
B=/Applications/Blender.app/Contents/MacOS/Blender
build() { $P scripts/m12_build_city3d.py "$@" 2>&1 | tail -1 | $P -c "import sys, json; print(json.loads(sys.stdin.read())['dir'])"; }
for args in "--scenario flat_city --scenario-seed 1 --mode HR" \
            "--terrain-tile USA_SAN_e958_n7125 --mode HR" \
            "--scenario narrow_river --scenario-seed 2 --mode HR" \
            "--scenario wide_river --scenario-seed 3 --mode HR" \
            "--scenario deep_valley --scenario-seed 2 --mode HRV" \
            "--scenario coastal_city --scenario-seed 1 --mode HR" \
            "--scenario wide_river --scenario-seed 3 --mode none" \
            "--scenario steep_hillside --scenario-seed 1 --mode R"; do
  D=$(build $args --model TN_s0)
  $B -b --factory-startup --python-exit-code 1 --python blender/m11_scene.py -- --tile "$D" --res 1400 > "$D/blender.log" 2>&1
  echo "$D blender $?"
done
$P scripts/m12_big_landscape.py --scenario rolling_hills --scenario-seed 4 --n 3 --model TN_s0 --mode HR
D=outputs/m12/cities/big_TN_s0_HR_rolling_hills4_3x3_s2050
$B -b --factory-startup --python-exit-code 1 --python blender/m11_scene.py -- --tile "$D" --res 2000 > "$D/blender.log" 2>&1
echo "$D blender $?"
$P scripts/m11_godot.py --root outputs/m12/cities --no-bench --tiles $(ls outputs/m12/cities)
