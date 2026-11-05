#!/bin/bash
# Milestone 13 demonstration cities (downstream illustration; not part of any road-generation result).
# Flat, rolling hills, steep hills, river valley, mountain pass, coast: the selected method M* and the strongest
# baseline (A, standard) on IDENTICAL terrain and seed; first sample (no cherry-picking); M13 downstream fixes on.
set -e
cd "$(dirname "$0")/.."
P=.venv/bin/python
B=/Applications/Blender.app/Contents/MacOS/Blender
read PRE OV < <($P -c "import json;s=json.load(open('outputs/m13/selection.json'))['M_star'];print(s['preset'], json.dumps(s['overrides']).replace(' ',''))")
for sc in "flat_city 1" "rolling_hills 4" "steep_hillside 1" "narrow_river 2" "mountain_pass 0" "coastal_city 1"; do
  set -- $sc
  for m in "$PRE $OV" "standard {}"; do
    set -- $sc $m
    D=$($P scripts/m12_build_city3d.py --scenario $1 --scenario-seed $2 --model TN_s0 --sampler $3 --overrides "$4" --m13-fixes 2>&1 | tail -1 | $P -c "import sys,json;print(json.loads(sys.stdin.read())['dir'])")
    $B -b --factory-startup --python-exit-code 1 --python blender/m11_scene.py -- --tile "$D" --res 1400 > "$D/blender.log" 2>&1
    echo "$D blender $?"
  done
done
$P scripts/m11_godot.py --root outputs/m12/cities --no-bench --tiles $(ls outputs/m12/cities | grep "_tc-")
mkdir -p outputs/m13/done && date > outputs/m13/done/demos
