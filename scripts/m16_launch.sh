#!/bin/bash
# Run one M16 plan on W parallel CPU workers (2 threads each); resumable.   bash scripts/m16_launch.sh <plan> [W]
cd "$(dirname "$0")/.."
W=${2:-5}
mkdir -p logs
for k in $(seq 0 $((W - 1))); do
  .venv/bin/python scripts/m16_run.py --plan "configs/m16_plans/$1.json" --worker "$k/$W" > "logs/m16_$1_$k.log" 2>&1 &
done
wait
echo "plan $1 done"
