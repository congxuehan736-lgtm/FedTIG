#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --dataset cifar100
  --num_classes 100
  --num_clients 20
  --num_online_clients 8
  --num_rounds 200
  --num_epochs_local_training 5
  --batch_size_local_training 32
  --lr_local_training 0.1
  --imb_factor 0.02
  --non_iid_alpha 0.5
  --rs_alpha 0.25
  --seed 42
  --device cuda
  --path_cifar100 data/CIFAR100
)

python src/algorithm/fedlf_cifar100_baseline.py "${COMMON[@]}"
python src/algorithm/fedlf_cifar100_ready.py "${COMMON[@]}"
