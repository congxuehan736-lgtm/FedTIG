#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

for beta in 0.0 0.05 0.125 0.20 0.30; do
  python src/algorithm/fedtig_beta_pareto_cifar10.py \
    --mode a \
    --beta "$beta" \
    --dataset cifar10 \
    --num_classes 10 \
    --num_clients 20 \
    --num_online_clients 8 \
    --num_rounds 200 \
    --num_epochs_local_training 10 \
    --batch_size_local_training 32 \
    --lr_local_training 0.1 \
    --imb_factor 0.01 \
    --non_iid_alpha 0.5 \
    --rs_alpha 0.5 \
    --seed 42 \
    --device cuda \
    --path_cifar10 data/CIFAR10
done
