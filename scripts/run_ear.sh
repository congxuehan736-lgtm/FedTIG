#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

for mode in fedlf s c a; do
  python src/algorithm/fedtig_ear_direct_cifar10.py \
    --mode "$mode" --dataset cifar10 --num_classes 10 --num_clients 20 \
    --num_online_clients 4 --num_rounds 200 --num_epochs_local_training 10 \
    --batch_size_local_training 32 --lr_local_training 0.1 --imb_factor 0.01 \
    --non_iid_alpha 0.5 --rs_alpha 0.5 --s_epsilon 0.75 --c_alpha 0.70 \
    --c_warmup 30 --tail_agg_weight 1.50 --target_class 9 \
    --controlled_host_count 2 --seed 42 --device cuda --path_cifar10 data/CIFAR10
done
