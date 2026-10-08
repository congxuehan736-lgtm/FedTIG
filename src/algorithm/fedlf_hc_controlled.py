# -*- coding: utf-8 -*-
"""
fedlf_hc_controlled.py
======================

Controlled Effective-Training-Rarity experiment for CIFAR-10-LT.

Goal
----
Keep the total number of samples of one target tail class fixed, and change
ONLY the number of clients that host that class:

    N_target fixed
    h_target in {1, 2, 4, 8, 16}

The default target is class 9. Under CIFAR-10-LT IF=100, class 9 has N_9=50.

This script uses:
    - pure FedLF local learner
    - uniform random client selection
    - standard sample-size-weighted FedAvg
    - fixed learning rate 0.1
    - no S / no C / no A
    - fixed Round-200 reporting, no test rollback

Thus it isolates the effect of class hosting concentration / exposure.

Controlled partition construction
---------------------------------
1. Build the ordinary long-tailed + Dirichlet partition using seed=42.
2. Keep every NON-target sample on exactly the same client as in the base
   partition.
3. Remove all target-class samples from all clients.
4. Redistribute exactly the same target samples over h target-host clients.
5. Host sets are nested and deterministic:
       h=1 hosts = first 1 client of a fixed seed-derived host order
       h=2 hosts = first 2
       h=4 hosts = first 4
       ...
   This makes cross-h comparisons paired and reproducible.

The script asserts:
    - target sample identity set is unchanged
    - target total N is unchanged
    - no non-target sample moves
    - no sample is duplicated/lost
    - actual host count equals requested h

For uniform sampling without replacement, the theoretical exposure probability
is:
    P_exp = 1 - C(K-h, m) / C(K, m)

Outputs
-------
Logs/hc_cifar10_class9_h{h}_IF100_alpha0p5_seed42.log
outputs/hc_cifar10_class9_h{h}_IF100_alpha0p5_seed42/
    config.json
    metrics.csv
    history.pkl
    checkpoint.pth
    final_model.pth

Special sanity mode
-------------------
Use --host_count 0 to keep the original untouched Dirichlet partition.
That is only for a short 5-round implementation sanity check; it is NOT one
of the controlled h points.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from Model.log_model import setup_logging
from Model.Resnet8 import ResNet_cifar
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import (
    classify_label,
    show_clients_data_distribution,
    Indices2Dataset,
)
from Dataset.sample_dirichlet import clients_indices


ENABLE_TEST_ROLLBACK = False


# =============================================================================
# CLI / reproducibility
# =============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-10-LT fixed-N / varied-h controlled exposure experiment"
        )
    )

    root = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_online_clients", type=int, default=8)

    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--num_epochs_local_training", type=int, default=10)
    parser.add_argument("--batch_size_local_training", type=int, default=32)
    parser.add_argument("--batch_size_test", type=int, default=500)
    parser.add_argument("--lr_local_training", type=float, default=0.1)

    parser.add_argument("--non_iid_alpha", type=float, default=0.5)
    parser.add_argument("--imb_type", type=str, default="exp")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--rs_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--path_cifar10",
        type=str,
        default=os.path.join(root, "data/CIFAR10/"),
    )

    parser.add_argument(
        "--target_class",
        type=int,
        default=9,
        help="Controlled tail class. Default: class 9.",
    )
    parser.add_argument(
        "--host_count",
        type=int,
        default=1,
        choices=[0, 1, 2, 4, 8, 16],
        help=(
            "Number of clients hosting the target class. "
            "0 keeps the untouched base partition for sanity only."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
    )

    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_sha256(list_client2indices):
    payload = [
        [int(v) for v in list(indices)]
        for indices in list_client2indices
    ]
    raw = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_eval_thresholds(dataset, imb_factor):
    ratio = float(imb_factor)

    if dataset == "cifar10":
        if ratio >= 0.099:
            return 1500, 600
        return 1500, 200

    raise ValueError(
        "This script is intentionally restricted to CIFAR-10."
    )


# =============================================================================
# Controlled target-class redistribution
# =============================================================================
def labels_from_dataset(dataset):
    """
    Return labels without relying on a specific torchvision attribute name.
    CIFAR datasets normally expose .targets; fallback is provided for safety.
    """
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    labels = []
    for _, y in dataset:
        labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def class_counts_per_client_from_indices(
    list_client2indices,
    labels,
    num_classes,
):
    result = []

    for indices in list_client2indices:
        counts = np.zeros(
            num_classes,
            dtype=np.int64,
        )

        if len(indices) > 0:
            ys = labels[np.asarray(indices, dtype=np.int64)]
            binc = np.bincount(
                ys,
                minlength=num_classes,
            )
            counts[:] = binc[:num_classes]

        result.append(counts.tolist())

    return result


def fixed_host_order(num_clients, seed, target_class):
    """
    A label-independent, deterministic client permutation.

    The order is nested across h:
      h=1 uses order[:1]
      h=2 uses order[:2]
      ...
    """
    rng = np.random.RandomState(
        int(seed) + 10000 + int(target_class)
    )
    return [
        int(v)
        for v in rng.permutation(num_clients)
    ]


def build_controlled_partition(
    base_partition,
    labels,
    target_indices,
    target_class,
    host_count,
    seed,
):
    """
    Preserve all non-target assignments and redistribute only target_class.

    Returns:
        new_partition
        host_order
        active_hosts
        host_target_counts
    """
    base = [
        [int(v) for v in client_indices]
        for client_indices in base_partition
    ]

    num_clients = len(base)

    if host_count == 0:
        # Untouched base partition, used only for implementation sanity.
        host_order = fixed_host_order(
            num_clients,
            seed,
            target_class,
        )
        actual_hosts = [
            k
            for k, indices in enumerate(base)
            if any(
                int(labels[idx]) == int(target_class)
                for idx in indices
            )
        ]
        counts = {
            int(k): int(
                sum(
                    int(labels[idx]) == int(target_class)
                    for idx in base[k]
                )
            )
            for k in actual_hosts
        }
        return base, host_order, actual_hosts, counts

    if host_count > num_clients:
        raise ValueError(
            f"host_count={host_count} exceeds K={num_clients}"
        )

    target_indices = [
        int(v)
        for v in target_indices
    ]
    target_set = set(target_indices)

    # Verify all controlled indices really belong to the target class.
    bad_target = [
        idx
        for idx in target_indices
        if int(labels[idx]) != int(target_class)
    ]
    if bad_target:
        raise RuntimeError(
            "Target index set contains non-target labels."
        )

    # Collect original union / non-target assignments for audit.
    original_union = []
    original_non_target_by_client = []

    for indices in base:
        original_union.extend(indices)
        original_non_target_by_client.append(
            [
                idx
                for idx in indices
                if idx not in target_set
            ]
        )

    if len(original_union) != len(set(original_union)):
        raise RuntimeError(
            "Base partition contains duplicated sample indices."
        )

    original_target_in_partition = set(
        idx
        for idx in original_union
        if int(labels[idx]) == int(target_class)
    )

    if original_target_in_partition != target_set:
        raise RuntimeError(
            "The target-class sample set in the base partition does not "
            "match the long-tail target index set."
        )

    # Only target examples are removed.
    new_partition = [
        list(v)
        for v in original_non_target_by_client
    ]

    host_order = fixed_host_order(
        num_clients,
        seed,
        target_class,
    )
    active_hosts = host_order[:host_count]

    # Keep target sample ordering identical across h values.
    target_rng = np.random.RandomState(
        int(seed) + 20000 + int(target_class)
    )
    ordered_target = [
        int(v)
        for v in target_rng.permutation(
            np.asarray(
                target_indices,
                dtype=np.int64,
            )
        )
    ]

    chunks = np.array_split(
        np.asarray(ordered_target, dtype=np.int64),
        host_count,
    )

    host_target_counts = {}

    for host, chunk in zip(active_hosts, chunks):
        chunk_list = [
            int(v)
            for v in chunk.tolist()
        ]
        new_partition[host].extend(chunk_list)
        host_target_counts[int(host)] = len(chunk_list)

    # ---------------------------------------------------------------------
    # Strong assertions: fixed N, fixed non-target assignments, no loss/dup.
    # ---------------------------------------------------------------------
    new_union = [
        idx
        for client_indices in new_partition
        for idx in client_indices
    ]

    if len(new_union) != len(set(new_union)):
        raise RuntimeError(
            "Controlled partition contains duplicated sample indices."
        )

    if set(new_union) != set(original_union):
        raise RuntimeError(
            "Controlled partition lost or added training samples."
        )

    for k in range(num_clients):
        new_non_target = [
            idx
            for idx in new_partition[k]
            if int(labels[idx]) != int(target_class)
        ]

        if new_non_target != original_non_target_by_client[k]:
            raise RuntimeError(
                "A non-target sample moved clients, which violates "
                "the controlled-design requirement."
            )

    new_target_set = set(
        idx
        for idx in new_union
        if int(labels[idx]) == int(target_class)
    )

    if new_target_set != target_set:
        raise RuntimeError(
            "Target sample identity set changed."
        )

    actual_hosts = [
        k
        for k in range(num_clients)
        if any(
            int(labels[idx]) == int(target_class)
            for idx in new_partition[k]
        )
    ]

    if len(actual_hosts) != host_count:
        raise RuntimeError(
            f"Requested h={host_count} but obtained h={len(actual_hosts)}"
        )

    return (
        new_partition,
        host_order,
        actual_hosts,
        host_target_counts,
    )


def theoretical_exposure_probability(K, m, h):
    """
    P_exp = 1 - C(K-h,m) / C(K,m)
    for uniform client sampling without replacement.
    """
    K = int(K)
    m = int(m)
    h = int(h)

    if h <= 0:
        return float("nan")

    if m <= 0:
        return 0.0

    if h >= K:
        return 1.0

    if K - h < m:
        return 1.0

    return (
        1.0
        - math.comb(K - h, m)
        / math.comb(K, m)
    )


# =============================================================================
# FedLF local learner
# =============================================================================
class DecorrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        n, m = mat.shape
        assert n == m

        if n <= 1:
            return mat.new_zeros(1)

        return (
            mat.flatten()[:-1]
            .view(n - 1, n + 1)[:, 1:]
            .flatten()
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[0] <= 1:
            return x.new_tensor(0.0)

        n = x.shape[0]

        x = x - x.mean(
            dim=0,
            keepdim=True,
        )
        x = x / torch.sqrt(
            self.eps
            + x.var(
                dim=0,
                keepdim=True,
                unbiased=False,
            )
        )

        corr_mat = x.t().matmul(x)
        return (
            self._off_diagonal(corr_mat)
            .pow(2)
            .mean()
            / n
        )


class Global(object):
    def __init__(self, args):
        self.device = args.device
        self.dataset = args.dataset
        self.imb_factor = args.imb_factor

        self.syn_model = ResNet_cifar(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        ).to(args.device)

        self.smoothed_centers = torch.zeros(
            (args.num_classes, 256),
            device=args.device,
        )
        self.ema_decay = 0.9
        self.center_initialized = False

    def update_smoothed_centers(
        self,
        new_centers,
    ):
        new_centers = new_centers.detach()

        if not torch.isfinite(
            new_centers
        ).all():
            return

        if not self.center_initialized:
            self.smoothed_centers = (
                new_centers.clone()
            )
            self.center_initialized = True
        else:
            self.smoothed_centers = (
                self.ema_decay
                * self.smoothed_centers
                + (1.0 - self.ema_decay)
                * new_centers
            )

    def get_smoothed_centers(self):
        return (
            self.smoothed_centers
            .detach()
        )

    def download_params(self):
        return copy.deepcopy(
            self.syn_model.state_dict()
        )

    def fedavg(
        self,
        local_states,
        local_sizes,
    ):
        if not local_states:
            raise ValueError(
                "No local states to aggregate."
            )

        total = float(sum(local_sizes))
        if total <= 0:
            raise ValueError(
                "Selected clients contain no data."
            )

        result = copy.deepcopy(
            local_states[0]
        )

        for name in result:
            first = local_states[0][name]

            if not torch.is_floating_point(
                first
            ):
                result[name] = first.clone()
                continue

            value = torch.zeros_like(first)

            for state, size in zip(
                local_states,
                local_sizes,
            ):
                value += (
                    state[name]
                    * float(size)
                )

            result[name] = value / total

        return result

    def evaluate(
        self,
        params,
        data_test,
        batch_size_test,
        class_counts,
        target_class,
        num_classes,
    ):
        """
        One-pass evaluation:
          Global / Head / Middle / Tail / target-class / per-class
        """
        self.syn_model.load_state_dict(
            params
        )
        self.syn_model.eval()

        majority_threshold, tail_threshold = (
            get_eval_thresholds(
                self.dataset,
                self.imb_factor,
            )
        )

        correct = np.zeros(
            num_classes,
            dtype=np.int64,
        )
        total = np.zeros(
            num_classes,
            dtype=np.int64,
        )

        loader = DataLoader(
            data_test,
            batch_size=batch_size_test,
            shuffle=False,
        )

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(
                    self.device
                )
                labels = labels.to(
                    self.device
                )

                _, logits = self.syn_model(
                    images
                )
                pred = logits.argmax(
                    dim=1
                )

                for y, p in zip(
                    labels,
                    pred,
                ):
                    yi = int(y.item())
                    total[yi] += 1
                    correct[yi] += int(
                        p.item() == yi
                    )

        per_class = np.divide(
            correct,
            np.maximum(total, 1),
            dtype=np.float64,
        )

        head_classes = [
            c
            for c, n in enumerate(
                class_counts
            )
            if int(n) > majority_threshold
        ]
        tail_classes = [
            c
            for c, n in enumerate(
                class_counts
            )
            if int(n) < tail_threshold
        ]
        middle_classes = [
            c
            for c in range(num_classes)
            if (
                c not in head_classes
                and c not in tail_classes
            )
        ]

        def group_acc(classes):
            if not classes:
                return 0.0
            numerator = int(
                correct[classes].sum()
            )
            denominator = int(
                total[classes].sum()
            )
            if denominator <= 0:
                return 0.0
            return numerator / denominator

        global_acc = float(
            correct.sum()
            / max(total.sum(), 1)
        )

        return {
            "global": global_acc,
            "head": group_acc(
                head_classes
            ),
            "middle": group_acc(
                middle_classes
            ),
            "tail": group_acc(
                tail_classes
            ),
            "target": float(
                per_class[
                    int(target_class)
                ]
            ),
            "per_class": [
                float(v)
                for v in per_class.tolist()
            ],
            "head_classes": head_classes,
            "middle_classes": middle_classes,
            "tail_classes": tail_classes,
        }


class Local(object):
    def __init__(
        self,
        data_client,
        args,
    ):
        self.data_client = data_client
        self.device = args.device

        self.criterion = (
            CrossEntropyLoss()
            .to(args.device)
        )
        self.feddecorr = DecorrLoss()

        self.local_model = ResNet_cifar(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        ).to(args.device)

        self.optimizer = SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
        )

    def local_train(
        self,
        args,
        global_params,
        dist,
        smoothed_centers,
    ):
        """
        Pure FedLF:
          L = L_A + 0.01 L_C + 0.01 L_D
        No S / C / A intervention.
        Fixed local LR = args.lr_local_training.
        """
        transform_train = (
            transforms.Compose([
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(),
            ])
        )

        self.local_model.load_state_dict(
            global_params
        )
        self.local_model.train()

        # FedLF adaptive logit adjustment.
        dist = dist.to(
            self.device
        )

        if (
            dist.numel()
            != args.num_classes
        ):
            tmp = torch.zeros(
                args.num_classes,
                device=self.device,
            )
            num_copy = min(
                len(dist),
                args.num_classes,
            )
            tmp[:num_copy] = (
                dist[:num_copy]
            )
            dist = tmp

        cdist = (
            dist
            / dist.max().clamp_min(
                1e-8
            )
        )
        cdist = (
            cdist
            * (1.0 - args.rs_alpha)
            + args.rs_alpha
        )
        cdist = cdist.clamp(
            0.50,
            1.00,
        ).reshape(1, -1)

        feature_centers = (
            smoothed_centers
            .to(self.device)
            .detach()
        )

        if (
            not torch.isfinite(
                feature_centers
            ).all()
            or feature_centers
            .abs()
            .sum()
            < 1e-8
        ):
            feature_centers = (
                self.local_model
                .classifier
                .weight
                .detach()
                .clone()
            )

        with torch.no_grad():
            gap = torch.cdist(
                feature_centers,
                feature_centers,
                p=2,
            )

            mask = ~torch.eye(
                args.num_classes,
                dtype=torch.bool,
                device=self.device,
            )

            valid_gap = gap[mask]

            max_gap = (
                valid_gap.max()
                if valid_gap.numel() > 0
                else torch.tensor(
                    1.0,
                    device=self.device,
                )
            )

            gap_val = float(
                torch.clamp(
                    max_gap,
                    min=0.05,
                    max=2.0,
                ).item()
            )

        # Formal Protocol-A uses FIXED lr=0.1.
        for group in (
            self.optimizer.param_groups
        ):
            group["lr"] = float(
                args.lr_local_training
            )

        for _ in range(
            args.num_epochs_local_training
        ):
            loader = DataLoader(
                self.data_client,
                batch_size=(
                    args.batch_size_local_training
                ),
                shuffle=True,
            )

            for images, labels in loader:
                images = images.to(
                    self.device
                )
                labels = labels.to(
                    self.device
                )

                images = (
                    transform_train(images)
                )

                hs, _ = (
                    self.local_model(
                        images
                    )
                )

                ws = (
                    self.local_model
                    .classifier
                    .weight
                )

                logits = (
                    cdist
                    * hs.mm(
                        ws.transpose(
                            0,
                            1,
                        )
                    )
                )

                loss_a = (
                    self.criterion(
                        logits,
                        labels,
                    )
                )

                features_square = (
                    torch.sum(
                        hs.pow(2),
                        dim=1,
                        keepdim=True,
                    )
                )
                centers_square = (
                    torch.sum(
                        feature_centers.pow(2),
                        dim=1,
                        keepdim=True,
                    )
                )
                f_into_c = hs.matmul(
                    feature_centers.t()
                )

                dist_2 = (
                    features_square
                    - 2 * f_into_c
                    + centers_square.t()
                )
                dist_2 = torch.sqrt(
                    dist_2.clamp_min(
                        1e-8
                    )
                )

                one_hot = F.one_hot(
                    labels,
                    args.num_classes,
                ).to(self.device)

                dist_2 = (
                    dist_2
                    + one_hot * gap_val
                )

                loss_c = (
                    self.criterion(
                        -dist_2,
                        labels,
                    )
                )
                loss_d = (
                    self.feddecorr(hs)
                )

                loss = (
                    loss_a
                    + 0.01 * loss_c
                    + 0.01 * loss_d
                )

                if not torch.isfinite(
                    loss
                ):
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(),
                    max_norm=20.0,
                )

                self.optimizer.step()

        return copy.deepcopy(
            self.local_model.state_dict()
        )


# =============================================================================
# Main experiment
# =============================================================================
def run():
    args = build_parser().parse_args()

    if args.dataset != "cifar10":
        raise ValueError(
            "Use --dataset cifar10."
        )
    if args.num_classes != 10:
        raise ValueError(
            "Use --num_classes 10."
        )
    if not (
        0 <= args.target_class
        < args.num_classes
    ):
        raise ValueError(
            "target_class out of range."
        )

    seed_everything(args.seed)

    paper_if = int(
        round(
            1.0
            / float(args.imb_factor)
        )
    )
    alpha_tag = str(
        args.non_iid_alpha
    ).replace(".", "p")

    h_tag = (
        "base"
        if args.host_count == 0
        else f"h{args.host_count}"
    )

    run_name = (
        f"hc_cifar10"
        f"_class{args.target_class}"
        f"_{h_tag}"
        f"_IF{paper_if}"
        f"_alpha{alpha_tag}"
        f"_seed{args.seed}"
    )

    os.makedirs(
        "Logs",
        exist_ok=True,
    )
    output_dir = os.path.join(
        "outputs",
        run_name,
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    log_filename = (
        run_name + ".log"
    )
    log_path = os.path.join(
        "Logs",
        log_filename,
    )
    metrics_path = os.path.join(
        output_dir,
        "metrics.csv",
    )
    history_path = os.path.join(
        output_dir,
        "history.pkl",
    )
    config_path = os.path.join(
        output_dir,
        "config.json",
    )
    checkpoint_path = os.path.join(
        output_dir,
        "checkpoint.pth",
    )
    final_path = os.path.join(
        output_dir,
        "final_model.pth",
    )

    logger = setup_logging(
        log_filename
    )

    logger.info("=" * 90)
    logger.info(
        "FIXED-N / VARIED-h CONTROLLED EXPERIMENT"
    )
    logger.info(
        "algorithm: pure FedLF | S=OFF C=OFF A=OFF"
    )
    logger.info(
        "dataset:CIFAR-10-LT IF=%d imb_factor=%s alpha=%s",
        paper_if,
        args.imb_factor,
        args.non_iid_alpha,
    )
    logger.info(
        "K=%d m=%d participation=%.4f",
        args.num_clients,
        args.num_online_clients,
        (
            float(args.num_online_clients)
            / float(args.num_clients)
        ),
    )
    logger.info(
        "rounds=%d E=%d batch=%d fixed_lr=%s seed=%d",
        args.num_rounds,
        args.num_epochs_local_training,
        args.batch_size_local_training,
        args.lr_local_training,
        args.seed,
    )
    logger.info(
        "target_class=%d requested_host_count=%d",
        args.target_class,
        args.host_count,
    )
    logger.info(
        "client_sampling=UNIFORM RANDOM WITHOUT REPLACEMENT"
    )
    logger.info(
        "aggregation=STANDARD SAMPLE-WEIGHTED FedAvg"
    )
    logger.info(
        "test rollback=%s",
        ENABLE_TEST_ROLLBACK,
    )
    logger.info(
        "Log: %s",
        os.path.abspath(
            log_path
        ),
    )
    logger.info(
        "Output: %s",
        os.path.abspath(
            output_dir
        ),
    )
    logger.info("=" * 90)

    transform_all = (
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (
                    0.4914,
                    0.4822,
                    0.4465,
                ),
                (
                    0.2023,
                    0.1994,
                    0.2010,
                ),
            ),
        ])
    )

    train_data = (
        datasets.CIFAR10(
            args.path_cifar10,
            train=True,
            download=True,
            transform=transform_all,
        )
    )
    test_data = (
        datasets.CIFAR10(
            args.path_cifar10,
            train=False,
            download=True,
            transform=transform_all,
        )
    )

    labels = labels_from_dataset(
        train_data
    )

    list_label2indices = (
        classify_label(
            train_data,
            args.num_classes,
        )
    )

    (
        global_class_counts,
        longtail_label_indices,
    ) = train_long_tail(
        copy.deepcopy(
            list_label2indices
        ),
        args.num_classes,
        args.imb_factor,
        args.imb_type,
    )

    target_indices = (
        copy.deepcopy(
            longtail_label_indices[
                args.target_class
            ]
        )
    )

    target_N = len(
        target_indices
    )

    base_partition = clients_indices(
        copy.deepcopy(
            longtail_label_indices
        ),
        args.num_classes,
        args.num_clients,
        args.non_iid_alpha,
        args.seed,
    )

    base_fingerprint = (
        partition_sha256(
            base_partition
        )
    )

    (
        controlled_partition,
        host_order,
        active_hosts,
        host_target_counts,
    ) = build_controlled_partition(
        base_partition=(
            base_partition
        ),
        labels=labels,
        target_indices=(
            target_indices
        ),
        target_class=(
            args.target_class
        ),
        host_count=(
            args.host_count
        ),
        seed=args.seed,
    )

    controlled_fingerprint = (
        partition_sha256(
            controlled_partition
        )
    )

    client_class_counts = (
        class_counts_per_client_from_indices(
            controlled_partition,
            labels,
            args.num_classes,
        )
    )

    # Optional human-readable distribution print, identical concept to the
    # rest of the project. This also acts as a visual audit.
    _ = show_clients_data_distribution(
        train_data,
        controlled_partition,
        args.num_classes,
    )

    actual_target_total = int(
        sum(
            row[
                args.target_class
            ]
            for row in client_class_counts
        )
    )
    actual_host_count = int(
        sum(
            row[
                args.target_class
            ] > 0
            for row in client_class_counts
        )
    )

    if actual_target_total != target_N:
        raise RuntimeError(
            "Target N changed: "
            f"{target_N} -> "
            f"{actual_target_total}"
        )

    if (
        args.host_count > 0
        and actual_host_count
        != args.host_count
    ):
        raise RuntimeError(
            "Actual host count mismatch."
        )

    theoretical_p = (
        theoretical_exposure_probability(
            K=args.num_clients,
            m=args.num_online_clients,
            h=actual_host_count,
        )
    )

    logger.info(
        "Global class counts: %s",
        list(
            map(
                int,
                global_class_counts,
            )
        ),
    )
    logger.info(
        "Target class %d N=%d (FIXED)",
        args.target_class,
        target_N,
    )
    logger.info(
        "Host order (nested across h): %s",
        host_order,
    )
    logger.info(
        "Actual host clients: %s",
        active_hosts,
    )
    logger.info(
        "Target samples per active host: %s",
        host_target_counts,
    )
    logger.info(
        "Actual h=%d | theoretical P_exp=%.6f",
        actual_host_count,
        theoretical_p,
    )
    logger.info(
        "Base partition SHA256: %s",
        base_fingerprint,
    )
    logger.info(
        "Controlled partition SHA256: %s",
        controlled_fingerprint,
    )
    logger.info(
        "Client sizes after target redistribution: %s",
        [
            int(len(v))
            for v in controlled_partition
        ],
    )

    majority_threshold, tail_threshold = (
        get_eval_thresholds(
            args.dataset,
            args.imb_factor,
        )
    )
    tail_classes = [
        c
        for c, n in enumerate(
            global_class_counts
        )
        if int(n) < tail_threshold
    ]

    logger.info(
        "Paper thresholds: Head > %d, Tail < %d; Tail classes=%s",
        majority_threshold,
        tail_threshold,
        tail_classes,
    )

    config = {
        "experiment": (
            "fixed_target_N_varied_host_count"
        ),
        "algorithm": "FedLF",
        "S": False,
        "C": False,
        "A": False,
        "dataset": "CIFAR-10-LT",
        "IF": paper_if,
        "imb_factor": float(
            args.imb_factor
        ),
        "non_iid_alpha": float(
            args.non_iid_alpha
        ),
        "rs_alpha": float(
            args.rs_alpha
        ),
        "K": int(
            args.num_clients
        ),
        "m": int(
            args.num_online_clients
        ),
        "rounds": int(
            args.num_rounds
        ),
        "local_epochs": int(
            args.num_epochs_local_training
        ),
        "batch_size": int(
            args.batch_size_local_training
        ),
        "lr": float(
            args.lr_local_training
        ),
        "lr_schedule": "fixed",
        "seed": int(
            args.seed
        ),
        "target_class": int(
            args.target_class
        ),
        "target_N": int(
            target_N
        ),
        "requested_host_count": int(
            args.host_count
        ),
        "actual_host_count": int(
            actual_host_count
        ),
        "host_order": host_order,
        "active_hosts": active_hosts,
        "host_target_counts": (
            host_target_counts
        ),
        "theoretical_exposure": (
            theoretical_p
        ),
        "base_partition_sha256": (
            base_fingerprint
        ),
        "controlled_partition_sha256": (
            controlled_fingerprint
        ),
        "control_rule": (
            "all non-target sample-to-client "
            "assignments fixed; only target "
            "class redistributed"
        ),
        "test_rollback": False,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "round",
        "selected_clients",
        "target_exposure_event",
        "target_selected_samples",
        "selected_total_samples",
        "target_exposure_mass",
        "cumulative_empirical_exposure",
        "theoretical_exposure",
        "target_class_accuracy",
        "tail_accuracy",
        "head_accuracy",
        "middle_accuracy",
        "global_accuracy",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

    global_model = Global(
        args
    )

    initial_state = (
        global_model
        .syn_model
        .state_dict()
    )
    global_model.update_smoothed_centers(
        initial_state[
            "classifier.weight"
        ].detach()
    )

    total_clients = list(
        range(
            args.num_clients
        )
    )
    indices2data = (
        Indices2Dataset(
            train_data
        )
    )

    # SAME selected-client random sequence for every h condition.
    client_rng = (
        np.random.RandomState(
            args.seed
        )
    )

    history = {
        "round": [],
        "selected_clients": [],
        "target_exposure_event": [],
        "target_selected_samples": [],
        "target_exposure_mass": [],
        "cumulative_empirical_exposure": [],
        "theoretical_exposure": [],
        "target_class_accuracy": [],
        "tail_accuracy": [],
        "head_accuracy": [],
        "middle_accuracy": [],
        "global_accuracy": [],
        "per_class_accuracy": [],
        "config": config,
    }

    best_target = -1.0
    best_target_round = 0
    best_tail = -1.0
    best_tail_round = 0
    exposure_hits = 0

    for r in tqdm(
        range(
            1,
            args.num_rounds + 1,
        ),
        desc=f"hc-{h_tag}",
    ):
        online_clients = [
            int(v)
            for v in (
                client_rng.choice(
                    total_clients,
                    size=min(
                        args.num_online_clients,
                        args.num_clients,
                    ),
                    replace=False,
                )
            )
        ]

        # Direct target-class exposure.
        target_selected_samples = int(
            sum(
                client_class_counts[
                    client
                ][
                    args.target_class
                ]
                for client
                in online_clients
            )
        )

        selected_total_samples = int(
            sum(
                len(
                    controlled_partition[
                        client
                    ]
                )
                for client
                in online_clients
            )
        )

        exposure_event = int(
            target_selected_samples > 0
        )
        exposure_hits += (
            exposure_event
        )
        empirical_p = (
            exposure_hits
            / float(r)
        )

        exposure_mass = (
            target_selected_samples
            / float(
                selected_total_samples
            )
            if selected_total_samples
            > 0
            else 0.0
        )

        logger.info(
            "Round %d selected clients: %s | "
            "target exposure=%d samples=%d mass=%.6f "
            "empirical_P=%.6f theory_P=%.6f",
            r,
            sorted(
                online_clients
            ),
            exposure_event,
            target_selected_samples,
            exposure_mass,
            empirical_p,
            theoretical_p,
        )

        global_params = (
            global_model
            .download_params()
        )
        centres = (
            global_model
            .get_smoothed_centers()
        )

        local_states = []
        local_sizes = []

        for client in online_clients:
            counts = torch.tensor(
                client_class_counts[
                    client
                ],
                dtype=torch.float32,
            )

            dist = (
                counts
                / counts.sum()
                .clamp_min(
                    1e-8
                )
            )

            indices2data.load(
                controlled_partition[
                    client
                ]
            )

            local_sizes.append(
                len(indices2data)
            )

            local = Local(
                data_client=(
                    indices2data
                ),
                args=args,
            )

            state = (
                local.local_train(
                    args=args,
                    global_params=(
                        copy.deepcopy(
                            global_params
                        )
                    ),
                    dist=dist,
                    smoothed_centers=(
                        centres
                    ),
                )
            )

            local_states.append(
                copy.deepcopy(
                    state
                )
            )

        accepted_params = (
            global_model.fedavg(
                local_states,
                local_sizes,
            )
        )

        metrics = (
            global_model.evaluate(
                params=accepted_params,
                data_test=test_data,
                batch_size_test=(
                    args.batch_size_test
                ),
                class_counts=(
                    global_class_counts
                ),
                target_class=(
                    args.target_class
                ),
                num_classes=(
                    args.num_classes
                ),
            )
        )

        global_model.syn_model.load_state_dict(
            copy.deepcopy(
                accepted_params
            )
        )
        global_model.update_smoothed_centers(
            accepted_params[
                "classifier.weight"
            ].detach()
        )

        target_acc = (
            metrics["target"]
        )
        tail_acc = (
            metrics["tail"]
        )

        if target_acc > best_target:
            best_target = (
                target_acc
            )
            best_target_round = r

        if tail_acc > best_tail:
            best_tail = tail_acc
            best_tail_round = r

        logger.info(
            "Round %d | Global %.4f | Head %.4f | "
            "Middle %.4f | Tail %.4f | "
            "Class-%d %.4f | "
            "best_class%d %.4f@%d",
            r,
            metrics["global"],
            metrics["head"],
            metrics["middle"],
            metrics["tail"],
            args.target_class,
            target_acc,
            args.target_class,
            best_target,
            best_target_round,
        )

        row = {
            "round": int(r),
            "selected_clients": (
                json.dumps(
                    sorted(
                        online_clients
                    )
                )
            ),
            "target_exposure_event": (
                exposure_event
            ),
            "target_selected_samples": (
                target_selected_samples
            ),
            "selected_total_samples": (
                selected_total_samples
            ),
            "target_exposure_mass": (
                exposure_mass
            ),
            "cumulative_empirical_exposure": (
                empirical_p
            ),
            "theoretical_exposure": (
                theoretical_p
            ),
            "target_class_accuracy": (
                target_acc
            ),
            "tail_accuracy": (
                metrics["tail"]
            ),
            "head_accuracy": (
                metrics["head"]
            ),
            "middle_accuracy": (
                metrics["middle"]
            ),
            "global_accuracy": (
                metrics["global"]
            ),
        }

        with open(
            metrics_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )
            )
            writer.writerow(
                row
            )

        history[
            "round"
        ].append(
            int(r)
        )
        history[
            "selected_clients"
        ].append(
            sorted(
                online_clients
            )
        )
        history[
            "target_exposure_event"
        ].append(
            exposure_event
        )
        history[
            "target_selected_samples"
        ].append(
            target_selected_samples
        )
        history[
            "target_exposure_mass"
        ].append(
            exposure_mass
        )
        history[
            "cumulative_empirical_exposure"
        ].append(
            empirical_p
        )
        history[
            "theoretical_exposure"
        ].append(
            theoretical_p
        )
        history[
            "target_class_accuracy"
        ].append(
            target_acc
        )
        history[
            "tail_accuracy"
        ].append(
            metrics["tail"]
        )
        history[
            "head_accuracy"
        ].append(
            metrics["head"]
        )
        history[
            "middle_accuracy"
        ].append(
            metrics["middle"]
        )
        history[
            "global_accuracy"
        ].append(
            metrics["global"]
        )
        history[
            "per_class_accuracy"
        ].append(
            metrics["per_class"]
        )

        if (
            r % args.save_every
            == 0
            or r
            == args.num_rounds
        ):
            with open(
                history_path,
                "wb",
            ) as f:
                pickle.dump(
                    history,
                    f,
                )

            torch.save(
                {
                    "round": r,
                    "state_dict": (
                        copy.deepcopy(
                            accepted_params
                        )
                    ),
                    "config": config,
                    "empirical_exposure": (
                        empirical_p
                    ),
                    "target_class_accuracy": (
                        target_acc
                    ),
                    "tail_accuracy": (
                        tail_acc
                    ),
                },
                checkpoint_path,
            )

    torch.save(
        {
            "state_dict": (
                copy.deepcopy(
                    global_model
                    .syn_model
                    .state_dict()
                )
            ),
            "config": config,
            "final_empirical_exposure": (
                exposure_hits
                / float(
                    args.num_rounds
                )
            ),
            "final_target_accuracy": (
                history[
                    "target_class_accuracy"
                ][-1]
            ),
            "final_tail_accuracy": (
                history[
                    "tail_accuracy"
                ][-1]
            ),
        },
        final_path,
    )

    final_empirical = (
        exposure_hits
        / float(
            args.num_rounds
        )
    )

    logger.info("=" * 90)
    logger.info(
        "FINISHED FIXED-N / VARIED-h EXPERIMENT"
    )
    logger.info(
        "target_class=%d N=%d actual_h=%d",
        args.target_class,
        target_N,
        actual_host_count,
    )
    logger.info(
        "Theoretical exposure: %.6f",
        theoretical_p,
    )
    logger.info(
        "Empirical exposure over %d rounds: %.6f",
        args.num_rounds,
        final_empirical,
    )
    logger.info(
        "Final class-%d accuracy: %.4f",
        args.target_class,
        history[
            "target_class_accuracy"
        ][-1],
    )
    logger.info(
        "Final Tail accuracy: %.4f",
        history[
            "tail_accuracy"
        ][-1],
    )
    logger.info(
        "Final Global accuracy: %.4f",
        history[
            "global_accuracy"
        ][-1],
    )
    logger.info(
        "Best class-%d accuracy "
        "(diagnostic only): %.4f @ round %d",
        args.target_class,
        best_target,
        best_target_round,
    )
    logger.info(
        "Best Tail accuracy "
        "(diagnostic only): %.4f @ round %d",
        best_tail,
        best_tail_round,
    )
    logger.info(
        "Metrics CSV: %s",
        os.path.abspath(
            metrics_path
        ),
    )
    logger.info(
        "History: %s",
        os.path.abspath(
            history_path
        ),
    )
    logger.info(
        "Config: %s",
        os.path.abspath(
            config_path
        ),
    )
    logger.info(
        "Checkpoint: %s",
        os.path.abspath(
            checkpoint_path
        ),
    )
    logger.info(
        "Final model: %s",
        os.path.abspath(
            final_path
        ),
    )
    logger.info("=" * 90)


if __name__ == "__main__":
    run()
