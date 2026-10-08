# -*- coding: utf-8 -*-
"""
fedtig_ear_direct_cifar10.py
============================

Experiment 11: E/A/R Direct Mechanism Evidence.

Dedicated CIFAR-10-LT mechanism stress-test:
- CIFAR-10-LT, IF=100, Dirichlet alpha=0.5
- K=20, m=4 by default
- controlled class 9: N_9=50 fixed, h_9=2 fixed
- ResNet-8, 200 rounds, E=10, batch=32, fixed lr=0.1, seed=42
- no test-set rollback

Formal module parameters:
- S epsilon=0.75
- C alpha=0.70, warmup=30
- A tail_agg_weight=1.50 -> beta=0.125

Recommended modes:
    fedlf, s, c, a
Optional:
    fedtig

Direct diagnostics:
Exposure:
    target_exposure_event, target_selected_samples,
    tail_exposure_rate, tail_exposure_mass
Acquisition:
    ||grad W_tail||_F / ||grad W_classifier||_F
Retention:
    ordinary FedAvg tail update versus post-A tail update,
    measured by projection and cosine against the selected
    tail-specialist local update direction.

Diagnostics are passive and never feed back into training.
"""
"""
fedtig_cifar100_unified.py
==========================

Unified CIFAR-100-LT experiment runner for FedLF / FedTIG stage ablations.

Design goal
-----------
Use ONE shared training scaffold and switch only the FedTIG governance modules:

    mode= fedlf : S=0 C=0 A=0
    mode= s     : S=1 C=0 A=0
    mode= c     : S=0 C=1 A=0
    mode= a     : S=0 C=0 A=1
    mode= sc    : S=1 C=1 A=0
    mode= sa    : S=1 C=0 A=1
    mode= ca    : S=0 C=1 A=1
    mode= fedtig: S=1 C=1 A=1

The implementation follows the FedTIG manuscript definitions:

S / Exposure Governance
    p_c = (N_c + 1) / (sum_j N_j + C)
    I_c = -log p_c
    q_k,c = n_k,c / N_k
    R_k = sum_c q_k,c I_c
    H_k = -sum_c q_k,c log(q_k,c) / log C
    D_k = q_k^T D q_k
where D is the cosine-distance matrix between server-maintained class centres.
After client-wise min-max normalization:
    s_k = 0.25 + 0.75 * (R~_k + H~_k + D~_k) / 3
Participation fairness:
    g_k = 1 / sqrt(1 + m_k)
With epsilon=0.75 and m=8, approximately 2 focused slots are sampled
proportional to s_k*g_k, and the remaining ~6 slots are uniform exploration.

C / Acquisition Governance
For locally present classes:
    a_k,c = sqrt(N_k / max(n_k,c, 1))
then mean-normalize over present classes and clip to [0.60, 1.80].

For sample i:
    margin_i = sim(h_i, centre_yi) - max_{c != yi} sim(h_i, centre_c)
    u_i = 1 + sigmoid(-margin_i / sigma_margin)
After the first c_warmup rounds, use:
    omega_i = c_alpha * a_k,yi + (1-c_alpha) * u_i
During warmup, semantic uncertainty is disabled and only a_k,yi is used.
C reweights only the main FedLF classification term; centre/decorrelation
terms are unchanged.

A / Retention Governance
First compute ordinary sample-size-weighted FedAvg. For each tail class c,
form a tail-specialist average using only selected clients that actually
contain class c:
    r_c = w_tail_c - wbar_c
    w_next_c = wbar_c + (1 + beta) * r_c
    beta = clip(0.25 * (tail_agg_weight - 1), 0, 0.35)
Default tail_agg_weight=1.50 -> beta=0.125 -> residual factor=1.125.
If no selected client contains c, retain the previous global classifier row.

Important
---------
- Fixed-round reporting only. No test-set rollback.
- Head/Tail thresholds follow FedLF:
    CIFAR-100 IF=100/50: Head > 200, Tail < 20
    CIFAR-100 IF=10:     Head > 300, Tail < 60
- Default CIFAR-100 protocol keeps the previously used late LR schedule
  (0.1 -> 0.05 from rounds 101..200) so existing same-protocol FedLF remains
  a valid reference. Use --lr_schedule fixed only when explicitly desired.
"""

import argparse
import csv
import copy
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
# CLI
# =============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 11: CIFAR-10 E/A/R direct mechanism evidence"
    )

    path_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--mode",
        type=str,
        default="fedtig",
        choices=["fedlf", "s", "c", "a", "sc", "sa", "ca", "fedtig"],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_online_clients", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--num_epochs_local_training", type=int, default=10)
    parser.add_argument("--batch_size_local_training", type=int, default=32)
    parser.add_argument("--batch_size_test", type=int, default=500)
    parser.add_argument("--lr_local_training", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--non_iid_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imb_type", type=str, default="exp")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--rs_alpha", type=float, default=0.5)

    parser.add_argument(
        "--path_cifar10",
        type=str,
        default=os.path.join(path_dir, "data/CIFAR10/"),
    )
    parser.add_argument(
        "--path_cifar100",
        type=str,
        default=os.path.join(path_dir, "data/CIFAR100/"),
    )

    # Formal FedTIG defaults from the manuscript.
    parser.add_argument("--s_epsilon", type=float, default=0.75)
    parser.add_argument("--c_alpha", type=float, default=0.70)
    parser.add_argument("--c_warmup", type=int, default=30)
    parser.add_argument("--tail_agg_weight", type=float, default=1.50)

    # Shared optimizer schedule for Protocol-C (CIFAR-100).
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="fixed",
        choices=["late_decay", "fixed"],
        help="late_decay keeps the existing CIFAR-100 0.1->0.05 schedule.",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Checkpoint/history interval in rounds.",
    )

    parser.add_argument(
        "--target_class",
        type=int,
        default=9,
        help="Controlled target tail class.",
    )
    parser.add_argument(
        "--controlled_host_count",
        type=int,
        default=2,
        choices=[2],
        help="Fixed host-client count for target class.",
    )

    return parser


def mode_flags(mode):
    mode = mode.lower()
    return {
        "S": mode in {"s", "sc", "sa", "fedtig"},
        "C": mode in {"c", "sc", "ca", "fedtig"},
        "A": mode in {"a", "sa", "ca", "fedtig"},
    }


# =============================================================================
# Reproducibility / evaluation
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_eval_thresholds(dataset, imb_factor):
    ratio = float(imb_factor)

    if dataset == "cifar10":
        if ratio >= 0.099:
            return 1500, 600
        return 1500, 200

    if dataset == "cifar100":
        if ratio >= 0.099:
            return 300, 60
        return 200, 20

    raise ValueError("Unsupported dataset: {}".format(dataset))


def partition_sha256(list_client2indices):
    payload = []
    for indices in list_client2indices:
        payload.append([int(v) for v in list(indices)])

    raw = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def hmean(head, tail):
    if head + tail <= 0:
        return 0.0
    return 2.0 * head * tail / (head + tail)


def minmax_np(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= eps:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


# =============================================================================
# Experiment-11 controlled partition and direct E/A/R diagnostics
# =============================================================================
def labels_from_dataset(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    labels = []
    for _, y in dataset:
        labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def fixed_host_order(num_clients, seed, target_class):
    rng = np.random.RandomState(
        int(seed) + 10000 + int(target_class)
    )
    return [int(v) for v in rng.permutation(num_clients)]


def build_controlled_partition(
    base_partition,
    labels,
    target_indices,
    target_class,
    host_count,
    seed,
):
    base = [
        [int(v) for v in indices]
        for indices in base_partition
    ]
    num_clients = len(base)

    target_indices = [int(v) for v in target_indices]
    target_set = set(target_indices)

    if host_count <= 0 or host_count > num_clients:
        raise ValueError("Invalid controlled host count.")

    if any(
        int(labels[idx]) != int(target_class)
        for idx in target_indices
    ):
        raise RuntimeError(
            "Target index set contains non-target labels."
        )

    original_union = []
    non_target_by_client = []

    for indices in base:
        original_union.extend(indices)
        non_target_by_client.append(
            [idx for idx in indices if idx not in target_set]
        )

    if len(original_union) != len(set(original_union)):
        raise RuntimeError(
            "Base partition contains duplicated sample indices."
        )

    original_target_set = {
        idx
        for idx in original_union
        if int(labels[idx]) == int(target_class)
    }

    if original_target_set != target_set:
        raise RuntimeError(
            "Target sample identity mismatch in base partition."
        )

    new_partition = [
        list(v)
        for v in non_target_by_client
    ]

    host_order = fixed_host_order(
        num_clients,
        seed,
        target_class,
    )
    active_hosts = host_order[:host_count]

    target_rng = np.random.RandomState(
        int(seed) + 20000 + int(target_class)
    )
    ordered_target = [
        int(v)
        for v in target_rng.permutation(
            np.asarray(target_indices, dtype=np.int64)
        )
    ]

    chunks = np.array_split(
        np.asarray(ordered_target, dtype=np.int64),
        host_count,
    )

    host_target_counts = {}

    for host, chunk in zip(active_hosts, chunks):
        chunk_list = [int(v) for v in chunk.tolist()]
        new_partition[int(host)].extend(chunk_list)
        host_target_counts[int(host)] = len(chunk_list)

    new_union = [
        idx
        for indices in new_partition
        for idx in indices
    ]

    if len(new_union) != len(set(new_union)):
        raise RuntimeError(
            "Controlled partition contains duplicate indices."
        )

    if set(new_union) != set(original_union):
        raise RuntimeError(
            "Controlled partition lost or added samples."
        )

    for k in range(num_clients):
        new_non_target = [
            idx
            for idx in new_partition[k]
            if int(labels[idx]) != int(target_class)
        ]
        if new_non_target != non_target_by_client[k]:
            raise RuntimeError(
                "A non-target sample moved clients."
            )

    actual_hosts = [
        k
        for k in range(num_clients)
        if any(
            int(labels[idx]) == int(target_class)
            for idx in new_partition[k]
        )
    ]

    if len(actual_hosts) != int(host_count):
        raise RuntimeError(
            "Actual host count does not match request."
        )

    return (
        new_partition,
        host_order,
        actual_hosts,
        host_target_counts,
    )


def theoretical_exposure_probability(K, m, h):
    K = int(K)
    m = int(m)
    h = int(h)

    if h <= 0:
        return float("nan")
    if K - h < m:
        return 1.0

    return (
        1.0
        - math.comb(K - h, m)
        / math.comb(K, m)
    )


def _safe_vector_norm(tensor):
    if tensor is None:
        return 0.0

    tensor = tensor.detach().float()

    if tensor.numel() == 0:
        return 0.0

    value = torch.linalg.vector_norm(tensor)

    if not torch.isfinite(value):
        return 0.0

    return float(value.item())


def _safe_cosine(a, b, eps=1e-12):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)

    denom = (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    if (
        not torch.isfinite(denom)
        or float(denom.item()) <= eps
    ):
        return 0.0

    value = torch.dot(a, b) / denom

    if not torch.isfinite(value):
        return 0.0

    return float(value.item())


def _safe_projection(global_delta, local_delta, eps=1e-12):
    g = global_delta.detach().float().reshape(-1)
    l = local_delta.detach().float().reshape(-1)

    denom = torch.dot(l, l) + eps
    value = torch.dot(g, l) / denom

    if not torch.isfinite(value):
        return 0.0

    return float(value.item())


def standard_fedavg_for_diagnostics(
    local_states,
    local_sizes,
):
    if not local_states:
        return None

    total = float(sum(local_sizes))

    if total <= 0:
        return None

    result = copy.deepcopy(local_states[0])

    for name in result:
        first = local_states[0][name]

        if not torch.is_floating_point(first):
            result[name] = first.clone()
            continue

        value = torch.zeros_like(first)

        for state, n_k in zip(
            local_states,
            local_sizes,
        ):
            value += state[name] * float(n_k)

        result[name] = value / total

    return result


def compute_exposure_diagnostics(
    online_clients,
    client_class_counts,
    tail_classes,
    target_class,
):
    selected_total = int(
        sum(
            sum(client_class_counts[int(k)])
            for k in online_clients
        )
    )

    per_class_event = {}
    per_class_selected = {}

    for c in tail_classes:
        count_c = int(
            sum(
                int(client_class_counts[int(k)][int(c)])
                for k in online_clients
            )
        )
        per_class_selected[int(c)] = count_c
        per_class_event[int(c)] = int(count_c > 0)

    selected_tail = int(sum(per_class_selected.values()))

    target_selected = int(
        sum(
            int(client_class_counts[int(k)][int(target_class)])
            for k in online_clients
        )
    )

    return {
        "target_exposure_event": int(target_selected > 0),
        "target_selected_samples": target_selected,
        "tail_exposure_rate": (
            float(np.mean(list(per_class_event.values())))
            if per_class_event
            else 0.0
        ),
        "tail_exposure_mass": (
            float(selected_tail) / float(selected_total)
            if selected_total > 0
            else 0.0
        ),
        "tail_selected_samples": selected_tail,
        "selected_total_samples": selected_total,
        "per_class_event": per_class_event,
        "per_class_selected_samples": per_class_selected,
    }


def compute_retention_diagnostics(
    old_global,
    fedavg_base,
    post_state,
    local_states,
    local_sizes,
    local_class_counts,
    tail_classes,
):
    empty = {
        "retention_num_tail_classes": 0,
        "retention_pre_cos": 0.0,
        "retention_post_cos": 0.0,
        "retention_pre_projection": 0.0,
        "retention_post_projection": 0.0,
        "retention_projection_gain": 0.0,
        "retention_cos_gain": 0.0,
        "tail_local_update_norm": 0.0,
        "fedavg_tail_update_norm": 0.0,
        "final_tail_update_norm": 0.0,
        "per_class": {},
    }

    if (
        old_global is None
        or fedavg_base is None
        or post_state is None
        or "classifier.weight" not in old_global
        or "classifier.weight" not in fedavg_base
        or "classifier.weight" not in post_state
    ):
        return empty

    old_w = old_global["classifier.weight"]
    base_w = fedavg_base["classifier.weight"]
    post_w = post_state["classifier.weight"]

    pre_cos = []
    post_cos = []
    pre_proj = []
    post_proj = []
    local_norms = []
    base_norms = []
    post_norms = []
    details = {}

    for c in tail_classes:
        contributors = [
            i
            for i, counts in enumerate(local_class_counts)
            if int(counts[int(c)]) > 0
        ]

        if not contributors:
            continue

        total = float(
            sum(local_sizes[i] for i in contributors)
        )

        if total <= 0:
            continue

        specialist = None

        for i in contributors:
            row = (
                local_states[i]["classifier.weight"][int(c)]
                * float(local_sizes[i])
            )
            specialist = row if specialist is None else specialist + row

        specialist = specialist / total

        local_delta = specialist - old_w[int(c)]
        pre_delta = base_w[int(c)] - old_w[int(c)]
        post_delta = post_w[int(c)] - old_w[int(c)]

        pc = _safe_cosine(pre_delta, local_delta)
        qc = _safe_cosine(post_delta, local_delta)
        pp = _safe_projection(pre_delta, local_delta)
        qp = _safe_projection(post_delta, local_delta)

        pre_cos.append(pc)
        post_cos.append(qc)
        pre_proj.append(pp)
        post_proj.append(qp)
        local_norms.append(_safe_vector_norm(local_delta))
        base_norms.append(_safe_vector_norm(pre_delta))
        post_norms.append(_safe_vector_norm(post_delta))

        details[int(c)] = {
            "num_contributors": int(len(contributors)),
            "pre_cos": pc,
            "post_cos": qc,
            "pre_projection": pp,
            "post_projection": qp,
        }

    if not details:
        return empty

    return {
        "retention_num_tail_classes": int(len(details)),
        "retention_pre_cos": float(np.mean(pre_cos)),
        "retention_post_cos": float(np.mean(post_cos)),
        "retention_pre_projection": float(np.mean(pre_proj)),
        "retention_post_projection": float(np.mean(post_proj)),
        "retention_projection_gain": float(
            np.mean(post_proj) - np.mean(pre_proj)
        ),
        "retention_cos_gain": float(
            np.mean(post_cos) - np.mean(pre_cos)
        ),
        "tail_local_update_norm": float(np.mean(local_norms)),
        "fedavg_tail_update_norm": float(np.mean(base_norms)),
        "final_tail_update_norm": float(np.mean(post_norms)),
        "per_class": details,
    }


# =============================================================================
# S: Exposure Governance
# =============================================================================
def class_cosine_distance_matrix(class_centres):
    """Cosine distance matrix between server-maintained class centres."""
    centres = class_centres.detach().float()
    centres = F.normalize(centres, p=2, dim=1, eps=1e-8)
    sim = centres @ centres.t()
    dist = (1.0 - sim).clamp(min=0.0, max=2.0)
    dist.fill_diagonal_(0.0)
    return dist.cpu().numpy()


def compute_exposure_scores(
    client_class_counts,
    global_class_counts,
    class_centres,
):
    """Compute manuscript R/H/D semantic-density score for every client."""
    counts = np.asarray(client_class_counts, dtype=np.float64)
    global_counts = np.asarray(global_class_counts, dtype=np.float64)

    num_clients, num_classes = counts.shape

    prior = (global_counts + 1.0) / (
        float(global_counts.sum()) + float(num_classes)
    )
    info = -np.log(np.clip(prior, 1e-12, None))

    dist_mat = class_cosine_distance_matrix(class_centres)

    rarity = np.zeros(num_clients, dtype=np.float64)
    entropy = np.zeros(num_clients, dtype=np.float64)
    dispersion = np.zeros(num_clients, dtype=np.float64)

    log_c = max(math.log(max(num_classes, 2)), 1e-12)

    for k in range(num_clients):
        nk = float(counts[k].sum())
        if nk <= 0:
            continue

        q = counts[k] / nk

        rarity[k] = float(np.sum(q * info))

        nz = q > 0
        entropy[k] = float(
            -np.sum(q[nz] * np.log(q[nz])) / log_c
        )

        dispersion[k] = float(q @ dist_mat @ q)

    r_norm = minmax_np(rarity)
    h_norm = minmax_np(entropy)
    d_norm = minmax_np(dispersion)

    score = 0.25 + 0.75 * (
        r_norm + h_norm + d_norm
    ) / 3.0

    return score, rarity, entropy, dispersion


def sample_clients_exposure_governance(
    total_clients,
    client_scores,
    participation_counts,
    num_online_clients,
    epsilon,
    rng,
):
    """
    Explicit focused+uniform mixture matching the manuscript text.

    epsilon = uniform-exploration fraction.
    For epsilon=.75 and m=8:
        focus slots = round((1-.75)*8) = 2
        uniform slots = 6
    """
    total_clients = np.asarray(list(total_clients), dtype=np.int64)
    m = min(int(num_online_clients), len(total_clients))

    if m <= 0:
        return []

    focus_slots = int(round((1.0 - float(epsilon)) * m))
    focus_slots = max(0, min(focus_slots, m))
    uniform_slots = m - focus_slots

    score = np.asarray(client_scores, dtype=np.float64)
    debt = 1.0 / np.sqrt(
        1.0 + np.asarray(participation_counts, dtype=np.float64)
    )

    weighted = np.maximum(score * debt, 1e-12)
    weighted = weighted / weighted.sum()

    selected = []

    if focus_slots > 0:
        focus = rng.choice(
            total_clients,
            size=focus_slots,
            replace=False,
            p=weighted,
        )
        selected.extend([int(v) for v in focus])

    remaining = np.asarray(
        [v for v in total_clients if int(v) not in set(selected)],
        dtype=np.int64,
    )

    if uniform_slots > 0:
        uniform = rng.choice(
            remaining,
            size=uniform_slots,
            replace=False,
        )
        selected.extend([int(v) for v in uniform])

    return selected


# =============================================================================
# FedLF feature decorrelation
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
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        if x.ndim != 2:
            return x.new_tensor(0.0)

        n, _ = x.shape
        if n <= 1:
            return x.new_tensor(0.0)

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(
            self.eps
            + x.var(
                dim=0,
                keepdim=True,
                unbiased=False,
            )
        )

        corr_mat = torch.matmul(x.t(), x)
        loss = self._off_diagonal(corr_mat).pow(2).mean()
        return loss / n


# =============================================================================
# Global / server
# =============================================================================
class Global(object):
    def __init__(self, num_classes, device, args):
        self.device = device
        self.num_classes = num_classes
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
        ).to(device)

        self.smoothed_centers = torch.zeros(
            (num_classes, 256),
            device=device,
        )
        self.ema_decay = 0.9
        self.center_initialized = False

    def update_smoothed_centers(self, new_centers):
        new_centers = new_centers.detach()

        if not torch.isfinite(new_centers).all():
            return

        if not self.center_initialized:
            self.smoothed_centers = new_centers.clone()
            self.center_initialized = True
        else:
            self.smoothed_centers = (
                self.ema_decay * self.smoothed_centers
                + (1.0 - self.ema_decay) * new_centers
            )

    def get_smoothed_centers(self):
        return self.smoothed_centers.detach()

    def download_params(self):
        return copy.deepcopy(self.syn_model.state_dict())

    def initialize_for_model_fusion(
        self,
        list_dicts_local_params,
        list_nums_local_data,
        enable_a=False,
        tail_classes=None,
        client_class_counts=None,
        old_global=None,
        tail_agg_weight=1.50,
    ):
        """Standard FedAvg plus optional FedTIG-A on tail classifier rows."""
        if not list_dicts_local_params:
            raise ValueError("No local parameters to aggregate.")

        total_num = float(sum(list_nums_local_data))
        if total_num <= 0:
            raise ValueError("Total local sample count must be positive.")

        global_params = copy.deepcopy(list_dicts_local_params[0])

        # Base FedAvg.
        for name_param in global_params:
            first = list_dicts_local_params[0][name_param]

            if not torch.is_floating_point(first):
                global_params[name_param] = first.clone()
                continue

            weighted_sum = torch.zeros_like(first)

            for local_state, num_data in zip(
                list_dicts_local_params,
                list_nums_local_data,
            ):
                weighted_sum = weighted_sum + (
                    local_state[name_param] * float(num_data)
                )

            global_params[name_param] = weighted_sum / total_num

        if not enable_a:
            return global_params

        tail_classes = list(tail_classes or [])
        if not tail_classes:
            return global_params

        beta = float(
            np.clip(
                0.25 * (float(tail_agg_weight) - 1.0),
                0.0,
                0.35,
            )
        )

        for name_param in ("classifier.weight", "classifier.bias"):
            if name_param not in global_params:
                continue

            for c in tail_classes:
                tail_client_indices = []

                if client_class_counts is not None:
                    for local_idx, counts in enumerate(client_class_counts):
                        if int(counts[c]) > 0:
                            tail_client_indices.append(local_idx)

                # No tail information this round: preserve prior classifier row.
                if not tail_client_indices:
                    if old_global is not None and name_param in old_global:
                        global_params[name_param][c] = (
                            old_global[name_param][c].clone()
                        )
                    continue

                tail_total = float(
                    sum(
                        list_nums_local_data[i]
                        for i in tail_client_indices
                    )
                )

                tail_avg = None
                for i in tail_client_indices:
                    value = (
                        list_dicts_local_params[i][name_param][c]
                        * float(list_nums_local_data[i])
                    )
                    tail_avg = value if tail_avg is None else tail_avg + value

                tail_avg = tail_avg / tail_total

                base_row = global_params[name_param][c].clone()
                residual = tail_avg - base_row

                global_params[name_param][c] = (
                    base_row + (1.0 + beta) * residual
                )

        return global_params

    def global_eval_more(
        self,
        params,
        data_test,
        batch_size_test,
        class_counts,
    ):
        self.syn_model.load_state_dict(params)
        self.syn_model.eval()

        majority_threshold, minority_threshold = get_eval_thresholds(
            self.dataset,
            self.imb_factor,
        )

        many_correct = many_total = 0
        medium_correct = medium_total = 0
        few_correct = few_total = 0

        test_loader = DataLoader(
            data_test,
            batch_size=batch_size_test,
            shuffle=False,
        )

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, outputs = self.syn_model(images)
                predicts = outputs.argmax(dim=-1)

                for label, predict in zip(labels, predicts):
                    y = int(label.item())
                    samples_num = int(class_counts[y])
                    correct = int(predict.item() == y)

                    if samples_num > majority_threshold:
                        many_total += 1
                        many_correct += correct
                    elif samples_num < minority_threshold:
                        few_total += 1
                        few_correct += correct
                    else:
                        medium_total += 1
                        medium_correct += correct

        def acc(correct, total):
            if total <= 0:
                return 0.0
            return round(correct / total, 4)

        many_acc = acc(many_correct, many_total)
        medium_acc = acc(medium_correct, medium_total)
        few_acc = acc(few_correct, few_total)

        return many_acc, medium_acc, few_acc, many_acc, few_acc

    def global_eval(self, params, data_test, batch_size_test):
        self.syn_model.load_state_dict(params)
        self.syn_model.eval()

        test_loader = DataLoader(
            data_test,
            batch_size=batch_size_test,
            shuffle=False,
        )

        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, outputs = self.syn_model(images)
                predicts = outputs.argmax(dim=-1)
                correct += int((predicts == labels).sum().item())

        return correct / len(data_test)


# =============================================================================
# Local client
# =============================================================================
class Local(object):
    def __init__(
        self,
        data_client,
        class_list,
        args,
        tail_classes=None,
    ):
        self.data_client = data_client
        self.class_list = list(class_list)
        self.device = args.device
        self.tail_classes = [
            int(v)
            for v in (tail_classes or [])
        ]

        self.criterion = CrossEntropyLoss().to(args.device)
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

        self.last_ear_stats = {
            "acq_steps": 0,
            "tail_grad_share_mean": 0.0,
            "tail_grad_norm_mean": 0.0,
            "classifier_grad_norm_mean": 0.0,
            "tail_label_fraction": 0.0,
        }

    def _class_acquisition_weights(self, args):
        """
        a_k,c = sqrt(N_k / max(n_k,c,1)),
        normalized over present classes and clipped to [0.60, 1.80].
        """
        counts = torch.tensor(
            self.class_list,
            dtype=torch.float32,
            device=self.device,
        )

        present = counts > 0
        weights = torch.zeros_like(counts)

        if present.any():
            n_k = counts[present].sum().clamp_min(1.0)

            raw = torch.sqrt(
                n_k / counts[present].clamp_min(1.0)
            )

            raw = raw / raw.mean().clamp_min(1e-8)
            raw = raw.clamp(0.60, 1.80)
            weights[present] = raw

        return weights

    def local_train(
        self,
        args,
        global_params,
        dist,
        smoothed_centers,
        current_round,
        enable_c,
    ):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        # FedLF adaptive logit adjustment.
        dist = dist.to(self.device)

        if dist.numel() != args.num_classes:
            tmp = torch.zeros(args.num_classes, device=self.device)
            num_copy = min(len(dist), args.num_classes)
            tmp[:num_copy] = dist[:num_copy]
            dist = tmp

        cdist = dist / dist.max().clamp_min(1e-8)
        cdist = cdist * (1.0 - args.rs_alpha) + args.rs_alpha
        cdist = cdist.clamp(0.50, 1.00).reshape(1, -1)

        # Shared server semantic/class centres.
        feature_centers = smoothed_centers.to(self.device).detach()

        if (
            not torch.isfinite(feature_centers).all()
            or feature_centers.abs().sum() < 1e-8
        ):
            feature_centers = (
                self.local_model.classifier.weight.detach().clone()
            )

        # Shared bounded centre-gap implementation.
        with torch.no_grad():
            gap = torch.cdist(feature_centers, feature_centers, p=2)
            mask = ~torch.eye(
                args.num_classes,
                dtype=torch.bool,
                device=self.device,
            )
            valid_gap = gap[mask]

            if valid_gap.numel() > 0:
                max_gap = valid_gap.max()
            else:
                max_gap = torch.tensor(1.0, device=self.device)

            gap_val = float(
                torch.clamp(
                    max_gap,
                    min=0.05,
                    max=2.0,
                ).item()
            )

        # Shared CIFAR-100 schedule.
        base_lr = float(args.lr_local_training)

        if args.lr_schedule == "fixed":
            local_lr = base_lr
        else:
            if current_round <= 100:
                local_lr = base_lr
            else:
                frac = min((current_round - 100) / 100.0, 1.0)
                local_lr = base_lr * (1.0 - 0.5 * frac)

        for group in self.optimizer.param_groups:
            group["lr"] = local_lr

        class_acq_weights = None
        if enable_c:
            class_acq_weights = self._class_acquisition_weights(args)

        ear_share_sum = 0.0
        ear_tail_norm_sum = 0.0
        ear_total_norm_sum = 0.0
        ear_steps = 0
        ear_tail_labels = 0
        ear_total_labels = 0

        ear_tail_tensor = None
        if self.tail_classes:
            ear_tail_tensor = torch.tensor(
                self.tail_classes,
                dtype=torch.long,
                device=self.device,
            )

        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(
                self.data_client,
                batch_size=args.batch_size_local_training,
                shuffle=True,
            )

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = transform_train(images)

                hs, _ = self.local_model(images)
                ws = self.local_model.classifier.weight

                logits = cdist * hs.mm(ws.transpose(0, 1))

                # ---------------------------------------------------------
                # FedLF main classification term, optionally governed by C.
                # ---------------------------------------------------------
                if not enable_c:
                    loss1 = self.criterion(logits, labels)
                else:
                    a_sample = class_acq_weights[labels]

                    # First c_warmup rounds:
                    # do not use semantic uncertainty because centres are
                    # not yet sufficiently stable.
                    if current_round <= int(args.c_warmup):
                        omega = a_sample
                    else:
                        h_norm = F.normalize(hs, p=2, dim=1, eps=1e-8)
                        c_norm = F.normalize(
                            feature_centers,
                            p=2,
                            dim=1,
                            eps=1e-8,
                        )
                        sim = h_norm @ c_norm.t()

                        true_sim = sim.gather(
                            1,
                            labels.view(-1, 1),
                        ).squeeze(1)

                        competitor = sim.clone()
                        competitor.scatter_(
                            1,
                            labels.view(-1, 1),
                            -1e9,
                        )
                        strongest_other = competitor.max(dim=1).values
                        margin = true_sim - strongest_other

                        sigma_m = margin.detach().std(
                            unbiased=False
                        ).clamp_min(1e-6)

                        uncertainty = (
                            1.0
                            + torch.sigmoid(
                                -margin / sigma_m
                            )
                        ).detach()

                        uncertainty = (
                            uncertainty
                            / uncertainty.mean().clamp_min(1e-8)
                        )

                        omega = (
                            float(args.c_alpha) * a_sample
                            + (1.0 - float(args.c_alpha)) * uncertainty
                        )

                    # Preserve relative weights while keeping the effective
                    # step size comparable to ordinary CE.
                    omega = omega / omega.mean().clamp_min(1e-8)

                    loss_per_sample = F.cross_entropy(
                        logits,
                        labels,
                        reduction="none",
                    )
                    loss1 = (loss_per_sample * omega).mean()

                # FedLF class-centre objective L_C.
                features_square = torch.sum(
                    hs.pow(2),
                    dim=1,
                    keepdim=True,
                )
                centers_square = torch.sum(
                    feature_centers.pow(2),
                    dim=1,
                    keepdim=True,
                )
                features_into_centers = hs.matmul(feature_centers.t())

                dist_2 = (
                    features_square
                    - 2 * features_into_centers
                    + centers_square.t()
                )
                dist_2 = torch.sqrt(dist_2.clamp_min(1e-8))

                one_hot = F.one_hot(
                    labels,
                    args.num_classes,
                ).to(self.device)

                dist_2 = dist_2 + one_hot * gap_val
                loss2 = self.criterion(-dist_2, labels)

                # FedLF feature decorrelation L_D.
                loss_decorr = self.feddecorr(hs)

                loss = (
                    loss1
                    + 0.01 * loss2
                    + 0.01 * loss_decorr
                )

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad_w = self.local_model.classifier.weight.grad

                    if (
                        grad_w is not None
                        and torch.isfinite(grad_w).all()
                    ):
                        total_norm = torch.linalg.vector_norm(grad_w)

                        if (
                            ear_tail_tensor is not None
                            and ear_tail_tensor.numel() > 0
                        ):
                            tail_norm = torch.linalg.vector_norm(
                                grad_w[ear_tail_tensor]
                            )
                        else:
                            tail_norm = torch.tensor(
                                0.0,
                                device=grad_w.device,
                            )

                        share = (
                            tail_norm
                            / total_norm.clamp_min(1e-12)
                        )

                        if torch.isfinite(share):
                            ear_share_sum += float(share.item())
                            ear_tail_norm_sum += float(tail_norm.item())
                            ear_total_norm_sum += float(total_norm.item())
                            ear_steps += 1

                    ear_total_labels += int(labels.numel())

                    if ear_tail_tensor is not None:
                        ear_tail_labels += int(
                            torch.isin(
                                labels,
                                ear_tail_tensor,
                            ).sum().item()
                        )

                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(),
                    max_norm=20.0,
                )

                self.optimizer.step()

        self.last_ear_stats = {
            "acq_steps": int(ear_steps),
            "tail_grad_share_mean": (
                float(ear_share_sum / ear_steps)
                if ear_steps > 0
                else 0.0
            ),
            "tail_grad_norm_mean": (
                float(ear_tail_norm_sum / ear_steps)
                if ear_steps > 0
                else 0.0
            ),
            "classifier_grad_norm_mean": (
                float(ear_total_norm_sum / ear_steps)
                if ear_steps > 0
                else 0.0
            ),
            "tail_label_fraction": (
                float(ear_tail_labels) / float(ear_total_labels)
                if ear_total_labels > 0
                else 0.0
            ),
        }

        return copy.deepcopy(self.local_model.state_dict())


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment():
    args = build_parser().parse_args()
    flags = mode_flags(args.mode)

    if args.dataset != "cifar10":
        raise ValueError(
            "Experiment 11 is restricted to CIFAR-10."
        )

    if int(args.num_classes) != 10:
        raise ValueError("CIFAR-10 requires --num_classes 10")

    if int(args.controlled_host_count) != 2:
        raise ValueError(
            "Experiment 11 requires controlled_host_count=2."
        )

    if not (0.0 <= args.s_epsilon <= 1.0):
        raise ValueError("--s_epsilon must be in [0,1]")

    if not (0.0 <= args.c_alpha <= 1.0):
        raise ValueError("--c_alpha must be in [0,1]")

    seed_everything(args.seed)

    paper_if = int(round(1.0 / float(args.imb_factor)))
    alpha_tag = str(args.non_iid_alpha).replace(".", "p")
    eps_tag = str(args.s_epsilon).replace(".", "p")
    ca_tag = str(args.c_alpha).replace(".", "p")
    taw_tag = str(args.tail_agg_weight).replace(".", "p")

    run_name = (
        f"ear_direct_{args.dataset}_{args.mode}"
        f"_IF{paper_if}"
        f"_alpha{alpha_tag}"
        f"_h{args.controlled_host_count}"
        f"_m{args.num_online_clients}"
        f"_seed{args.seed}"
        f"_eps{eps_tag}"
        f"_ca{ca_tag}"
        f"_w{args.c_warmup}"
        f"_taw{taw_tag}"
    )

    log_dir = "Logs"
    output_dir = os.path.join("outputs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_filename = run_name + ".log"
    log_path = os.path.join(log_dir, log_filename)
    history_path = os.path.join(output_dir, "head_tail_history.pkl")
    ear_history_path = os.path.join(output_dir, "ear_history.pkl")
    ear_csv_path = os.path.join(output_dir, "ear_metrics.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    final_model_path = os.path.join(output_dir, "final_model.pth")
    config_path = os.path.join(output_dir, "config.json")

    logger = setup_logging(log_filename)

    beta = float(
        np.clip(
            0.25 * (float(args.tail_agg_weight) - 1.0),
            0.0,
            0.35,
        )
    )

    logger.info("=" * 90)
    logger.info("EXPERIMENT 11: E/A/R DIRECT MECHANISM EVIDENCE")
    logger.info("Run name: %s", run_name)
    logger.info("Mode: %s | S=%s C=%s A=%s",
                args.mode, flags["S"], flags["C"], flags["A"])
    logger.info("dataset:%s, num_classes:%d", args.dataset, args.num_classes)
    logger.info(
        "imb_factor:%s (IF=%d), non_iid_alpha:%s, rs_alpha:%s",
        args.imb_factor,
        paper_if,
        args.non_iid_alpha,
        args.rs_alpha,
    )
    logger.info(
        "K=%d, online=%d, participation=%.4f",
        args.num_clients,
        args.num_online_clients,
        float(args.num_online_clients) / float(args.num_clients),
    )
    logger.info(
        "rounds=%d, local_epochs=%d, batch=%d, base_lr=%s, lr_schedule=%s",
        args.num_rounds,
        args.num_epochs_local_training,
        args.batch_size_local_training,
        args.lr_local_training,
        args.lr_schedule,
    )
    logger.info(
        "S epsilon=%.4f | C alpha=%.4f warmup=%d | "
        "A tail_agg_weight=%.4f beta=%.4f residual_factor=%.4f",
        args.s_epsilon,
        args.c_alpha,
        args.c_warmup,
        args.tail_agg_weight,
        beta,
        1.0 + beta,
    )
    logger.info("seed=%d", args.seed)
    logger.info("test-set rollback enabled: %s", ENABLE_TEST_ROLLBACK)
    logger.info("Log file: %s", os.path.abspath(log_path))
    logger.info("Output directory: %s", os.path.abspath(output_dir))
    logger.info("=" * 90)

    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    if args.dataset == "cifar10":
        data_local_training = datasets.CIFAR10(
            args.path_cifar10,
            train=True,
            download=True,
            transform=transform_all,
        )
        data_global_test = datasets.CIFAR10(
            args.path_cifar10,
            train=False,
            download=True,
            transform=transform_all,
        )
    else:
        data_local_training = datasets.CIFAR100(
            args.path_cifar100,
            train=True,
            download=True,
            transform=transform_all,
        )
        data_global_test = datasets.CIFAR100(
            args.path_cifar100,
            train=False,
            download=True,
            transform=transform_all,
        )

    # Long-tail construction.
    list_label2indices = classify_label(
        data_local_training,
        args.num_classes,
    )

    class_counts, list_label2indices_train_new = train_long_tail(
        copy.deepcopy(list_label2indices),
        args.num_classes,
        args.imb_factor,
        args.imb_type,
    )

    # Base Dirichlet partition.
    base_partition = clients_indices(
        copy.deepcopy(list_label2indices_train_new),
        args.num_classes,
        args.num_clients,
        args.non_iid_alpha,
        args.seed,
    )

    base_fingerprint = partition_sha256(base_partition)

    labels_np = labels_from_dataset(data_local_training)
    target_indices = copy.deepcopy(
        list_label2indices_train_new[int(args.target_class)]
    )

    (
        list_client2indices,
        host_order,
        active_hosts,
        host_target_counts,
    ) = build_controlled_partition(
        base_partition=base_partition,
        labels=labels_np,
        target_indices=target_indices,
        target_class=int(args.target_class),
        host_count=int(args.controlled_host_count),
        seed=int(args.seed),
    )

    original_dict_per_client = show_clients_data_distribution(
        data_local_training,
        list_client2indices,
        args.num_classes,
    )

    fingerprint = partition_sha256(list_client2indices)
    target_N = int(len(target_indices))
    actual_h = int(len(active_hosts))
    target_theoretical_random_exposure = (
        theoretical_exposure_probability(
            K=args.num_clients,
            m=args.num_online_clients,
            h=actual_h,
        )
    )

    majority_threshold, tail_threshold = get_eval_thresholds(
        args.dataset,
        args.imb_factor,
    )

    tail_classes = [
        i
        for i, cnt in enumerate(class_counts)
        if int(cnt) < tail_threshold
    ]

    logger.info("Global class counts: %s", list(map(int, class_counts)))
    logger.info(
        "Controlled target class %d | N=%d | h=%d | hosts=%s | per_host=%s",
        int(args.target_class),
        target_N,
        actual_h,
        active_hosts,
        host_target_counts,
    )
    logger.info(
        "Random-sampling theoretical target Exposure at m=%d: %.6f",
        args.num_online_clients,
        target_theoretical_random_exposure,
    )
    logger.info("Base partition SHA256: %s", base_fingerprint)
    logger.info(
        "Paper thresholds: Head > %d, Tail < %d",
        majority_threshold,
        tail_threshold,
    )
    logger.info(
        "Tail classes (<%d): %s",
        tail_threshold,
        tail_classes,
    )
    logger.info(
        "Client data sizes: %s",
        [int(len(v)) for v in list_client2indices],
    )
    logger.info("Partition SHA256: %s", fingerprint)

    config = {
        "implementation": "FedTIG_unified_formal_v1",
        "mode": args.mode,
        "S": bool(flags["S"]),
        "C": bool(flags["C"]),
        "A": bool(flags["A"]),
        "dataset": args.dataset,
        "IF": paper_if,
        "imb_factor": float(args.imb_factor),
        "non_iid_alpha": float(args.non_iid_alpha),
        "rs_alpha": float(args.rs_alpha),
        "num_clients": int(args.num_clients),
        "num_online_clients": int(args.num_online_clients),
        "participation_rate": float(args.num_online_clients) / float(args.num_clients),
        "num_rounds": int(args.num_rounds),
        "local_epochs": int(args.num_epochs_local_training),
        "batch_size": int(args.batch_size_local_training),
        "base_lr": float(args.lr_local_training),
        "lr_schedule": args.lr_schedule,
        "seed": int(args.seed),
        "s_epsilon": float(args.s_epsilon),
        "c_alpha": float(args.c_alpha),
        "c_warmup": int(args.c_warmup),
        "tail_agg_weight": float(args.tail_agg_weight),
        "beta": beta,
        "head_threshold": int(majority_threshold),
        "tail_threshold": int(tail_threshold),
        "tail_classes": tail_classes,
        "partition_sha256": fingerprint,
        "base_partition_sha256": base_fingerprint,
        "target_class": int(args.target_class),
        "target_N": target_N,
        "controlled_host_count": actual_h,
        "active_hosts": [int(v) for v in active_hosts],
        "host_target_counts": {
            str(k): int(v)
            for k, v in host_target_counts.items()
        },
        "random_target_exposure_theory": float(
            target_theoretical_random_exposure
        ),
        "mechanism_protocol": (
            "fixed N_target and h_target; m=4 stress test "
            "to avoid natural Exposure saturation"
        ),
        "test_rollback": False,
        "S_definition": "R/H/class-centre-cosine-dispersion + participation debt + focused/uniform mixture",
        "C_definition": "present-class sqrt inverse-frequency + cosine-margin uncertainty; uncertainty OFF during warmup",
        "A_definition": "FedAvg base + bounded tail-specialist residual on classifier rows",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    global_model = Global(
        num_classes=args.num_classes,
        device=args.device,
        args=args,
    )

    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)

    # Initialize server class centres once from classifier rows.
    initial_state = global_model.syn_model.state_dict()
    global_model.update_smoothed_centers(
        initial_state["classifier.weight"].detach()
    )

    random_state = np.random.RandomState(args.seed)
    participation_counts = np.zeros(args.num_clients, dtype=np.int64)

    re_trained_acc = []
    ft_many = []
    ft_medium = []
    ft_few = []
    head_acc_history = []
    tail_acc_history = []
    hmean_history = []
    selected_clients_history = []
    exposure_event_history = []
    exposure_mass_history = []
    semantic_score_history = []

    best_tail = -1.0
    best_global = -1.0
    best_round = 0
    best_state = copy.deepcopy(global_model.download_params())

    ear_fieldnames = [
        "round",
        "mode",
        "selected_clients",
        "target_exposure_event",
        "target_selected_samples",
        "tail_exposure_rate",
        "tail_exposure_mass",
        "tail_selected_samples",
        "selected_total_samples",
        "acq_steps",
        "tail_grad_share",
        "tail_grad_norm",
        "classifier_grad_norm",
        "tail_label_fraction",
        "retention_num_tail_classes",
        "retention_pre_cos",
        "retention_post_cos",
        "retention_pre_projection",
        "retention_post_projection",
        "retention_projection_gain",
        "retention_cos_gain",
        "tail_local_update_norm",
        "fedavg_tail_update_norm",
        "final_tail_update_norm",
        "global_acc",
        "head_acc",
        "middle_acc",
        "tail_acc",
        "hmean",
    ]

    with open(
        ear_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=ear_fieldnames,
        ).writeheader()

    ear_round_history = []
    target_exposure_history = []
    tail_exposure_rate_history = []
    tail_grad_share_history = []
    retention_pre_projection_history = []
    retention_post_projection_history = []

    for r in tqdm(
        range(1, args.num_rounds + 1),
        desc=f"unified-{args.mode}",
    ):
        # ---------------------------------------------------------
        # S / Exposure Governance
        # ---------------------------------------------------------
        if flags["S"]:
            centres_for_sampling = global_model.get_smoothed_centers()

            scores, rarity, entropy, dispersion = compute_exposure_scores(
                original_dict_per_client,
                class_counts,
                centres_for_sampling,
            )

            online_clients = sample_clients_exposure_governance(
                total_clients,
                scores,
                participation_counts,
                args.num_online_clients,
                args.s_epsilon,
                random_state,
            )

            semantic_score_history.append(
                [float(v) for v in scores]
            )
        else:
            online_clients = random_state.choice(
                total_clients,
                size=min(args.num_online_clients, len(total_clients)),
                replace=False,
            )
            online_clients = [int(v) for v in online_clients]
            semantic_score_history.append(None)

        for client in online_clients:
            participation_counts[int(client)] += 1

        selected_clients_history.append(
            sorted([int(v) for v in online_clients])
        )

        logger.info(
            "Round %d selected clients: %s",
            r,
            sorted([int(v) for v in online_clients]),
        )

        exposure_diag = compute_exposure_diagnostics(
            online_clients=online_clients,
            client_class_counts=original_dict_per_client,
            tail_classes=tail_classes,
            target_class=int(args.target_class),
        )

        exposure_event = int(
            exposure_diag["target_exposure_event"]
        )
        exposure_mass = float(
            exposure_diag["tail_exposure_mass"]
        )

        exposure_event_history.append(exposure_event)
        exposure_mass_history.append(exposure_mass)

        # ---------------------------------------------------------
        # Local training
        # ---------------------------------------------------------
        old_global = global_model.download_params()
        global_params = copy.deepcopy(old_global)
        smoothed_centers = global_model.get_smoothed_centers()

        list_dicts_local_params = []
        list_nums_local_data = []
        list_client_class_counts = []
        list_local_ear_stats = []

        for client in online_clients:
            client = int(client)

            cnts = torch.tensor(
                original_dict_per_client[client],
                dtype=torch.float32,
            )
            dist = cnts / cnts.sum().clamp_min(1e-8)

            indices2data.load(list_client2indices[client])
            data_client = indices2data

            list_nums_local_data.append(len(data_client))
            list_client_class_counts.append(
                np.asarray(
                    original_dict_per_client[client],
                    dtype=np.int64,
                )
            )

            local_model = Local(
                data_client=data_client,
                class_list=original_dict_per_client[client],
                args=args,
                tail_classes=tail_classes,
            )

            local_out = int(
                local_model.local_model.classifier.weight.shape[0]
            )
            global_out = int(
                global_params["classifier.weight"].shape[0]
            )

            if local_out != global_out:
                raise RuntimeError(
                    "Classifier class-count mismatch: "
                    f"local={local_out}, global={global_out}, "
                    f"expected={args.num_classes}"
                )

            local_params = local_model.local_train(
                args=args,
                global_params=copy.deepcopy(global_params),
                dist=dist,
                smoothed_centers=smoothed_centers,
                current_round=r,
                enable_c=flags["C"],
            )

            list_dicts_local_params.append(
                copy.deepcopy(local_params)
            )
            list_local_ear_stats.append(
                copy.deepcopy(local_model.last_ear_stats)
            )

        diagnostic_fedavg_base = (
            standard_fedavg_for_diagnostics(
                list_dicts_local_params,
                list_nums_local_data,
            )
        )

        # ---------------------------------------------------------
        # A / Retention Governance
        # ---------------------------------------------------------
        accepted_params = global_model.initialize_for_model_fusion(
            list_dicts_local_params,
            list_nums_local_data,
            enable_a=flags["A"],
            tail_classes=tail_classes,
            client_class_counts=list_client_class_counts,
            old_global=old_global,
            tail_agg_weight=args.tail_agg_weight,
        )

        retention_diag = compute_retention_diagnostics(
            old_global=old_global,
            fedavg_base=diagnostic_fedavg_base,
            post_state=accepted_params,
            local_states=list_dicts_local_params,
            local_sizes=list_nums_local_data,
            local_class_counts=list_client_class_counts,
            tail_classes=tail_classes,
        )

        total_acq_steps = int(
            sum(
                int(s.get("acq_steps", 0))
                for s in list_local_ear_stats
            )
        )

        def _step_weighted_mean(key):
            if total_acq_steps <= 0:
                return 0.0

            return float(
                sum(
                    float(s.get(key, 0.0))
                    * int(s.get("acq_steps", 0))
                    for s in list_local_ear_stats
                )
                / total_acq_steps
            )

        acquisition_diag = {
            "acq_steps": total_acq_steps,
            "tail_grad_share": _step_weighted_mean(
                "tail_grad_share_mean"
            ),
            "tail_grad_norm": _step_weighted_mean(
                "tail_grad_norm_mean"
            ),
            "classifier_grad_norm": _step_weighted_mean(
                "classifier_grad_norm_mean"
            ),
            "tail_label_fraction": _step_weighted_mean(
                "tail_label_fraction"
            ),
        }

        # Fixed-round evaluation. Never test-driven rollback.
        many, medium, few, head_acc, tail_acc = (
            global_model.global_eval_more(
                accepted_params,
                data_global_test,
                args.batch_size_test,
                class_counts,
            )
        )

        global_acc = global_model.global_eval(
            accepted_params,
            data_global_test,
            args.batch_size_test,
        )

        global_model.syn_model.load_state_dict(
            copy.deepcopy(accepted_params)
        )

        global_model.update_smoothed_centers(
            accepted_params["classifier.weight"].detach()
        )

        hm = hmean(head_acc, tail_acc)

        re_trained_acc.append(global_acc)
        ft_many.append(many)
        ft_medium.append(medium)
        ft_few.append(few)
        head_acc_history.append((r, head_acc))
        tail_acc_history.append((r, tail_acc))
        hmean_history.append((r, hm))

        if (
            tail_acc > best_tail
            or (
                abs(tail_acc - best_tail) < 1e-8
                and global_acc > best_global
            )
        ):
            best_tail = tail_acc
            best_global = global_acc
            best_round = r
            best_state = copy.deepcopy(accepted_params)

        logger.info(
            "Round %d | Global %.4f | Head %.4f | Middle %.4f | "
            "Tail %.4f | Hmean %.4f | ExpEvent %d | ExpMass %.6f | "
            "best_tail %.4f@%d",
            r,
            global_acc,
            head_acc,
            medium,
            tail_acc,
            hm,
            exposure_event,
            exposure_mass,
            best_tail,
            best_round,
        )

        logger.info(
            "EAR Round %d | targetExp %d targetSamples %d "
            "tailExpRate %.4f | gradShare %.6f | "
            "Retention proj %.6f -> %.6f (gain %.6f) | "
            "cos %.6f -> %.6f",
            r,
            exposure_diag["target_exposure_event"],
            exposure_diag["target_selected_samples"],
            exposure_diag["tail_exposure_rate"],
            acquisition_diag["tail_grad_share"],
            retention_diag["retention_pre_projection"],
            retention_diag["retention_post_projection"],
            retention_diag["retention_projection_gain"],
            retention_diag["retention_pre_cos"],
            retention_diag["retention_post_cos"],
        )

        ear_row = {
            "round": int(r),
            "mode": args.mode,
            "selected_clients": json.dumps(
                sorted([int(v) for v in online_clients])
            ),
            "target_exposure_event": int(
                exposure_diag["target_exposure_event"]
            ),
            "target_selected_samples": int(
                exposure_diag["target_selected_samples"]
            ),
            "tail_exposure_rate": float(
                exposure_diag["tail_exposure_rate"]
            ),
            "tail_exposure_mass": float(
                exposure_diag["tail_exposure_mass"]
            ),
            "tail_selected_samples": int(
                exposure_diag["tail_selected_samples"]
            ),
            "selected_total_samples": int(
                exposure_diag["selected_total_samples"]
            ),
            "acq_steps": int(acquisition_diag["acq_steps"]),
            "tail_grad_share": float(
                acquisition_diag["tail_grad_share"]
            ),
            "tail_grad_norm": float(
                acquisition_diag["tail_grad_norm"]
            ),
            "classifier_grad_norm": float(
                acquisition_diag["classifier_grad_norm"]
            ),
            "tail_label_fraction": float(
                acquisition_diag["tail_label_fraction"]
            ),
            "retention_num_tail_classes": int(
                retention_diag["retention_num_tail_classes"]
            ),
            "retention_pre_cos": float(
                retention_diag["retention_pre_cos"]
            ),
            "retention_post_cos": float(
                retention_diag["retention_post_cos"]
            ),
            "retention_pre_projection": float(
                retention_diag["retention_pre_projection"]
            ),
            "retention_post_projection": float(
                retention_diag["retention_post_projection"]
            ),
            "retention_projection_gain": float(
                retention_diag["retention_projection_gain"]
            ),
            "retention_cos_gain": float(
                retention_diag["retention_cos_gain"]
            ),
            "tail_local_update_norm": float(
                retention_diag["tail_local_update_norm"]
            ),
            "fedavg_tail_update_norm": float(
                retention_diag["fedavg_tail_update_norm"]
            ),
            "final_tail_update_norm": float(
                retention_diag["final_tail_update_norm"]
            ),
            "global_acc": float(global_acc),
            "head_acc": float(head_acc),
            "middle_acc": float(medium),
            "tail_acc": float(tail_acc),
            "hmean": float(hm),
        }

        with open(
            ear_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.DictWriter(
                f,
                fieldnames=ear_fieldnames,
            ).writerow(ear_row)

        ear_round_history.append(
            {
                "round": int(r),
                "mode": args.mode,
                "selected_clients": sorted(
                    [int(v) for v in online_clients]
                ),
                "exposure": copy.deepcopy(exposure_diag),
                "acquisition": copy.deepcopy(acquisition_diag),
                "retention": copy.deepcopy(retention_diag),
                "performance": {
                    "global": float(global_acc),
                    "head": float(head_acc),
                    "middle": float(medium),
                    "tail": float(tail_acc),
                    "hmean": float(hm),
                },
            }
        )

        target_exposure_history.append(
            int(exposure_diag["target_exposure_event"])
        )
        tail_exposure_rate_history.append(
            float(exposure_diag["tail_exposure_rate"])
        )
        tail_grad_share_history.append(
            float(acquisition_diag["tail_grad_share"])
        )
        retention_pre_projection_history.append(
            float(retention_diag["retention_pre_projection"])
        )
        retention_post_projection_history.append(
            float(retention_diag["retention_post_projection"])
        )

        with open(ear_history_path, "wb") as f:
            pickle.dump(
                {
                    "rounds": ear_round_history,
                    "target_exposure_event": target_exposure_history,
                    "tail_exposure_rate": tail_exposure_rate_history,
                    "tail_grad_share": tail_grad_share_history,
                    "retention_pre_projection": (
                        retention_pre_projection_history
                    ),
                    "retention_post_projection": (
                        retention_post_projection_history
                    ),
                    "config": config,
                },
                f,
            )

        if (
            r % int(args.save_every) == 0
            or r == args.num_rounds
        ):
            history = {
                "global": re_trained_acc,
                "head": head_acc_history,
                "tail": tail_acc_history,
                "hmean": hmean_history,
                "many": ft_many,
                "medium": ft_medium,
                "few": ft_few,
                "selected_clients": selected_clients_history,
                "tail_exposure_event": exposure_event_history,
                "tail_exposure_mass": exposure_mass_history,
                "semantic_scores": semantic_score_history,
                "target_exposure_event_direct": target_exposure_history,
                "tail_exposure_rate_direct": tail_exposure_rate_history,
                "tail_grad_share": tail_grad_share_history,
                "retention_pre_projection": retention_pre_projection_history,
                "retention_post_projection": retention_post_projection_history,
                "participation_counts": participation_counts.tolist(),
                "best_tail": best_tail,
                "best_round": best_round,
                "partition_sha256": fingerprint,
                "config": config,
            }

            with open(history_path, "wb") as f:
                pickle.dump(history, f)

            torch.save(
                {
                    "round": r,
                    "state_dict": copy.deepcopy(accepted_params),
                    "best_state_dict": copy.deepcopy(best_state),
                    "best_tail": best_tail,
                    "best_global": best_global,
                    "best_round": best_round,
                    "config": config,
                },
                checkpoint_path,
            )

            logger.info(
                "Checkpoint @%d | Global %.4f | Head %.4f | "
                "Middle %.4f | Tail %.4f | Hmean %.4f",
                r,
                global_acc,
                head_acc,
                medium,
                tail_acc,
                hm,
            )

    final_state = global_model.syn_model.state_dict()

    torch.save(
        {
            "state_dict": copy.deepcopy(final_state),
            "best_state_dict": copy.deepcopy(best_state),
            "best_tail": best_tail,
            "best_global": best_global,
            "best_round": best_round,
            "config": config,
        },
        final_model_path,
    )

    logger.info("=" * 90)
    logger.info("FINISHED UNIFIED %s", args.mode.upper())
    logger.info("Final Global: %.4f", re_trained_acc[-1])
    logger.info("Final Head: %.4f", head_acc_history[-1][1])
    logger.info("Final Middle: %.4f", ft_medium[-1])
    logger.info("Final Tail: %.4f", tail_acc_history[-1][1])
    logger.info("Final Hmean: %.4f", hmean_history[-1][1])
    logger.info(
        "Best Tail (diagnostic only; NOT fixed-round main result): "
        "%.4f @ round %d",
        best_tail,
        best_round,
    )
    logger.info(
        "Mean target Exposure over all rounds: %.6f",
        float(np.mean(target_exposure_history))
        if target_exposure_history
        else 0.0,
    )
    logger.info(
        "Mean tail gradient share over all rounds: %.6f",
        float(np.mean(tail_grad_share_history))
        if tail_grad_share_history
        else 0.0,
    )
    logger.info(
        "Mean retention projection pre/post: %.6f -> %.6f",
        float(np.mean(retention_pre_projection_history))
        if retention_pre_projection_history
        else 0.0,
        float(np.mean(retention_post_projection_history))
        if retention_post_projection_history
        else 0.0,
    )
    logger.info("EAR CSV: %s", os.path.abspath(ear_csv_path))
    logger.info("EAR history: %s", os.path.abspath(ear_history_path))
    logger.info("History file: %s", os.path.abspath(history_path))
    logger.info("Checkpoint file: %s", os.path.abspath(checkpoint_path))
    logger.info("Final model file: %s", os.path.abspath(final_model_path))
    logger.info("Config file: %s", os.path.abspath(config_path))
    logger.info("Partition SHA256: %s", fingerprint)
    logger.info("=" * 90)


if __name__ == "__main__":
    run_experiment()
