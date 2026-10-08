# -*- coding: utf-8 -*-
"""
fedtig_beta_pareto_cifar10.py
=============================

Experiment 12: A-strength / Head-Tail Pareto analysis.

Purpose
-------
Isolate Retention Governance (A) under the MAIN CIFAR-10 protocol and vary
only beta:

    beta in {0.00, 0.05, 0.125, 0.20, 0.30}

Important:
- A ON with beta=0 is NOT the same as A OFF.
- beta=0 still applies the tail-specialist residual with factor 1.0.
- A OFF / FedLF should be shown as a separate baseline point.

Protocol
--------
CIFAR-10-LT IF=100
Dirichlet alpha=0.5
K=20, m=8 (40% participation)
ResNet-8
200 rounds
10 local epochs
batch=32
fixed lr=0.1
seed=42
S=OFF, C=OFF, A=ON
no test-set rollback
fixed Round-200 reporting

Mapping used by the implementation:
    beta = 0.25 * (tail_agg_weight - 1)
so:
    beta=0.00  -> tail_agg_weight=1.0
    beta=0.05  -> tail_agg_weight=1.2
    beta=0.125 -> tail_agg_weight=1.5
    beta=0.20  -> tail_agg_weight=1.8
    beta=0.30  -> tail_agg_weight=2.2

Outputs per run:
    Logs/beta_cifar10_beta{tag}_IF100_alpha0p5_seed42.log
    outputs/beta_cifar10_beta{tag}_IF100_alpha0p5_seed42/
        config.json
        metrics.csv
        head_tail_history.pkl
        checkpoint.pth
        final_model.pth
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
        description="Experiment 12: CIFAR-10 A-beta Head-Tail Pareto sweep"
    )

    path_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--mode",
        type=str,
        default="a",
        choices=["a"],
        help="Experiment 12 isolates A only.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_online_clients", type=int, default=8)
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
    parser.add_argument(
        "--beta",
        type=float,
        default=0.125,
        choices=[0.0, 0.05, 0.125, 0.20, 0.30],
        help="Retention residual strength beta.",
    )

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
    def __init__(self, data_client, class_list, args):
        self.data_client = data_client
        self.class_list = list(class_list)
        self.device = args.device

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

                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(),
                    max_norm=20.0,
                )

                self.optimizer.step()

        return copy.deepcopy(self.local_model.state_dict())


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment():
    args = build_parser().parse_args()
    flags = mode_flags(args.mode)

    if args.dataset != "cifar10":
        raise ValueError("Experiment 12 is restricted to CIFAR-10.")

    if int(args.num_classes) != 10:
        raise ValueError("CIFAR-10 requires --num_classes 10")

    if int(args.num_clients) != 20:
        raise ValueError("Experiment 12 requires K=20.")

    if int(args.num_online_clients) != 8:
        raise ValueError("Experiment 12 requires m=8 (40% participation).")

    if int(args.num_epochs_local_training) != 10:
        raise ValueError("Experiment 12 requires local_epochs=10.")

    if args.lr_schedule != "fixed":
        raise ValueError("Experiment 12 requires fixed lr.")

    args.tail_agg_weight = 1.0 + 4.0 * float(args.beta)

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
    beta_tag = str(args.beta).replace(".", "p")

    run_name = (
        f"beta_cifar10_beta{beta_tag}"
        f"_IF{paper_if}"
        f"_alpha{alpha_tag}"
        f"_seed{args.seed}"
    )

    log_dir = "Logs"
    output_dir = os.path.join("outputs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_filename = run_name + ".log"
    log_path = os.path.join(log_dir, log_filename)
    history_path = os.path.join(output_dir, "head_tail_history.pkl")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    final_model_path = os.path.join(output_dir, "final_model.pth")
    config_path = os.path.join(output_dir, "config.json")

    logger = setup_logging(log_filename)

    beta = float(args.beta)

    beta_from_weight = float(
        np.clip(
            0.25 * (float(args.tail_agg_weight) - 1.0),
            0.0,
            0.35,
        )
    )

    if abs(beta - beta_from_weight) > 1e-12:
        raise RuntimeError("beta/tail_agg_weight mapping mismatch.")

    logger.info("=" * 90)
    logger.info("EXPERIMENT 12: A-BETA / HEAD-TAIL PARETO")
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

    # Dirichlet client partition.
    list_client2indices = clients_indices(
        copy.deepcopy(list_label2indices_train_new),
        args.num_classes,
        args.num_clients,
        args.non_iid_alpha,
        args.seed,
    )

    original_dict_per_client = show_clients_data_distribution(
        data_local_training,
        list_client2indices,
        args.num_classes,
    )

    fingerprint = partition_sha256(list_client2indices)

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

    metrics_fields = [
        "round",
        "beta",
        "tail_agg_weight",
        "global_acc",
        "head_acc",
        "middle_acc",
        "tail_acc",
        "hmean",
        "tail_exposure_event",
        "tail_exposure_mass",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=metrics_fields,
        ).writeheader()

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

        # Lightweight exposure diagnostics.
        selected_tail_count = 0
        selected_total_count = 0

        for client in online_clients:
            counts_k = original_dict_per_client[int(client)]
            selected_total_count += int(sum(counts_k))
            selected_tail_count += int(
                sum(counts_k[c] for c in tail_classes)
            )

        exposure_event = int(selected_tail_count > 0)
        exposure_mass = (
            float(selected_tail_count) / float(selected_total_count)
            if selected_total_count > 0
            else 0.0
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

        with open(
            metrics_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.DictWriter(
                f,
                fieldnames=metrics_fields,
            ).writerow(
                {
                    "round": int(r),
                    "beta": float(beta),
                    "tail_agg_weight": float(args.tail_agg_weight),
                    "global_acc": float(global_acc),
                    "head_acc": float(head_acc),
                    "middle_acc": float(medium),
                    "tail_acc": float(tail_acc),
                    "hmean": float(hm),
                    "tail_exposure_event": int(exposure_event),
                    "tail_exposure_mass": float(exposure_mass),
                }
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
    logger.info("FINISHED BETA PARETO | beta=%.3f", beta)
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
    logger.info("Metrics CSV: %s", os.path.abspath(metrics_path))
    logger.info("History file: %s", os.path.abspath(history_path))
    logger.info("Checkpoint file: %s", os.path.abspath(checkpoint_path))
    logger.info("Final model file: %s", os.path.abspath(final_model_path))
    logger.info("Config file: %s", os.path.abspath(config_path))
    logger.info("Partition SHA256: %s", fingerprint)
    logger.info("=" * 90)


if __name__ == "__main__":
    run_experiment()
