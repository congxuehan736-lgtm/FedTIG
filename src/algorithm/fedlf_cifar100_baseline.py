# -*- coding: utf-8 -*-
"""
fedlf_cifar100_baseline.py
==========================

Purpose
-------
A SAME-PROTOCOL FedLF baseline for the CIFAR-100-LT FedTIG experiment.

This file is deliberately built from the shared training scaffold used by the
current CIFAR-100 FedTIG script, but disables the three FedTIG modules:

    S / Exposure governance: semantic client sampling       -> OFF
    C / Acquisition governance: confidence/tail reweighting -> OFF
    A / Retention governance: tail residual aggregation     -> OFF

What is kept identical/shared with the current CIFAR-100 FedTIG scaffold:
    - CIFAR-100-LT construction
    - Dirichlet client partitioning
    - ResNet-8 backbone
    - FedLF adaptive logit adjustment
    - FedLF class-centre loss
    - FedLF feature-decorrelation loss
    - shared centre/EMA stabilization used by this project implementation
    - the same late LR schedule used by the current CIFAR-100 FedTIG run
    - K / online clients / rounds / local epochs / batch size / seed
    - fixed-round (Round 200) reporting; no test-set rollback

Server aggregation is ordinary sample-size-weighted FedAvg.
Client participation is uniform random sampling without replacement.

IMPORTANT
---------
This is the correct baseline for a FAIR *within-project* comparison against the
current CIFAR-100 FedTIG run. It is not claimed to be a byte-for-byte copy of
the authors' public FedLF repository. The goal here is to hold the shared
training scaffold fixed and remove only FedTIG S/C/A.

Recommended location:
    D:\\Desktop\\FedLF_project\\FedLF\\fedlf_cifar100_baseline.py
"""

from Model.log_model import setup_logging
from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import (
    classify_label,
    show_clients_data_distribution,
    Indices2Dataset,
)
from Dataset.sample_dirichlet import clients_indices

import copy
import hashlib
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm


# Fixed-round reporting only. Never use test accuracy to roll model parameters back.
ENABLE_TEST_ROLLBACK = False


# =============================================================================
# Reproducibility / evaluation definitions
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_eval_thresholds(dataset, imb_factor):
    """FedLF paper Head/Tail thresholds.

    Project argument ``imb_factor`` is minority/majority ratio:
      0.01 -> IF=100
      0.02 -> IF=50
      0.10 -> IF=10

    CIFAR-100-LT:
      IF=100/50 -> Head > 200, Tail < 20
      IF=10     -> Head > 300, Tail < 60
    """
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
    """Audit fingerprint for the generated client partition."""
    payload = []
    for indices in list_client2indices:
        payload.append([int(v) for v in list(indices)])

    raw = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# =============================================================================
# FedLF feature decorrelation -- kept identical to current project scaffold
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
# Server
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

        # Shared project stabilization for the FedLF centre loss.
        # This is retained in BOTH FedLF baseline and current FedTIG so the
        # S/C/A comparison does not change the underlying local learner.
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
    ):
        """STANDARD sample-size-weighted FedAvg. FedTIG-A is disabled."""
        if not list_dicts_local_params:
            raise ValueError("No local parameters to aggregate.")

        total_num = float(sum(list_nums_local_data))
        if total_num <= 0:
            raise ValueError("Total local sample count must be positive.")

        global_params = copy.deepcopy(list_dicts_local_params[0])

        for name_param in global_params:
            first = list_dicts_local_params[0][name_param]

            # Keep integer buffers valid.
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

        return global_params

    def global_eval_more(
        self,
        params,
        data_test,
        batch_size_test,
        class_counts,
    ):
        """Head / Middle / Tail evaluation using FedLF paper thresholds."""
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
    def __init__(self, data_client, args):
        self.data_client = data_client
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

        # Same optimizer family as current project implementation.
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
        current_round,
    ):
        """FedLF local learner with FedTIG-C disabled.

        Kept:
          L_A: FedLF adaptive logit adjustment
          L_C: shared FedLF centre loss
          L_D: FedLF decorrelation loss

        Removed:
          confidence-driven weighting
          inverse-frequency sample weighting
          dynamic tail_boost
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        # ------------------------------------------------------------------
        # FedLF adaptive logit adjustment -- shared with current FedTIG
        # ------------------------------------------------------------------
        dist = dist.to(self.device)

        if dist.numel() != args.num_classes:
            tmp = torch.zeros(args.num_classes, device=self.device)
            num_copy = min(len(dist), args.num_classes)
            tmp[:num_copy] = dist[:num_copy]
            dist = tmp

        cdist = dist / dist.max().clamp_min(1e-8)
        cdist = cdist * (1.0 - args.rs_alpha) + args.rs_alpha

        # Keep the same numerical guard as current FedTIG implementation.
        cdist = cdist.clamp(0.50, 1.00).reshape(1, -1)

        # ------------------------------------------------------------------
        # Shared centre representation used by the current project scaffold
        # ------------------------------------------------------------------
        feature_centers = smoothed_centers.to(self.device).detach()

        if (
            not torch.isfinite(feature_centers).all()
            or feature_centers.abs().sum() < 1e-8
        ):
            feature_centers = (
                self.local_model.classifier.weight.detach().clone()
            )

        # Same bounded centre-gap implementation as current FedTIG so that
        # S/C/A, not a centre-loss rewrite, is the experimental difference.
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

        # ------------------------------------------------------------------
        # SAME LR schedule as the current CIFAR-100 FedTIG run.
        # Keeping this schedule is important for a same-protocol comparison.
        # ------------------------------------------------------------------
        base_lr = float(args.lr_local_training)

        if current_round <= 100:
            local_lr = base_lr
        else:
            frac = min((current_round - 100) / 100.0, 1.0)
            local_lr = base_lr * (1.0 - 0.5 * frac)

        for group in self.optimizer.param_groups:
            group["lr"] = local_lr

        # ------------------------------------------------------------------
        # Local training
        # ------------------------------------------------------------------
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

                # FedLF L_A. No confidence/tail sample weighting.
                logits = cdist * hs.mm(ws.transpose(0, 1))
                loss1 = self.criterion(logits, labels)

                # FedLF centre loss L_C -- same implementation as FedTIG.
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

                # Same high safety ceiling as current FedTIG implementation.
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(),
                    max_norm=20.0,
                )

                self.optimizer.step()

        return copy.deepcopy(self.local_model.state_dict())


# =============================================================================
# Main
# =============================================================================
def fedlf_baseline():
    args = args_parser()

    if args.dataset != "cifar100":
        raise ValueError(
            "fedlf_cifar100_baseline.py requires --dataset cifar100"
        )
    if int(args.num_classes) != 100:
        raise ValueError(
            "CIFAR-100 requires --num_classes 100"
        )

    seed_everything(args.seed)

    paper_if = int(round(1.0 / float(args.imb_factor)))
    alpha_tag = str(args.non_iid_alpha).replace(".", "p")

    run_name = (
        "fedlf_baseline_cifar100"
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
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    final_model_path = os.path.join(output_dir, "final_model.pth")
    config_path = os.path.join(output_dir, "config.json")

    logger = setup_logging(log_filename)

    logger.info("=" * 80)
    logger.info("CIFAR-100-LT SAME-PROTOCOL FedLF BASELINE")
    logger.info("Run name: %s", run_name)
    logger.info("dataset:%s, num_classes:%s", args.dataset, args.num_classes)
    logger.info(
        "imb_factor:%s (IF=%d), non_iid_alpha:%s, rs_alpha:%s",
        args.imb_factor,
        paper_if,
        args.non_iid_alpha,
        args.rs_alpha,
    )
    logger.info("num_clients:%d", args.num_clients)
    logger.info("num_online_clients:%d", args.num_online_clients)
    logger.info(
        "participation_rate:%.4f",
        float(args.num_online_clients) / float(args.num_clients),
    )
    logger.info("num_rounds:%d", args.num_rounds)
    logger.info("local_epochs:%d", args.num_epochs_local_training)
    logger.info("batch_size:%d", args.batch_size_local_training)
    logger.info("base_lr:%s", args.lr_local_training)
    logger.info("late_lr_schedule: SHARED WITH CURRENT FedTIG (0.1 -> 0.05)")
    logger.info("seed:%d", args.seed)
    logger.info("FedLF local learner: L_A + 0.01 L_C + 0.01 L_D")
    logger.info("S / semantic client sampling: DISABLED")
    logger.info("C / confidence-tail reweighting: DISABLED")
    logger.info("A / tail residual aggregation: DISABLED")
    logger.info("client selection: UNIFORM RANDOM WITHOUT REPLACEMENT")
    logger.info("server aggregation: STANDARD SAMPLE-WEIGHTED FedAvg")
    logger.info("test-set rollback enabled: %s", ENABLE_TEST_ROLLBACK)
    logger.info("Log file: %s", os.path.abspath(log_path))
    logger.info("Output directory: %s", os.path.abspath(output_dir))
    logger.info("=" * 80)

    # Same normalization used by current FedTIG script.
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    data_local_training = datasets.CIFAR100(
        args.path_cifar100,
        train=True,
        download=True,
        transform=transform_all,
    )
    data_global_test = datasets.CIFAR100(
        args.path_cifar100,
        train=False,
        transform=transform_all,
    )

    # Same long-tail + Dirichlet pipeline as current FedTIG run.
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

    logger.info(
        "Global class counts: %s",
        list(map(int, class_counts)),
    )
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
        [int(len(indices)) for indices in list_client2indices],
    )
    logger.info("Partition SHA256: %s", fingerprint)

    config = {
        "algorithm": "FedLF_same_protocol_baseline",
        "dataset": "CIFAR-100-LT",
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
        "late_lr_schedule": "shared_with_current_fedtig_0.1_to_0.05",
        "seed": int(args.seed),
        "S": False,
        "C": False,
        "A": False,
        "client_sampling": "uniform_random_without_replacement",
        "aggregation": "sample_weighted_fedavg",
        "head_threshold": int(majority_threshold),
        "tail_threshold": int(tail_threshold),
        "tail_classes": tail_classes,
        "partition_sha256": fingerprint,
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

    # Initialize the shared centre state exactly once from classifier rows.
    initial_state = global_model.syn_model.state_dict()
    global_model.update_smoothed_centers(
        initial_state["classifier.weight"].detach()
    )

    random_state = np.random.RandomState(args.seed)

    re_trained_acc = []
    ft_many = []
    ft_medium = []
    ft_few = []
    head_acc_history = []
    tail_acc_history = []
    selected_clients_history = []

    best_tail = -1.0
    best_global = -1.0
    best_round = 0
    best_state = copy.deepcopy(global_model.download_params())

    for r in tqdm(
        range(1, args.num_rounds + 1),
        desc="fedlf-baseline-training",
    ):
        # ------------------------------------------------------------------
        # S OFF: uniform random client selection
        # ------------------------------------------------------------------
        online_clients = random_state.choice(
            total_clients,
            size=min(args.num_online_clients, len(total_clients)),
            replace=False,
        )
        online_clients = [int(v) for v in online_clients]
        selected_clients_history.append(sorted(online_clients))

        logger.info(
            "Round %d selected clients: %s",
            r,
            sorted(online_clients),
        )

        global_params = global_model.download_params()
        smoothed_centers = global_model.get_smoothed_centers()

        list_dicts_local_params = []
        list_nums_local_data = []

        for client in online_clients:
            cnts = torch.tensor(
                original_dict_per_client[client],
                dtype=torch.float32,
            )
            dist = cnts / cnts.sum().clamp_min(1e-8)

            indices2data.load(list_client2indices[client])
            data_client = indices2data

            list_nums_local_data.append(len(data_client))

            local_model = Local(
                data_client=data_client,
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
                    "Classifier class-count mismatch: local={}, global={}, expected={}".format(
                        local_out,
                        global_out,
                        args.num_classes,
                    )
                )

            local_params = local_model.local_train(
                args=args,
                global_params=copy.deepcopy(global_params),
                dist=dist,
                smoothed_centers=smoothed_centers,
                current_round=r,
            )

            list_dicts_local_params.append(copy.deepcopy(local_params))

        # ------------------------------------------------------------------
        # A OFF: ordinary FedAvg
        # ------------------------------------------------------------------
        fedavg_params = global_model.initialize_for_model_fusion(
            list_dicts_local_params,
            list_nums_local_data,
        )

        # Fixed reporting, no rollback.
        accepted_params = fedavg_params

        many, medium, few, head_acc, tail_acc = global_model.global_eval_more(
            accepted_params,
            data_global_test,
            args.batch_size_test,
            class_counts,
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

        re_trained_acc.append(global_acc)
        ft_many.append(many)
        ft_medium.append(medium)
        ft_few.append(few)
        head_acc_history.append((r, head_acc))
        tail_acc_history.append((r, tail_acc))

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
            "Round %d | Global %.4f | Head %.4f | Tail %.4f | "
            "Many %.4f | Medium %.4f | Few %.4f | best_tail %.4f@%d",
            r,
            global_acc,
            head_acc,
            tail_acc,
            many,
            medium,
            few,
            best_tail,
            best_round,
        )

        if r % 10 == 0 or r == args.num_rounds:
            with open(history_path, "wb") as f:
                pickle.dump(
                    {
                        "global": re_trained_acc,
                        "head": head_acc_history,
                        "tail": tail_acc_history,
                        "many": ft_many,
                        "medium": ft_medium,
                        "few": ft_few,
                        "selected_clients": selected_clients_history,
                        "best_tail": best_tail,
                        "best_round": best_round,
                        "partition_sha256": fingerprint,
                    },
                    f,
                )

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
                "Checkpoint @%d | Global %.4f | Head %.4f | Middle %.4f | Tail %.4f",
                r,
                global_acc,
                head_acc,
                medium,
                tail_acc,
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

    logger.info("=" * 80)
    logger.info("FINISHED SAME-PROTOCOL FedLF BASELINE")
    logger.info("Final Global: %.4f", re_trained_acc[-1])
    logger.info("Final Head: %.4f", head_acc_history[-1][1])
    logger.info("Final Middle: %.4f", ft_medium[-1])
    logger.info("Final Tail: %.4f", tail_acc_history[-1][1])
    logger.info(
        "Best Tail (diagnostic only; NOT the fixed Round-200 main result): %.4f @ round %d",
        best_tail,
        best_round,
    )
    logger.info("History file: %s", os.path.abspath(history_path))
    logger.info("Checkpoint file: %s", os.path.abspath(checkpoint_path))
    logger.info("Final model file: %s", os.path.abspath(final_model_path))
    logger.info("Config file: %s", os.path.abspath(config_path))
    logger.info("Partition SHA256: %s", fingerprint)
    logger.info("=" * 80)


if __name__ == "__main__":
    fedlf_baseline()
