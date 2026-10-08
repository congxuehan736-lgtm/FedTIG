# THESIS_FINAL_ABLATION_V1
from Model.log_model import setup_logging
from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import (
    classify_label,
    show_clients_data_distribution,
    Indices2Dataset,
    get_class_num,
)
from Dataset.sample_dirichlet import clients_indices
from Dataset.Gradient_matching_loss import match_loss
from Dataset.param_aug import DiffAugment

import argparse
import copy
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm


SEMANTIC_DENSITIES = None


# ================================================================
# Ablation CLI
# These arguments are parsed here and removed from sys.argv before
# the project's original options.args_parser() sees the remaining args.
# ================================================================
def _parse_ablation_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["fedlf", "s", "c", "a", "sc", "sa", "ca", "full"],
    )
    parser.add_argument("--semantic_epsilon", type=float, default=0.75)
    parser.add_argument("--confidence_alpha", type=float, default=0.70)
    parser.add_argument("--confidence_warmup", type=int, default=30)
    parser.add_argument("--tail_agg_weight", type=float, default=1.50)
    parser.add_argument("--run_name", type=str, default=None)

    known, remaining = parser.parse_known_args()
    # Leave only the project's original arguments for options.args_parser().
    sys.argv = [sys.argv[0]] + remaining

    flags = {
        "fedlf": (False, False, False),
        "s":     (True,  False, False),
        "c":     (False, True,  False),
        "a":     (False, False, True),
        "sc":    (True,  True,  False),
        "sa":    (True,  False, True),
        "ca":    (False, True,  True),
        "full":  (True,  True,  True),
    }
    known.use_s, known.use_c, known.use_a = flags[known.variant]

    if not (0.0 <= known.semantic_epsilon <= 1.0):
        raise ValueError("--semantic_epsilon must be in [0, 1].")
    if not (0.0 <= known.confidence_alpha <= 1.0):
        raise ValueError("--confidence_alpha must be in [0, 1].")
    if known.confidence_warmup < 0:
        raise ValueError("--confidence_warmup must be >= 0.")
    if known.tail_agg_weight < 1.0:
        raise ValueError("--tail_agg_weight must be >= 1.0.")

    if known.run_name is None:
        known.run_name = known.variant

    return known


ABL = _parse_ablation_cli()


# ================================================================
# Reproducibility
# ================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Innovation 1: semantic-density sampling (revised)
#
# IMPORTANT SCIENTIFIC CHANGE:
# The previous implementation mapped unrelated external density anchors
# to client IDs.  That is not a valid client-semantic signal for CIFAR.
# This version computes each client's informativeness from the ACTUAL
# federated data distribution and the current global class prototypes:
#   1) rarity information       : favors globally under-represented classes;
#   2) label entropy            : favors locally informative/diverse clients;
#   3) prototype dispersion     : favors semantically diverse class mixtures.
# The three normalized terms are equally weighted.  Sampling also contains
# an explicit uniform-exploration part and a participation-debt correction
# so high-scoring clients cannot monopolize training.
# ================================================================
def _minmax01(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def compute_client_semantic_scores(
    client_class_counts,
    global_class_counts,
    class_centers,
):
    """Return one bounded informativeness score per client.

    Scores are computed only from quantities available inside the training
    problem; no external corpus or client-ID mapping is used.
    """
    if isinstance(client_class_counts, dict):
        ordered_keys = sorted(client_class_counts)
        expected_keys = list(range(len(client_class_counts)))
        if ordered_keys != expected_keys:
            raise ValueError(
                "client_class_counts dict keys must be contiguous client IDs 0..N-1"
            )
        counts = np.asarray(
            [client_class_counts[k] for k in ordered_keys],
            dtype=np.float64,
        )
    else:
        counts = np.asarray(client_class_counts, dtype=np.float64)
    global_counts = np.asarray(global_class_counts, dtype=np.float64)

    if counts.ndim != 2:
        raise ValueError("client_class_counts must be a [clients, classes] matrix")

    num_clients, num_classes = counts.shape
    eps = 1e-12

    # Global rarity information: -log p(c), normalized later across clients.
    global_prior = (global_counts + 1.0) / (
        float(global_counts.sum()) + float(num_classes)
    )
    class_information = -np.log(global_prior + eps)

    # Current semantic geometry from EMA classifier centres.
    if torch.is_tensor(class_centers):
        centers = class_centers.detach().float().cpu()
    else:
        centers = torch.as_tensor(class_centers, dtype=torch.float32)

    if centers.ndim != 2 or centers.shape[0] != num_classes:
        # Safe fallback: semantic dispersion becomes zero; rarity/entropy
        # still provide a valid data-dependent sampling signal.
        semantic_distance = np.zeros((num_classes, num_classes), dtype=np.float64)
    else:
        centers = F.normalize(centers, p=2, dim=1)
        cosine = centers @ centers.t()
        semantic_distance = (1.0 - cosine).clamp(0.0, 2.0).cpu().numpy()

    rarity_values = []
    entropy_values = []
    dispersion_values = []

    for i in range(num_clients):
        row = counts[i]
        total = float(row.sum())
        if total <= 0:
            rarity_values.append(0.0)
            entropy_values.append(0.0)
            dispersion_values.append(0.0)
            continue

        p = row / total

        rarity_values.append(float(np.sum(p * class_information)))

        nonzero = p[p > 0]
        entropy = -float(np.sum(nonzero * np.log(nonzero + eps)))
        entropy /= max(np.log(float(num_classes)), eps)
        entropy_values.append(entropy)

        # Expected pairwise semantic separation under the client's class mix.
        dispersion = float(p @ semantic_distance @ p)
        dispersion_values.append(dispersion)

    rarity_n = _minmax01(rarity_values)
    entropy_n = _minmax01(entropy_values)
    dispersion_n = _minmax01(dispersion_values)

    # Locked final S (validated V2):
    # retain equal contributions from rarity, label entropy, and
    # semantic prototype dispersion.
    combined = (rarity_n + entropy_n + dispersion_n) / 3.0

    # Keep a non-zero floor: the score affects only the focused part of
    # sampling, while the remaining slots are sampled uniformly.
    scores = 0.25 + 0.75 * combined
    return np.asarray(scores, dtype=np.float64)


def semantic_sample_clients(
    total_clients,
    semantic_scores,
    num_online_clients,
    epsilon,
    rng,
    participation_counts=None,
):
    """Exploration-preserving semantic sampling.

    epsilon is the fraction of slots assigned to uniform exploration.
    For example, epsilon=0.75 with 8 online clients means about 2 focused
    semantic-density slots + 6 uniformly sampled slots.
    """
    total_clients = np.asarray(list(total_clients), dtype=np.int64)
    k = min(int(num_online_clients), len(total_clients))
    if k <= 0:
        return np.asarray([], dtype=np.int64)

    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    k_focus = int(round((1.0 - epsilon) * k))
    k_focus = max(0, min(k_focus, k))

    scores = np.asarray(semantic_scores, dtype=np.float64)
    if scores.shape[0] != len(total_clients):
        raise ValueError("semantic_scores must have one value per client")

    scores = np.nan_to_num(scores, nan=0.25, posinf=1.0, neginf=0.25)
    scores = np.maximum(scores, 1e-6)

    if participation_counts is not None:
        debt = 1.0 / np.sqrt(1.0 + np.asarray(participation_counts, dtype=np.float64))
        if debt.shape[0] == scores.shape[0]:
            scores = scores * debt

    chosen = []
    remaining = total_clients.copy()

    if k_focus > 0:
        p = scores / scores.sum()
        focus = rng.choice(
            total_clients,
            size=k_focus,
            replace=False,
            p=p,
        )
        chosen.extend(map(int, focus))
        remaining = np.asarray(
            [c for c in total_clients if int(c) not in set(chosen)],
            dtype=np.int64,
        )

    k_uniform = k - len(chosen)
    if k_uniform > 0:
        uniform = rng.choice(
            remaining,
            size=k_uniform,
            replace=False,
        )
        chosen.extend(map(int, uniform))

    return np.asarray(chosen, dtype=np.int64)

# ================================================================
# Optional t-SNE diagnostic
# ================================================================
def tsne_evaluation(model, dataloader, save_path, device):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                output, _ = model(inputs)
                features.append(
                    output.detach().cpu().numpy()
                )
                labels.append(targets.numpy())

        if not features:
            return

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        if len(features) < 10:
            return

        perplexity = min(
            30,
            max(5, len(features) // 20),
            len(features) - 1,
        )

        tsne = TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
        )

        projected = tsne.fit_transform(features)

        plt.figure(figsize=(10, 10))
        for i in range(10):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    projected[idx, 0],
                    projected[idx, 1],
                    label=str(i),
                )

        plt.legend()
        plt.savefig(save_path)
        plt.close()

    except Exception as exc:
        logging.getLogger(__name__).warning(
            "t-SNE skipped: %s", exc
        )


# ================================================================
# FedLF feature decorrelation
# ================================================================
class DecorrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        n, m = mat.shape
        assert n == m
        if n <= 1:
            return mat.new_zeros(1)
        return mat.flatten()[:-1].view(n - 1, n + 1)[
            :, 1:
        ].flatten()

    def forward(self, x):
        if x.ndim != 2:
            return x.new_tensor(0.0)

        N, C = x.shape
        if N <= 1:
            return x.new_tensor(0.0)

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(
            self.eps + x.var(
                dim=0,
                keepdim=True,
                unbiased=False,
            )
        )

        corr_mat = torch.matmul(x.t(), x)
        loss = self._off_diagonal(
            corr_mat
        ).pow(2).mean()

        return loss / N


# ================================================================
# Global/server
# ================================================================
class Global(object):
    def __init__(
        self,
        num_classes,
        device,
        args,
        num_of_feature,
    ):
        self.device = device
        self.num_classes = num_classes
        self.num_of_feature = num_of_feature

        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []

        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []

        self.feature_syn = torch.randn(
            size=(
                args.num_classes * self.num_of_feature,
                256,
            ),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )

        self.label_syn = torch.tensor(
            np.repeat(
                np.arange(args.num_classes),
                self.num_of_feature,
            ),
            dtype=torch.long,
            device=args.device,
        )

        self.optimizer_feature = SGD(
            [self.feature_syn],
            lr=args.lr_feature,
        )

        self.criterion = CrossEntropyLoss().to(args.device)

        self.syn_model = ResNet_cifar(
            resnet_size=8,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        ).to(device)

        self.feature_net = nn.Linear(
            256,
            args.num_classes,
        ).to(args.device)

        self.smoothed_centers = torch.zeros(
            (num_classes, 256),
            device=device,
        )
        self.ema_decay = 0.9
        self.center_initialized = False

    # ------------------------------------------------------------
    # EMA centres
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Original FedLF gradient matching
    # ------------------------------------------------------------
    def update_feature_syn(
        self,
        args,
        global_params,
        list_clients_gradient,
    ):
        feature_net_params = self.feature_net.state_dict()

        if (
            "classifier.weight" not in global_params
            or "classifier.bias" not in global_params
        ):
            return

        feature_net_params["weight"] = (
            global_params["classifier.weight"]
        )
        feature_net_params["bias"] = (
            global_params["classifier.bias"]
        )

        self.feature_net.load_state_dict(
            feature_net_params
        )
        self.feature_net.train()

        net_global_parameters = list(
            self.feature_net.parameters()
        )

        gw_real_all = {
            c: [] for c in range(self.num_classes)
        }

        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                if class_num in gw_real_all:
                    gw_real_all[class_num].append(
                        gradient
                    )

        gw_real_avg = {
            c: [] for c in range(self.num_classes)
        }

        for c in range(self.num_classes):
            gradients = gw_real_all[c]

            if not gradients:
                continue

            avg = []

            for p_idx in range(len(gradients[0])):
                avg.append(
                    sum(
                        g[p_idx]
                        for g in gradients
                    ) / len(gradients)
                )

            gw_real_avg[c] = avg

        self.optimizer_feature.zero_grad()

        total_loss = None

        for c in range(self.num_classes):
            if not gw_real_avg[c]:
                continue

            start = (
                c * self.num_of_feature
            )
            end = (
                (c + 1) * self.num_of_feature
            )

            feature_c = self.feature_syn[
                start:end
            ]
            label_c = self.label_syn[
                start:end
            ]

            batch_size = args.batch_real

            if len(feature_c) > batch_size:
                idx = torch.randperm(
                    len(feature_c),
                    device=feature_c.device,
                )[:batch_size]

                feature_c = feature_c[idx]
                label_c = label_c[idx]

            output_c = self.feature_net(
                feature_c
            )

            loss_ce = self.criterion(
                output_c,
                label_c,
            )

            gw_syn = torch.autograd.grad(
                loss_ce,
                net_global_parameters,
                create_graph=True,
            )

            loss_match = match_loss(
                gw_syn,
                gw_real_avg[c],
                args,
            )

            total_loss = (
                loss_match
                if total_loss is None
                else total_loss + loss_match
            )

        if (
            total_loss is not None
            and torch.isfinite(total_loss)
        ):
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [self.feature_syn],
                10.0,
            )

            self.optimizer_feature.step()

    # ------------------------------------------------------------
    # Corrected tail-aware aggregation
    # ------------------------------------------------------------
    def initialize_for_model_fusion(
        self,
        list_dicts_local_params,
        list_nums_local_data,
        tail_classes=None,
        client_class_counts=None,
        grad_rew_weight=1.0,
    ):
        """
        Standard FedAvg for the whole model.

        For tail classifier rows:
          - ordinary FedAvg remains the base;
          - only clients actually containing that tail class contribute
            a residual tail update;
          - the residual is amplified by a bounded coefficient.

        Crucially, we never multiply the absolute classifier row by 1.2/2.0.
        This avoids cumulative norm explosion.

        If no selected client has a particular tail class, its classifier
        row is retained from the previous global model.
        """
        if not list_dicts_local_params:
            raise ValueError("No local parameters.")

        total_num = float(
            sum(list_nums_local_data)
        )

        global_params = copy.deepcopy(
            list_dicts_local_params[0]
        )

        for name_param in global_params:
            weighted_sum = None

            for local_state, num_data in zip(
                list_dicts_local_params,
                list_nums_local_data,
            ):
                value = (
                    local_state[name_param]
                    * float(num_data)
                )

                weighted_sum = (
                    value
                    if weighted_sum is None
                    else weighted_sum + value
                )

            global_params[name_param] = (
                weighted_sum / total_num
            )

        if not tail_classes:
            return global_params

        # Keep tail intervention moderate.
        # grad_rew_weight=2 => beta=0.25.
        beta = float(
            np.clip(
                0.25 * (grad_rew_weight - 1.0),
                0.0,
                0.35,
            )
        )

        # Need the old global classifier rows for the
        # "no tail client selected" case.
        old_global = getattr(
            self,
            "_last_global_state",
            None,
        )

        for name_param in (
            "classifier.weight",
            "classifier.bias",
        ):
            if name_param not in global_params:
                continue

            for c in tail_classes:
                tail_client_indices = []

                if client_class_counts is not None:
                    for local_idx, counts in enumerate(
                        client_class_counts
                    ):
                        if (
                            int(counts[c]) > 0
                        ):
                            tail_client_indices.append(
                                local_idx
                            )

                if not tail_client_indices:
                    # No tail information this round:
                    # do not let head-only clients erase the tail row.
                    if (
                        old_global is not None
                        and name_param in old_global
                    ):
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
                    local_state = (
                        list_dicts_local_params[i]
                    )
                    value = (
                        local_state[name_param][c]
                        * float(list_nums_local_data[i])
                    )

                    tail_avg = (
                        value
                        if tail_avg is None
                        else tail_avg + value
                    )

                tail_avg = (
                    tail_avg / tail_total
                )

                # Base FedAvg row.
                base_row = global_params[
                    name_param
                ][c].clone()

                # Tail-specialist residual.
                residual = (
                    tail_avg - base_row
                )

                # Bounded correction.
                global_params[name_param][c] = (
                    base_row
                    + (1.0 + beta) * residual
                )

        return global_params

    # ------------------------------------------------------------
    # Head/Tail evaluation
    # ------------------------------------------------------------
    def global_eval_more(
        self,
        fedavg_params,
        data_test,
        batch_size_test,
        class_counts,
    ):
        self.syn_model.load_state_dict(
            fedavg_params
        )
        self.syn_model.eval()

        majority_threshold = 1500

        # For CIFAR-10-LT IF=100:
        # classes 7/8/9 are below 200.
        minority_threshold = (
            600
            if class_counts[0] < 600
            else 200
        )

        head_threshold = 200

        many_correct = many_total = 0
        medium_correct = medium_total = 0
        few_correct = few_total = 0
        head_correct = head_total = 0
        tail_correct = tail_total = 0

        test_loader = DataLoader(
            data_test,
            batch_size=batch_size_test,
            shuffle=False,
        )

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, outputs = self.syn_model(
                    images
                )

                predicts = outputs.argmax(
                    dim=-1
                )

                for label, predict in zip(
                    labels,
                    predicts,
                ):
                    y = int(label.item())

                    samples_num = int(
                        class_counts[y]
                    )

                    correct = int(
                        predict.item() == y
                    )

                    if samples_num > majority_threshold:
                        many_total += 1
                        many_correct += correct
                    elif samples_num < minority_threshold:
                        few_total += 1
                        few_correct += correct
                    else:
                        medium_total += 1
                        medium_correct += correct

                    if samples_num >= head_threshold:
                        head_total += 1
                        head_correct += correct
                    else:
                        tail_total += 1
                        tail_correct += correct

        def acc(correct, total):
            return (
                round(
                    correct / total,
                    4,
                )
                if total > 0
                else 0.0
            )

        return (
            acc(many_correct, many_total),
            acc(medium_correct, medium_total),
            acc(few_correct, few_total),
            acc(head_correct, head_total),
            acc(tail_correct, tail_total),
        )

    def global_eval(
        self,
        params,
        data_test,
        batch_size_test,
    ):
        self.syn_model.load_state_dict(
            params
        )
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

                _, outputs = self.syn_model(
                    images
                )

                predicts = outputs.argmax(
                    dim=-1
                )

                correct += int(
                    (predicts == labels).sum().item()
                )

        return correct / len(data_test)

    def download_params(self):
        return copy.deepcopy(
            self.syn_model.state_dict()
        )


# ================================================================
# Local client
# ================================================================
class Local(object):
    def __init__(
        self,
        data_client,
        class_list,
    ):
        args = args_parser()

        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list

        self.criterion = CrossEntropyLoss().to(
            args.device
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

        # IMPORTANT:
        # Restore the project's original optimizer.
        # Do not introduce momentum/weight decay here.
        self.optimizer = SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
        )

    # ------------------------------------------------------------
    # Original gradient matching
    # ------------------------------------------------------------
    def compute_gradient(
        self,
        global_params,
        args,
    ):
        list_class, _ = get_class_num(
            self.class_compose
        )

        images_all = []
        labels_all = []

        indices_class = {
            class_index: []
            for class_index in list_class
        }

        for i in range(len(self.data_client)):
            x, y = self.data_client[i]

            images_all.append(
                x.unsqueeze(0)
            )
            labels_all.append(int(y))

        if not images_all:
            return {}

        images_all = torch.cat(
            images_all,
            dim=0,
        ).to(args.device)

        for i, label in enumerate(labels_all):
            if label in indices_class:
                indices_class[label].append(i)

        def get_images(
            class_index,
            num_images,
        ):
            idxs = indices_class.get(
                class_index,
                [],
            )

            if not idxs:
                return None

            idxs = np.random.permutation(
                idxs
            )[:num_images]

            return images_all[idxs]

        self.local_model.load_state_dict(
            global_params
        )

        self.local_model.eval()

        net_parameters = list(
            self.local_model.classifier.parameters()
        )

        criterion = CrossEntropyLoss().to(
            args.device
        )

        truth_gradient_all = {
            c: [] for c in list_class
        }

        truth_gradient_avg = {
            c: [] for c in list_class
        }

        for _ in range(10):
            for c in list_class:
                img_real = get_images(
                    c,
                    args.batch_real,
                )

                if img_real is None:
                    continue

                if getattr(args, "dsa", False):
                    seed = (
                        int(time.time() * 1000)
                        % 100000
                    )

                    img_real = DiffAugment(
                        img_real,
                        args.dsa_strategy,
                        seed=seed,
                        param=args.dsa_param,
                    )

                lab_real = torch.full(
                    (
                        img_real.shape[0],
                    ),
                    c,
                    device=args.device,
                    dtype=torch.long,
                )

                _, output_real = (
                    self.local_model(
                        img_real
                    )
                )

                loss_real = criterion(
                    output_real,
                    lab_real,
                )

                gw_real = torch.autograd.grad(
                    loss_real,
                    net_parameters,
                )

                truth_gradient_all[c].append(
                    [
                        g.detach().clone()
                        for g in gw_real
                    ]
                )

        for c in list_class:
            gradients = (
                truth_gradient_all[c]
            )

            if not gradients:
                continue

            avg = []

            for p_idx in range(
                len(gradients[0])
            ):
                avg.append(
                    sum(
                        g[p_idx]
                        for g in gradients
                    )
                    / len(gradients)
                )

            truth_gradient_avg[c] = avg

        return truth_gradient_avg

    # ------------------------------------------------------------
    # Innovation 2:
    # confidence-driven Logit Adjustment
    # ------------------------------------------------------------
    def local_train(
        self,
        args,
        global_params,
        dist,
        smoothed_centers,
        tail_classes,
        current_round,
    ):
        """
        Thesis-final local training.

        C=OFF restores the previous FedLF-aligned baseline:
          1) Adaptive Logit Adjustment
          2) per-client local class-centre optimization
          3) FedLF feature decorrelation
          4) fixed local LR

        C=ON only ADDS confidence/class-aware sample weighting.
        Innovation A is server-side only and does not change local training.
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        # FedLF baseline LR: fixed for every variant.
        for group in self.optimizer.param_groups:
            group["lr"] = float(args.lr_local_training)

        # --------------------------------------------------------
        # FedLF baseline: LOCAL class centres
        # --------------------------------------------------------
        local_centers = torch.zeros(
            (args.num_classes, 256),
            device=self.device,
        )
        class_counts = torch.zeros(
            args.num_classes,
            device=self.device,
        )

        pre_loader = DataLoader(
            dataset=self.data_client,
            batch_size=args.batch_size_local_training,
            shuffle=True,
        )

        with torch.no_grad():
            for images, labels in pre_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = transform_train(images)

                features, _ = self.local_model(images)

                for i in range(features.size(0)):
                    local_centers[labels[i]] += features[i]
                    class_counts[labels[i]] += 1

        valid_classes = class_counts != 0
        local_centers[valid_classes] /= (
            class_counts[valid_classes].unsqueeze(1)
        )
        local_centers[~valid_classes] = 1e-8

        gap_matrix = (
            torch.ones(
                (args.num_classes, args.num_classes),
                device=self.device,
            )
            * 1e9
        )

        for i in range(args.num_classes):
            for j in range(i):
                dis = torch.norm(
                    local_centers[i] - local_centers[j],
                    p=2,
                )
                gap_matrix[i, j] = dis
                gap_matrix[j, i] = dis

        upper_mask = (
            torch.triu(
                torch.ones_like(gap_matrix),
                diagonal=1,
            )
            > 0
        )
        valid_gaps = gap_matrix[upper_mask]
        max_gap = (
            torch.max(valid_gaps)
            if valid_gaps.numel() > 0
            else torch.tensor(1.0, device=self.device)
        )

        # --------------------------------------------------------
        # C-only preparation
        # --------------------------------------------------------
        class_weights = torch.ones(
            args.num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        present = class_counts > 0

        if present.any():
            present_counts = class_counts[present].float()
            reference = present_counts.mean().clamp_min(1.0)

            class_weights[present] = torch.sqrt(
                reference / present_counts.clamp_min(1.0)
            )
            class_weights[present] = (
                class_weights[present]
                / class_weights[present].mean().clamp_min(1e-6)
            )
            class_weights[present] = class_weights[present].clamp(
                0.60, 1.80
            )

        semantic_centers = (
            smoothed_centers.to(self.device).detach()
            if smoothed_centers is not None
            else None
        )

        if (
            semantic_centers is None
            or semantic_centers.ndim != 2
            or semantic_centers.shape[0] != args.num_classes
            or not torch.isfinite(semantic_centers).all()
            or semantic_centers.abs().sum() < 1e-8
        ):
            semantic_centers = (
                self.local_model.classifier.weight.detach().clone()
            )

        # --------------------------------------------------------
        # FedLF Adaptive Logit Adjustment
        # --------------------------------------------------------
        dist = dist.to(self.device)

        if dist.numel() != args.num_classes:
            tmp = torch.zeros(
                args.num_classes,
                device=self.device,
            )
            n = min(len(dist), args.num_classes)
            tmp[:n] = dist[:n]
            dist = tmp

        cdist = (
            dist
            / dist.max().clamp_min(1e-8)
        )
        cdist = (
            cdist * (1.0 - args.rs_alpha)
            + args.rs_alpha
        )
        cdist = cdist.reshape((1, -1))

        alpha = float(ABL.confidence_alpha)

        # --------------------------------------------------------
        # Local epochs
        # --------------------------------------------------------
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(
                dataset=self.data_client,
                batch_size=args.batch_size_local_training,
                shuffle=True,
            )

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = transform_train(images)

                hs, _ = self.local_model(images)
                ws = self.local_model.classifier.weight

                logits = (
                    cdist
                    * hs.mm(ws.transpose(0, 1))
                )

                # Innovation C: additive sample weighting.
                c_active = (
                    ABL.use_c
                    and current_round > ABL.confidence_warmup
                )

                if c_active:
                    hs_norm = F.normalize(hs, p=2, dim=1)
                    center_norm = F.normalize(
                        semantic_centers,
                        p=2,
                        dim=1,
                    )

                    similarities = hs_norm.mm(center_norm.t())
                    row_index = torch.arange(
                        labels.shape[0],
                        device=self.device,
                    )
                    true_sim = similarities[row_index, labels]

                    competing = similarities.clone()
                    competing[row_index, labels] = -1e9
                    best_other = competing.max(dim=1).values
                    margin = true_sim - best_other

                    margin_scale = (
                        margin.detach().std().clamp_min(0.10)
                    )
                    uncertainty = (
                        1.0
                        + torch.sigmoid(-margin / margin_scale)
                    ).detach()
                    uncertainty = (
                        uncertainty
                        / uncertainty.mean().clamp_min(1e-6)
                    ).clamp(0.70, 1.50)

                    class_component = class_weights[labels]
                    class_component = (
                        class_component
                        / class_component.mean().clamp_min(1e-6)
                    )

                    sample_weights = (
                        alpha * class_component
                        + (1.0 - alpha) * uncertainty
                    )
                    sample_weights = (
                        sample_weights
                        / sample_weights.mean().clamp_min(1e-6)
                    ).clamp(0.50, 2.00)

                    loss_per_sample = F.cross_entropy(
                        logits,
                        labels,
                        reduction="none",
                    )
                    loss1 = (
                        loss_per_sample * sample_weights
                    ).mean()
                else:
                    loss1 = self.criterion(
                        logits,
                        labels,
                    )

                # FedLF baseline local class-centre optimization.
                features_square = torch.sum(
                    hs.pow(2),
                    dim=1,
                    keepdim=True,
                )
                centers_square = torch.sum(
                    local_centers.pow(2),
                    dim=1,
                    keepdim=True,
                )
                features_into_centers = hs.matmul(
                    local_centers.t()
                )

                dist_2 = (
                    features_square
                    - 2 * features_into_centers
                    + centers_square.t()
                )
                dist_2 = torch.sqrt(
                    dist_2.clamp_min(1e-8)
                )

                one_hot = F.one_hot(
                    labels,
                    args.num_classes,
                ).to(self.device)

                gap_value = min(
                    float(max_gap.item()),
                    100.0,
                )
                dist_2 = (
                    dist_2
                    + one_hot * gap_value
                )

                loss2 = self.criterion(
                    -dist_2,
                    labels,
                )

                # FedLF feature decorrelation is baseline, not C.
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
                self.optimizer.step()

        return copy.deepcopy(
            self.local_model.state_dict()
        )


# ================================================================
# Main FedLF
# ================================================================
def fedlf():
    log_path = f"{ABL.run_name}.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = setup_logging(log_path)

    args = args_parser()

    seed_everything(args.seed)

    output_dir = os.path.join(
        "outputs",
        ABL.run_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "========== THESIS FINAL FedLF S/C/A ABLATION ==========\\n"
        "variant=%s | S=%s | C=%s | A=%s\\n"
        "imb_factor:%s, non_iid:%s, rs_alpha:%s\\n"
        "lr_local_training:%s (fixed FedLF baseline LR)\\n"
        "num_rounds:%s, local_epochs:%s\\n"
        "batch_size_local_training:%s\\n"
        "num_online_clients:%s\\n"
        "S semantic_epsilon:%s\\n"
        "C confidence_alpha:%s, warmup:%s\\n"
        "A server_tail_agg_weight:%s\\n"
        "baseline:AdaptiveLogit+LocalCenters+FedDecorr+FedAvg\\n"
        "formal_protocol:no_test_rollback\\n",
        ABL.variant,
        "ON" if ABL.use_s else "OFF",
        "ON" if ABL.use_c else "OFF",
        "ON" if ABL.use_a else "OFF",
        args.imb_factor,
        args.non_iid_alpha,
        args.rs_alpha,
        args.lr_local_training,
        args.num_rounds,
        args.num_epochs_local_training,
        args.batch_size_local_training,
        args.num_online_clients,
        ABL.semantic_epsilon,
        ABL.confidence_alpha,
        ABL.confidence_warmup,
        ABL.tail_agg_weight,
    )

    random_state = np.random.RandomState(
        args.seed
    )

    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    if args.dataset == "cifar10":
        data_local_training = (
            datasets.CIFAR10(
                args.path_cifar10,
                train=True,
                download=True,
                transform=transform_all,
            )
        )

        data_global_test = (
            datasets.CIFAR10(
                args.path_cifar10,
                train=False,
                transform=transform_all,
            )
        )

    elif args.dataset == "cifar100":
        data_local_training = (
            datasets.CIFAR100(
                args.path_cifar100,
                train=True,
                download=True,
                transform=transform_all,
            )
        )

        data_global_test = (
            datasets.CIFAR100(
                args.path_cifar100,
                train=False,
                transform=transform_all,
            )
        )

    else:
        raise ValueError(
            "Unknown dataset: {}".format(
                args.dataset
            )
        )

    # ------------------------------------------------------------
    # Long-tail split
    # ------------------------------------------------------------
    list_label2indices = classify_label(
        data_local_training,
        args.num_classes,
    )

    a, list_label2indices_train_new = (
        train_long_tail(
            copy.deepcopy(
                list_label2indices
            ),
            args.num_classes,
            args.imb_factor,
            args.imb_type,
        )
    )

    list_client2indices = clients_indices(
        copy.deepcopy(
            list_label2indices_train_new
        ),
        args.num_classes,
        args.num_clients,
        args.non_iid_alpha,
        args.seed,
    )

    original_dict_per_client = (
        show_clients_data_distribution(
            data_local_training,
            list_client2indices,
            args.num_classes,
        )
    )

    global_model = Global(
        num_classes=args.num_classes,
        device=args.device,
        args=args,
        num_of_feature=args.num_of_feature,
    )

    total_clients = list(
        range(args.num_clients)
    )

    indices2data = Indices2Dataset(
        data_local_training
    )

    re_trained_acc = []
    ft_many = []
    ft_medium = []
    ft_few = []

    head_acc_history = []
    tail_acc_history = []

    # Participation counter used only by S to prevent repeatedly
    # selecting the same high-score clients.
    participation_counts = np.zeros(
        args.num_clients,
        dtype=np.int64,
    )

    # Innovation 3.
    tail_threshold = 200

    tail_classes = [
        i
        for i, cnt in enumerate(a)
        if cnt < tail_threshold
    ]

    logger.info(
        "Global class counts: %s",
        list(map(int, a)),
    )

    logger.info(
        "Tail classes (<200): %s",
        tail_classes,
    )

    # ------------------------------------------------------------
    # Initial centres
    #
    # IMPORTANT:
    # Use the classifier weights as the initial centre exactly once,
    # rather than leaving all centres zero for the first local round.
    # ------------------------------------------------------------
    initial_state = (
        global_model.syn_model.state_dict()
    )

    global_model.update_smoothed_centers(
        initial_state[
            "classifier.weight"
        ].detach()
    )

    # Previous accepted global state.
    previous_global = (
        global_model.download_params()
    )

    # Best actual tail model.
    best_tail = -1.0
    best_global = -1.0
    best_round = 0
    best_state = copy.deepcopy(
        previous_global
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    for r in tqdm(
        range(
            1,
            args.num_rounds + 1,
        ),
        desc="server-training",
    ):
        # --------------------------------------------------------
        # Innovation 1: semantic density sampling
        # --------------------------------------------------------
        if ABL.use_s:
            semantic_scores = compute_client_semantic_scores(
                original_dict_per_client,
                a,
                global_model.get_smoothed_centers(),
            )
            online_clients = semantic_sample_clients(
                total_clients,
                semantic_scores,
                args.num_online_clients,
                epsilon=ABL.semantic_epsilon,
                rng=random_state,
                participation_counts=participation_counts,
            )
        else:
            semantic_scores = None
            online_clients = random_state.choice(
                total_clients,
                size=min(
                    args.num_online_clients,
                    len(total_clients),
                ),
                replace=False,
            )

        for _client in online_clients:
            participation_counts[int(_client)] += 1

        if ABL.use_s and semantic_scores is not None:
            selected_scores = [
                float(semantic_scores[int(c)])
                for c in online_clients
            ]
            selected_tail = sum(
                int(original_dict_per_client[int(c)][t])
                for c in online_clients
                for t in tail_classes
            )
            selected_total = sum(
                int(sum(original_dict_per_client[int(c)]))
                for c in online_clients
            )
            logger.info(
                "Round %d S-diagnostic | score_mean %.4f | "
                "selected_tail_fraction %.4f | participation_range %d-%d",
                r,
                float(np.mean(selected_scores)) if selected_scores else 0.0,
                float(selected_tail) / max(float(selected_total), 1.0),
                int(participation_counts.min()),
                int(participation_counts.max()),
            )

        logger.info(
            "Round %d selected clients: %s",
            r,
            sorted(
                map(
                    int,
                    online_clients,
                )
            ),
        )

        global_params = (
            global_model.download_params()
        )

        syn_feature_params = copy.deepcopy(
            global_params
        )

        # Temporary classifier for gradient matching.
        temp_model = nn.Linear(
            256,
            args.num_classes,
        ).to(args.device)

        syn_params = temp_model.state_dict()

        syn_feature_params[
            "classifier.weight"
        ] = syn_params["weight"]

        syn_feature_params[
            "classifier.bias"
        ] = syn_params["bias"]

        list_clients_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []

        # New: counts corresponding exactly to each local model.
        list_client_class_counts = []

        smoothed_centers = (
            global_model.get_smoothed_centers()
        )

        for client in online_clients:
            client = int(client)

            cnts = torch.tensor(
                original_dict_per_client[
                    client
                ],
                dtype=torch.float32,
            )

            dist = (
                cnts
                / cnts.sum().clamp_min(
                    1e-8
                )
            )

            indices2data.load(
                list_client2indices[
                    client
                ]
            )

            data_client = indices2data

            list_nums_local_data.append(
                len(data_client)
            )

            list_client_class_counts.append(
                np.asarray(
                    original_dict_per_client[
                        client
                    ],
                    dtype=np.int64,
                )
            )

            local_model = Local(
                data_client=data_client,
                class_list=(
                    original_dict_per_client[
                        client
                    ]
                ),
            )

            try:
                truth_gradient = (
                    local_model.compute_gradient(
                        copy.deepcopy(
                            syn_feature_params
                        ),
                        args,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Gradient matching skipped "
                    "for client %d: %s",
                    client,
                    exc,
                )
                truth_gradient = {}

            list_clients_gradient.append(
                copy.deepcopy(
                    truth_gradient
                )
            )

            local_params = (
                local_model.local_train(
                    args=args,
                    global_params=copy.deepcopy(
                        global_params
                    ),
                    dist=dist,
                    smoothed_centers=(
                        smoothed_centers
                    ),
                    tail_classes=(
                        tail_classes
                    ),
                    current_round=r,
                )
            )

            list_dicts_local_params.append(
                copy.deepcopy(
                    local_params
                )
            )

        # --------------------------------------------------------
        # Tail aggregation
        # --------------------------------------------------------
        #
        # Keep the requested project value grad_rew_weight=2.0,
        # but implement it as bounded residual amplification.
        #
        grad_rew_weight = (
            float(ABL.tail_agg_weight)
            if ABL.use_a
            else 1.0
        )

        fedavg_params = (
            global_model.initialize_for_model_fusion(
                list_dicts_local_params,
                list_nums_local_data,
                tail_classes=(
                    tail_classes
                    if ABL.use_a
                    else None
                ),
                client_class_counts=(
                    list_client_class_counts
                    if ABL.use_a
                    else None
                ),
                grad_rew_weight=(
                    grad_rew_weight
                ),
            )
        )

        # --------------------------------------------------------
        # Gradient matching
        # --------------------------------------------------------
        try:
            global_model.update_feature_syn(
                args,
                copy.deepcopy(
                    syn_feature_params
                ),
                list_clients_gradient,
            )
        except Exception as exc:
            logger.warning(
                "Feature synthesis update skipped "
                "at round %d: %s",
                r,
                exc,
            )

        # --------------------------------------------------------
        # Evaluate BEFORE committing.
        # --------------------------------------------------------
        (
            many,
            medium,
            few,
            head_acc,
            tail_acc,
        ) = global_model.global_eval_more(
            fedavg_params,
            data_global_test,
            args.batch_size_test,
            a,
        )

        global_acc = (
            global_model.global_eval(
                fedavg_params,
                data_global_test,
                args.batch_size_test,
            )
        )

        # --------------------------------------------------------
        # Fixed-round scientific protocol: NO test-set rollback.
        #
        # Earlier code used test Tail accuracy to alter model parameters
        # after round 110. That leaks test information into training.
        # For formal experiments the test set is evaluation-only.
        # --------------------------------------------------------
        accepted_params = fedavg_params
        rollback = False

        # --------------------------------------------------------
        # Commit
        # --------------------------------------------------------
        global_model.syn_model.load_state_dict(
            copy.deepcopy(
                accepted_params
            )
        )

        previous_global = copy.deepcopy(
            accepted_params
        )

        # EMA update.
        global_model.update_smoothed_centers(
            accepted_params[
                "classifier.weight"
            ].detach()
        )

        # Store the state for "no tail client selected" protection.
        global_model._last_global_state = (
            copy.deepcopy(
                accepted_params
            )
        )

        # --------------------------------------------------------
        # History
        # --------------------------------------------------------
        re_trained_acc.append(
            global_acc
        )

        ft_many.append(many)
        ft_medium.append(medium)
        ft_few.append(few)

        head_acc_history.append(
            (r, head_acc)
        )

        tail_acc_history.append(
            (r, tail_acc)
        )

        # Best actual tail model.
        if (
            tail_acc > best_tail
            or (
                abs(
                    tail_acc
                    - best_tail
                ) < 1e-8
                and global_acc
                > best_global
            )
        ):
            best_tail = tail_acc
            best_global = global_acc
            best_round = r
            best_state = copy.deepcopy(
                accepted_params
            )

        logger.info(
            "Round %d | Global %.4f | "
            "Head %.4f | Tail %.4f | "
            "Many %.4f | Medium %.4f | "
            "Few %.4f | "
            "best_tail %.4f@%d | "
            "rollback=%s",
            r,
            global_acc,
            head_acc,
            tail_acc,
            many,
            medium,
            few,
            best_tail,
            best_round,
            rollback,
        )

        # --------------------------------------------------------
        # Checkpoint every 10 rounds
        # --------------------------------------------------------
        if (
            r % 10 == 0
            or r == args.num_rounds
        ):
            with open(
                os.path.join(
                    output_dir,
                    "head_tail_history.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "head": (
                            head_acc_history
                        ),
                        "tail": (
                            tail_acc_history
                        ),
                        "global": (
                            re_trained_acc
                        ),
                        "many": ft_many,
                        "medium": ft_medium,
                        "few": ft_few,
                        "best_tail": best_tail,
                        "best_round": best_round,
                    },
                    f,
                )

            torch.save(
                {
                    "round": r,
                    "state_dict": copy.deepcopy(
                        accepted_params
                    ),
                    "best_state_dict": copy.deepcopy(
                        best_state
                    ),
                    "best_tail": best_tail,
                    "best_global": best_global,
                    "best_round": best_round,
                },
                os.path.join(
                    output_dir,
                    "fedlf_corrected_checkpoint.pth",
                ),
            )

            logger.info(
                "Global Accuracy: %s",
                re_trained_acc,
            )

            logger.info(
                "Majority Class Accuracy: %s",
                ft_many,
            )

            logger.info(
                "Medium Class Accuracy: %s",
                ft_medium,
            )

            logger.info(
                "Minority Class Accuracy: %s",
                ft_few,
            )

            logger.info(
                "=== Head-Tail Evaluation ==="
            )

            logger.info(
                "Head Accuracy: %.4f",
                head_acc,
            )

            logger.info(
                "Tail Accuracy: %.4f",
                tail_acc,
            )

            # t-SNE is diagnostic only.
            try:
                test_loader = DataLoader(
                    data_global_test,
                    batch_size=(
                        args.batch_size_test
                    ),
                    shuffle=False,
                )

                save_dir = os.path.join(
                    output_dir,
                    "Dimensionality_reduction",
                    "vsloss_feature",
                )

                os.makedirs(
                    save_dir,
                    exist_ok=True,
                )

                save_path = os.path.join(
                    save_dir,
                    f"tsne_epoch_{r}.png",
                )

                tsne_evaluation(
                    global_model.syn_model,
                    test_loader,
                    save_path,
                    args.device,
                )

            except Exception as exc:
                logger.warning(
                    "t-SNE failed: %s",
                    exc,
                )

    # ============================================================
    # Final save
    # ============================================================
    final_state = (
        global_model.syn_model.state_dict()
    )

    torch.save(
        {
            "state_dict": copy.deepcopy(
                final_state
            ),
            "best_state_dict": copy.deepcopy(
                best_state
            ),
            "best_tail": best_tail,
            "best_global": best_global,
            "best_round": best_round,
        },
        os.path.join(
            output_dir,
            "fedlf_corrected_final.pth",
        ),
    )

    logger.info(
        "========== FINISHED =========="
    )

    logger.info(
        "Final Global: %.4f",
        re_trained_acc[-1]
        if re_trained_acc
        else 0.0,
    )

    logger.info(
        "Final Head: %.4f",
        head_acc_history[-1][1]
        if head_acc_history
        else 0.0,
    )

    logger.info(
        "Final Tail: %.4f",
        tail_acc_history[-1][1]
        if tail_acc_history
        else 0.0,
    )

    logger.info(
        "Best Tail: %.4f @ round %d",
        best_tail,
        best_round,
    )


# ================================================================
# Revised innovation validation launcher
# Run this FIRST after replacing the file:
#   python .\run_ablation.py --validate_revised
# It reruns baseline + revised S/C + verified A for 100 rounds.
# ================================================================



# ================================================================
# Refined S/C validation launcher
#
# First-stage validation already established the unchanged baseline and
# a strong A result. This stage reruns ONLY S and C.
#
# Usage:
#   python .\run_ablation.py --validate_refined
# ================================================================


# ================================================================
# Final formal 8-run launcher
# Usage:
#   python .\run_ablation.py --run_all_formal
# ================================================================
def _run_all_formal():
    common = [
        "--dataset", "cifar10",
        "--num_classes", "10",
        "--num_clients", "20",
        "--num_online_clients", "8",
        "--num_rounds", "200",
        "--num_epochs_local_training", "10",
        "--batch_size_local_training", "32",
        "--lr_local_training", "0.1",
        "--imb_factor", "0.01",
        "--non_iid_alpha", "0.5",
        "--rs_alpha", "0.5",
        "--seed", "42",
    ]

    final_s = ["--semantic_epsilon", "0.75"]
    final_c = [
        "--confidence_alpha", "0.70",
        "--confidence_warmup", "30",
    ]
    final_a = [
        "--tail_agg_weight", "1.50",
    ]

    runs = [
        ("FINAL_FedLF_r200", "fedlf", []),
        ("FINAL_S_r200", "s", final_s),
        ("FINAL_C_r200", "c", final_c),
        ("FINAL_A_r200", "a", final_a),
        ("FINAL_SC_r200", "sc", final_s + final_c),
        ("FINAL_SA_r200", "sa", final_s + final_a),
        ("FINAL_CA_r200", "ca", final_c + final_a),
        ("FINAL_FULL_r200", "full", final_s + final_c + final_a),
    ]

    print("=" * 78)
    print("THESIS FINAL 8-RUN ABLATION | RESTORED FedLF BASELINE")
    print("CIFAR-10-LT | IF100 | non-IID alpha=0.5 | seed=42 | 200 rounds")
    print("S: V2 semantic-density sampling, epsilon=0.75 (2 focused + 6 uniform)")
    print("C: additive confidence/class-aware weighting, alpha=0.70, warmup=30")
    print("A: server-only bounded residual aggregation, agg=1.50")
    print("=" * 78)

    script = os.path.abspath(__file__)

    for idx, (run_name, variant, extra) in enumerate(runs, start=1):
        print()
        print("=" * 78)
        print(f"[{idx}/8] START {run_name} | variant={variant}")
        print("=" * 78)

        cmd = [
            sys.executable,
            script,
            "--variant", variant,
            "--run_name", run_name,
        ] + common + extra

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(
                f"Experiment failed: {run_name}, exit code={result.returncode}"
            )

        print(f"[{idx}/8] FINISHED {run_name}")

    print()
    print("=" * 78)
    print("ALL 8 FINAL 200-ROUND ABLATION RUNS FINISHED")
    print("=" * 78)
    print("Expected logs:")
    for run_name, _, _ in runs:
        print(f"  {run_name}.log")


if __name__ == "__main__":
    if "--run_all_formal" in sys.argv:
        _run_all_formal()
    else:
        fedlf()
