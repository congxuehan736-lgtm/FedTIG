# -*- coding: utf-8 -*-
"""
run_baselines.py
================
统一运行 FedLF 项目的 CIFAR-10-LT baseline。

运行方式：
    python run_baselines.py --seed 42

会依次运行：
    FedAvg
    FedProx
    FedBN
    FedRS
    FEDIC
    Focal Loss
    CReFF

注意：
1. 不修改项目原有 algorithm/*.py。
2. 每个算法实际调用项目原来的 main.py。
3. 每个算法单独保存日志。
4. 某一个算法失败后立即停止，避免后面的实验在错误环境下继续跑。
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


METHODS = [
    ("fedavg", "FedAvg"),
    ("fedprox", "FedProx"),
    ("fedbn", "FedBN"),
    ("fedrs", "FedRS"),
    ("fedic", "FEDIC"),
    ("focalloss", "Focal Loss"),
    ("creff", "CReFF"),
]


def make_command(args, algorithm):
    root = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(root, "main.py")

    return [
        sys.executable,
        main_py,

        "--algorithm", algorithm,
        "--dataset", "cifar10",

        "--num_clients", str(args.num_clients),
        "--num_rounds", str(args.num_rounds),
        "--num_channels", "3",
        "--num_epochs_local_training", str(args.local_epochs),
        "--batch_size_local_training", str(args.batch_size),
        "--num_online_clients", str(args.online_clients),
        "--num_classes", "10",

        "--batch_size_test", "500",
        "--lr_local_training", str(args.lr),

        "--non_iid_alpha", str(args.alpha),
        "--seed", str(args.seed),

        "--imb_type", "exp",
        "--imb_factor", str(args.imb_factor),

        # FedProx
        "--mu", str(args.mu),

        # FedRS
        "--rs_alpha", str(args.rs_alpha),

        # Focal Loss
        "--alpha", str(args.focal_alpha),
        "--gamma", str(args.focal_gamma),

        # CReFF
        "--match_epoch", str(args.match_epoch),
        "--crt_epoch", str(args.crt_epoch),
        "--batch_real", str(args.batch_real),
        "--num_of_feature", str(args.num_of_feature),
        "--lr_feature", str(args.lr_feature),
        "--lr_net", str(args.lr_net),

        # 项目已有参数
        "--method", args.method,
        "--dsa_strategy", args.dsa_strategy,
    ]


def run_algorithm(args, algorithm, display_name):
    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "Logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"cifar10_{algorithm}_seed{args.seed}.log"
    )

    cmd = make_command(args, algorithm)

    start = datetime.now()

    header = (
        "\n"
        + "=" * 80 + "\n"
        + f"START: {display_name}\n"
        + f"time: {start:%Y-%m-%d %H:%M:%S}\n"
        + f"algorithm: {algorithm}\n"
        + f"dataset: CIFAR-10-LT\n"
        + f"IF: {1.0 / args.imb_factor:g}\n"
        + f"imb_factor: {args.imb_factor}\n"
        + f"clients: {args.num_clients}\n"
        + f"online_clients: {args.online_clients}\n"
        + f"alpha: {args.alpha}\n"
        + f"rounds: {args.num_rounds}\n"
        + f"local_epochs: {args.local_epochs}\n"
        + f"batch_size: {args.batch_size}\n"
        + f"lr: {args.lr}\n"
        + f"seed: {args.seed}\n"
        + f"log: {log_path}\n"
        + "=" * 80 + "\n"
    )

    print(header, flush=True)

    # 覆盖旧日志，避免把不同实验混在一起。
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(header)
        log.write("COMMAND:\n")
        log.write(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        log.write("\n\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        finally:
            if process.stdout is not None:
                process.stdout.close()

        return_code = process.wait()

        end = datetime.now()
        footer = (
            "\n"
            + "=" * 80 + "\n"
            + f"END: {display_name}\n"
            + f"time: {end:%Y-%m-%d %H:%M:%S}\n"
            + f"return_code: {return_code}\n"
            + "=" * 80 + "\n"
        )

        print(footer, flush=True)
        log.write(footer)
        log.flush()

    if return_code != 0:
        raise RuntimeError(
            f"{display_name} 运行失败。\n"
            f"请查看日志：{log_path}"
        )

    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Run all CIFAR-10-LT baselines for FedLF."
    )

    parser.add_argument("--seed", type=int, default=42)

    # 论文复现实验配置
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--online_clients", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)

    # FedProx
    parser.add_argument("--mu", type=float, default=0.01)

    # FedRS
    parser.add_argument("--rs_alpha", type=float, default=0.25)

    # Focal Loss
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # CReFF
    parser.add_argument("--match_epoch", type=int, default=100)
    parser.add_argument("--crt_epoch", type=int, default=300)
    parser.add_argument("--batch_real", type=int, default=32)
    parser.add_argument("--num_of_feature", type=int, default=100)
    parser.add_argument("--lr_feature", type=float, default=0.1)
    parser.add_argument("--lr_net", type=float, default=0.01)

    parser.add_argument("--method", type=str, default="DSA")
    parser.add_argument(
        "--dsa_strategy",
        type=str,
        default="color_crop_cutout_flip_scale_rotate",
    )

    # 可以只跑一个，排查问题时非常有用。
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=[
            "all",
            "fedavg",
            "fedprox",
            "fedbn",
            "fedrs",
            "fedic",
            "focalloss",
            "creff",
        ],
    )

    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isfile(os.path.join(root, "main.py")):
        raise FileNotFoundError(
            "当前目录没有 main.py。\n"
            f"当前目录：{root}\n"
            "请把 run_baselines.py 放在 D:\\Desktop\\FedLF_project\\FedLF\\ 下。"
        )

    if args.imb_factor <= 0:
        parser.error("--imb_factor must be > 0")
    if args.num_clients <= 0:
        parser.error("--num_clients must be > 0")
    if not 1 <= args.online_clients <= args.num_clients:
        parser.error("--online_clients must be between 1 and num_clients")
    if args.alpha <= 0:
        parser.error("--alpha must be > 0")

    selected = METHODS
    if args.only != "all":
        selected = [x for x in METHODS if x[0] == args.only]

    print("\n" + "=" * 80)
    print("FedLF CIFAR-10-LT BASELINE EXPERIMENT")
    print("=" * 80)
    print(f"Project : {root}")
    print(f"Seed    : {args.seed}")
    print(f"IF      : {1.0 / args.imb_factor:g}")
    print(f"Clients : {args.num_clients}")
    print(f"Online  : {args.online_clients}")
    print(f"Alpha   : {args.alpha}")
    print(f"Rounds  : {args.num_rounds}")
    print(f"Epochs  : {args.local_epochs}")
    print(f"Batch   : {args.batch_size}")
    print(f"LR      : {args.lr}")
    print("Methods :", ", ".join(name for _, name in selected))
    print("=" * 80)

    completed = []

    for i, (algorithm, display_name) in enumerate(selected, 1):
        print(
            f"\n>>> [{i}/{len(selected)}] {display_name} START\n",
            flush=True,
        )

        log_path = run_algorithm(
            args,
            algorithm,
            display_name,
        )

        completed.append((display_name, log_path))

        print(
            f"\n>>> [{i}/{len(selected)}] {display_name} DONE\n",
            flush=True,
        )

    print("\n" + "=" * 80)
    print("ALL BASELINES COMPLETED")
    print("=" * 80)

    for name, log_path in completed:
        print(f"{name:12s} -> {log_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
