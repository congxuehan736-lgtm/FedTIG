# FedTIG Reproducibility Guide

## 1. Environment

The archived experiment environment used Python 3.11 with PyTorch 2.7.1 and torchvision 0.22.1. Install the released dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\\.venv\\Scripts\\Activate.ps1`.

## 2. Data

Place CIFAR data under:

```text
data/
├── CIFAR10/
└── CIFAR100/
```

Raw datasets are not distributed in this repository.

## 3. Main CIFAR-10 FedTIG run

```bash
bash scripts/run_cifar10.sh
```

Main protocol: IF=100, Dirichlet alpha=0.5, 20 clients, 8 online clients, 200 rounds, 10 local epochs, batch size 32, learning rate 0.1, seed 42.

## 4. Full S/C/A ablation

```bash
bash scripts/run_ablation.sh
```

Reference logs: `results/ablation/`.

## 5. Unified external baselines

```bash
PYTHONPATH=src python src/algorithm/baseline_cifar10.py --seed 42 --imb_factor 0.01 --alpha 0.5 --num_clients 20 --online 8 --rounds 200 --local_epochs 10 --batch_size 32 --lr 0.1 --data_root data/CIFAR10
```

Reference logs/CSVs: `results/baselines/`.

## 6. Random-seed robustness

Reference full-FedTIG logs for seeds 42, 123, 456, and 789 are stored in `results/seeds/`.

## 7. Structural Exposure experiment

```bash
bash scripts/run_host_count.sh
```

Reference logs: `results/host_count/`.

## 8. Participation sensitivity

```bash
bash scripts/run_participation.sh
```

Reference logs: `results/participation/`. The archived 40% point (`m=8`) reuses the identical `h=2,m=8` run from `results/host_count/`; see `results/participation/README.md`.

## 9. E/A/R diagnostics

```bash
bash scripts/run_ear.sh
```

Reference logs: `results/ear/`.

## 10. Retention sensitivity

```bash
bash scripts/run_sensitivity.sh
```

The sweep is beta = 0, 0.05, 0.125, 0.20, 0.30, corresponding to manuscript retention factors 1.00, 1.05, 1.125, 1.20, and 1.30.

## 11. CIFAR-100-LT cross-dataset experiment

```bash
bash scripts/run_cifar100.sh
```

Reference logs: `results/cifar100/`.

The archived CIFAR-100 configuration is a distinct cross-dataset protocol (IF=50 and 5 local epochs with its recorded CIFAR-100-specific schedule). This difference must be disclosed in the manuscript or replaced by a rerun under the main unified hyperparameters.

## 12. Reproducibility checklist

Before comparing results, verify the same dataset split, random seed, client count, participation ratio, local epochs, learning rate, imbalance factor, Dirichlet parameter, and S/C/A settings.
