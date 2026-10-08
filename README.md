# FedTIG

Official implementation and reproduction package for the manuscript:

**FedTIG: Effective Training Rarity via Exposure--Acquisition--Retention Governance for Federated Long-Tailed Learning**

FedTIG studies federated long-tailed learning from an **Exposure--Acquisition--Retention (E--A--R)** perspective. The central idea is that global class frequency alone does not determine the effective training opportunity of a class under partial client participation.

## Method overview

FedTIG contains three coordinated modules:

- **S -- Exposure governance:** semantic-density-aware client sampling changes which class information enters the current communication round.
- **C -- Acquisition governance:** class- and uncertainty-aware local weighting reallocates optimization budget toward exposed tail classes and difficult samples.
- **A -- Retention governance:** tail-specialist retention aggregation strengthens the preservation of tail-class update directions at the server.

The three stages jointly determine the class-wise effective-update strength used in the manuscript's effective training rarity formulation.

## Repository structure

```text
FedTIG/
├── README.md
├── REPRODUCIBILITY.md
├── CODE_STRUCTURE.md
├── THIRD_PARTY_NOTICE.md
├── requirements.txt
├── .gitignore
├── configs/
├── scripts/
├── results/
└── src/
    ├── algorithm/
    ├── Dataset/
    └── Model/
```

> **Important:** the released repository should use directory names that match the imports in the source code. In the current implementation, the code imports `Dataset.*` and `Model.*`; therefore Linux users require the corresponding directory names to match exactly, or the imports must be refactored consistently to lowercase.

## Environment

Recommended environment:

- Python 3.10
- PyTorch 2.x
- CUDA-enabled NVIDIA GPU for full experiments

Create and activate an isolated environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For reproducibility, we recommend replacing loose lower bounds in `requirements.txt` with the exact package versions used for the reported experiments.

## Dataset preparation

The repository uses the public CIFAR-10 and CIFAR-100 datasets through `torchvision`. The datasets are **not redistributed** in this repository.

A convenient layout is:

```text
data/
├── CIFAR10/
└── CIFAR100/
```

The training scripts can download CIFAR automatically when `download=True` is enabled. Dataset paths can also be specified explicitly through command-line arguments such as `--path_cifar10` and `--path_cifar100`.

## Main CIFAR-10-LT setting

The principal CIFAR-10-LT protocol used in the manuscript is:

| Setting | Value |
|---|---:|
| Number of clients | 20 |
| Participating clients per round | 8 |
| Participation rate | 40% |
| Communication rounds | 200 |
| Local epochs | 10 |
| Batch size | 32 |
| Learning rate | 0.1 |
| Dirichlet non-IID parameter | 0.5 |
| Long-tail imbalance factor | IF = 100 (`imb_factor=0.01`) |
| Backbone | ResNet-8 |
| Main seed | 42 |

### FedTIG hyperparameters

| Manuscript quantity | Code argument | Value |
|---|---|---:|
| S exploration ratio `epsilon` | `--semantic_epsilon` | 0.75 |
| C mixing coefficient `alpha` | `--confidence_alpha` | 0.70 |
| C warmup | `--confidence_warmup` | 30 rounds |
| A implementation strength | `--tail_agg_weight` | 1.50 |
| Manuscript retention factor `lambda_A` | internal residual factor | 1.125 |

The current implementation parameterizes A through `tail_agg_weight`. For the official setting,

```text
beta = 0.25 * (tail_agg_weight - 1.0) = 0.125
lambda_A = 1 + beta = 1.125
```

which matches the retention factor reported in the manuscript.

## Running the full FedTIG model on CIFAR-10-LT

From the repository root, expose `src/` on `PYTHONPATH` and run the `full` variant:

```bash
PYTHONPATH=src python src/algorithm/run_ablation.py \
  --variant full \
  --run_name FedTIG_CIFAR10_seed42 \
  --semantic_epsilon 0.75 \
  --confidence_alpha 0.70 \
  --confidence_warmup 30 \
  --tail_agg_weight 1.50 \
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
```

On Windows PowerShell, set the module path before running:

```powershell
$env:PYTHONPATH = "src"
python src/algorithm/run_ablation.py --variant full --run_name FedTIG_CIFAR10_seed42 --semantic_epsilon 0.75 --confidence_alpha 0.70 --confidence_warmup 30 --tail_agg_weight 1.50 --dataset cifar10 --num_classes 10 --num_clients 20 --num_online_clients 8 --num_rounds 200 --num_epochs_local_training 10 --batch_size_local_training 32 --lr_local_training 0.1 --imb_factor 0.01 --non_iid_alpha 0.5 --rs_alpha 0.5 --seed 42 --device cuda --path_cifar10 data/CIFAR10
```

## Full S/C/A ablation

The ablation runner supports all eight combinations:

| Variant | S | C | A |
|---|:---:|:---:|:---:|
| `fedlf` |  |  |  |
| `s` | ✓ |  |  |
| `c` |  | ✓ |  |
| `a` |  |  | ✓ |
| `sc` | ✓ | ✓ |  |
| `sa` | ✓ |  | ✓ |
| `ca` |  | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ |

To run the manuscript's complete eight-run ablation suite:

```bash
PYTHONPATH=src python src/algorithm/run_ablation.py --run_all_formal
```

Reference logs are stored under `results/ablation/`.

## Unified CIFAR-10 baselines

The repository contains implementations/re-implementations for:

- FedAvg
- FedProx
- FedBN
- FedRS
- FEDIC
- CReFF

Run the unified baseline suite with the manuscript training budget:

```bash
PYTHONPATH=src python src/algorithm/baseline_cifar10.py \
  --seed 42 \
  --imb_factor 0.01 \
  --alpha 0.5 \
  --num_clients 20 \
  --online 8 \
  --rounds 200 \
  --local_epochs 10 \
  --batch_size 32 \
  --lr 0.1 \
  --data_root data/CIFAR10
```

These baseline implementations use a unified local data/training pipeline. They should be described as **re-implementations under a unified protocol** rather than byte-for-byte reproductions of the original authors' repositories.

## Controlled support-redistribution experiment

This experiment fixes the global number of target-class samples and changes the number of clients hosting the target class.

Example for `h=1`:

```bash
PYTHONPATH=src python src/algorithm/fedlf_hc_controlled.py \
  --target_class 9 \
  --host_count 1 \
  --num_clients 20 \
  --num_online_clients 8 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --imb_factor 0.01 \
  --non_iid_alpha 0.5 \
  --seed 42 \
  --path_cifar10 data/CIFAR10
```

Repeat with:

```text
--host_count 1
--host_count 2
--host_count 4
--host_count 8
--host_count 16
```

Reference logs are stored under `results/host_count/`.

## Partial-participation sensitivity

The participation experiment fixes the target-class hosting-client count at `h=2` and varies the number of online clients:

```bash
PYTHONPATH=src python src/algorithm/fedlf_participation_controlled.py \
  --target_class 9 \
  --host_count 2 \
  --num_online_clients 2 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --imb_factor 0.01 \
  --non_iid_alpha 0.5 \
  --seed 42 \
  --path_cifar10 data/CIFAR10
```

Repeat with `--num_online_clients 2`, `4`, `8`, and `16`, corresponding to 10%, 20%, 40%, and 80% participation.

Reference logs are stored under `results/participation/`.

## E--A--R mechanism diagnostics

The direct diagnostics use the controlled stress setting `N_9=50`, `h_9=2`, and `m=4`.

Example:

```bash
PYTHONPATH=src python src/algorithm/fedtig_ear_direct_cifar10.py \
  --mode fedtig \
  --dataset cifar10 \
  --num_clients 20 \
  --num_online_clients 4 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --s_epsilon 0.75 \
  --c_alpha 0.70 \
  --c_warmup 30 \
  --tail_agg_weight 1.50 \
  --target_class 9 \
  --controlled_host_count 2 \
  --seed 42 \
  --path_cifar10 data/CIFAR10
```

Available modes are `fedlf`, `s`, `c`, `a`, `sc`, `sa`, `ca`, and `fedtig`.

Reference logs are stored under `results/ear/`.

## Retention sensitivity

The retention-sensitivity script isolates module A and evaluates:

```text
beta in {0.00, 0.05, 0.125, 0.20, 0.30}
```

Example:

```bash
PYTHONPATH=src python src/algorithm/fedtig_beta_pareto_cifar10.py \
  --beta 0.125 \
  --dataset cifar10 \
  --num_clients 20 \
  --num_online_clients 8 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --seed 42 \
  --path_cifar10 data/CIFAR10
```

Reference logs are stored under `results/beta_sensitivity/`.

## CIFAR-100-LT

The repository includes the exact source files and reference logs used for the reported CIFAR-100-LT cross-dataset experiment. The released run uses IF=50 (`imb_factor=0.02`), Dirichlet parameter `0.5`, 20 clients, 8 online clients, 200 rounds, 5 local epochs, batch size 32, and seed 42.

Run both the same-protocol FedLF reference and the FedTIG cross-dataset configuration with:

```bash
bash scripts/run_cifar100.sh
```

Reference logs are stored under `results/cifar100/`.

> **Protocol note.** The archived CIFAR-100 FedTIG run uses a cross-dataset Protocol-C configuration implemented in `fedlf_cifar100_ready.py` (including 5 local epochs and its recorded CIFAR-100-specific S/C/A schedule). This is not identical to the main CIFAR-10 hyperparameter setting. The manuscript should disclose this difference explicitly, or the CIFAR-100 experiment should be rerun under a fully unified configuration before submission.

## Results included in this repository

The repository contains reference logs for:

- S/C/A ablation
- retention sensitivity
- CIFAR-100-LT diagnostics
- E--A--R mechanism diagnostics
- hosting-client support redistribution
- partial-participation sensitivity

For a publication release, we recommend including a compact machine-readable summary (for example `results/summary.csv`) that maps each manuscript table/figure to the corresponding command, seed, and output log.

## Reproducibility notes

- Use the same random seed, long-tail split, Dirichlet partition, number of clients, participation rate, local epochs, and learning rate when comparing methods.
- The main paper protocol uses 10 local epochs on CIFAR-10-LT.
- The baseline implementations in this repository are unified re-implementations; differences from original repositories should be documented.
- The source tree should not contain `__pycache__/` directories or `.pyc` files.
- Do not commit raw CIFAR datasets, private server paths, API keys, checkpoints containing sensitive data, or large temporary logs.

See `REPRODUCIBILITY.md` and `CODE_STRUCTURE.md` for additional details.

## Citation

The manuscript is currently under preparation/submission. Please use the following placeholder until a DOI or preprint identifier is available:

```bibtex
@article{cong2026fedtig,
  title   = {FedTIG: Effective Training Rarity via Exposure--Acquisition--Retention Governance for Federated Long-Tailed Learning},
  author  = {Cong, Xuehan and Wu, Tianhao and Bao, Xingxing and Zhao, Fang and Tian, Qiao},
  year    = {2026},
  note    = {Manuscript}
}
```

The citation entry should be updated after the paper receives a DOI or public preprint identifier.

## License and third-party code

Before release, verify the licenses of all code adapted from external repositories. Do not relicense third-party code under MIT unless its original license permits this. Preserve required copyright notices and attribution files.

See `THIRD_PARTY_NOTICE.md`. Add a repository-wide license only after verifying all upstream licensing and attribution requirements.

## Contact

For questions about the manuscript or code, please contact the corresponding author:

**Qiao Tian**  
Harbin Engineering University  
Email: `tianqiao@hrbeu.edu.cn`
