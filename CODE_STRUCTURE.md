# FedTIG Code Structure

## Official experiment entry points

- `src/algorithm/run_ablation.py`: main CIFAR-10 FedTIG implementation and S/C/A ablations.
- `src/algorithm/baseline_cifar10.py`: unified CIFAR-10 baseline suite (FedAvg, FedProx, FedBN, FedRS, FEDIC, CReFF).
- `src/algorithm/fedlf_hc_controlled.py`: fixed-global-frequency / varied-host-count experiment.
- `src/algorithm/fedlf_participation_controlled.py`: participation-rate sensitivity.
- `src/algorithm/fedtig_ear_direct_cifar10.py`: E/A/R mechanism diagnostics.
- `src/algorithm/fedtig_beta_pareto_cifar10.py`: Retention-strength sensitivity.
- `src/algorithm/fedlf_cifar100_baseline.py`: same-protocol FedLF reference for CIFAR-100-LT.
- `src/algorithm/fedlf_cifar100_ready.py`: archived FedTIG CIFAR-100-LT cross-dataset run.

## Supporting modules

- `src/Dataset/`: dataset utilities and long-tail partition construction.
- `src/Model/`: model definitions, including ResNet-8.
- `src/algorithm/fedprox.py`, `svgfedbn.py`, `svgfedrs.py`, `svgfedic.py`, `svgCReFF.py`: baseline components used by the unified baseline runner.
- `src/utils/`: auxiliary utilities.

## Reproduction scripts

- `scripts/run_cifar10.sh`
- `scripts/run_ablation.sh`
- `scripts/run_host_count.sh`
- `scripts/run_participation.sh`
- `scripts/run_ear.sh`
- `scripts/run_sensitivity.sh`
- `scripts/run_cifar100.sh`

Run scripts from a Bash environment (Linux, WSL, Git Bash, or macOS). They set `PYTHONPATH` automatically.

## Reference results

Curated logs corresponding to manuscript experiments are stored in `results/`. Generated checkpoints, raw datasets, caches, and virtual environments are intentionally excluded.
