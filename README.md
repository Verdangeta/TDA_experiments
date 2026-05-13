# The Density of Cross-Persistence Diagrams and Its Applications

This repository contains the cleaned publication code for the paper:

> Alexander Mironenko, Evgeny Burnaev, Serguei Barannikov, **The Density of Cross-Persistence Diagrams and Its Applications**. IEEE Xplore document [11417786](https://ieeexplore.ieee.org/document/11417786).

The original repository had code split between two server branches (`master` and `ZH_exps`). This branch keeps only the files that correspond to experiments described in the paper, clears notebook outputs, removes scratch notebooks, and replaces server-specific experiment code with a safer CLI entry point.

## Repository contents

| File | Paper section | Purpose |
| --- | --- | --- |
| `01_mtd_density_estimation.ipynb` | Section V, Appendix G | MTD density estimation for image point-cloud classes and noise-sensitivity plots. |
| `02_cross_ripsnet_synthetic.ipynb` | Section VI-D | Cross-RipsNet experiments on synthetic unions of circles. |
| `03_cross_ripsnet_modelnet10.ipynb` | Section VI-E | Cross-RipsNet experiments for ModelNet10-derived 3D point clouds. |
| `04_cross_ripsnet_text.ipynb` | Section VI-E | Cross-RipsNet experiments for GPT/Human text embeddings. |
| `05_gravitational_waves_mtd.ipynb` | Section VII-A | Cross-persistence/MTD feature generation for gravitational-wave classification. |
| `06_human_gpt_density_analysis.ipynb` | Appendix E | Qualitative density analysis for human-written vs AI-generated text point clouds. |
| `distance_matrix_exp.py` | Section VI, Table I-style runs | CLI for Cross-RipsNet variants with and without cross-distance-matrix features. |
| `Topological_classifier.py` | Section VII | scikit-learn compatible topological feature generator for time series. |
| `utils.py` | Shared | Cross-RipsNet layers, distance utilities, density losses, and metric helpers. |
| `requirements.txt` | Setup | Python dependencies used by the notebooks and scripts. |

Removed as publication noise: checkpoint files, notebook outputs, server-checking notebooks, generic time-series scratch work, curve-drawing scratch work, duplicate encoding experiments, and unrelated anti-noise/prototype notebooks.

## Setup

The experiments were developed for a GPU Python environment. Python 3.10 is recommended because `giotto-tda` has historically had tighter Python-version constraints than the rest of the stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The MTop-Divergence implementation is imported as `mtd`. Install it from the upstream project used in the paper:

```bash
git clone https://github.com/IlyaTrofimov/MTopDiv external/MTopDiv
pip install -e external/MTopDiv
```

If your environment does not provide CUDA, use the CPU options where available (`--pdist-device cpu` in `distance_matrix_exp.py`) and edit notebook cells that pass `pdist_device="cuda"` to `mtd.calc_cross_barcodes`.

## Data and generated artifacts

Large datasets and generated intermediate files are intentionally not committed. The notebooks/scripts expect the following local layout when reproducing the full experiments:

```text
Data/
  cross_ripsnet_3d_exp/
    3d_shapes_pc_train_2500_30_boostrap
    3d_shapes_indexes_2500_30_boostrap
    3d_shapes_PI_2500_30_boostrap
  cross_ripsnet_text_exp/
    human_gpt3_davinci_003_pc_train
    human_gpt3_davinci_003_train_indexes
    human_gpt3_davinci_003_PI_10000_50_boostrap
    human_gpt3_davinci_003_pds_10000_50_boostrap
RipsNet_exp/
  cross_pd_circles_3000_strat_boost_10_data.npy
  cross_pd_circles_3000_strat_boost_10_indexes.npy
  cross_pd_circles_3000_strat_boost_10_PI.npy
density_exp/
  # cached MTD/cross-barcode outputs produced by 01_mtd_density_estimation.ipynb
runs/
  # script outputs
```

Public datasets used by the notebooks include MNIST/OpenML, CIFAR, COIL20, CIFAR100, ModelNet10-derived point clouds, gravitational-wave examples from giotto-tda, and GPT/Human text embeddings prepared as described in the paper.

## How to run

### Notebooks

Start Jupyter from the repository root so the relative paths above resolve correctly:

```bash
jupyter lab
```

Then run the notebooks in the numbered order if you want the broad paper walkthrough. Some notebooks consume cached artifacts produced by earlier long-running barcode computations; rerunning all cells from scratch can be expensive.

### Cross-RipsNet CLI

The script reproduces the Cross-RipsNet model comparisons once the precomputed point-cloud pairs and target densities are available:

```bash
python distance_matrix_exp.py --task synthetic --data-root . --output-dir runs/cross_ripsnet --pdist-device cuda:0
python distance_matrix_exp.py --task 3d_shapes --data-root . --output-dir runs/cross_ripsnet --pdist-device cuda:0
python distance_matrix_exp.py --task textual --data-root . --output-dir runs/cross_ripsnet --pdist-device cuda:0
```

Use `--pdist-device cpu` for CPU-only runs. The resulting metrics are written to `runs/cross_ripsnet/<run-name>_<task>_metrics.json`.

## Notes for reviewers/reusers

- All notebooks are cleared of outputs and execution counts.
- No private experiment tracker credentials are required or stored.
- Hard-coded server GPU selection cells were removed from notebooks.
- `Data/`, `RipsNet_exp/`, `density_exp/`, and `runs/` are ignored by Git because they contain large downloaded or generated artifacts.
