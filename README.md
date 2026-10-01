# Copula Active Subspaces

Code and saved results for the two papers

> J. Chen and P. J. van Leeuwen, *Copula Active Subspaces I: A Score-Covariance Method for Reduced-Order Non-Gaussian Density Estimation*. Submitted to the SIAM/ASA Journal on Uncertainty Quantification; arXiv:2609.36142.
>
> J. Chen and P. J. van Leeuwen, *Copula Active Subspaces II: Error Decomposition, A Posteriori Estimation, and Sharpness of the Bounds*. Submitted to the SIAM/ASA Journal on Uncertainty Quantification.

The version of this repository used for Part I is the release tagged `v1.0-part1`, and the version used for Part II is the release tagged `v1.0-part2`.

Copula Active Subspaces (CAS) estimate a non-Gaussian noise density from samples. A componentwise rank transform separates the marginal distributions from the dependence between coordinates; the leading eigenvectors of the copula score covariance span a low-dimensional subspace that carries the dependence; and Hermite score matching estimates the density on that subspace. The estimated log-density and its gradient are available in closed form.

## Installation

The code needs Python 3.11 or later with NumPy, SciPy, Matplotlib and scikit-learn. From the repository root, either

```
conda env create -f environment.yml
conda activate cas
```

or

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The pinned versions are those that produced the results in the papers. Compiling the papers also requires a LaTeX installation with `pdflatex`.

## Reproducing the figures

```
python reproduce.py
```

redraws every figure of both papers from the saved results in `cache/`, in about a minute. The figures are written to `out/` and copied to `paper/`, where the LaTeX sources include them. Their labels are typeset with LaTeX when `latex`, `dvipng` and the cm-super fonts are installed, as for the papers, and otherwise with Matplotlib's own math renderer, which gives the same figures in slightly different type. To compile the two papers and their supplementary materials:

```
python reproduce.py --pdf
```

Every number reported in the text and tables of both papers is also computed from the saved results. The command

```
python reproduce.py --check-numbers
```

recomputes each of them and reports any that differ from the LaTeX sources.

## Recomputing the results

Every figure and table is drawn from results saved in `cache/`, so reproducing the figures requires no experiment to be run. The results themselves are recomputed by

```
python reproduce.py --recompute ITEM
```

with `ITEM` one of the names in the table below; the figures are redrawn afterwards. `python reproduce.py --recompute all` recomputes every result, which takes several days on one machine. An experiment skips the results that are already saved, so a result is recomputed only after its file is deleted from `cache/`. The option `--workers N` sets the number of parallel processes (default 4). Experiments print their progress and diagnostics as they run; recomputing `part1-pca` also writes to `out/logs/sm1_level_bound.txt` the number of Stage-2 solves, at each rank, that end at the round cap with the level bound unmet, which the Part I supplement reports.

| Paper | Figures and tables | `ITEM` | Experiment scripts in `experiments/` |
|---|---|---|---|
| Part I | Figures 2–4 and the results of Section 5 | `part1-main` | `exp_noise_entropy.py`, `exp_section51_subspace.py`, `exp_paper_run.py` |
| Part I | Figure 5 | `part1-posteriors` | `exp_section53_hero.py` |
| Part I | normalizer accuracy, reported in the appendix | `part1-normalizer` | `exp_normalizer_stability.py` |
| Part I supplement | Tables SM1 and SM2 | `part1-pca` | `exp_sm1_pca_ablation.py` |
| Part I supplement | Figure SM2 | `part1-random-subspaces` | `exp_random_subspace_oracle_kl.py` |
| Part I supplement | Table SM3 | `part1-tail-sum` | `exp_ehat_tightness.py` |
| Part I supplement | Tables SM4–SM7 | `part1-regularizer` | `exp_sm3_c_stability.py`, `exp_sm3_rh2_kappa.py`, `exp_regularizer_ablation.py` |
| Part I supplement | Tables SM8 and SM9 | `part1-centering` | `exp_centering_ablation.py`, `exp_k2_sweep_centering.py` |
| Part II | reference quantities used by Figures 1–3 | `part2-reference` | `exp_stage1_reference.py` |
| Part II | Figure 1; supplement Figures SM3 and SM4 | `part2-decomposition` | `exp_delta_sub_Z.py`, `exp_stage2_oracle_kl.py` |
| Part II | Figures 2 and 3 | `part2-a-posteriori` | `exp_stage1_aposteriori.py`, `exp_apost_endtoend.py` |
| Part II | estimate of the truncation term, reported in the text | `part2-truncation` | `exp_hier_trunc.py` |
| Part II | Section 5, ratio of the bound to the divergence at the estimated subspace | `part2-certificate` | `exp_stage1_cert.py`, `exp_delta_sub_Z.py` |
| Part II supplement | Table SM1 | `part2-sharper-bound` | `exp_jhi_supplement.py` |

The Part II items read the results of the Part II items listed above them. Three figures have no saved results and are computed when they are drawn: Figure 1 of Part I, which is also Figure SM1 of the Part II supplement; Figure SM1 of the Part I supplement; and Figure SM2 of the Part II supplement. `python reproduce.py --list` prints the same information.

## Using the estimator

```python
import numpy as np
from cas import ReducedDensityModel, sample_noise

rng = np.random.default_rng(0)
x = sample_noise(20000, rng)        # samples of the Example 1 noise, d = 20
model = ReducedDensityModel(r=4, K=4, q=2, K_inner=5, q_inner=3, seed=0).fit(x)
log_density = model.evaluate_log_density(x[:10])
```

Here `r` is the dimension of the subspace, `(K, q)` the total degree and interaction order of the Hermite expansion that identifies it, and `(K_inner, q_inner)` those of the expansion on the subspace. The experiments choose both pairs by cross-validation with `cas.kq_selection.fit_cas_autoselect`.

## Repository structure

`src/cas/` contains the estimator. `ranking.py` implements the componentwise rank transform and the marginal density estimates, and `hermite.py` and `features.py` the Hermite basis and its derivatives. `stage1.py` and `hermite_score_matching.py` implement the first stage, which estimates the copula score by Hermite score matching and takes the leading eigenvectors of its second-moment matrix. `stage2.py`, `ridge.py` and `constrained.py` implement the second stage, which estimates the density on the subspace with its regularizer and constraints. `reduced_density.py` and `densities.py` assemble the full estimator and define the baselines. `noise.py`, `even_fold.py` and `conformal_cube.py` define the three example noise laws, and `bip.py` the Bayesian inference problem. The remaining modules compute reference quantities used only by the experiments, such as the reference subspace (`analytic_oracle.py`) and reference Kullback–Leibler divergences (`conditional_kl.py`, `quad_marginalization.py`, `stage1_reference.py`, `oracle_reduced.py`).

`experiments/` contains one script per experiment, each writing its results to `cache/`, and `figures/` the scripts that draw the figures from those results. `cache/` holds the saved results; `out/` receives the figures and holds three summaries of the Part II decomposition that its figures read. `paper/` holds the LaTeX sources of both papers and their supplementary materials. `tests/` holds the unit tests, which run with `python -m pytest tests` in about two minutes. `scripts/` holds numerical checks of the constrained Stage-2 solve, of the held-out choice of the regularization constants and of the quadrature estimator, a script that rebuilds a Part II summary file from the saved results, and `check_paper_numbers.py`, which `python reproduce.py --check-numbers` runs.

## Citation

```bibtex
@article{ChenVanLeeuwenCAS1,
  author  = {Chen, Joshua and van Leeuwen, Peter Jan},
  title   = {Copula Active Subspaces {I}: A Score-Covariance Method for
             Reduced-Order Non-{G}aussian Density Estimation},
  note    = {Submitted to the SIAM/ASA Journal on Uncertainty Quantification},
  eprint  = {2609.36142},
  archivePrefix = {arXiv},
  year    = {2026}
}

@article{ChenVanLeeuwenCAS2,
  author  = {Chen, Joshua and van Leeuwen, Peter Jan},
  title   = {Copula Active Subspaces {II}: Error Decomposition, A Posteriori
             Estimation, and Sharpness of the Bounds},
  note    = {Submitted to the SIAM/ASA Journal on Uncertainty Quantification},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
