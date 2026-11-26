# CPDEM

Constitutive Parameterized Deep Energy Method for solid mechanics. Three minimal PyTorch examples train shared material-conditioned displacement networks by minimizing potential energy. Coordinates and material samples are generated at runtime; no training dataset is required.

## Examples

| Script | Problem | Material range |
| --- | --- | --- |
| `examples/bar1d_linear.py` | Linear bar, body force `f(x)=x`, fixed left end, free right end | `E` in `[0.5, 3]` |
| `examples/beam2d_linear.py` | Plane-strain linear cantilever, right traction `(0, -5)` | `E` in `[100, 500]`, `nu` in `[0.25, 0.35]` |
| `examples/beam2d_neohookean.py` | Plane-strain compressible Neo-Hookean cantilever, right traction `(0, -2)` | `E` in `[800, 1200]`, `nu` in `[0.25, 0.35]` |

The bar occupies `[-1, 1]`; both beams occupy `[0, 4] x [0, 1]` with a fixed left edge. Essential boundary conditions are imposed through the displacement ansatz.

## Usage

Requires Python 3.10 or newer. CPU and CUDA are supported.

```bash
python -m pip install -r requirements.txt
python examples/bar1d_linear.py
python examples/beam2d_linear.py
python examples/beam2d_neohookean.py
```

Common options: `--device cpu`, `--steps 1500`, `--lbfgs-steps 100`, `--seed 2025`, and `--output outputs/model.pt`. Checkpoints contain the network state, sampled materials, and run configuration. These are compact training examples, not scripts reproducing all paper experiments.

## Source

Extracted from the CPDEM server project: `run_bar1d_generalization.py`, `run_p2inn_vs_cpdem_plane_strain.py`, and `run_neohookean_ablation.py`. Network and energy definitions are retained; entry points are simplified. FEM solvers, plotting, baseline comparisons, reference datasets, and experiment results are excluded.
