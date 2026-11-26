# CPDEM

Constitutive Parameterized Deep Energy Method for solid mechanics. Representative PyTorch examples train shared material-conditioned displacement networks by minimizing potential energy. Coordinates and material samples are generated at runtime.

## Examples

| Script | Problem | Material range |
| --- | --- | --- |
| `bar1d_linear.py` | Linear bar, body force `f(x)=x`, fixed left end, free right end | `E` in `[0.5, 3]` |
| `beam2d_linear.py` | Plane-strain linear cantilever, right traction `(0, -5)` | `E` in `[100, 500]`, `nu` in `[0.25, 0.35]` |
| `beam2d_neohookean.py` | Plane-strain compressible Neo-Hookean cantilever, right traction `(0, -2)` | `E` in `[800, 1200]`, `nu` in `[0.25, 0.35]` |

The bar occupies `[-1, 1]`; both beams occupy `[0, 4] x [0, 1]` with a fixed left edge. Essential boundary conditions are imposed through the displacement ansatz.

## Usage

Requires Python 3.10 or newer. CPU and CUDA are supported.

```bash
python -m pip install -r requirements.txt
python bar1d_linear.py
python beam2d_linear.py
python beam2d_neohookean.py
```

Common options: `--device cpu`, `--steps 1500`, `--lbfgs-steps 100`, `--seed 2025`, and `--output outputs/model.pt`. Checkpoints contain the network state, sampled materials, and run configuration.

## Source

Extracted from the CPDEM server project: `run_bar1d_generalization.py`, `run_p2inn_vs_cpdem_plane_strain.py`, and `run_neohookean_ablation.py`. The package retains the network and energy definitions with simplified training entry points for the representative examples.

## Citation

```bibtex
@article{liang2026constitutive,
  title={Constitutive parameterized deep energy method for solid mechanics problems with random material parameters},
  author={Liang, Zhangyong and Zhu, Jihong and Gao, Huanhuan},
  journal={arXiv preprint arXiv:2603.26030},
  year={2026}
}
```

## Contact

[zyliang1994@tju.edu.cn](mailto:zyliang1994@tju.edu.cn)
