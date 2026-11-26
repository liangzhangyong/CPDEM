from __future__ import annotations

import torch
from torch import nn

from CPDEM import CPDEM, gauss_interval, parse_training_arguments, setup_training, train_and_save

E_MIN, E_MAX = .5, 3.


def potential_energy(model: nn.Module, E_values: torch.Tensor, xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    n_E, n_q = (E_values.shape[0], xq.shape[0])
    E = E_values.repeat_interleave(n_q, dim=0)
    x = xq.repeat(n_E, 1).clone().detach().requires_grad_(True)
    u = model(E, x)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    density = 0.5 * E * du_dx.square() - x * u
    return (density.reshape(n_E, n_q) * wq.reshape(1, n_q)).sum(dim=1).mean()


def main():
    args = parse_training_arguments('bar1d_linear', steps=2500, nqx=64)
    device = setup_training(args)
    model = CPDEM('bar1d_linear').to(device)
    xq, wq = gauss_interval(args.nqx, -1, 1, device)
    samples = torch.quasirandom.SobolEngine(1, scramble=True, seed=args.seed + 101).draw(32)
    materials = (E_MIN + (E_MAX - E_MIN) * samples).to(device)
    train_and_save(model, materials, lambda: potential_energy(model, materials, xq, wq), args, gradient_clip=10.)


if __name__ == '__main__':
    main()
