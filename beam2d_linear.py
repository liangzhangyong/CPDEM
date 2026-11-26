from __future__ import annotations

import torch
from torch import nn

from CPDEM import CPDEM, beam_quadrature as quadrature, displacement as inference
from CPDEM import LENGTH, HEIGHT, parse_training_arguments, repeat_cases, setup_training, train_and_save

TRACTION_Y = -5.


def field_and_stress(model: nn.Module, params: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    raw = model(params, X)
    u = X[:, 0:1] / LENGTH * raw
    dux = torch.autograd.grad(u[:, 0:1], X, torch.ones_like(u[:, 0:1]), create_graph=True, retain_graph=True)[0]
    duy = torch.autograd.grad(u[:, 1:2], X, torch.ones_like(u[:, 1:2]), create_graph=True, retain_graph=True)[0]
    exx, eyy, gamma = (dux[:, 0:1], duy[:, 1:2], dux[:, 1:2] + duy[:, 0:1])
    E, nu = (params[:, 0:1], params[:, 1:2])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    sxx = (lam + 2.0 * mu) * exx + lam * eyy
    syy = lam * exx + (lam + 2.0 * mu) * eyy
    sxy = mu * gamma
    return (u, (exx, eyy, gamma, sxx, syy, sxy, mu, lam))

def potential(model: nn.Module, parameters: torch.Tensor, Xq: torch.Tensor, wq: torch.Tensor, Xright: torch.Tensor, wright: torch.Tensor) -> torch.Tensor:
    params, X = repeat_cases(parameters, Xq)
    _, (exx, eyy, gamma, sxx, syy, sxy, _, _) = field_and_stress(model, params, X)
    density = 0.5 * (sxx * exx + syy * eyy + sxy * gamma)
    ncase, nq = (parameters.shape[0], Xq.shape[0])
    internal = (density.reshape(ncase, nq) * wq.reshape(1, nq)).sum(1)
    p_right, Xr = repeat_cases(parameters, Xright)
    ur = Xr[:, 0:1] / LENGTH * model(p_right, Xr)
    external = (TRACTION_Y * ur[:, 1:2].reshape(ncase, Xright.shape[0]) * wright.reshape(1, -1)).sum(1)
    return (internal - external).mean()


def main():
    args = parse_training_arguments('beam2d_linear', steps=1200, nqx=24)
    device = setup_training(args)
    model = CPDEM('beam2d_linear').to(device)
    materials = torch.tensor([(E, nu) for E in (100., 300., 500.)
        for nu in (.25, .30, .35)], dtype=torch.float32, device=device)
    grid = quadrature(args.nqx, args.nqy, device)
    train_and_save(model, materials, lambda: potential(model, materials, *grid), args)


if __name__ == '__main__':
    main()
