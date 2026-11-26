from __future__ import annotations

import torch
from torch import nn

from CPDEM import CPDEM, beam_quadrature as quadrature, displacement as inference
from CPDEM import LENGTH, HEIGHT, parse_training_arguments, setup_training, train_and_save

TRACTION_Y = -2.


def field(model: nn.Module, p: torch.Tensor, X: torch.Tensor, create_graph: bool=True):
    raw = model(p, X)
    u = X[:, :1] / LENGTH * raw
    gu = []
    for i in range(2):
        gu.append(torch.autograd.grad(u[:, i].sum(), X, create_graph=create_graph, retain_graph=True)[0])
    F = torch.stack((torch.stack((1 + gu[0][:, 0], gu[0][:, 1]), 1), torch.stack((gu[1][:, 0], 1 + gu[1][:, 1]), 1)), 1)
    J = torch.linalg.det(F)
    E, nu = (p[:, 0], p[:, 1])
    mu, lam = (E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)))
    logJ = torch.log(torch.clamp(J, min=1e-06))
    I1 = F.square().sum((1, 2))
    psi = 0.5 * mu * (I1 - 2) - mu * logJ + 0.5 * lam * logJ.square()
    FinvT = torch.linalg.inv(F).transpose(1, 2)
    P = mu[:, None, None] * F + (lam * logJ - mu)[:, None, None] * FinvT
    return (u, P, J, psi)

def potential(model, materials, Xq, wq, Xr, wr):
    nmat, nq = (materials.shape[0], Xq.shape[0])
    p = materials.repeat_interleave(nq, 0)
    X = Xq.repeat(nmat, 1).detach().clone().requires_grad_(True)
    _, _, J, psi = field(model, p, X)
    internal = (psi.reshape(nmat, nq) * wq.reshape(1, nq)).sum(1)
    pr = materials.repeat_interleave(Xr.shape[0], 0)
    xr = Xr.repeat(nmat, 1)
    ur = xr[:, :1] / LENGTH * model(pr, xr)
    external = (TRACTION_Y * ur[:, 1].reshape(nmat, -1) * wr.reshape(1, -1)).sum(1)
    barrier = 10000.0 * torch.relu(0.2 - J).square().mean()
    return ((internal - external).mean() + barrier, J.min())


def main():
    args = parse_training_arguments('beam2d_neohookean', steps=1500, nqx=24)
    device = setup_training(args)
    model = CPDEM('beam2d_neohookean').to(device)
    materials = torch.tensor([(E, nu) for E in (800., 1000., 1200.)
        for nu in (.25, .30, .35)], dtype=torch.float32, device=device)
    grid = quadrature(args.nqx, args.nqy, device)
    train_and_save(model, materials, lambda: potential(model, materials, *grid)[0], args)


if __name__ == '__main__':
    main()
