from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

LENGTH, HEIGHT, TRACTION_Y = 4.0, 1.0, -2.0
E_MIN, E_MAX, NU_MIN, NU_MAX = 800.0, 1200.0, 0.25, 0.35


def init(module: nn.Module, gain: float=0.1) -> None:
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for layer in layers:
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    nn.init.xavier_normal_(layers[-1].weight, gain=gain)

class CPDEM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.pe1, self.pe2, self.pe3 = (nn.Linear(8, 100), nn.Linear(100, 100), nn.Linear(100, 50))
        self.ce1, self.ce2, self.ce3 = (nn.Linear(2, 50), nn.Linear(50, 50), nn.Linear(50, 50))
        self.d1, self.d2, self.d3, self.d4 = (nn.Linear(100, 100), nn.Linear(100, 100), nn.Linear(100, 50), nn.Linear(50, 50))
        self.out_ux, self.out_uy, self.hyper = (nn.Linear(50, 1), nn.Linear(50, 1), nn.Linear(10, 50))
        self.act = nn.Tanh()
        init(self)
        nn.init.xavier_normal_(self.hyper.weight, gain=0.01)

    def features(self, p: torch.Tensor) -> torch.Tensor:
        E, nu = (p[:, :1], p[:, 1:])
        En, nn_ = ((E - E_MIN) / (E_MAX - E_MIN), (nu - NU_MIN) / (NU_MAX - NU_MIN))
        mu = E / (2 * (1 + nu))
        mun = (mu - E_MIN / (2 * (1 + NU_MAX))) / (E_MAX / (2 * (1 + NU_MIN)) - E_MIN / (2 * (1 + NU_MAX)))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        lam0 = E_MIN * NU_MIN / ((1 + NU_MIN) * (1 - 2 * NU_MIN))
        lam1 = E_MAX * NU_MAX / ((1 + NU_MAX) * (1 - 2 * NU_MAX))
        ln = (lam - lam0) / (lam1 - lam0)
        bulk = lam + 2 * mu / 3
        b0 = lam0 + 2 * E_MIN / (2 * (1 + NU_MAX)) / 3
        b1 = lam1 + 2 * E_MAX / (2 * (1 + NU_MIN)) / 3
        bn = (bulk - b0) / (b1 - b0)
        return torch.cat((En, nn_, mun, ln, bn, En.square(), nn_.square(), En * nn_), 1)

    def forward(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        f = self.features(p)
        pn = self.act(self.pe1(f))
        pn = self.act(self.pe2(pn))
        pn = self.act(self.pe3(pn))
        xn = torch.cat((x[:, :1] / LENGTH, x[:, 1:] / HEIGHT), 1)
        c = self.act(self.ce1(xn))
        c = self.act(self.ce2(c))
        c = self.act(self.ce3(c))
        d = self.act(self.d1(torch.cat((pn, c), 1)))
        d = self.act(self.d2(d))
        d = self.act(self.d3(d))
        d = self.act(self.d4(d + self.hyper(torch.cat((f, xn), 1))))
        return torch.cat((self.out_ux(d), self.out_uy(d)), 1) * (1000.0 / p[:, :1])

def gauss(n: int, low: float, high: float, device: torch.device):
    q, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor((q + 1) * (high - low) / 2 + low, device=device, dtype=torch.float32), torch.tensor(w * (high - low) / 2, device=device, dtype=torch.float32))

def quadrature(nx: int, ny: int, device: torch.device):
    x, wx = gauss(nx, 0, LENGTH, device)
    y, wy = gauss(ny, 0, HEIGHT, device)
    X = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
    w = (wx[:, None] * wy[None, :]).reshape(-1, 1)
    right = torch.stack((torch.full_like(y, LENGTH), y), 1)
    return (X, w, right, wy[:, None])

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

def inference(model, parameters, coordinates):
    return coordinates[:, :1] / LENGTH * model(parameters, coordinates)


def main():
    parser = argparse.ArgumentParser(description="CPDEM: beam2d neohookean")
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--nqx', type=int, default=24)
    parser.add_argument('--nqy', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('outputs/beam2d_neohookean.pt'))
    args = parser.parse_args()
    if args.steps < 1 or args.lbfgs_steps < 0 or args.nqx < 2 or args.nqy < 2 or args.threads < 1:
        parser.error('Steps and thread count must be positive; quadrature sizes must be at least two.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = CPDEM().to(device)
    materials = torch.tensor([(E, nu) for E in (800.0, 1000.0, 1200.0)
        for nu in (0.25, 0.30, 0.35)], dtype=torch.float32, device=device)
    Xq, wq, Xright, wright = quadrature(args.nqx, args.nqy, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = potential(model, materials, Xq, wq, Xright, wright)[0]
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite potential energy.')
        loss.backward()
        if not isinstance(optimizer, torch.optim.LBFGS):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        return loss

    for step in range(1, args.steps + 1):
        loss = closure()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f'step={step} potential={loss.detach().item():.8e}', flush=True)
    if args.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5,
            max_iter=args.lbfgs_steps, history_size=20, line_search_fn='strong_wolfe')
        optimizer.step(closure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'materials': materials.detach().cpu(),
        'case': 'beam2d_neohookean', 'config': {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}}, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
