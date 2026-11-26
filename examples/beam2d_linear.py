from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

LENGTH, HEIGHT, TRACTION_Y = 4.0, 1.0, -5.0


class CPDEM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('E_min', torch.tensor(100.0))
        self.register_buffer('E_max', torch.tensor(500.0))
        self.register_buffer('nu_min', torch.tensor(0.25))
        self.register_buffer('nu_max', torch.tensor(0.35))
        self.pe1, self.pe2, self.pe3 = (nn.Linear(8, 100), nn.Linear(100, 100), nn.Linear(100, 50))
        self.ce1, self.ce2, self.ce3 = (nn.Linear(2, 50), nn.Linear(50, 50), nn.Linear(50, 50))
        self.d1, self.d2, self.d3, self.d4 = (nn.Linear(100, 100), nn.Linear(100, 100), nn.Linear(100, 50), nn.Linear(50, 50))
        self.out_ux, self.out_uy, self.hyper = (nn.Linear(50, 1), nn.Linear(50, 1), nn.Linear(10, 50))
        self.act = nn.Tanh()
        for layer in (self.pe1, self.pe2, self.pe3, self.ce1, self.ce2, self.ce3, self.d1, self.d2, self.d3, self.d4):
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        for layer in (self.out_ux, self.out_uy):
            nn.init.xavier_normal_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.hyper.weight, gain=0.01)
        nn.init.zeros_(self.hyper.bias)

    def physics_features(self, E: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        E_n = (E - self.E_min) / (self.E_max - self.E_min + 1e-08)
        nu_n = (nu - self.nu_min) / (self.nu_max - self.nu_min + 1e-08)
        mu = E / (2.0 * (1.0 + nu))
        mu_min = self.E_min / (2.0 * (1.0 + self.nu_max))
        mu_max = self.E_max / (2.0 * (1.0 + self.nu_min))
        mu_n = (mu - mu_min) / (mu_max - mu_min + 1e-08)
        c11 = E / (1.0 - nu.square() + 1e-08)
        c11_min = self.E_min / (1.0 - self.nu_max.square() + 1e-08)
        c11_max = self.E_max / (1.0 - self.nu_min.square() + 1e-08)
        c11_n = (c11 - c11_min) / (c11_max - c11_min + 1e-08)
        c12 = E * nu / (1.0 - nu.square() + 1e-08)
        c12_min = self.E_min * self.nu_min / (1.0 - self.nu_min.square() + 1e-08)
        c12_max = self.E_max * self.nu_max / (1.0 - self.nu_max.square() + 1e-08)
        c12_n = (c12 - c12_min) / (c12_max - c12_min + 1e-08)
        return torch.cat((E_n, nu_n, mu_n, c11_n, c12_n, E_n.square(), nu_n.square(), E_n * nu_n), dim=1)

    def forward(self, params: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        E, nu = (params[:, 0:1], params[:, 1:2])
        features = self.physics_features(E, nu)
        p = self.act(self.pe1(features))
        p = self.act(self.pe2(p))
        p = self.act(self.pe3(p))
        c = self.act(self.ce1(X))
        c = self.act(self.ce2(c))
        c = self.act(self.ce3(c))
        d = self.act(self.d1(torch.cat((p, c), dim=1)))
        d = self.act(self.d2(d))
        d = self.act(self.d3(d))
        d = self.act(self.d4(d + self.hyper(torch.cat((features, X), dim=1))))
        return torch.cat((self.out_ux(d), self.out_uy(d)), dim=1) * (300.0 / (E + 1e-08))

def gauss_interval(n: int, low: float, high: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mapped = 0.5 * (high - low) * nodes + 0.5 * (high + low)
    scaled = 0.5 * (high - low) * weights
    return (torch.as_tensor(mapped[:, None], dtype=torch.float32, device=device), torch.as_tensor(scaled[:, None], dtype=torch.float32, device=device))

def quadrature(nqx: int, nqy: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, wx = gauss_interval(nqx, 0.0, LENGTH, device)
    y, wy = gauss_interval(nqy, 0.0, HEIGHT, device)
    X = torch.stack(torch.meshgrid(x[:, 0], y[:, 0], indexing='ij'), dim=-1).reshape(-1, 2)
    w = (wx @ wy.T).reshape(-1, 1)
    right = torch.cat((torch.full_like(y, LENGTH), y), dim=1)
    return (X, w, right, wy)

def repeat_cases(parameters: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (parameters.repeat_interleave(X.shape[0], dim=0), X.repeat(parameters.shape[0], 1).detach().clone().requires_grad_(True))

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

def inference(model: nn.Module, params: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return X[:, 0:1] / LENGTH * model(params, X)

def main():
    parser = argparse.ArgumentParser(description="CPDEM: beam2d linear")
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--nqx', type=int, default=24)
    parser.add_argument('--nqy', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('outputs/beam2d_linear.pt'))
    args = parser.parse_args()
    if args.steps < 1 or args.lbfgs_steps < 0 or args.nqx < 2 or args.nqy < 2 or args.threads < 1:
        parser.error('Steps and thread count must be positive; quadrature sizes must be at least two.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = CPDEM().to(device)
    materials = torch.tensor([(E, nu) for E in (100.0, 300.0, 500.0)
        for nu in (0.25, 0.30, 0.35)], dtype=torch.float32, device=device)
    Xq, wq, Xright, wright = quadrature(args.nqx, args.nqy, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = potential(model, materials, Xq, wq, Xright, wright)
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
        'case': 'beam2d_linear', 'config': {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}}, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
