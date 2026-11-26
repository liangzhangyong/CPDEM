from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

E_MIN, E_MAX = 0.5, 3.0


class CPDEM(nn.Module):

    def __init__(self, width: int) -> None:
        super().__init__()
        self.parameter_encoder = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.coordinate_encoder = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, E: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        E_unit = (E - E_MIN) / (E_MAX - E_MIN)
        z_p = self.parameter_encoder(E_unit)
        z_x = self.coordinate_encoder(x)
        return (x + 1.0) * self.decoder(torch.cat((z_p, z_x), dim=1))

def gauss_rule(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(w[:, None], dtype=torch.float32))

def potential_energy(model: nn.Module, E_values: torch.Tensor, xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    n_E, n_q = (E_values.shape[0], xq.shape[0])
    E = E_values.repeat_interleave(n_q, dim=0)
    x = xq.repeat(n_E, 1).clone().detach().requires_grad_(True)
    u = model(E, x)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    density = 0.5 * E * du_dx.square() - x * u
    return (density.reshape(n_E, n_q) * wq.reshape(1, n_q)).sum(dim=1).mean()

def main():
    parser = argparse.ArgumentParser(description="CPDEM: bar1d linear")
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--nqx', type=int, default=64)
    parser.add_argument('--nqy', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('outputs/bar1d_linear.pt'))
    args = parser.parse_args()
    if args.steps < 1 or args.lbfgs_steps < 0 or args.nqx < 2 or args.nqy < 2 or args.threads < 1:
        parser.error('Steps and thread count must be positive; quadrature sizes must be at least two.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = CPDEM(width=32).to(device)
    xq, wq = gauss_rule(args.nqx)
    xq, wq = xq.to(device), wq.to(device)
    samples = torch.quasirandom.SobolEngine(1, scramble=True, seed=args.seed + 101).draw(32)
    materials = (E_MIN + (E_MAX - E_MIN) * samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = potential_energy(model, materials, xq, wq)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite potential energy.')
        loss.backward()
        if not isinstance(optimizer, torch.optim.LBFGS):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
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
        'case': 'bar1d_linear', 'config': {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}}, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
