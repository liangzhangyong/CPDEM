from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

CASES = ('bar1d_linear', 'beam2d_linear', 'beam2d_neohookean')
LENGTH, HEIGHT = 4.0, 1.0


class CPDEM(nn.Module):
    def __init__(self, case='beam2d_linear', width=32):
        super().__init__()
        if case not in CASES:
            raise ValueError(f'Unknown case: {case}')
        self.case, self.width = case, width
        if case == 'bar1d_linear':
            self.parameter_encoder = nn.Sequential(
                nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
            self.coordinate_encoder = nn.Sequential(
                nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
            self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            return
        if case == 'beam2d_linear':
            for name, value in [('E_min', 100.), ('E_max', 500.), ('nu_min', .25), ('nu_max', .35)]:
                self.register_buffer(name, torch.tensor(value))
        self.pe1, self.pe2, self.pe3 = nn.Linear(8, 100), nn.Linear(100, 100), nn.Linear(100, 50)
        self.ce1, self.ce2, self.ce3 = nn.Linear(2, 50), nn.Linear(50, 50), nn.Linear(50, 50)
        self.d1, self.d2 = nn.Linear(100, 100), nn.Linear(100, 100)
        self.d3, self.d4 = nn.Linear(100, 50), nn.Linear(50, 50)
        self.out_ux, self.out_uy, self.hyper = nn.Linear(50, 1), nn.Linear(50, 1), nn.Linear(10, 50)
        self.act = nn.Tanh()
        if case == 'beam2d_linear':
            layers = (self.pe1, self.pe2, self.pe3, self.ce1, self.ce2, self.ce3,
                      self.d1, self.d2, self.d3, self.d4)
            for layer in layers:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in (self.out_ux, self.out_uy):
                nn.init.xavier_normal_(layer.weight, gain=.1)
                nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.hyper.bias)
        else:
            layers = [layer for layer in self.modules() if isinstance(layer, nn.Linear)]
            for layer in layers:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.xavier_normal_(layers[-1].weight, gain=.1)
        nn.init.xavier_normal_(self.hyper.weight, gain=.01)

    def features(self, parameters):
        E, nu = parameters[:, :1], parameters[:, 1:2]
        if self.case == 'beam2d_linear':
            En = (E - self.E_min) / (self.E_max - self.E_min + 1e-8)
            nun = (nu - self.nu_min) / (self.nu_max - self.nu_min + 1e-8)
            mu = E / (2 * (1 + nu))
            mu0 = self.E_min / (2 * (1 + self.nu_max))
            mu1 = self.E_max / (2 * (1 + self.nu_min))
            mun = (mu - mu0) / (mu1 - mu0 + 1e-8)
            c11 = E / (1 - nu.square() + 1e-8)
            c110 = self.E_min / (1 - self.nu_max.square() + 1e-8)
            c111 = self.E_max / (1 - self.nu_min.square() + 1e-8)
            c11n = (c11 - c110) / (c111 - c110 + 1e-8)
            c12 = E * nu / (1 - nu.square() + 1e-8)
            c120 = self.E_min * self.nu_min / (1 - self.nu_min.square() + 1e-8)
            c121 = self.E_max * self.nu_max / (1 - self.nu_max.square() + 1e-8)
            c12n = (c12 - c120) / (c121 - c120 + 1e-8)
            return torch.cat((En, nun, mun, c11n, c12n, En.square(), nun.square(), En * nun), 1)
        En, nun = (E - 800.) / 400., (nu - .25) / .10
        mu = E / (2 * (1 + nu))
        mun = (mu - 800. / (2 * 1.35)) / (1200. / (2 * 1.25) - 800. / (2 * 1.35))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        lam0, lam1 = 800. * .25 / (1.25 * .5), 1200. * .35 / (1.35 * .3)
        ln = (lam - lam0) / (lam1 - lam0)
        bulk = lam + 2 * mu / 3
        b0, b1 = lam0 + 2 * 800. / (2 * 1.35) / 3, lam1 + 2 * 1200. / (2 * 1.25) / 3
        bn = (bulk - b0) / (b1 - b0)
        return torch.cat((En, nun, mun, ln, bn, En.square(), nun.square(), En * nun), 1)

    def forward(self, parameters, coordinates):
        if self.case == 'bar1d_linear':
            p = self.parameter_encoder((parameters - .5) / 2.5)
            x = self.coordinate_encoder(coordinates)
            return (coordinates + 1) * self.decoder(torch.cat((p, x), 1))
        features = self.features(parameters)
        p = self.act(self.pe1(features))
        p = self.act(self.pe2(p))
        p = self.act(self.pe3(p))
        x = coordinates if self.case == 'beam2d_linear' else coordinates / coordinates.new_tensor([LENGTH, HEIGHT])
        c = self.act(self.ce1(x))
        c = self.act(self.ce2(c))
        c = self.act(self.ce3(c))
        d = self.act(self.d1(torch.cat((p, c), 1)))
        d = self.act(self.d2(d))
        d = self.act(self.d3(d))
        d = self.act(self.d4(d + self.hyper(torch.cat((features, x), 1))))
        raw = torch.cat((self.out_ux(d), self.out_uy(d)), 1)
        scale = 300. / (parameters[:, :1] + 1e-8) if self.case == 'beam2d_linear' else 1000. / parameters[:, :1]
        return raw * scale


def displacement(model, parameters, coordinates):
    raw = model(parameters, coordinates)
    return raw if model.case == 'bar1d_linear' else coordinates[:, :1] / LENGTH * raw


def gauss_interval(n, low, high, device='cpu'):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = .5 * (high - low) * nodes + .5 * (high + low)
    weights = .5 * (high - low) * weights
    return (torch.as_tensor(nodes[:, None], dtype=torch.float32, device=device),
            torch.as_tensor(weights[:, None], dtype=torch.float32, device=device))


def beam_quadrature(nx, ny, device='cpu'):
    x, wx = gauss_interval(nx, 0, LENGTH, device)
    y, wy = gauss_interval(ny, 0, HEIGHT, device)
    coordinates = torch.stack(torch.meshgrid(x[:, 0], y[:, 0], indexing='ij'), -1).reshape(-1, 2)
    right = torch.cat((torch.full_like(y, LENGTH), y), 1)
    return coordinates, (wx @ wy.T).reshape(-1, 1), right, wy


def repeat_cases(parameters, coordinates):
    return (parameters.repeat_interleave(len(coordinates), 0),
            coordinates.repeat(len(parameters), 1).detach().clone().requires_grad_(True))


def parse_training_arguments(case, steps, nqx):
    parser = argparse.ArgumentParser(description=f'CPDEM: {case}')
    parser.add_argument('--steps', type=int, default=steps)
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--nqx', type=int, default=nqx)
    parser.add_argument('--nqy', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path(f'outputs/{case}.pt'))
    args = parser.parse_args()
    if args.steps < 1 or args.lbfgs_steps < 0 or args.lr <= 0 or args.nqx < 2 or args.nqy < 2 or args.threads < 1:
        parser.error('Use positive steps, learning rate, and thread count; quadrature sizes must be at least two.')
    return args


def setup_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    return torch.device(args.device)


def save_checkpoint(model, materials, args):
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['width'] = model.width
    torch.save({'state_dict': model.state_dict(), 'materials': materials.detach().cpu(),
                'case': model.case, 'config': config}, path)


def train_and_save(model, materials, objective, args, gradient_clip=1000.):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite potential energy.')
        loss.backward()
        if not isinstance(optimizer, torch.optim.LBFGS):
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        return loss

    for step in range(1, args.steps + 1):
        loss = closure()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f'step={step} potential={loss.detach().item():.8e}', flush=True)
    if args.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=args.lbfgs_steps,
                                      history_size=20, line_search_fn='strong_wolfe')
        optimizer.step(closure)
    save_checkpoint(model, materials, args)
    print(f'Saved {args.output}')


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    width = checkpoint.get('config', {}).get('width', 32)
    model = CPDEM(checkpoint['case'], width=width).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def predict(model, parameters, coordinates):
    prototype = next(model.parameters())
    parameters = torch.as_tensor(parameters, dtype=prototype.dtype, device=prototype.device)
    coordinates = torch.as_tensor(coordinates, dtype=prototype.dtype, device=prototype.device)
    dimensions = 1 if model.case == 'bar1d_linear' else 2
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    if coordinates.ndim == 1 and dimensions == 1:
        coordinates = coordinates[:, None]
    if coordinates.ndim != 2 or coordinates.shape[1] != dimensions:
        raise ValueError(f'Coordinates must have shape (N, {dimensions}).')
    if parameters.ndim != 2 or parameters.shape[1] != dimensions:
        raise ValueError(f'Material parameters must have shape (N, {dimensions}).')
    if parameters.shape[0] == 1:
        parameters = parameters.expand(coordinates.shape[0], -1)
    if parameters.shape[0] != coordinates.shape[0]:
        raise ValueError('Provide one material or one material per coordinate.')
    if not torch.isfinite(parameters).all() or not torch.isfinite(coordinates).all():
        raise ValueError('Coordinates and material parameters must be finite.')
    if (parameters[:, 0] <= 0).any():
        raise ValueError('Young modulus must be positive.')
    if dimensions == 2 and ((parameters[:, 1] <= -1) | (parameters[:, 1] >= .5)).any():
        raise ValueError('Poisson ratio must lie between -1 and 0.5.')
    with torch.inference_mode():
        return displacement(model, parameters, coordinates)
