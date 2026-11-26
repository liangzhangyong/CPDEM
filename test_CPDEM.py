from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from CPDEM import CASES, CPDEM, beam_quadrature, gauss_interval
from CPDEM import load_checkpoint, predict, save_checkpoint

MATERIALS = {'bar1d_linear': [1.75], 'beam2d_linear': [250., .28],
             'beam2d_neohookean': [950., .28]}


class ExactBar(nn.Module):
    def forward(self, E, x):
        return (.5 * x - x.pow(3) / 6 + 1 / 3) / E


class ZeroBeam(nn.Module):
    def forward(self, parameters, coordinates):
        return coordinates * 0


class TestCPDEM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        torch.set_num_threads(1)

    def test_displacements_and_clamps(self):
        for case in CASES:
            with self.subTest(case=case):
                model = CPDEM(case)
                x = [[-1.], [0.], [1.]] if case == 'bar1d_linear' else [[0., .2], [0., .8], [4., .5]]
                u = predict(model, MATERIALS[case], x)
                self.assertEqual(tuple(u.shape), (3, 1 if case == 'bar1d_linear' else 2))
                self.assertTrue(torch.isfinite(u).all())
                boundary = u[:1] if case == 'bar1d_linear' else u[:2]
                torch.testing.assert_close(boundary, torch.zeros_like(boundary), atol=0, rtol=0)

    def test_energy_gradients(self):
        for case in CASES:
            with self.subTest(case=case):
                module = importlib.import_module(case)
                model = CPDEM(case)
                materials = torch.tensor([MATERIALS[case]])
                if case == 'bar1d_linear':
                    loss = module.potential_energy(model, materials, *gauss_interval(8, -1, 1))
                else:
                    loss = module.potential(model, materials, *beam_quadrature(4, 2))
                    if case == 'beam2d_neohookean':
                        loss = loss[0]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_bar_analytical_energy(self):
        from bar1d_linear import potential_energy
        materials = torch.tensor([[.75], [1.75], [2.5]])
        actual = potential_energy(ExactBar(), materials, *gauss_interval(32, -1, 1))
        expected = (-2 / (15 * materials)).mean()
        torch.testing.assert_close(actual, expected)

    def test_linear_undeformed_state(self):
        from beam2d_linear import field_and_stress
        coordinates = torch.tensor([[1., .3], [3., .7]], requires_grad=True)
        parameters = torch.tensor([[200., .28], [400., .32]])
        u, values = field_and_stress(ZeroBeam(), parameters, coordinates)
        for value in (u, *values[:6]):
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)

    def test_neohookean_identity(self):
        from beam2d_neohookean import field
        coordinates = torch.tensor([[1., .3], [3., .7]], requires_grad=True)
        parameters = torch.tensor([[900., .28], [1100., .32]])
        u, stress, determinant, energy = field(ZeroBeam(), parameters, coordinates)
        for value in (u, stress, energy):
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        torch.testing.assert_close(determinant, torch.ones_like(determinant), atol=0, rtol=0)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            for case in CASES:
                with self.subTest(case=case):
                    model = CPDEM(case, width=16 if case == 'bar1d_linear' else 32)
                    path = Path(temporary) / (case + '.pt')
                    materials = torch.tensor([MATERIALS[case]])
                    save_checkpoint(model, materials, argparse.Namespace(output=path))
                    restored, checkpoint = load_checkpoint(path)
                    coordinates = [[0.]] if case == 'bar1d_linear' else [[2., .5]]
                    expected = predict(model, materials, coordinates)
                    actual = predict(restored, materials, coordinates)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertEqual(checkpoint['case'], case)

    def test_material_broadcast(self):
        model = CPDEM('beam2d_linear')
        coordinates = torch.tensor([[1., .2], [2., .5], [3., .8]])
        material = torch.tensor([[250., .28]])
        torch.testing.assert_close(predict(model, material, coordinates),
                                   predict(model, material.repeat(3, 1), coordinates))

    def test_invalid_inputs(self):
        model = CPDEM('beam2d_linear')
        for material in ([0., .3], [200., .5], [200., float('nan')]):
            with self.subTest(material=material), self.assertRaises(ValueError):
                predict(model, material, [[2., .5]])
        with self.assertRaises(ValueError):
            predict(model, [200., .3], [[1.]])


def evaluate_checkpoint(args):
    torch.set_num_threads(args.threads)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    if model.case == 'bar1d_linear':
        coordinates = torch.linspace(-1, 1, args.nx)[:, None]
    else:
        x, y = torch.linspace(0, 4, args.nx), torch.linspace(0, 1, args.ny)
        coordinates = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
    u = predict(model, args.material, coordinates).cpu()
    if not torch.isfinite(u).all():
        raise FloatingPointError('Non-finite displacement prediction.')
    print(f'case={model.case} material={args.material} points={len(coordinates)}')
    names = ['u'] if model.case == 'bar1d_linear' else ['ux', 'uy']
    for index, name in enumerate(names):
        print(f'{name}: min={u[:, index].min().item():.8e} max={u[:, index].max().item():.8e}')
    if model.case == 'bar1d_linear':
        reference = ExactBar()(torch.tensor(args.material), coordinates)
        error = torch.linalg.vector_norm(u - reference) / torch.linalg.vector_norm(reference)
        print(f'relative_L2={error.item():.8e}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow((['x'] if model.case == 'bar1d_linear' else ['x', 'y']) + names)
            writer.writerows(torch.cat((coordinates, u), 1).tolist())
        print(f'Saved {args.output}')


def main():
    parser = argparse.ArgumentParser(description='CPDEM checks and checkpoint evaluation')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--material', type=float, nargs='+')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--nx', type=int, default=101)
    parser.add_argument('--ny', type=int, default=21)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        if args.material is None or args.nx < 2 or args.ny < 2 or args.threads < 1:
            parser.error('Provide --material and positive thread count; grid sizes must be at least two.')
        evaluate_checkpoint(args)
    else:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestCPDEM))
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == '__main__':
    main()
