"""An interrupted run must reproduce uninterrupted CPU optimizer updates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from augmentation.hqgan.training import train


class FixtureDataset:
    identity = {'fixture': 'resume-test'}

    def __init__(self, interrupt=False):
        self.calls = 0
        self.interrupt = interrupt

    def __len__(self):
        return 4

    def __getitem__(self, index):
        self.calls += 1
        if self.interrupt and self.calls > 2:
            raise RuntimeError('simulated interruption')
        return torch.full((3, 64, 64), index / 4)


class ResumeTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_parameters(self):
        settings = yaml.safe_load((ROOT / 'config/hqgan_sutd.yaml').read_text())
        settings.update(resolution=64, width=1/32, batch_size=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory:
            full, resumed = Path(directory)/'full', Path(directory)/'resumed'
            train(FixtureDataset(), settings, full, 2, 'cpu', smoke=True)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                train(FixtureDataset(True), settings, resumed, 2, 'cpu', smoke=True)
            train(FixtureDataset(), settings, resumed, 2, 'cpu', smoke=True, resume=True)
            expected = torch.load(full/'last.pt', weights_only=True)
            actual = torch.load(resumed/'last.pt', weights_only=True)
            for network in ('generator', 'discriminator', 'generator_ema'):
                for key in expected[network]:
                    torch.testing.assert_close(actual[network][key], expected[network][key], rtol=0, atol=0)
            self.assertEqual((full/'losses.jsonl').read_text(), (resumed/'losses.jsonl').read_text())
            self.assertEqual(json.loads((resumed/'completion.json').read_text())['status'], 'complete')
            # The sampled weights are an average, not the last step, and carry measured statistics.
            self.assertFalse(all(torch.equal(expected['generator'][k], expected['generator_ema'][k])
                                 for k in expected['generator'] if k.endswith('weight')))
            self.assertTrue(any(k.endswith('running_var') and not torch.all(v == 1)
                                for k, v in expected['generator_ema'].items()))
            for run in (full, resumed):
                self.assertEqual(sorted(p.name for p in (run/'sheets').glob('*.png')),
                                 ['step_000001.png', 'step_000002.png'])
                self.assertEqual(len(list(run.glob('sample_*.png'))), 4)
            self.assertEqual((full/'sheets/step_000002.png').read_bytes(), (resumed/'sheets/step_000002.png').read_bytes())
            self.assertEqual((full/'sample_0000.png').read_bytes(), (resumed/'sample_0000.png').read_bytes())

    def test_discriminator_and_generator_steps_both_use_train_mode_fakes(self):
        from unittest.mock import patch
        from augmentation.hqgan import training
        calls = []

        class Recording(training.Generator):
            def forward(self, z):
                calls.append((self.training, len(z)))
                return super().forward(z)

        settings = yaml.safe_load((ROOT / 'config/hqgan_sutd.yaml').read_text())
        settings.update(resolution=64, width=1/32, batch_size=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(training, 'Generator', Recording):
            train(FixtureDataset(), settings, Path(directory)/'run', 1, 'cpu', smoke=True)
        # First the discriminator's fakes, then the generator's own: the same batch-statistics mode.
        self.assertEqual(calls[:2], [(True, 2), (True, 2)])
        # Every eval-mode call is a single sample taken after statistics were measured for it.
        self.assertTrue(all(size == 1 for training_mode, size in calls if not training_mode))
        self.assertIn((True, 16), calls)

    def test_sampling_uses_the_averaged_weights(self):
        from augmentation.hqgan.training import sampling_generator, standing_statistics
        from augmentation.hqgan.model import Generator
        architecture = dict(resolution=64, latent_dim=16, width=1/32, hsc_alpha=1.0, hsc_magnitude=True)
        torch.manual_seed(0)
        live, averaged = Generator(**architecture), Generator(**architecture)
        ready = standing_statistics(averaged, 5, 'cpu', batches=4, batch_size=8)
        self.assertFalse(ready.training)
        self.assertTrue(all(m.momentum == 0.1 for m in ready.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        checkpoint = {'architecture': architecture, 'generator': live.state_dict(), 'generator_ema': ready.state_dict()}
        sampler = sampling_generator(checkpoint, 'cpu')
        z = torch.randn(2, 16)
        with torch.no_grad():
            torch.testing.assert_close(sampler(z), ready(z), rtol=0, atol=0)
            self.assertFalse(torch.allclose(sampler(z), live.eval()(z)))
            # Eval-mode sampling: one image does not depend on the others in its batch.
            torch.testing.assert_close(sampler(z)[:1], sampler(z[:1]), atol=1e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
