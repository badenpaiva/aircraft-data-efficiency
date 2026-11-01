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
            for network in ('generator', 'discriminator'):
                for key in expected[network]:
                    torch.testing.assert_close(actual[network][key], expected[network][key], rtol=0, atol=0)
            self.assertEqual((full/'losses.jsonl').read_text(), (resumed/'losses.jsonl').read_text())
            self.assertEqual(json.loads((resumed/'completion.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
