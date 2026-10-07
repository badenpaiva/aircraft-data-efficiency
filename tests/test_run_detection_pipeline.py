import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_detection_pipeline as runner


class RunnerTests(unittest.TestCase):
    def configuration(self, root):
        data = root / 'data.yaml'; data.write_text('names: [dent, crack, paint_peeling]\n')
        config = root / 'experiment.yaml'
        config.write_text(yaml.safe_dump({'task': {'type': 'detection'}, 'training': {
            'name': 'baseline', 'data': str(data), 'project': str(root / 'outputs')}}))
        return config

    def test_sequence_uses_new_weights_and_same_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.configuration(root); calls = []
            def run(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    weights = root / 'outputs/trial/weights/best.pt'
                    weights.parent.mkdir(parents=True); weights.write_bytes(b'test')
            with patch.object(runner.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(['--config', str(config), '--name', 'trial', '--smoke'])
            self.assertEqual(result, 0)
            self.assertEqual([Path(c[2]).name for c in calls],
                             ['train_detector.py','evaluate_detector.py','predict_samples.py'])
            self.assertTrue(all(c[c.index('--config')+1] == str(config) for c in calls))
            self.assertEqual(calls[1][calls[1].index('--weights')+1], str(root/'outputs/trial/weights/best.pt'))
            self.assertIn('--smoke', calls[0]); self.assertIn('--smoke', calls[1])
            self.assertEqual(calls[1][calls[1].index('--split')+1], 'val')

    def test_training_failure_stops_remaining_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.configuration(Path(tmp))
            with patch.object(runner.subprocess, 'run', side_effect=subprocess.CalledProcessError(7, 'train')) as run:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = runner.main(['--config', str(config), '--name', 'trial'])
            self.assertEqual(result, 7); self.assertEqual(run.call_count, 1)

    def test_existing_evaluation_directory_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.configuration(root)
            (root/'outputs/trial_evaluation').mkdir(parents=True)
            with patch.object(runner.subprocess, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main(['--config', str(config), '--name', 'trial'])
            run.assert_not_called()


if __name__ == '__main__': unittest.main()
