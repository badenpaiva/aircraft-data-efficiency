import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from training_progress import Progress, duration, attach_ultralytics_progress


class ProgressTests(unittest.TestCase):
    def test_resume_eta_uses_only_work_timed_in_current_session(self):
        now = [0.0]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            p = Progress('resumed', 100, initial=80, clock=lambda: now[0])
            now[0] = 20
            p.update(82)
        self.assertIn('ETA estimating...', output.getvalue())
        self.assertIn('ETA 00:03:00', output.getvalue())
        self.assertIn('82/100 steps (82.0%)', output.getvalue())

    def test_throttling_and_completion(self):
        now = [0.0]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            p = Progress('test', 4, clock=lambda: now[0])
            now[0] = 1
            p.update(1)
            p.update(2)
            now[0] = 2
            p.update(4)
        self.assertEqual(len(output.getvalue().splitlines()), 3)
        self.assertIn('ETA 00:00:00', output.getvalue())
        self.assertEqual(duration(90061), '1d 01:01:01')

    def test_ultralytics_callbacks_respect_resumed_epoch(self):
        callbacks = {}
        model = SimpleNamespace(add_callback=lambda name, fn: callbacks.update({name: fn}))
        trainer = SimpleNamespace(train_loader=[1, 2], start_epoch=2, epochs=3, epoch=2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            attach_ultralytics_progress(model)
            callbacks['on_train_start'](trainer)
            callbacks['on_train_batch_end'](trainer)
            callbacks['on_train_batch_end'](trainer)
            callbacks['on_fit_epoch_end'](trainer)
            trainer.epoch = 3
            callbacks['on_fit_epoch_end'](trainer)
        self.assertIn('4/6 batches', output.getvalue())
        self.assertIn('6/6 batches (100.0%)', output.getvalue())
        self.assertIn('epoch 3/3', output.getvalue())
        self.assertNotIn('epoch 4/3', output.getvalue())


if __name__ == '__main__':
    unittest.main()
