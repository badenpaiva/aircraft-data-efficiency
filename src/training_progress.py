"""Throttled, flushed terminal progress with resume-aware wall-time estimates."""
import time


def duration(seconds):
    if seconds is None:
        return 'estimating...'
    seconds = max(0, round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{days}d ' if days else '') + f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self, label, total, initial=0, unit='steps', interval=10, clock=None):
        self.label, self.total, self.initial = label, total, initial
        self.unit, self.interval = unit, interval
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.last_print = self.started
        self.update(initial, force=True)

    def update(self, completed, detail='', force=False):
        now = self.clock()
        if not force and completed != self.initial + 1 and completed < self.total and now-self.last_print < self.interval:
            return
        elapsed = max(0, now-self.started)
        done = completed-self.initial
        rate = done/elapsed if elapsed > 0 and done > 0 else None
        eta = max(0, self.total-completed)/rate if rate else None
        if completed >= self.total:
            eta = 0
        percent = 100*completed/self.total if self.total else 100
        speed = f'{rate:.2f} {self.unit}/s' if rate else 'speed estimating...'
        print(f'[{self.label}] {completed}/{self.total} {self.unit} ({percent:.1f}%) | '
              f'elapsed {duration(elapsed)} | ETA {duration(eta)} | {speed}'
              + (f' | {detail}' if detail else ''), flush=True)
        self.last_print = now


def attach_ultralytics_progress(model):
    """Keep native batch bars and add an estimate over all remaining epochs."""
    state = {}

    def start(trainer):
        batches = len(trainer.train_loader)
        initial = trainer.start_epoch*batches
        state.update(batches=batches, completed=initial,
                     progress=Progress('Detector training', trainer.epochs*batches,
                                       initial=initial, unit='batches'))

    def batch_end(trainer):
        state['completed'] += 1
        state['progress'].update(state['completed'], f'epoch {trainer.epoch+1}/{trainer.epochs}; training ETA')

    def epoch_end(trainer):
        # Ultralytics invokes this again after final evaluation with epoch advanced.
        if 'progress' in state and trainer.epoch < trainer.epochs:
            state['progress'].update(state['completed'],
                f'epoch {trainer.epoch+1}/{trainer.epochs} validation/checkpoint complete', force=True)

    model.add_callback('on_train_start', start)
    model.add_callback('on_train_batch_end', batch_end)
    model.add_callback('on_fit_epoch_end', epoch_end)
