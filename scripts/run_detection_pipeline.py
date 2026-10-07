"""Run baseline training, validation evaluation, and prediction previews in order."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from detection_pipeline import ROOT, load_config, resolve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--data', help='Override the prepared dataset YAML')
    parser.add_argument('--name', help='Run prefix; defaults to a unique UTC timestamp')
    parser.add_argument('--device', help='auto, cpu, or a CUDA device such as 0')
    parser.add_argument('--epochs', type=int, help='Override full-training epochs')
    parser.add_argument('--batch', type=int, help='Override training batch size')
    parser.add_argument('--limit', type=int, default=12, help='Number of validation prediction images')
    parser.add_argument('--smoke', action='store_true', help='Small one-epoch train/evaluation check')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them')
    args = parser.parse_args(argv)
    for value in (args.epochs, args.batch, args.limit):
        if value is not None and value <= 0:
            parser.error('epochs, batch and limit must be positive')
    if args.smoke and (args.epochs is not None or args.batch is not None):
        parser.error('--smoke fixes epochs=1 and batch=2; omit --epochs and --batch')

    config_path = resolve(args.config)
    config = load_config(config_path)
    name = args.name or (config['training']['name'] + ('_smoke' if args.smoke else '') +
                         datetime.now(timezone.utc).strftime('_%Y%m%d_%H%M%S_%fZ'))
    if Path(name).name != name or name in {'.', '..'}:
        parser.error('name must be a directory name, not a path')
    project = resolve(config['training']['project'])
    train_dir, eval_dir, predict_dir = [project / n for n in (name, name + '_evaluation', name + '_predictions')]
    for directory in (train_dir, eval_dir, predict_dir):
        if directory.exists():
            parser.error(f'Output already exists: {directory}; choose another --name')
    data = resolve(args.data or config['training']['data'])
    if not data.is_file():
        parser.error(f'Prepared dataset not found: {data}; run tools/prepare_detection.py first')
    weights = train_dir / 'weights' / 'best.pt'
    common = ['--config', str(config_path), '--data', str(data)]
    if args.device:
        common += ['--device', args.device]
    train = ['--name', name]
    evaluate = ['--weights', str(weights), '--name', eval_dir.name, '--split', 'val']
    if args.smoke:
        train += ['--smoke']; evaluate += ['--smoke']
    for flag, value in (('--epochs', args.epochs), ('--batch', args.batch)):
        if value is not None:
            train += [flag, str(value)]
    predict = ['--weights', str(weights), '--name', predict_dir.name, '--limit', str(args.limit)]
    stages = [('Training', 'train_detector.py', train),
              ('Evaluation', 'evaluate_detector.py', evaluate),
              ('Predictions', 'predict_samples.py', predict)]
    for index, (label, script, options) in enumerate(stages, 1):
        command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *common, *options]
        print(f'\n[{index}/3] {label}', flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        if index > 1 and not weights.is_file():
            print(f'Stopped: training did not create {weights}', file=sys.stderr)
            return 1
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f'Stopped: {label.lower()} failed (exit {exc.returncode}). Earlier outputs are preserved.', file=sys.stderr)
            return exc.returncode
        except KeyboardInterrupt:
            print('\nPipeline interrupted. Earlier outputs are preserved.', file=sys.stderr)
            return 130
    if not args.dry_run:
        print(f'\nPipeline complete.\nWeights: {weights}\nMetrics: {eval_dir / "metrics.json"}\nPrediction images: {predict_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
