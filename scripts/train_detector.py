"""Train the configured D1 baseline. Use --smoke for a small one-epoch check."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from detection_pipeline import (resolve, load_config, check_dataset, device_name, versions,
                                training_args, write_json, sha256, smoke_subset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--data')
    parser.add_argument('--name')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imgsz', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate the export and print the plan without training')
    args = parser.parse_args()
    config = load_config(args.config)
    for value in (args.epochs, args.batch, args.imgsz):
        if value is not None and value <= 0:
            parser.error('epochs, batch and imgsz must be positive')
    data, counts, manifest, runtime = check_dataset(args.data or config['training']['data'], config)
    name = args.name or (config['training']['name'] + ('_smoke' if args.smoke else ''))
    if Path(name).name != name or name in {'.','..'}:
        parser.error('name must be a directory name, not a path')
    output = resolve(config['training']['project']) / name
    if output.exists():
        parser.error(f'Run directory already exists: {output}; use a new --name')
    device = device_name(args.device or config['training']['device'])
    plan = {'dataset': str(data), 'quality': counts['quality'], 'images': counts['images'],
            'model': config['training']['model'], 'device': device, 'smoke': args.smoke,
            'output': str(output)}
    if args.dry_run:
        print(plan); return
    output.mkdir(parents=True)
    if args.smoke:
        runtime = smoke_subset(runtime, output)
    runtime_path = output / 'runtime_data.yaml'
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding='utf-8')
    kwargs = training_args(config, runtime_path, output, device,
                           1 if args.smoke else args.epochs, 2 if args.smoke else args.batch,
                           args.imgsz or (128 if args.smoke else None), args.seed)
    write_json(output / 'run_manifest.json', {**plan, 'status': 'started', 'versions': versions(),
               'config': config, 'resolved_training_args': kwargs,
               'split_manifest_sha256': sha256(data.parent / 'split_manifest.json')})
    (output / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    from ultralytics import YOLO
    started = time.perf_counter()
    try:
        model = YOLO(config['training']['model'])
        model.train(**kwargs)
        write_json(output / 'completion.json', {'status': 'complete', 'quality': counts['quality'],
                   'smoke': args.smoke, 'train_time_sec': time.perf_counter()-started,
                   'best_weights': str(output / 'weights/best.pt')})
    except Exception as exc:
        write_json(output / 'completion.json', {'status':'failed','error':str(exc)})
        raise
    print(f'Completed: {output}')


if __name__ == '__main__':
    main()
