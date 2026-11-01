"""Prepare traceable training-only defect patches for HQGAN."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from detection_pipeline import load_config, resolve
from augmentation.hqgan.data import prepare_patches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--data', help='Frozen detector dataset YAML')
    parser.add_argument('--output', required=True, help='New patch export directory')
    parser.add_argument('--subset', help='JSON list of training export stems; defaults to all training images')
    parser.add_argument('--min-side', type=int, default=8, help='Minimum native crop width and height in pixels')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        stems = json.loads(resolve(args.subset).read_text()) if args.subset else None
        if args.subset and not isinstance(stems, list):
            raise ValueError('Subset file must contain a JSON list, not null or an object')
        report = prepare_patches(args.data or config['training']['data'], config,
                                 resolve(args.output), stems, args.min_side)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Preparation failed: {exc}\n')
    print(json.dumps({k: report[k] for k in ('status', 'source_image_count', 'patch_count',
                                          'patches_by_class', 'skipped_count')}, indent=2))
    print(f'Prepared patches: {resolve(args.output)}\nNext: scripts/train_hqgan.py (see docs/hqgan.md).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
