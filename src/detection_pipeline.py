"""Shared configuration, dataset validation and reporting for the baseline."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
# Ultralytics checks the parent directory before creating its settings subfolder.
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.ultralytics' / 'matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text(encoding='utf-8-sig'))
    if config['task']['type'] != 'detection':
        raise ValueError('This pipeline implements detection only')
    return config


def check_dataset(data_path, config):
    """Check exported contents, class mapping, group leakage and review policy."""
    data_path = resolve(data_path)
    spec = yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    if spec['names'] != config['task']['core_classes']:
        raise ValueError('Dataset class order differs from experiment core_classes')
    base = data_path.parent
    counts = json.loads((base / 'counts.json').read_text())
    manifest = json.loads((base / 'split_manifest.json').read_text())
    if not manifest:
        raise ValueError('Dataset is empty')
    if counts.get('quality') not in {'reviewed', 'provisional'}:
        raise ValueError('Dataset is missing its review-quality metadata; rebuild it')
    strict = config['datasets']['annotation_policy']['require_domain_and_completeness_review']
    if strict and (counts['quality'] != 'reviewed' or any(r.get('review_status') != 'approved' for r in manifest)):
        raise ValueError('Manual review is required: rebuild a reviewed dataset before training')
    membership, seen, class_counts = {}, set(), {s: [0] * len(spec['names']) for s in ('train','val','test')}
    image_index = {}
    for split in class_counts:
        for image in (base / split / 'images').glob('*'):
            image_index.setdefault((split,image.stem),[]).append(image)
    for row in manifest:
        split, stem = row['split'], row['export_stem']
        if split not in class_counts or stem in seen:
            raise ValueError('Invalid split or repeated image in split manifest')
        seen.add(stem)
        membership.setdefault(row['group'], set()).add(split)
        images = image_index.get((split,stem), [])
        label = base / split / 'labels' / (stem + '.txt')
        if len(images) != 1 or not label.is_file():
            raise ValueError(f'Missing or ambiguous image/label pair: {stem}')
        if sha256(images[0]) != row['image_sha256'] or sha256(label) != row['export_label_sha256']:
            raise ValueError(f'Exported data changed: {stem}; rebuild the dataset')
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Expected detection box: {label}')
            cid = int(parts[0])
            if not 0 <= cid < len(spec['names']):
                raise ValueError(f'Invalid class: {label}')
            class_counts[split][cid] += 1
    if any(len(splits) != 1 for splits in membership.values()):
        raise ValueError('Related-image group crosses dataset splits')
    if sum(len(paths) for paths in image_index.values()) != len(manifest):
        raise ValueError('Export contains images outside the frozen split manifest')
    if any(n == 0 for values in class_counts.values() for n in values):
        raise ValueError('Every split must contain all core classes')
    # Resolve paths from the actual export location (also works after copying to Colab).
    runtime = {'path': str(base), 'names': spec['names'],
               **{split: str(base / split / 'images') for split in class_counts}}
    return data_path, counts, manifest, runtime


def device_name(requested):
    import torch
    return ('0' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def versions():
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('ultralytics','torch','torchvision','numpy','Pillow')}}


def training_args(config, data, output, device, epochs=None, batch=None, imgsz=None, seed=None):
    train = config['training']
    return dict(data=str(data), project=str(output.parent), name=output.name,
                exist_ok=True, epochs=epochs or train['epochs'], batch=batch or train['batch_size'],
                imgsz=imgsz or train['image_size'], device=device, workers=train['workers'],
                seed=config['seeds'][0] if seed is None else seed,
                deterministic=config['compute']['deterministic'], patience=train['patience'],
                optimizer=train['optimizer'], lr0=train['learning_rate'],
                cache=False, amp=False, plots=True, val=True, save=True,
                close_mosaic=0, **train['augmentation'])


def metrics_dict(metrics, names):
    # These P/R values are library-selected operating points, not fixed-confidence recall.
    rows = []
    for i, cid in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        rows.append(dict(class_id=int(cid), name=names[int(cid)], precision=float(p), recall=float(r),
                         ap50=float(ap50), ap50_95=float(ap)))
    return {'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
            'per_class': rows, 'precision_recall_note': 'Ultralytics-selected confidence operating point',
            'speed_ms_per_image': {k: float(v) for k,v in metrics.speed.items()}}


def checkpoint_matches(model, names):
    actual = [model.names[k] for k in sorted(model.names)]
    if actual != names:
        raise ValueError(f'Checkpoint classes {actual} do not match dataset classes {names}')


def match_counts(predictions, targets, n_classes, iou_threshold=0.5):
    """Class-aware one-to-one matching; rows are cid,confidence,x1,y1,x2,y2 and cid,x1,y1,x2,y2."""
    totals = [{'tp':0,'fp':0,'fn':0} for _ in range(n_classes)]
    used = set()
    for prediction in sorted(predictions, key=lambda row: -row[1]):
        cid = int(prediction[0]); box = np.asarray(prediction[2:], dtype=float)
        best, best_iou = None, -1.0
        for index, target in enumerate(targets):
            if index in used or int(target[0]) != cid: continue
            gt = np.asarray(target[1:], dtype=float)
            overlap = np.maximum(0, np.minimum(box[2:],gt[2:])-np.maximum(box[:2],gt[:2]))
            intersection = float(np.prod(overlap))
            union = float(np.prod(box[2:]-box[:2]) + np.prod(gt[2:]-gt[:2]) - intersection)
            iou = intersection/union if union > 0 else 0
            if iou > best_iou: best, best_iou = index, iou
        if best is not None and best_iou >= iou_threshold:
            used.add(best); totals[cid]['tp'] += 1
        else: totals[cid]['fp'] += 1
    for index, target in enumerate(targets):
        if index not in used: totals[int(target[0])]['fn'] += 1
    return totals


def smoke_subset(runtime, output, per_split=9):
    """Small deterministic train/val-only sample; never reads the test images."""
    result = dict(runtime)
    for split in ('train', 'val'):
        images = sorted(Path(runtime[split]).glob('*'))
        selected, covered = [], set()
        for image in images:
            label = image.parent.parent / 'labels' / (image.stem + '.txt')
            classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
            if classes - covered:
                selected.append(image); covered |= classes
        for image in images:
            if len(selected) >= per_split:
                break
            if image not in selected:
                selected.append(image)
        file = output / f'smoke_{split}.txt'
        file.write_text(''.join(str(p) + '\n' for p in selected), encoding='utf-8')
        result[split] = str(file)
    result.pop('test', None)
    return result
