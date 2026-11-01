"""Traceable training-only crops, independent of the unverified HQGAN architecture."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from detection_pipeline import check_dataset, sha256, write_json


def selected_rows(manifest, stems=None):
    train = {row['export_stem']: row for row in manifest if row['split'] == 'train'}
    if stems is None:
        stems = sorted(train)
    if (not isinstance(stems, list) or not stems or
            any(not isinstance(s, str) for s in stems) or len(set(stems)) != len(stems)):
        raise ValueError('Subset must be a nonempty JSON list of unique export stems')
    if set(stems) - train.keys():
        raise ValueError('Subset contains unknown or non-training images')
    return [train[stem] for stem in sorted(stems)]


def pixel_box(line, width, height, n_classes):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError('Expected a YOLO detection box')
    cid = int(parts[0])
    cx, cy, w, h = map(float, parts[1:])
    if (not 0 <= cid < n_classes or not all(math.isfinite(v) for v in (cx, cy, w, h))
            or w <= 0 or h <= 0 or min(cx-w/2, cy-h/2) < -1e-6
            or max(cx+w/2, cy+h/2) > 1+1e-6):
        raise ValueError('Invalid or out-of-bounds detection box')
    # Outward rounding retains the full annotated box; no fabricated segmentation mask.
    box = (max(0, math.floor((cx-w/2)*width)), max(0, math.floor((cy-h/2)*height)),
           min(width, math.ceil((cx+w/2)*width)), min(height, math.ceil((cy+h/2)*height)))
    return cid, box


def prepare_patches(data, config, output, stems=None, min_side=8):
    """Export native-resolution RGB patches and provenance without touching the source."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    if isinstance(min_side, bool) or not isinstance(min_side, int) or min_side < 1:
        raise ValueError('min_side must be a positive integer')
    data, counts, manifest, runtime = check_dataset(data, config)
    if output.is_relative_to(data.parent):
        raise ValueError('Patch output must be outside the frozen dataset')
    rows = selected_rows(manifest, stems)
    names = runtime['names']
    sources = [{k: row[k] for k in ('export_stem', 'group', 'image_sha256', 'export_label_sha256')}
               for row in rows]
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    images = {p.stem: p for p in Path(runtime['train']).glob('*')}
    output.mkdir(parents=True, exist_ok=False)
    records, skipped = [], []
    try:
        for row in rows:
            stem = row['export_stem']
            image = images[stem]
            label = data.parent / 'train' / 'labels' / (stem + '.txt')
            # Check again at consumption time, after the whole-export validation.
            if sha256(image) != row['image_sha256'] or sha256(label) != row['export_label_sha256']:
                raise ValueError(f'Source changed during preparation: {stem}')
            with Image.open(image) as source:
                rgb = source.convert('RGB')
                for index, line in enumerate(label.read_text().splitlines()):
                    cid, box = pixel_box(line, *rgb.size, len(names))
                    provenance = {'source_stem': stem, 'source_split': 'train', 'group': row['group'],
                                  'annotation_index': index, 'class_id': cid, 'class_name': names[cid],
                                  'pixel_box_xyxy': list(box), 'source_image_sha256': row['image_sha256'],
                                  'source_label_sha256': row['export_label_sha256']}
                    if min(box[2]-box[0], box[3]-box[1]) < min_side:
                        skipped.append({**provenance, 'reason': 'below_min_side'})
                        continue
                    relative = Path('patches') / str(cid) / f'{stem}_{index:04d}.png'
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    rgb.crop(box).save(target)
                    records.append({**provenance, 'patch': relative.as_posix(), 'sha256': sha256(target)})
        class_counts = Counter(r['class_name'] for r in records)
        if any(class_counts[name] == 0 for name in names):
            raise ValueError('Selected training subset lacks usable patches for every core class')
        write_json(output / 'patches.json', records)
        write_json(output / 'skipped.json', skipped)
        report = {'status': 'complete', 'stage': 'hqgan_data_preparation',
                  'model_implemented': False, 'quality': counts['quality'],
                  'source_data': str(data), 'split_manifest_sha256': sha256(data.parent / 'split_manifest.json'),
                  'training_subset_sha256': identity, 'training_sources': sources,
                  'source_image_count': len(rows), 'patch_count': len(records),
                  'patches_by_class': dict(class_counts), 'skipped_count': len(skipped),
                  'min_side': min_side, 'resize': None,
                  'warning': 'Bounding-box crops can contain background and other defects; these are not masks or certified isolated defects.'}
        write_json(output / 'manifest.json', report)
        return report
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'error': str(exc)})
        raise
