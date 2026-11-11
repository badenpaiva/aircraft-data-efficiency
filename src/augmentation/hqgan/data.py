"""Traceable training-only crops, independent of the unverified HQGAN architecture."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
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


def crop_settings(crop=None, min_side=8):
    """Validated crop rule; the defaults reproduce the first export (tight box, as annotated).

    margin: context added on every side, as a fraction of that side of the box.
    min_box_side: boxes with a shorter side are skipped instead of being enlarged.
    canonical_orientation: turn tall boxes a quarter turn so the long side is horizontal.
    """
    settings = {'margin': 0.0, 'min_box_side': min_side, 'canonical_orientation': False, **(crop or {})}
    if set(settings) != {'margin', 'min_box_side', 'canonical_orientation'}:
        raise ValueError('Unknown crop setting')
    margin, side = settings['margin'], settings['min_box_side']
    if isinstance(side, bool) or not isinstance(side, int) or side < 1:
        raise ValueError('min_side must be a positive integer')
    if (isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin)
            or not 0 <= margin <= 0.5):
        raise ValueError('Crop margin must be a fraction in [0, 0.5]')
    if not isinstance(settings['canonical_orientation'], bool):
        raise ValueError('canonical_orientation must be true or false')
    settings['margin'] = float(margin)
    return settings


def margin_pixels(length, margin):
    """Context pixels beside a box side; shared by the crop and by the paste that mirrors it."""
    return math.floor(length*margin + 0.5)


def context_patch(rgb, box, crop):
    """Annotated box with its margin, as a PIL image, plus the window and whether it was turned.

    The box fills the same central part of every patch. A generated patch therefore carries
    its label by construction: the same central part, mapped through the paste rectangle.
    Context that would fall outside the photograph is mirrored instead of invented.
    """
    x1, y1, x2, y2 = box
    px, py = margin_pixels(x2-x1, crop['margin']), margin_pixels(y2-y1, crop['margin'])
    window = (x1-px, y1-py, x2+px, y2+py)
    inside = (max(0, window[0]), max(0, window[1]), min(rgb.width, window[2]), min(rgb.height, window[3]))
    pixels = np.asarray(rgb.crop(inside))
    pads = ((inside[1]-window[1], window[3]-inside[3]), (inside[0]-window[0], window[2]-inside[2]), (0, 0))
    if any(v for pair in pads for v in pair):
        pixels = np.pad(pixels, pads, mode='reflect')
    turned = crop['canonical_orientation'] and (y2-y1) > (x2-x1)
    if turned:
        pixels = np.rot90(pixels)  # counter-clockwise; study.augmentation.paste_patch turns it back
    return Image.fromarray(np.ascontiguousarray(pixels)), list(window), bool(turned)


def prepare_patches(data, config, output, stems=None, min_side=8, crop=None):
    """Export native-resolution RGB patches and provenance without touching the source."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    crop = crop_settings(crop, min_side)
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
                    if min(box[2]-box[0], box[3]-box[1]) < crop['min_box_side']:
                        skipped.append({**provenance, 'reason': 'below_min_side'})
                        continue
                    patch, window, turned = context_patch(rgb, box, crop)
                    relative = Path('patches') / str(cid) / f'{stem}_{index:04d}.png'
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    patch.save(target)
                    records.append({**provenance, 'crop_window_xyxy': window, 'quarter_turn': turned,
                                    'patch': relative.as_posix(), 'sha256': sha256(target)})
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
                  'min_side': crop['min_box_side'], 'crop': crop, 'resize': None,
                  'warning': 'Bounding-box crops can contain background and other defects; these are not masks or certified isolated defects.'}
        write_json(output / 'manifest.json', report)
        return report
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'error': str(exc)})
        raise
