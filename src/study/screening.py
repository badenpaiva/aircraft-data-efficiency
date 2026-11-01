"""Deterministic technical triage. Passing never certifies defect semantics."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from detection_pipeline import sha256, write_json
from .data import identity, target_label

VERSION = 1
DEFAULTS = {'min_side': 16, 'min_gray_std': 5.0, 'min_laplacian_variance': 15.0,
            'near_duplicate_hamming': 2}


def settings_for(settings=None):
    result = {**DEFAULTS, **(settings or {})}
    if set(result) != set(DEFAULTS):
        raise ValueError('Unknown screening setting')
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid screening setting: {key}')
    if not isinstance(result['min_side'], int) or result['min_side'] < 1:
        raise ValueError('min_side must be a positive integer')
    if not isinstance(result['near_duplicate_hamming'], int) or result['near_duplicate_hamming'] > 64:
        raise ValueError('near_duplicate_hamming must be an integer in [0,64]')
    return result


def subject(row, stage):
    if stage == 'candidates':
        return {k: row.get(k) for k in ('id', 'path', 'sha256', 'class_id', 'class_name',
                                      'checkpoint', 'checkpoint_sha256', 'gan_identity')}
    if stage == 'composites':
        return row['record']
    raise ValueError('Screening stage must be candidates or composites')


def item_id(row, stage):
    return row['id'] if stage == 'candidates' else row['record']['export_stem']


def file_info(row, stage):
    return (row['path'], row['sha256']) if stage == 'candidates' else (row['record']['image_path'], row['record']['image_sha256'])


def fingerprint(row, stage, subset, settings):
    return identity({'version': VERSION, 'stage': stage, 'subset': subset,
                     'settings': settings, 'subject': subject(row, stage)})


def require_screening(row, stage, subset):
    receipt = row.get('screening', {})
    if receipt.get('version') != VERSION or receipt.get('stage') != stage:
        raise ValueError(f'Run screen --stage {stage} before approving images')
    settings = settings_for(receipt.get('settings'))
    if receipt.get('fingerprint') != fingerprint(row, stage, subset, settings):
        raise ValueError('Screening is stale; screen and review again')
    if receipt.get('outcome') == 'reject':
        raise ValueError('Screening rejected this image; it cannot be approved')
    if receipt.get('outcome') not in {'pass', 'needs_review'}:
        raise ValueError('Invalid screening outcome')
    if receipt['outcome'] == 'needs_review' and not row.get('screening_override_reason', '').strip():
        raise ValueError('Flagged image requires a human screening_override_reason')


def scan_bundle(path, stage, settings=None, names=None):
    """Screen all rows, save receipts and previews, and invalidate prior approvals."""
    path = Path(path); bundle = json.loads(path.read_text())
    settings = settings_for(settings); rows = bundle[stage]
    ids = [item_id(row, stage) for row in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate review item IDs')
    output = path.parent / (stage + '_screening'); output.mkdir(exist_ok=True)
    pixels_seen, hashes_seen, checkpoint_hashes = {}, [], {}
    summaries = []
    for index, row in enumerate(rows):
        identifier = item_id(row, stage); image_path, expected = file_info(row, stage)
        reasons, flags, metrics, preview = [], [], {}, None
        try:
            if sha256(image_path) != expected: reasons.append('image_hash_mismatch')
            with Image.open(image_path) as source:
                source.load(); picture = source.convert('RGB')
            gray = np.asarray(picture.convert('L'))
            metrics = {'width': picture.width, 'height': picture.height,
                       'gray_std': float(gray.std()),
                       'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_64F).var())}
            if min(picture.size) < settings['min_side']: flags.append('small_image')
            if stage == 'composites':
                r = row['record']; p = r['augmentation']['placement']
                x, y, w, h = p['rectangle_xywh']
                if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>picture.width or y+h>picture.height:
                    reasons.append('invalid_composite_region')
                else:
                    # Score the inserted region, not the full background image.
                    gray = gray[y:y+h, x:x+w]
                    if min(w,h) < settings['min_side']: flags.append('small_inserted_region')
                    metrics['gray_std'] = float(gray.std())
                    metrics['laplacian_variance'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sha256(r['label_path']) != r['export_label_sha256']: reasons.append('label_hash_mismatch')
                if Path(r['label_path']).read_text() != target_label(r['targets']): reasons.append('labels_disagree_with_targets')
                if (r['width'], r['height']) != picture.size: reasons.append('dimension_mismatch')
                for target in r['targets']:
                    c, x1, y1, x2, y2 = target
                    if not all(isinstance(v, (int,float)) and math.isfinite(v) for v in target) or int(c)!=c or c<0 or (names and c>=len(names)) or not (0<=x1<x2<=1 and 0<=y1<y2<=1):
                        reasons.append('invalid_target_box')
                        break
            else:
                checkpoint = row['checkpoint']
                if checkpoint not in checkpoint_hashes: checkpoint_hashes[checkpoint] = sha256(checkpoint)
                if checkpoint_hashes[checkpoint] != row['checkpoint_sha256']: reasons.append('checkpoint_hash_mismatch')
                if not isinstance(row['class_id'],int) or row['class_id']<0 or (names and (row['class_id']>=len(names) or row['class_name']!=names[row['class_id']])):
                    reasons.append('invalid_class')
            if metrics['gray_std'] < settings['min_gray_std']: flags.append('low_contrast_or_blank')
            if metrics['laplacian_variance'] < settings['min_laplacian_variance']: flags.append('low_detail_or_blur')
            pixel_hash = identity([picture.size, hashlib.sha256(picture.tobytes()).hexdigest()])
            phash = int(str(imagehash.phash(picture)), 16)
            if not reasons:
                if pixel_hash in pixels_seen:
                    reasons.append('exact_duplicate_of:' + pixels_seen[pixel_hash])
                else:
                    for other_hash, other_id in hashes_seen:
                        if (phash ^ other_hash).bit_count() <= settings['near_duplicate_hamming']:
                            flags.append('near_duplicate_of:' + other_id); break
                    pixels_seen[pixel_hash] = identifier; hashes_seen.append((phash,identifier))
            preview = picture.copy()
            draw = ImageDraw.Draw(preview)
            if stage == 'composites':
                for c,x1,y1,x2,y2 in row['record']['targets']:
                    if 0<=x1<x2<=1 and 0<=y1<y2<=1:
                        draw.rectangle((x1*picture.width,y1*picture.height,x2*picture.width,y2*picture.height),outline='red',width=2)
                        draw.text((x1*picture.width,y1*picture.height),str(names[int(c)] if names and 0<=int(c)<len(names) else c),fill='red')
            preview = ImageOps.contain(preview, (640,640))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            reasons.append('unreadable_or_invalid:' + type(exc).__name__)
        outcome = 'reject' if reasons else ('needs_review' if flags else 'pass')
        receipt = {'version': VERSION, 'stage': stage, 'settings': settings,
                   'fingerprint': fingerprint(row,stage,bundle['study_subset_identity'],settings),
                   'outcome': outcome, 'reasons': sorted(set(reasons+flags)), 'metrics': metrics}
        row['screening'] = receipt
        # A re-screen is a new review round; never retain old semantic approval.
        row.update(approved=False, reviewer='', screening_override_reason='', review_notes='')
        for key in ('defect_present','class_correct','labels_complete','realistic'):
            if key in row: row[key] = False
        if stage == 'candidates': row['defect_box_xyxy'] = None
        filename = f'{index:05d}.jpg'
        canvas = Image.new('RGB',(680,740),'white'); draw = ImageDraw.Draw(canvas)
        draw.text((10,10),f'{index:05d} {identifier}\n{outcome}\n'+ '\n'.join(receipt['reasons']),fill='black')
        if preview: canvas.paste(preview,(10,100))
        canvas.save(output/filename)
        summaries.append({'id': identifier, 'outcome': outcome, 'reasons': '; '.join(receipt['reasons']),
                          'preview': filename, 'original': image_path, 'metrics': json.dumps(metrics)})
    bundle['screening_summary'] = {'version': VERSION, 'stage': stage, 'settings': settings,
                                   'outcomes': dict(Counter(row['outcome'] for row in summaries))}
    # Replace the review queue only after all images have been processed.
    temporary = path.with_suffix('.json.tmp'); write_json(temporary,bundle); temporary.replace(path)
    write_json(output/'summary.json',bundle['screening_summary'])
    with (output/'index.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','outcome','reasons','preview','original','metrics'])
        writer.writeheader(); writer.writerows(summaries)
    return output
