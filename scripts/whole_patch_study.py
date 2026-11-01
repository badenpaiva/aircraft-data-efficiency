"""Exploratory, pass-only HQGAN composites with approximate whole-patch labels.

Separate entry point preserves the original study's frozen implementation.
No human approvals are created. Only validation evaluation is exposed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2
import numpy as np
from detection_pipeline import resolve, sha256, write_json
from study import runner
from study.augmentation import export_image, propose_placements
from study.data import identity
from study.screening import require_screening

METHOD = 'exploratory_pass_only_whole_patch_v1'
NOTE = ('Approximate whole-patch boxes; technical-pass candidates only; automatic '
        'non-overlapping placements; no human semantic, placement or composite review. '
        'Validation-only exploratory results, not manually annotated HQGAN results.')


def read(path):
    return json.loads(Path(path).read_text())


def select_candidates(bundle, subset, names):
    if bundle['study_subset_identity'] != subset['identity'] or bundle.get('smoke'):
        raise ValueError('Candidate subset mismatch or smoke candidates')
    selected = {}
    hashes = {}
    for row in bundle['candidates']:
        if row.get('screening', {}).get('outcome') != 'pass':
            continue
        require_screening(row, 'candidates', subset['identity'])
        if row['id'] in selected:
            raise ValueError('Duplicate candidate ID')
        cid = row['class_id']
        if not isinstance(cid, int) or not 0 <= cid < len(names) or names[cid] != row['class_name']:
            raise ValueError('Invalid candidate class')
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Candidate image changed')
        checkpoint = row['checkpoint']
        if checkpoint not in hashes:
            hashes[checkpoint] = sha256(checkpoint)
        if hashes[checkpoint] != row['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint changed')
        selected[row['id']] = row
    if {r['class_id'] for r in selected.values()} != set(range(len(names))):
        raise ValueError('Every class needs at least one screening-pass candidate')
    # Round robin classes; avoid class-sorted prefixes concentrating small quotas.
    groups = [[r for r in selected.values() if r['class_id'] == c] for c in range(len(names))]
    ordered = {}
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                row = group[i]
                ordered[row['id']] = row
    return ordered


def verify_records(rows):
    for row in rows:
        if sha256(row['image_path']) != row['image_sha256'] or sha256(row['label_path']) != row['export_label_sha256']:
            raise ValueError('Image or label changed: ' + row['export_stem'])


def make_composites(source, output, payload, key):
    subset = payload['subsets'][key]
    queue = source / 'gan' / key / 'candidates_review.json'
    bundle = read(queue)
    candidates = select_candidates(bundle, subset, payload['names'])
    real = runner.subset_records(payload, key)
    verify_records(real)
    count = math.ceil(len(real) * max(a['synthetic_ratio'] for a in payload['config']['arms'].values() if a['kind'] == 'hqgan'))
    directory = output / 'whole_patch' / key
    manifest = directory / 'manifest.json'
    inputs = {'method': METHOD, 'source_plan': payload['plan_identity'],
              'candidate_queue_sha256': sha256(queue), 'subset_identity': subset['identity'],
              'count': count, 'script_sha256': sha256(__file__)}
    if manifest.exists():
        cached = read(manifest)
        if cached['inputs'] != inputs:
            raise ValueError('Preparation inputs changed; use a new output directory')
        verify_records(cached['records'])
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    placements = propose_placements(real, candidates, count, subset['seed'])
    backgrounds = {r['export_stem']: r for r in real}
    records = []
    for i, placement in enumerate(placements):
        bg = backgrounds[placement['background_stem']]
        candidate = candidates[placement['candidate_id']]
        x, y, w, h = placement['rectangle_xywh']
        background = cv2.imread(bg['image_path'])
        patch = cv2.imread(candidate['path'])
        if background is None or patch is None:
            raise ValueError('Unreadable blend input')
        patch = cv2.resize(patch, (w, h))
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
        blended = cv2.seamlessClone(patch, background, mask, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
        box = [candidate['class_id'], x / bg['width'], y / bg['height'],
               (x + w) / bg['width'], (y + h) / bg['height']]
        provenance = {'method': METHOD, 'quality': NOTE, 'human_reviewed': False,
                      'defect_box_xyxy': [0, 0, 1, 1], 'candidate': candidate,
                      'placement': placement, 'study_subset_identity': subset['identity']}
        stem = 'whole_patch_' + identity([inputs, placement])[:24]
        records.append(export_image(directory, stem, blended, bg['targets'] + [box], bg, provenance))
        if i % 50 == 0 or i + 1 == count:
            print(f'{key}: blended {i + 1}/{count}', flush=True)
    result = {'inputs': inputs, 'pass_candidates': len(candidates), 'records': records,
              'quality': NOTE, 'human_reviewed': False}
    write_json(manifest, result)
    return result


def prepare(source, output, seed):
    source, output = resolve(source), resolve(output)
    if source == output:
        raise ValueError('Use a separate exploratory output directory')
    payload = runner.load_plan(source)
    if payload['smoke']:
        raise ValueError('Cannot use smoke studies')
    keys = [k for k, s in payload['subsets'].items() if s['seed'] == seed]
    if not keys:
        raise ValueError('Seed not in source study')
    if (output / 'plan.json').exists():
        existing = runner.load_plan(output)
        validate_extension(output, existing)
        if existing['whole_patch']['source_plan_identity'] != payload['plan_identity'] or existing['config']['seeds'] != [seed]:
            raise ValueError('Existing exploratory study has different inputs')
        return
    # Check every candidate queue before creating any composites.
    for key in keys:
        select_candidates(read(source / 'gan' / key / 'candidates_review.json'), payload['subsets'][key], payload['names'])
    manifests = {}
    for key in keys:
        make_composites(source, output, payload, key)
        manifests[key] = sha256(output / 'whole_patch' / key / 'manifest.json')
    derived = copy.deepcopy(payload)
    derived.pop('plan_identity')
    derived['config']['seeds'] = [seed]
    derived['config']['arms'] = {k: v for k, v in derived['config']['arms'].items() if v['kind'] == 'hqgan'}
    derived['subsets'] = {k: derived['subsets'][k] for k in keys}
    derived['jobs'] = [j for j in derived['jobs'] if j['subset'] in keys and j['arm'] in derived['config']['arms']]
    derived['counts']['quality'] = NOTE
    derived['whole_patch'] = {'method': METHOD, 'note': NOTE, 'source_study': str(source),
                             'source_plan_identity': payload['plan_identity'],
                             'script_sha256': sha256(__file__), 'manifest_hashes': manifests}
    derived['plan_identity'] = identity(derived)
    write_json(output / 'plan.json', derived)
    write_json(output / 'status.json', {j['id']: {'status': 'pending'} for j in derived['jobs']})
    (output / 'README.md').write_text('# Exploratory whole-patch study\n\n' + NOTE +
        '\n\nUse scripts/whole_patch_study.py run for this study. Original review approvals are unchanged.\n')
    print(f'Prepared {len(derived["jobs"])} exploratory detector jobs at {output}', flush=True)


def validate_extension(directory, payload):
    extension = payload.get('whole_patch', {})
    if extension.get('method') != METHOD or extension.get('script_sha256') != sha256(__file__):
        raise ValueError('Exploratory script changed or wrong study type')
    for key, expected in extension['manifest_hashes'].items():
        path = directory / 'whole_patch' / key / 'manifest.json'
        if sha256(path) != expected:
            raise ValueError('Exploratory composite manifest changed')


def whole_patch_records(directory, payload, job):
    validate_extension(directory, payload)
    real = runner.subset_records(payload, job['subset'])
    records = read(directory / 'whole_patch' / job['subset'] / 'manifest.json')['records']
    count = math.ceil(len(real) * payload['config']['arms'][job['arm']]['synthetic_ratio'])
    records = sorted(records, key=lambda r: identity([job['seed'], r['export_stem']]))
    if len(records) < count:
        raise ValueError('Insufficient distinct composites for requested ratio')
    records = records[:count]
    verify_records(real + records)
    return real + records, len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    p = commands.add_parser('run')
    p.add_argument('--study', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--models', nargs='+')
    p.add_argument('--arms', nargs='+')
    p.add_argument('--limit-jobs', type=int)
    p.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.output, args.seed)
        return 0
    directory = resolve(args.study)
    payload = runner.load_plan(directory)
    validate_extension(directory, payload)
    if args.limit_jobs is not None and args.limit_jobs < 1:
        parser.error('--limit-jobs must be positive')
    for selected, available in [(args.models, payload['config']['models']), (args.arms, payload['config']['arms'])]:
        if selected and not set(selected) <= set(available):
            parser.error('Unknown model or arm')
    # Local process adapter only: reviewed study implementation stays untouched.
    original = runner.train_records
    try:
        runner.train_records = whole_patch_records
        states = runner.run(directory, args.device, args.models, args.arms, args.limit_jobs, args.retry_failed)
    finally:
        runner.train_records = original
    from study.report import report
    print('Exploratory report:', report(directory), flush=True)
    return int(any(s['status'] == 'failed' for s in states.values()))


if __name__ == '__main__':
    raise SystemExit(main())
