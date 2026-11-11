"""Frozen input inventory, deterministic nested grouped scarcity subsets and exports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from detection_pipeline import check_dataset, sha256, write_json


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inventory(data, experiment):
    data, counts, manifest, runtime = check_dataset(data, experiment)
    result = []
    files = {(split,p.stem):p for split in ('train','val','test') for p in Path(runtime[split]).glob('*')}
    for row in manifest:
        image = files[row['split'],row['export_stem']]
        label = data.parent/row['split']/'labels'/(row['export_stem']+'.txt')
        with Image.open(image) as im:
            width,height = im.size
        boxes = []
        for line in label.read_text().splitlines():
            cid,cx,cy,w,h = map(float,line.split())
            if not (cid.is_integer() and 0 <= cid < len(runtime['names']) and
                    all(math.isfinite(x) for x in (cx,cy,w,h)) and w > 0 and h > 0 and
                    min(cx-w/2,cy-h/2) >= -1e-6 and max(cx+w/2,cy+h/2) <= 1+1e-6):
                raise ValueError(f'Invalid geometry: {label}')
            boxes.append([int(cid),max(0,cx-w/2),max(0,cy-h/2),min(1,cx+w/2),min(1,cy+h/2)])
        result.append({**row,'image_path':str(image),'label_path':str(label),
                       'width':width,'height':height,'targets':boxes})
    return result,counts,runtime['names'],sha256(data.parent/'split_manifest.json')


def nested_subsets(records, fractions, seed, n_classes, min_instances=5):
    """Grow whole-group subsets; report achieved size rather than claim exact fractions.

    Seeded randomised-greedy class/size balancing: each step takes one of the three
    best-balancing groups, chosen by the seed. With few large groups a plain greedy
    choice has no ties, so every seed would otherwise receive the same subset.
    Minimum coverage is established at the smallest fraction; if it cannot fit that
    budget, reject instead of padding.
    """
    if not fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError('Fractions must be unique values in (0,1]')
    groups = defaultdict(list)
    for row in records:
        if row['split'] == 'train': groups[row['group']].append(row)
    if not groups: raise ValueError('Empty training pool')
    def counts(rows):
        return np.array([sum(t[0]==c for r in rows for t in r['targets']) for c in range(n_classes)])
    sizes = {g:len(rs) for g,rs in groups.items()}
    features = {g:counts(rs) for g,rs in groups.items()}
    total = sum(sizes.values()); total_classes = sum(features.values())
    if min(total_classes) < min_instances: raise ValueError('Training pool lacks minimum class coverage')
    order = sorted(groups,key=lambda g:identity([seed,g]))
    rng = np.random.default_rng(int(identity(['subset choice',seed])[:16],16))
    def choose(ranked):
        # Stable sorts keep the seed's own order among equal scores.
        return ranked[int(rng.integers(min(3,len(ranked))))]
    selected = []; current = np.zeros(n_classes); n = 0
    smallest_budget = max(1, math.floor(total*min(fractions)))
    while min(current) < min_instances:
        feasible = [g for g in order if g not in selected and n+sizes[g] <= smallest_budget]
        deficit = np.maximum(0,min_instances-current)
        def coverage(g):
            return float(np.minimum(features[g],deficit).sum()/sizes[g])
        useful = sorted((g for g in feasible if coverage(g) > 0),key=lambda g:-coverage(g))
        if not useful: raise ValueError('Smallest fraction cannot satisfy grouped minimum class coverage')
        best = choose(useful)
        selected.append(best); n += sizes[best]; current += features[best]
    subsets = {}
    for fraction in sorted(fractions):
        target = max(1,math.floor(total*fraction))
        while True:
            eligible = [g for g in order if g not in selected and n+sizes[g] <= target]
            if not eligible: break
            desired = total_classes*fraction
            def score(g):
                return float(np.square((current+features[g]-desired)/np.maximum(desired,1)).sum()
                             + ((n+sizes[g]-target)/target)**2)
            best = choose(sorted(eligible,key=score))
            selected.append(best); n += sizes[best]; current += features[best]
        rows = sorted([r for g in selected for r in groups[g]],key=lambda r:r['export_stem'])
        subsets[str(fraction)] = {'fraction':fraction,'seed':seed,'requested_images':target,
            'images':len(rows),'actual_fraction':len(rows)/total,'groups':len(selected),
            'instances_per_class':counts(rows).tolist(),'stems':[r['export_stem'] for r in rows],
            'identity':identity([{k:r[k] for k in ('export_stem','image_sha256','export_label_sha256')} for r in rows])}
    return subsets


def bounded_sample(records, limit, n_classes):
    chosen = []; covered = set()
    for r in sorted(records,key=lambda r:r['export_stem']):
        cs = {t[0] for t in r['targets']}
        if cs-covered: chosen.append(r); covered |= cs
    if covered != set(range(n_classes)): raise ValueError('Smoke sample lacks a core class')
    for r in sorted(records,key=lambda r:r['export_stem']):
        if len(chosen) >= limit: break
        if r not in chosen: chosen.append(r)
    return chosen


def gan_smoke_sample(records, limit, n_classes):
    """Keep at least two usable crops per class for the GAN's batch-two smoke run."""
    chosen = []
    counts = np.zeros(n_classes, dtype=int)
    ordered = sorted(records, key=lambda r: r['export_stem'])
    for row in ordered:
        usable = np.zeros(n_classes, dtype=int)
        for c,x1,y1,x2,y2 in row['targets']:
            width = math.ceil(x2*row['width'])-math.floor(x1*row['width'])
            height = math.ceil(y2*row['height'])-math.floor(y1*row['height'])
            if min(width,height) >= 8: usable[int(c)] += 1
        if any((counts < 2) & (usable > 0)):
            chosen.append(row); counts += usable
        if min(counts) >= 2: break
    if min(counts) < 2 or len(chosen) > limit:
        raise ValueError('GAN smoke subset needs two usable patches per class within its image budget')
    for row in ordered:
        if len(chosen) >= limit: break
        if row not in chosen: chosen.append(row)
    return chosen


def runtime_yaml(output, train, val, test, names):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    paths = {}
    for split,rows in [('train',train),('val',val),('test',test)]:
        file = output/(split+'.txt')
        file.write_text(''.join(r['image_path']+'\n' for r in rows),encoding='utf-8')
        paths[split] = str(file.resolve())
    path = output/'data.yaml'
    path.write_text(yaml.safe_dump({'path':str(output.resolve()),'names':names,**paths}),encoding='utf-8')
    return path


def target_label(targets):
    return ''.join(f'{int(c)} {(x1+x2)/2:.12g} {(y1+y2)/2:.12g} {x2-x1:.12g} {y2-y1:.12g}\n'
                   for c,x1,y1,x2,y2 in targets)
