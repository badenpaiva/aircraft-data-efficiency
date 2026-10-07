"""Prepare configured detection sources and export reviewed or provisional images.
Raw files are read-only. See docs/prepare_detection.md for the two-stage workflow.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import imagehash
import yaml
from PIL import Image, ImageDraw, ImageOps
from review_d1 import parse_line

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
HINTS = {'D1': 'Innovation Hangar', 'D2': 'aircraft_skin_defects.v1', 'D4': 'airscraft-skin-crack-dent'}
CHECKS = ['domain_ok', 'labels_complete', 'unmapped_absent', 'visible_at_640', 'grouping_checked', 'unaugmented_real']
FIELDS = ['image', 'source', 'image_sha256', 'label_sha256', 'schema_sha256', 'action', 'reviewer', *CHECKS, 'background_verified', 'additional_group', 'notes']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_reviews(path):
    if not path: return {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = row.get('image')
        if not key or key in out: raise ValueError('Review CSV has missing or duplicate image keys')
        if row.get('action', '') not in {'', 'approve', 'exclude', 'repair', 'quarantine'}:
            raise ValueError(f'Unknown review action for {key}')
        out[key] = row
    return out


def normalized_annotations(text, names, source_map, target_names):
    annotations, issues, seen = [], [], set()
    duplicates = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        try:
            cid, kind, xy, bounds = parse_line(line, len(names))
        except ValueError as exc:
            issues.append(f'invalid_label_line_{line_no}:{exc}'); continue
        raw_class = names[cid]
        target = source_map.get(raw_class)
        if target not in target_names:
            issues.append(f'unmapped_class:{raw_class}')
        # Only remove identical source geometry, not overlapping objects.
        key = (cid, kind, tuple(xy))
        if key in seen:
            duplicates += 1; continue
        seen.add(key)
        x1, y1, x2, y2 = bounds
        annotations.append({'line': line_no, 'source_class': raw_class, 'kind': kind,
                            'coordinates': xy, 'target_class': target,
                            'target_id': target_names.index(target) if target in target_names else None,
                            'box': [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]})
    return annotations, sorted(set(issues)), duplicates


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def group_records(records, threshold):
    """Group similarities; do not delete near duplicates. Include quarantined bridges."""
    uf = UnionFind(len(records)); keys = {}
    for i, r in enumerate(records):
        stem = Path(r['image']).stem.split('.rf.')[0]
        # Known numeric variants are conservative within-source families.
        family = re.match(r'^(\d+)(?:_\d+)?(?:_|$)', stem)
        group_keys = [('bytes', r['image_sha256'])]
        if family: group_keys.append(('numeric_family', r['source'], family[1]))
        if '.rf.' in Path(r['image']).stem: group_keys.append(('rf_base', r['source'], stem))
        if r.get('additional_group'): group_keys.append(('reviewer_group', r['additional_group']))
        for key in group_keys:
            if key in keys: uf.union(i, keys[key])
            else: keys[key] = i
    valid = [(i, r['phash']) for i,r in enumerate(records) if r['phash'] is not None]
    for pos, (i, h) in enumerate(valid):
        for j, other in valid[pos+1:]:
            if (h ^ other).bit_count() <= threshold: uf.union(i, j)
    groups = defaultdict(list)
    for i, r in enumerate(records): groups[uf.find(i)].append(r)
    for members in groups.values():
        gid = hashlib.sha256('\n'.join(sorted(r['image'] for r in members)).encode()).hexdigest()[:20]
        for r in members: r['group'] = gid
    return list(groups.values())


def annotation_signature(r):
    return sorted((str(a['target_class'])+':'+a['source_class'], tuple(round(x,10) for x in a['box'])) for a in r['annotations'])


def assign_status(r, review, require_review=True):
    issues = r['issues']
    if any(x.startswith('unmapped_class:') for x in issues): return 'excluded_unmapped'
    if review and (review.get('image_sha256') != r['image_sha256'] or review.get('label_sha256') != (r['label_sha256'] or '') or review.get('schema_sha256') != r['schema_sha256']):
        return 'quarantine_stale_review'
    if review and review.get('action') == 'exclude': return 'excluded_review'
    if issues: return 'quarantine_invalid'
    if review and review.get('action') == 'approve':
        required = CHECKS + (['background_verified'] if not r['annotations'] else [])
        if review.get('reviewer', '').strip() and all(review.get(k, '').lower() == 'yes' for k in required):
            return 'approved'
        return 'quarantine_incomplete_approval'
    if review and review.get('action') in {'repair','quarantine'}: return 'quarantine_review'
    if r.get('prior_review_note'): return 'quarantine_prior_review'
    if not r['annotations']: return 'quarantine_empty'
    return 'pending_review' if require_review else 'provisional'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def scan(root, config, output, reviews=None, d4_order=None, overrides=None):
    target_names = config['task']['core_classes']
    datasets = config['datasets']
    require_review = datasets['annotation_policy'].get('require_domain_and_completeness_review', True)
    if not isinstance(require_review, bool):
        raise ValueError('require_domain_and_completeness_review must be true or false')
    selected = datasets.get('experiment_sources')
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError('experiment_sources must be a nonempty list of unique source names')
    supported = {datasets['source_ids'][did]: did for did in HINTS}
    if any(name not in supported for name in selected):
        raise ValueError('experiment_sources contains an unsupported detection source')
    if datasets.get('pool_supplementary') is not True and len(selected) > 1:
        raise ValueError('Multiple experiment_sources require pool_supplementary: true')
    source_ids = [supported[name] for name in selected]
    if config['datasets']['unmapped_class_policy'] != 'exclude_images':
        raise ValueError('This cleaner implements exclude_images only')
    prior_path = root / config['datasets']['annotation_policy']['review_ledger']
    prior = {r['image']: r for r in json.loads(prior_path.read_text())} if prior_path.exists() else {}
    decisions = read_reviews(reviews)
    records, orphan_labels = [], []
    for did in source_ids:
        hint = HINTS[did]
        matches = sorted(p for p in (root/'data/raw').iterdir() if p.is_dir() and hint.lower() in p.name.lower())
        if len(matches) != 1: raise ValueError(f'{did}: expected exactly one source folder, got {matches}')
        folder = matches[0]; spec = config['datasets']['sources'][config['datasets']['source_ids'][did]]
        yml = folder/'data.yaml'
        if yml.exists():
            names = yaml.safe_load(yml.read_text())['names']
            if isinstance(names, dict): names = [names[k] for k in sorted(names)]
        elif did == 'D4':
            names = d4_order.split(',') if d4_order else spec.get('source_label_names')
        else:
            names = None
        matched_labels = set()
        images = sorted(p for p in folder.glob('*/images/*') if p.suffix.lower() in EXTENSIONS)
        if not images: raise ValueError(f'{did}: no supported images in split/images directories')
        for image_path in images:
            rel = image_path.relative_to(root).as_posix()
            label = image_path.parent.parent/'labels'/f'{image_path.stem}.txt'
            matched_labels.add(label)
            label_rel = label.relative_to(root)
            effective = overrides/label_rel if overrides and (overrides/label_rel).exists() else label
            r = {'image': rel, 'source': did, 'original_label': label_rel.as_posix(),
                 'effective_label': str(effective.resolve()), 'image_sha256': digest(image_path),
                 'original_label_sha256': digest(label), 'label_sha256': digest(effective),
                 'annotations': [], 'issues': [], 'phash': None, 'duplicate_rows_removed': 0,
                 'class_order': names, 'class_order_evidence': 'local_data_yaml' if yml.exists() else ('explicit_cli_order' if d4_order else spec.get('class_order_evidence', 'configured_order')) if names else 'unconfirmed_numeric_order'}
            try:
                with Image.open(image_path) as im:
                    im.load(); r['size'] = list(im.size)
                    if im.getexif().get(274,1) != 1: r['issues'].append('nontrivial_exif_orientation')
                    r['phash'] = int(str(imagehash.phash(im)),16)
            except Exception as exc:
                r['issues'].append(f'corrupt_image:{type(exc).__name__}')
            if not effective.exists(): r['issues'].append('missing_label')
            elif names is None:
                r['issues'].append('unconfirmed_class_id_order')
                r['raw_label_text'] = effective.read_text(encoding='utf-8-sig')
            else:
                r['annotations'], issues, r['duplicate_rows_removed'] = normalized_annotations(effective.read_text(encoding='utf-8-sig'), names, spec['class_map'], target_names)
                r['issues'].extend(issues)
            if rel in prior:
                r['prior_review_note'] = prior[rel]['note']
                r['prior_review_label_sha256'] = prior[rel].get('label_sha256')
            r['schema_sha256'] = hashlib.sha256(json.dumps({'names':names,'mapping':spec['class_map'],'targets':target_names},sort_keys=True).encode()).hexdigest()
            r['empty_label'] = effective.exists() and not effective.read_text(encoding='utf-8-sig').strip()
            if did == 'D4': r['source_caution'] = 'Paper reports pre-augmentation; verify genuine unaugmented source and completeness of all target classes.'
            review = decisions.get(rel)
            if review and review.get('image_sha256') == r['image_sha256'] and review.get('label_sha256') == (r['label_sha256'] or '') and review.get('schema_sha256') == r['schema_sha256']:
                r['additional_group'] = review.get('additional_group','')
            r['status'] = assign_status(r, review, require_review)
            r['review'] = review
            records.append(r)
        orphan_labels.extend(p.relative_to(root).as_posix() for p in folder.glob('*/labels/*.txt') if p not in matched_labels)
    unknown = set(decisions)-{r['image'] for r in records}
    if unknown: raise ValueError(f'Reviews reference {len(unknown)} unknown images')
    threshold = config['dedup']['hamming_threshold']
    if not isinstance(threshold, int) or not 0 <= threshold <= 64: raise ValueError('Invalid pHash threshold')
    groups = group_records(records, threshold)
    exact = defaultdict(list)
    for r in records: exact[r['image_sha256']].append(r)
    # A conflicting annotation on an identical image needs adjudication first.
    for members in exact.values():
        if len(members) > 1 and any(annotation_signature(r) != annotation_signature(members[0]) or r['issues'] != members[0]['issues'] for r in members):
            for r in members:
                r['issues'].append('exact_image_label_conflict')
                if not r['status'].startswith('excluded'): r['status'] = 'quarantine_conflicting_labels'
        elif len(members) > 1:
            # Prefer an approved copy; do not propagate approval to other views.
            canonical = min(members, key=lambda r:(r['status'] != 'approved', r['image']))
            for r in members:
                if r is not canonical:
                    r['duplicate_of'] = canonical['image']; r['status'] = 'duplicate_exact'
    output.mkdir(parents=True, exist_ok=False)
    staging = output/'candidate_labels'; staging.mkdir()
    for r in records:
        r['export_stem'] = r['source']+'_'+hashlib.sha256(r['image'].encode()).hexdigest()[:24]
        if r['annotations'] and not r['issues'] and not r['status'].startswith('excluded'):
            text=''.join(f"{a['target_id']} "+' '.join(f'{v:.12g}' for v in a['box'])+'\n' for a in r['annotations'])
            (staging/(r['export_stem']+'.txt')).write_text(text)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with (output/'review.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
        for r in records:
            row={k:(r.get('review') or {}).get(k,'') for k in FIELDS}
            row.update({k:r[k] or '' for k in ['image','source','image_sha256','label_sha256','schema_sha256']})
            if r['status'] == 'quarantine_stale_review':
                row.update({k:'' for k in ['reviewer',*CHECKS,'background_verified']})
                row['action']='quarantine'; row['notes']='Previous review invalidated by changed image, label or schema. Review again.'
            if not row['notes']: row['notes']='; '.join(r['issues']) or r.get('prior_review_note',r.get('source_caution',''))
            writer.writerow(row)
    write_json(output/'groups.json', [{'group':g[0]['group'],'members':[r['image'] for r in g],'sources':sorted({r['source'] for r in g})} for g in groups])
    summary={'sources':{},'target_names':target_names,'status_counts':dict(Counter(r['status'] for r in records)),
             'prior_review_ledger_found':prior_path.is_file(), 'manual_review_required':require_review,
             'similarity_groups':len(groups),'cross_source_groups':sum(len({r['source'] for r in g})>1 for g in groups),
             'orphan_labels':orphan_labels,'warning':'Candidate labels are not an approved dataset. Similarity groups are conservative, not verified duplicates.',
             'duplicate_rows_removed':sum(r['duplicate_rows_removed'] for r in records),'threshold':threshold}
    for did in source_ids:
        subset=[r for r in records if r['source']==did]
        summary['sources'][did]={'images':len(subset),'empty_labels':sum(r['empty_label'] for r in subset),'statuses':dict(Counter(r['status'] for r in subset)),
                                'parsed_records':dict(Counter(a['source_class'] for r in subset for a in r['annotations']))}
    write_json(output/'summary.json',summary)
    write_json(output/'settings.json',config)
    write_json(output/'run_info.json',{'script_sha256':digest(Path(__file__)), 'python':sys.version, 'd4_cli_order':d4_order, 'class_orders':{did:next(r['class_order'] for r in records if r['source']==did) for did in source_ids}, 'source_tree_read_only':True})
    return records, summary


def export_reviewed(root, records, config, output):
    require_review=config['datasets']['annotation_policy'].get('require_domain_and_completeness_review', True)
    statuses={'approved'} if require_review else {'approved', 'provisional'}
    approved=[r for r in records if r['status'] in statuses]
    if not approved: raise ValueError('No approved images: review queue was saved; no training dataset exported')
    quality='provisional' if any(r['status']=='provisional' for r in approved) else 'reviewed'
    fractions=[config['split'][k] for k in ['train_pool','val','test']]
    if any(v <= 0 for v in fractions) or abs(sum(fractions)-1)>1e-9: raise ValueError('Invalid split fractions')
    splits=['train','val','test']; seed=config['split']['split_seed']
    groups=defaultdict(list)
    for r in approved: groups[r['group']].append(r)
    # Greedy multi-label group balancing: class presence, image counts and source.
    # Ratios are approximate because a related-image group is indivisible.
    classes=config['task']['core_classes']
    sources=sorted({r['source'] for r in approved}) if config['split'].get('stratify_by_source') else []
    features={}
    for gid,members in groups.items():
        features[gid]=np.array([len(members)]+[sum(any(a['target_class']==c for a in r['annotations']) for r in members) for c in classes]+[sum(r['source']==src for r in members) for src in sources],dtype=float)
    total=sum(features.values()); targets=np.array(fractions)[:,None]*total[None,:]
    current=np.zeros_like(targets); assignment={}
    order=sorted(groups,key=lambda gid:(-float(np.max(features[gid]/np.maximum(total,1))),hashlib.sha256(f'{seed}:{gid}'.encode()).hexdigest()))
    for gid in order:
        costs=[]
        for i in range(3):
            trial=current.copy(); trial[i]+=features[gid]
            costs.append(float(np.sum((trial-targets)**2/np.maximum(targets,1))))
        chosen=min(range(3),key=lambda i:(costs[i],i))
        current[chosen]+=features[gid]; assignment[gid]=splits[chosen]
    counts={sp:Counter(a['target_class'] for r in approved if assignment[r['group']]==sp for a in r['annotations']) for sp in splits}
    missing=[f'{sp}:{cls}' for sp in splits for cls in config['task']['core_classes'] if not counts[sp][cls]]
    if missing: raise ValueError('Export blocked: insufficient approved class coverage in '+', '.join(missing)+'. Review more independent groups; do not silently move related copies.')
    dest=output/'dataset'; dest.mkdir()
    exported=[]
    for sp in splits:
        (dest/sp/'images').mkdir(parents=True); (dest/sp/'labels').mkdir()
    for r in approved:
        sp=assignment[r['group']]; src=root/r['image']; stem=r['export_stem']
        if digest(src) != r['image_sha256'] or digest(Path(r['effective_label'])) != r['label_sha256']:
            raise ValueError(f'Input changed during export: {src}; discard this incomplete run')
        shutil.copy2(src,dest/sp/'images'/(stem+src.suffix.lower()))
        label=''.join(f"{a['target_id']} "+' '.join(f'{v:.12g}' for v in a['box'])+'\n' for a in r['annotations'])
        (dest/sp/'labels'/(stem+'.txt')).write_text(label)
        exported.append({'image':r['image'],'source':r['source'],'group':r['group'],'split':sp,'export_stem':stem,
                         'review_status':r['status'],'image_sha256':r['image_sha256'],
                         'export_label_sha256':digest(dest/sp/'labels'/(stem+'.txt'))})
    (dest/'data.yaml').write_text(yaml.safe_dump({'path':str(dest.resolve()),'train':'train/images','val':'val/images','test':'test/images','names':config['task']['core_classes']},sort_keys=False))
    write_json(dest/'split_manifest.json',exported)
    write_json(dest/'counts.json',{'quality':quality,'manual_review_required':require_review,
                                'images':dict(Counter(r['split'] for r in exported)),'annotations':counts,
                                'source_by_split':{sp:dict(Counter(r['source'] for r in exported if r['split']==sp)) for sp in splits},
                                'split_method':'seeded greedy grouped balance of image counts, class presence and configured source strata; approximate ratios',
                                'frozen_run':'Use this exported run for the entire experiment; rescanning changed membership can change group IDs and splits.'})


def contact_sheets(root, records, output):
    directory=output/'contact_sheets'; directory.mkdir()
    # All records retain their manifest order, including exclusions; review CSV is the authoritative key.
    for start in range(0,len(records),25):
        page=Image.new('RGB',(1250,1400),'#202020'); draw=ImageDraw.Draw(page)
        for k,r in enumerate(records[start:start+25]):
            x,y=(k%5)*250,(k//5)*280
            try:
                with Image.open(root/r['image']) as im: tile=ImageOps.contain(im.convert('RGB'),(246,238))
                td=ImageDraw.Draw(tile); w,h=tile.size
                for a in r['annotations']:
                    cx,cy,bw,bh=a['box']; td.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline='red',width=1)
                page.paste(tile,(x,y+38))
            except OSError: pass
            draw.text((x+2,y+2),f"row {start+k+2} {r['source']} {r['status'][:27]}",fill='white')
            draw.text((x+2,y+18),Path(r['image']).name[:36],fill='white')
        page.save(directory/f'batch_{start//25+1:03}.jpg',quality=90)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path,default=Path('config/experiment.yaml'))
    parser.add_argument('--output',type=Path,required=True,help='New output directory; never overwrite a run')
    parser.add_argument('--reviews',type=Path,help='Completed review.csv from a previous scan')
    parser.add_argument('--label-overrides',type=Path,help='Corrected source-format labels, mirroring data/raw/... relative paths')
    parser.add_argument('--d4-class-order',choices=['crack,dent','dent,crack'],help='Explicit numeric order: class 0,class 1')
    parser.add_argument('--export',action='store_true',help='Export eligible images under the configured review policy')
    parser.add_argument('--contact-sheets',action='store_true')
    args=parser.parse_args(argv); root=args.root.resolve()
    config=yaml.safe_load((root/args.config).read_text(encoding='utf-8-sig'))
    output=(root/args.output).resolve()
    raw=(root/'data/raw').resolve()
    if output.exists() or output==raw or output.is_relative_to(raw): parser.error('Use a new output directory outside data/raw')
    try:
        records,summary=scan(root,config,output,args.reviews,args.d4_class_order,args.label_overrides)
        if args.contact_sheets: contact_sheets(root,records,output)
        if args.export: export_reviewed(root,records,config,output)
    except (ValueError,OSError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr); return 2
    print(json.dumps(summary,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
