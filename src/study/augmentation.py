"""Offline classical augmentation and explicitly reviewed Generation-Poisson blending."""
import json
from pathlib import Path

import cv2
import numpy as np

from detection_pipeline import sha256, write_json
from augmentation.hqgan.data import crop_settings, margin_pixels
from .data import identity,target_label
from .metrics import iou
from .screening import require_screening


class ReviewRequired(ValueError):
    pass


def export_image(output, stem, image, targets, source, provenance):
    output = Path(output)
    (output/'images').mkdir(parents=True,exist_ok=True); (output/'labels').mkdir(exist_ok=True)
    file = output/'images'/(stem+'.png'); label = output/'labels'/(stem+'.txt')
    if not cv2.imwrite(str(file),image): raise OSError(f'Cannot write {file}')
    label.write_text(target_label(targets))
    return {**source,'export_stem':stem,'image_path':str(file.resolve()),'label_path':str(label.resolve()),
        'image_sha256':sha256(file),'export_label_sha256':sha256(label),'targets':targets,
        'width':image.shape[1],'height':image.shape[0],'augmentation':provenance,'split':'train'}


def classical(records, settings, output, seed):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed); result = []
    for row in records:
        image = cv2.imread(row['image_path']); h,w = image.shape[:2]
        for repetition in range(settings['copies_per_real']):
            angle = float(rng.uniform(-settings['degrees'],settings['degrees']))
            flip = bool(rng.random() < settings['flip_probability'])
            matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
            out = cv2.warpAffine(image,matrix,(w,h),borderValue=(114,114,114))
            if flip: out = cv2.flip(out,1)
            targets = []
            for c,x1,y1,x2,y2 in row['targets']:
                corners = np.array([[x1*w,y1*h,1],[x2*w,y1*h,1],[x1*w,y2*h,1],[x2*w,y2*h,1]])@matrix.T
                lower,upper = corners.min(0),corners.max(0)
                box = np.clip([lower[0]/w,lower[1]/h,upper[0]/w,upper[1]/h],0,1)
                if flip: box[[0,2]] = 1-box[[2,0]]
                if box[2]>box[0] and box[3]>box[1]: targets.append([c,*box.tolist()])
            noise = rng.normal(0,settings['noise_std']*255,out.shape)
            out = np.clip(out.astype(float)+noise,0,255).astype(np.uint8)
            stem = 'classical_'+identity([row['export_stem'],seed,repetition,settings])[:24]
            result.append(export_image(output,stem,out,targets,row,{'method':'classical',
                'source_stem':row['export_stem'],'angle':angle,'flip':flip,'seed':seed,'settings':settings}))
    write_json(output/'manifest.json',{'status':'complete','records':result})
    return result


def validate_candidates(samples, subset_identity, allow_smoke=False):
    if samples.get('study_subset_identity') != subset_identity:
        raise ValueError('Synthetic candidate training subset mismatch')
    if samples.get('smoke') and not allow_smoke:
        raise ValueError('Smoke GAN candidates cannot enter a research run')
    approved = {}
    for row in samples['candidates']:
        if not row.get('approved'): continue
        require_screening(row, 'candidates', subset_identity)
        if not row.get('reviewer') or not row.get('defect_present') or not row.get('class_correct'):
            raise ValueError('Approved candidate lacks semantic review')
        box = row.get('defect_box_xyxy')
        if not isinstance(box,list) or len(box)!=4 or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
            raise ValueError('Reviewed defect bounding box required; full crop is not an inferred label')
        if sha256(row['path']) != row['sha256']: raise ValueError('Candidate changed after review')
        if sha256(row['checkpoint']) != row['checkpoint_sha256']: raise ValueError('Generator checkpoint changed')
        if row['id'] in approved: raise ValueError('Duplicate candidate id')
        approved[row['id']] = row
    if not approved: raise ReviewRequired('No semantically approved generated candidates; complete candidates_review.json')
    return approved


def paste_patch(background, patch, placement):
    """Poisson-blend a generated patch into its rectangle, undoing the crop's quarter turn."""
    x,y,w,h = placement['rectangle_xywh']
    if placement.get('quarter_turn'): patch = np.ascontiguousarray(np.rot90(patch,-1))  # clockwise
    patch = cv2.resize(patch,(w,h))
    mask = np.full((h,w),255,dtype=np.uint8); mask[[0,-1],:]=0; mask[:,[0,-1]]=0
    return cv2.seamlessClone(patch,background,mask,(x+w//2,y+h//2),cv2.NORMAL_CLONE)


def turned_box(box, quarter_turn):
    """A box drawn on the upright generated patch, as it lies after paste_patch's quarter turn."""
    x1,y1,x2,y2 = box
    return [1-y2,x1,1-y1,x2] if quarter_turn else [x1,y1,x2,y2]


def propose_placements(records, candidates, count, seed):
    """Nonoverlapping proposals; a person still confirms appropriate aircraft locations.

    The rectangle mirrors the crop rule of the GAN that made the candidate: a real box of the
    class supplies the core, the crop margin is added around it, and a tall core means the
    patch is turned back from its long-side-horizontal training orientation.
    """
    rng = np.random.default_rng(seed); values = list(candidates.values()); proposals = []
    for index in range(count):
        sample = values[index%len(values)]
        rule = crop_settings((sample.get('gan_identity') or {}).get('crop'))
        found = None
        for _ in range(500):
            background = records[int(rng.integers(len(records)))]
            width,height = background['width'],background['height']
            # Match a real training defect's relative dimensions, not arbitrary inference.
            boxes = [t for r in records for t in r['targets'] if t[0]==sample['class_id']]
            if not boxes: raise ValueError('No size exemplars for generated class')
            _,x1,y1,x2,y2 = boxes[int(rng.integers(len(boxes)))]
            w = max(8,min(width-4,round((x2-x1)*width)))
            h = max(8,min(height-4,round((y2-y1)*height)))
            px,py = margin_pixels(w,rule['margin']),margin_pixels(h,rule['margin'])
            if w+2*px > width-4: w = width-4-2*px; px = margin_pixels(w,rule['margin'])
            if h+2*py > height-4: h = height-4-2*py; py = margin_pixels(h,rule['margin'])
            rw,rh = w+2*px,h+2*py
            x = int(rng.integers(2,width-rw-1)); y = int(rng.integers(2,height-rh-1))
            bounds = [x/width,y/height,(x+rw)/width,(y+rh)/height]
            if all(iou(bounds,t[1:])==0 for t in background['targets']):
                found = {'id':f'blend_{index:06d}','background_stem':background['export_stem'],
                    'background_sha256':background['image_sha256'],'candidate_id':sample['id'],
                    'rectangle_xywh':[x,y,rw,rh],'core_box_xywh':[x+px,y+py,w,h],
                    'quarter_turn':bool(rule['canonical_orientation'] and h > w),
                    'approved':False,'reviewer':'','placement_valid':False}
                break
        if found is None: raise ReviewRequired('Cannot propose a nonoverlapping placement; supply a reviewed placement manually')
        proposals.append(found)
    return proposals


def blend(records, candidate_file, placement_file, subset_identity, output, allow_smoke=False):
    bundle = json.loads(Path(candidate_file).read_text())
    candidates = validate_candidates(bundle,subset_identity,allow_smoke)
    placements = json.loads(Path(placement_file).read_text())
    if placements['study_subset_identity'] != subset_identity: raise ValueError('Placement subset mismatch')
    backgrounds = {r['export_stem']:r for r in records}
    valid = [p for p in placements['placements'] if p.get('approved')]
    if not valid: raise ReviewRequired('No reviewed placements; complete placements_review.json')
    # Validate all entries before writing any output.
    seen = set()
    for p in valid:
        if p['id'] in seen: raise ValueError('Duplicate placement id')
        seen.add(p['id'])
        if not p.get('reviewer') or not p.get('placement_valid'): raise ValueError('Placement approval incomplete')
        bg = backgrounds.get(p['background_stem']); sample = candidates.get(p['candidate_id'])
        if bg is None or sample is None: raise ValueError('Unapproved candidate or background outside training subset')
        if bg['image_sha256'] != p['background_sha256'] or sha256(bg['image_path']) != p['background_sha256']:
            raise ValueError('Background changed after review')
        x,y,w,h = p['rectangle_xywh']
        if any(not isinstance(v,int) for v in (x,y,w,h)) or min(x,y)<1 or min(w,h)<4 or x+w>=bg['width'] or y+h>=bg['height']:
            raise ValueError('Invalid blend rectangle')
        bounds = [x/bg['width'],y/bg['height'],(x+w)/bg['width'],(y+h)/bg['height']]
        if any(iou(bounds,t[1:])>0 for t in bg['targets']): raise ValueError('Placement overlaps an existing labeled defect')
    output = Path(output); output.mkdir(parents=True,exist_ok=False); result=[]
    for p in valid:
        bg=backgrounds[p['background_stem']]; sample=candidates[p['candidate_id']]
        background=cv2.imread(bg['image_path']); source=cv2.imread(sample['path'])
        x,y,w,h=p['rectangle_xywh']
        out=paste_patch(background,source,p)
        bx1,by1,bx2,by2=turned_box(sample['defect_box_xyxy'],p.get('quarter_turn',False))
        new=[sample['class_id'],(x+bx1*w)/bg['width'],(y+by1*h)/bg['height'],
             (x+bx2*w)/bg['width'],(y+by2*h)/bg['height']]
        provenance={'method':'hqgan_poisson','candidate':sample,'placement':p,
                    'study_subset_identity':subset_identity,'smoke':bundle.get('smoke',False),
                    'quality':'provisional_composite_requires_final_visual_review'}
        stem='synthetic_'+identity([subset_identity,p,sample['sha256']])[:24]
        result.append(export_image(output,stem,out,bg['targets']+[new],bg,provenance))
    # Final review binds approval to the actual blended pixels and labels.
    write_json(output/'review.json',{'study_subset_identity':subset_identity,'smoke':bundle.get('smoke',False),
        'candidate_review_sha256':sha256(candidate_file),'placement_review_sha256':sha256(placement_file),
        'composites':[{'approved':False,'reviewer':'','labels_complete':False,'realistic':False,
                       'record':r} for r in result]})
    return result


def approved_composites(path,subset_identity,records,allow_smoke=False):
    bundle=json.loads(Path(path).read_text()); sources={r['export_stem'] for r in records}
    if bundle['study_subset_identity'] != subset_identity: raise ValueError('Composite subset mismatch')
    if bundle.get('smoke') and not allow_smoke: raise ValueError('Smoke composites cannot enter research runs')
    result=[]
    for row in bundle['composites']:
        if not row.get('approved'): continue
        require_screening(row, 'composites', subset_identity)
        if not row.get('reviewer') or not row.get('labels_complete') or not row.get('realistic'):
            raise ValueError('Composite review incomplete')
        r=row['record']
        if r['augmentation']['placement']['background_stem'] not in sources: raise ValueError('Composite background outside subset')
        if sha256(r['image_path']) != r['image_sha256'] or sha256(r['label_path']) != r['export_label_sha256']:
            raise ValueError('Reviewed composite changed')
        if r['split'] != 'train' or r['augmentation']['study_subset_identity'] != subset_identity:
            raise ValueError('Invalid composite provenance')
        result.append(r)
    if not result: raise ReviewRequired('No final reviewed composites')
    return result
