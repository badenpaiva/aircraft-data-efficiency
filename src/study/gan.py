"""Subset-bound GAN training, review queues and reviewed blending."""
import json
import copy
import math
from pathlib import Path

import torch
import yaml
from PIL import Image,ImageDraw

from detection_pipeline import resolve,sha256,write_json
from augmentation.hqgan.data import prepare_patches
from augmentation.hqgan.training import PatchDataset,train,save_samples
from augmentation.hqgan.model import Generator
from .runner import load_plan,subset_records
from .augmentation import validate_candidates,propose_placements,blend
from .backends import torch_device
from .screening import scan_bundle, require_screening, item_id


def train_gans(directory,device='auto',subset=None,limit=None):
    directory=resolve(directory); plan=load_plan(directory); done=0
    device=torch_device(device)
    for key,s in plan['subsets'].items():
        if subset and key!=subset: continue
        if limit is not None and done>=limit: break
        done+=1; root=directory/'gan'/key; root.mkdir(parents=True,exist_ok=True)
        patch_dir=root/'patches'
        print(f'GAN subset {key}: preparing/checking training patches...',flush=True)
        if not (patch_dir/'manifest.json').exists():
            prepare_patches(plan['source_data'],plan['experiment'],patch_dir,s['stems'])
        candidates=[]; checkpoints=[]
        for cid,name in enumerate(plan['names']):
            print(f'GAN subset {key}: class {cid+1}/{len(plan["names"])} ({name})',flush=True)
            settings=copy.deepcopy(plan['gan_settings'])
            settings['seed']=s['seed']
            if plan['smoke']: settings.update(resolution=64,width=1/32,batch_size=2,checkpoint_every=1)
            dataset=PatchDataset(patch_dir,name,settings['resolution'],plan['experiment'])
            output=root/name/'training'; ckpt=output/'last.pt'
            completion=output/'completion.json'
            imports=json.loads((root/'imports.json').read_text()) if (root/'imports.json').exists() else {}
            if name in imports:
                ckpt=Path(imports[name]['checkpoint'])
                if sha256(ckpt)!=imports[name]['sha256']: raise ValueError('Imported checkpoint changed')
            elif not completion.exists() or json.loads(completion.read_text())['status']!='complete':
                train(dataset,settings,output,plan['config']['gan']['steps'],device,plan['smoke'],resume=ckpt.exists())
            checkpoint=torch.load(ckpt,map_location='cpu',weights_only=True)
            if checkpoint['identity']!=dataset.identity or checkpoint['settings']!=settings:
                raise ValueError('Cached GAN differs from study subset/settings')
            sample_dir=root/name/'candidates'
            if not (sample_dir/'samples.json').exists():
                sample_dir.mkdir(parents=True,exist_ok=True)
                g=Generator(**checkpoint['architecture']).to(device); g.load_state_dict(checkpoint['generator'])
                images=save_samples(g,sample_dir,plan['config']['gan']['candidates_per_class'],s['seed']+1000,device)
                write_json(sample_dir/'samples.json',images)
            for i,row in enumerate(json.loads((sample_dir/'samples.json').read_text())):
                if sha256(sample_dir/row['image'])!=row['sha256']: raise ValueError('Generated candidate changed')
                candidates.append({'id':f'{name}_{i:05d}','path':str((sample_dir/row['image']).resolve()),
                    'sha256':row['sha256'],'class_id':cid,'class_name':name,'checkpoint':str(ckpt.resolve()),
                    'checkpoint_sha256':sha256(ckpt),'gan_identity':dataset.identity,
                    'approved':False,'reviewer':'','defect_present':False,'class_correct':False,'defect_box_xyxy':None})
            checkpoints.append({'class_name':name,'checkpoint_sha256':sha256(ckpt)})
        review=root/'candidates_review.json'
        if not review.exists():
            write_json(review,{'study_subset_identity':s['identity'],'smoke':plan['smoke'],
                               'candidates':candidates,'checkpoints':checkpoints})
            scan_bundle(review, 'candidates', plan['config']['screening'], plan['names'])
        elif 'screening_summary' not in json.loads(review.read_text()):
            scan_bundle(review, 'candidates', plan['config']['screening'], plan['names'])
        print(f'GAN candidates ready for semantic review: {review}',flush=True)


def proposals(directory,key):
    directory=resolve(directory); plan=load_plan(directory); s=plan['subsets'][key]
    root=directory/'gan'/key; review=root/'candidates_review.json'
    samples=json.loads(review.read_text()); approved=validate_candidates(samples,s['identity'],plan['smoke'])
    count=math.ceil(s['images']*max(a['synthetic_ratio'] for a in plan['config']['arms'].values() if a['kind']=='hqgan'))
    path=root/'placements_review.json'
    if path.exists(): raise ValueError('Placement review already exists; preserve existing decisions')
    records=subset_records(plan,key); placements=propose_placements(records,approved,count,s['seed'])
    write_json(path,{'study_subset_identity':s['identity'],'placements':placements})
    previews=root/'placement_previews'; previews.mkdir(exist_ok=True)
    index={r['export_stem']:r for r in records}
    for i,p in enumerate(placements):
        row=index[p['background_stem']]
        with Image.open(row['image_path']) as source: image=source.convert('RGB')
        draw=ImageDraw.Draw(image); x,y,w,h=p['rectangle_xywh']
        draw.rectangle([x,y,x+w,y+h],outline='red',width=3)
        draw.text((x,y),p['id']+' '+p['candidate_id'],fill='red')
        image.thumbnail((800,800)); image.save(previews/(p['id']+'.jpg'))
    return path


def composites(directory,key):
    directory=resolve(directory); plan=load_plan(directory); s=plan['subsets'][key]
    root=directory/'gan'/key
    result = blend(subset_records(plan,key),root/'candidates_review.json',root/'placements_review.json',
                   s['identity'],root/'composites',plan['smoke'])
    scan_bundle(root/'composites/review.json', 'composites', plan['config']['screening'], plan['names'])
    return result


def review_file(directory, key, stage):
    root = directory/'gan'/key
    return root / {'candidates':'candidates_review.json', 'placements':'placements_review.json',
                   'composites':'composites/review.json'}[stage]


def screen(directory, key, stage):
    directory=resolve(directory); plan=load_plan(directory)
    if key not in plan['subsets']: raise ValueError('Unknown subset')
    path=review_file(directory,key,stage)
    bundle=json.loads(path.read_text())
    if bundle['study_subset_identity'] != plan['subsets'][key]['identity']:
        raise ValueError('Review queue belongs to another subset')
    return scan_bundle(path,stage,plan['config']['screening'],plan['names'])


def review_item(directory, key, stage, identifier, decision, reviewer, note='', box=None,
                defect_present=False, class_correct=False, placement_valid=False,
                realistic=False, labels_complete=False):
    """Record a named human's explicit decision; never infer semantic approval."""
    directory=resolve(directory); plan=load_plan(directory); subset=plan['subsets'][key]
    path=review_file(directory,key,stage); bundle=json.loads(path.read_text())
    if bundle['study_subset_identity'] != subset['identity']: raise ValueError('Review subset mismatch')
    matches=[r for r in bundle[stage] if (r['id'] if stage=='placements' else item_id(r,stage))==identifier]
    if len(matches)!=1: raise ValueError('Review ID must identify exactly one item')
    if not reviewer.strip(): raise ValueError('A human reviewer name is required')
    if decision not in {'approve','reject'}: raise ValueError('Unknown decision')
    apply_review_decision(matches[0],stage,subset['identity'],plan['smoke'],decision,reviewer,note,
                          box,defect_present,class_correct,placement_valid,realistic,labels_complete)
    temporary=path.with_suffix('.json.tmp'); write_json(temporary,bundle); temporary.replace(path)
    return path


def apply_review_decision(row,stage,subset_identity,smoke,decision,reviewer,note='',box=None,
                          defect_present=False,class_correct=False,placement_valid=False,
                          realistic=False,labels_complete=False):
    """Shared validation for CLI and atomic CVAT batch imports."""
    if not reviewer.strip(): raise ValueError('A human reviewer name is required')
    if decision not in {'approve','reject'}: raise ValueError('Unknown decision')
    row.update(approved=False,reviewer=reviewer.strip(),review_notes=note)
    if decision=='approve':
        if stage!='placements':
            row['screening_override_reason']=note
            require_screening(row,stage,subset_identity)
        if stage=='candidates':
            if not defect_present or not class_correct: raise ValueError('Confirm defect presence and class')
            row.update(defect_present=True,class_correct=True,defect_box_xyxy=box,approved=True)
            validate_candidates({'study_subset_identity':subset_identity,'smoke':smoke,'candidates':[row]},subset_identity,smoke)
        elif stage=='placements':
            if not placement_valid: raise ValueError('Confirm suitable aircraft-surface placement')
            row.update(placement_valid=True,approved=True)
        else:
            if not realistic or not labels_complete: raise ValueError('Confirm realism and complete labels')
            r=row['record']
            if sha256(r['image_path'])!=r['image_sha256'] or sha256(r['label_path'])!=r['export_label_sha256']:
                raise ValueError('Composite changed; screen and review again')
            row.update(realistic=True,labels_complete=True,approved=True)


def import_checkpoint(directory,key,checkpoint_path):
    directory=resolve(directory); plan=load_plan(directory); s=plan['subsets'][key]
    checkpoint_path=resolve(checkpoint_path)
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if ckpt.get('method')!='hqgan_paper_guided_reconstruction_v1' or ckpt['smoke']!=plan['smoke']:
        raise ValueError('Incompatible GAN method or smoke status')
    settings=copy.deepcopy(plan['gan_settings']); settings['seed']=s['seed']
    if plan['smoke']: settings.update(resolution=64,width=1/32,batch_size=2,checkpoint_every=1)
    if ckpt['settings']!=settings or ckpt['step']!=plan['config']['gan']['steps']:
        raise ValueError('GAN settings/budget do not match plan; set the pilot budget when creating the plan')
    root=directory/'gan'/key; root.mkdir(parents=True,exist_ok=True); patches=root/'patches'
    if not (patches/'manifest.json').exists():
        prepare_patches(plan['source_data'],plan['experiment'],patches,s['stems'])
    dataset=PatchDataset(patches,ckpt['identity']['class_name'],settings['resolution'],plan['experiment'])
    if ckpt['identity']!=dataset.identity:
        raise ValueError('GAN was trained on different real data; importing would invalidate the scarcity comparison')
    file=root/'imports.json'; imports=json.loads(file.read_text()) if file.exists() else {}
    imports[dataset.identity['class_name']]={'checkpoint':str(checkpoint_path),'sha256':sha256(checkpoint_path)}
    write_json(file,imports); return file
