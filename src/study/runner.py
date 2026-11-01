"""Immutable study plans and restartable detector jobs; final test evaluation is explicit."""
import copy
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import yaml

from detection_pipeline import ROOT,load_config,resolve,sha256,write_json,versions
from .data import inventory,nested_subsets,identity,bounded_sample,gan_smoke_sample,runtime_yaml
from .augmentation import classical,approved_composites,ReviewRequired
from .metrics import evaluate
from training_progress import Progress
from .screening import settings_for


def code_identity():
    files=sorted(list((ROOT/'src/study').glob('*.py'))+list((ROOT/'src/augmentation/hqgan').glob('*.py'))+[ROOT/'src/detection_pipeline.py',ROOT/'src/training_progress.py'])
    return identity({p.relative_to(ROOT).as_posix():sha256(p) for p in files})


def plan(config_path,output,smoke=False,gan_steps=None):
    config=yaml.safe_load(resolve(config_path).read_text()); experiment=load_config(config['experiment'])
    config['screening'] = settings_for(config.get('screening'))
    if gan_steps is not None:
        if gan_steps<1: raise ValueError('gan_steps must be positive')
        config['gan']['steps']=gan_steps
    if experiment['task']['type']!='detection': raise ValueError('Detection only')
    if not config['seeds'] or len(set(config['seeds']))!=len(config['seeds']): raise ValueError('Unique seeds required')
    if not config['models'] or any(m['backend'] not in ('ultralytics','yolox') for m in config['models'].values()):
        raise ValueError('Unsupported/empty model grid')
    if not config['arms'] or any(a['kind'] not in ('baseline','classical','hqgan') for a in config['arms'].values()):
        raise ValueError('Unsupported/empty detection arms; CHPDNet belongs to segmentation')
    if smoke:
        config=copy.deepcopy(config)
        config['fractions']=[.05,.1]; config['seeds']=[0]
        config['training'].update(epochs=1,image_size=128,batch_size=2,cpu_threads=2)
        config['evaluation']['bootstrap_samples']=50
        config['gan'].update(steps=2,candidates_per_class=4)
    records,counts,names,split_hash=inventory(experiment['training']['data'],experiment)
    subsets={}
    for seed in config['seeds']:
        for subset in nested_subsets(records,config['fractions'],seed,len(names),config['min_instances_per_class']).values():
            key=f'seed{seed}_f{round(subset["fraction"]*10000):05d}'
            if smoke:
                sampler = gan_smoke_sample if any(a['kind']=='hqgan' for a in config['arms'].values()) else bounded_sample
                rows=sampler([r for r in records if r['export_stem'] in subset['stems']],9,len(names))
                subset.update(stems=[r['export_stem'] for r in rows],images=len(rows),
                    actual_fraction=len(rows)/sum(r['split']=='train' for r in records),
                    groups=len({r['group'] for r in rows}),
                    identity=identity([{k:r[k] for k in ('export_stem','image_sha256','export_label_sha256')} for r in rows]),
                    instances_per_class=[sum(t[0]==c for r in rows for t in r['targets']) for c in range(len(names))])
            subsets[key]=subset
    jobs=[{'id':f'{key}__{arm}__{model}','subset':key,'arm':arm,'model':model,'seed':subsets[key]['seed'],
           'fraction':subsets[key]['fraction']} for key,arm,model in itertools.product(subsets,config['arms'],config['models'])]
    out=resolve(output); out.mkdir(parents=True,exist_ok=False)
    payload={'config':config,'experiment':experiment,'source_data':str(resolve(experiment['training']['data'])),
             'counts':counts,'names':names,'split_manifest_sha256':split_hash,'subsets':subsets,
             'jobs':jobs,'smoke':smoke,'code_identity':code_identity(),'environment':versions(),
             'platform':platform.platform(),'checkpoint_selection':'last epoch; validation is not used for cross-arm tuning',
             'records':records,'gan_settings':yaml.safe_load(resolve(config['gan']['config']).read_text()),
             'initial_weight_hashes':{n:sha256(resolve(s['weights'])) if resolve(s['weights']).is_file() else None for n,s in config['models'].items()}}
    payload['plan_identity']=identity(payload)
    write_json(out/'plan.json',payload)
    for key,subset in subsets.items():
        directory=out/'subsets'/key; directory.mkdir(parents=True)
        write_json(directory/'stems.json',subset['stems']); write_json(directory/'manifest.json',subset)
    write_json(out/'status.json',{j['id']:{'status':'pending'} for j in jobs})
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    return payload


def load_plan(directory,verify_data=True):
    directory=resolve(directory); payload=json.loads((directory/'plan.json').read_text())
    check=dict(payload); expected=check.pop('plan_identity')
    if identity(check)!=expected: raise ValueError('Study plan changed; create a new plan')
    if code_identity()!=payload['code_identity']: raise ValueError('Study implementation changed; create a fresh plan to avoid mixing code versions')
    if verify_data:
        _,_,_,split_hash=inventory(payload['source_data'],payload['experiment'])
        if split_hash!=payload['split_manifest_sha256']: raise ValueError('Frozen input split changed')
        for name,spec in payload['config']['models'].items():
            expected=payload['initial_weight_hashes'][name]
            if expected is None: raise ValueError('Weights were missing when planned; run setup and create a fresh plan')
            if sha256(resolve(spec['weights']))!=expected: raise ValueError('Initial model weights changed')
    return payload


def subset_records(payload,key):
    stems=set(payload['subsets'][key]['stems'])
    return [r for r in payload['records'] if r['split']=='train' and r['export_stem'] in stems]


def evaluation_records(payload,split):
    rows=[r for r in payload['records'] if r['split']==split]
    return bounded_sample(rows,9,len(payload['names'])) if payload['smoke'] else rows


def train_records(directory,payload,job):
    real=subset_records(payload,job['subset']); arm=payload['config']['arms'][job['arm']]
    if arm['kind']=='baseline': return real,0
    cache=directory/'augmentation'/job['subset']/job['arm']
    if arm['kind']=='classical':
        if (cache/'manifest.json').exists():
            augmented=json.loads((cache/'manifest.json').read_text())['records']
            for row in augmented:
                if sha256(row['image_path'])!=row['image_sha256'] or sha256(row['label_path'])!=row['export_label_sha256']:
                    raise ValueError('Cached augmentation changed')
        else:
            if cache.exists():
                # Preserve incomplete output and rebuild deterministically in a new attempt.
                attempt=1
                while cache.with_name(cache.name+f'_incomplete_{attempt}').exists(): attempt+=1
                cache.rename(cache.with_name(cache.name+f'_incomplete_{attempt}'))
            augmented=classical(real,arm,cache,job['seed'])
    else:
        review=directory/'gan'/job['subset']/'composites/review.json'
        if not review.exists(): raise ReviewRequired(f'Missing reviewed HQGAN composites: {review}')
        augmented=approved_composites(review,payload['subsets'][job['subset']]['identity'],real,payload['smoke'])
        count=math.ceil(len(real)*arm['synthetic_ratio'])
        # Same deterministic prefix for every ratio/model. Never silently recycle images.
        augmented=sorted(augmented,key=lambda r:identity([job['seed'],r['export_stem']]))
        if len(augmented)<count: raise ReviewRequired(f'Need {count} approved composites; have {len(augmented)}')
        augmented=augmented[:count]
    return real+augmented,len(augmented)


def save_preview(records,predictions,path,confidence,limit):
    from PIL import Image,ImageDraw
    path.mkdir(exist_ok=True)
    for i,row in enumerate(records[:limit]):
        with Image.open(row['image_path']) as source: image=source.convert('RGB')
        draw=ImageDraw.Draw(image)
        for c,x1,y1,x2,y2 in row['targets']:
            draw.rectangle((x1*image.width,y1*image.height,x2*image.width,y2*image.height),outline='lime',width=2)
        for c,score,x1,y1,x2,y2 in predictions[row['export_stem']]:
            if score<confidence: continue
            draw.rectangle((x1*image.width,y1*image.height,x2*image.width,y2*image.height),outline='red',width=2)
            draw.text((x1*image.width,y1*image.height),f'{c} {score:.2f}',fill='red')
        image.save(path/f'{i:03d}.jpg')


def run(directory,device='auto',models=None,arms=None,limit=None,retry=False,finalize=False):
    from .backends import Detector
    directory=resolve(directory)
    print('Checking frozen study, dataset and model hashes...',flush=True)
    payload=load_plan(directory)
    status=json.loads((directory/'status.json').read_text()); attempted=0
    if finalize:
        if payload['smoke']: raise ValueError('Smoke studies cannot evaluate the reserved test set')
        # Lock predeclared plan before touching test data; validation outcomes cannot alter the plan.
        write_json(directory/'test_lock.json',{'plan_identity':payload['plan_identity'],'code_identity':payload['code_identity'],
                    'note':'Test results must not be used to tune this frozen study.'})
    split='test' if finalize else 'val'
    eligible=[j for j in payload['jobs']
              if (not models or j['model'] in models) and (not arms or j['arm'] in arms)
              and not (status[j['id']].get('status')=='complete' and (directory/'runs'/j['id']/split/'metrics.json').exists())
              and (status[j['id']].get('status')!='failed' or retry)
              and (not finalize or (directory/'runs'/j['id']/'val/metrics.json').exists())]
    queue=Progress('Study queue - rough ETA',min(len(eligible),limit) if limit is not None else len(eligible),unit='attempts')
    for job in eligible:
        if models and job['model'] not in models: continue
        if arms and job['arm'] not in arms: continue
        previous=status[job['id']]; out=directory/'runs'/job['id']
        split='test' if finalize else 'val'; target=out/split/'metrics.json'
        if previous.get('status')=='complete' and target.exists(): continue
        if previous.get('status')=='failed' and not retry: continue
        if limit is not None and attempted>=limit: break
        if finalize and not (out/'val/metrics.json').exists(): continue
        attempted+=1
        print(f'\nStarting job {attempted}/{queue.total}: {job["id"]} | preparing training data',flush=True)
        try:
            config=payload['config']; rows,n_aug=train_records(directory,payload,job)
            out.mkdir(parents=True,exist_ok=True)
            training_identity=identity([{k:r[k] for k in ('export_stem','image_sha256','export_label_sha256')} for r in rows])
            job_manifest={'job':job,'plan_identity':payload['plan_identity'],'training_identity':training_identity,
                         'real_images':len(rows)-n_aug,'augmented_images':n_aug,'smoke':payload['smoke'],
                         'quality':payload['counts']['quality'],'subset':payload['subsets'][job['subset']]}
            if (out/'job.json').exists() and json.loads((out/'job.json').read_text())!=job_manifest:
                raise ValueError('Training inputs changed after job started; create a new study')
            write_json(out/'job.json',job_manifest)
            detector=Detector(config['models'][job['model']],payload['names'],config['training'],config['yolox'],device)
            train_complete=out/'training/train_complete.json'
            if train_complete.exists():
                meta=json.loads(train_complete.read_text())
                if sha256(meta['checkpoint'])!=meta['checkpoint_sha256']: raise ValueError('Trained checkpoint changed')
                detector.load(meta['checkpoint'])
            else:
                if finalize: raise ValueError('Cannot finalize an untrained cell')
                data_yaml=runtime_yaml(out/'data',rows,evaluation_records(payload,'val'),[],payload['names'])
                status[job['id']]={'status':'running'}; write_json(directory/'status.json',status)
                meta=detector.train(rows,evaluation_records(payload,'val'),data_yaml,out/'training',job['seed'],payload['smoke'])
            eval_rows=evaluation_records(payload,split)
            predictions,timings=detector.predict(eval_rows,config['evaluation'])
            print('Computing detection metrics and bootstrap intervals...',flush=True)
            metrics=evaluate(eval_rows,predictions,payload['names'],config['evaluation'],timings)
            metrics.update(job_manifest); metrics.update(split=split,training=meta,evaluation_protocol=config['evaluation'],
                evaluation_manifest_sha256=identity([{k:r[k] for k in ('export_stem','image_sha256','export_label_sha256')} for r in eval_rows]))
            target.parent.mkdir(exist_ok=True)
            write_json(target.parent/'predictions.json',predictions)
            write_json(target,metrics)
            save_preview(eval_rows,predictions,target.parent/'previews',config['evaluation']['operating_confidence'],config['evaluation']['preview_count'])
            status[job['id']]={'status':'complete','validation_metrics':str(out/'val/metrics.json'),
                               'test_metrics':str(target) if finalize else previous.get('test_metrics')}
            print(f'Complete {job["id"]} {split}: mAP50={metrics["map50"]:.4f}',flush=True)
        except ReviewRequired as exc:
            status[job['id']]={'status':'blocked_review','reason':str(exc)}
            print(f'Review required: {job["id"]}: {exc}',flush=True)
        except Exception as exc:
            status[job['id']]={'status':'complete' if finalize else 'failed',
                              'test_status':'failed' if finalize else None,'reason':str(exc),'traceback':traceback.format_exc()}
            print(f'Failed {job["id"]}: {exc}',flush=True)
        finally:
            write_json(directory/'status.json',status)
            queue.update(attempted,f'{status[job["id"]]["status"]}; includes selected attempts only, not GAN training/review',force=True)
    return status
