"""Ultralytics and official YOLOX model adapters with a common prediction format."""
import copy
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from detection_pipeline import resolve, sha256, torch_device, write_json
from training_progress import Progress, attach_ultralytics_progress
from .metrics import coco_metrics


def seed_all(seed,threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def synchronize(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def load_yolox(settings,n_classes):
    repo=resolve(settings['repository'])
    if not (repo/'yolox').is_dir(): raise RuntimeError('YOLOX source missing: run scripts/setup_detection_study.py')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=settings['commit']: raise ValueError('YOLOX checkout differs from pinned commit')
    if subprocess.check_output(['git','-C',str(repo),'diff','--name-only','--','yolox'],text=True).strip():
        raise ValueError('Official YOLOX model source has local edits')
    sys.path.insert(0,str(repo))
    from yolox.exp import get_exp
    exp=get_exp(None,'yolox-s'); exp.num_classes=n_classes
    return exp.get_model()


class Detector:
    def __init__(self,spec,names,settings,yolox_settings,device):
        self.spec,self.names,self.settings,self.yolox_settings=spec,names,settings,yolox_settings
        self.device=torch_device(device); self.model=None

    def train(self,train_rows,val_rows,data_yaml,output,seed,smoke=False):
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        timing=output/'timing.json'
        elapsed=json.loads(timing.read_text())['elapsed_sec'] if timing.exists() else 0.0
        tick=time.perf_counter()
        try:
            result=self._train(train_rows,val_rows,data_yaml,output,seed,smoke)
        finally:
            elapsed+=time.perf_counter()-tick
            write_json(timing,{'elapsed_sec':elapsed,'note':'Sum of observed training invocations; abrupt process termination can lose the final segment.'})
        result['train_time_sec']=elapsed
        write_json(output/'train_complete.json',result)
        return result

    def _train(self,train_rows,val_rows,data_yaml,output,seed,smoke=False):
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        seed_all(seed,self.settings['cpu_threads'])
        started=time.perf_counter()
        if self.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(self.device)
        if self.spec['backend']=='ultralytics':
            from ultralytics import YOLO
            checkpoint=output/'weights/last.pt'
            if checkpoint.exists():
                model=YOLO(str(checkpoint))
                attach_ultralytics_progress(model)
                # Ultralytics strips optimizer/epoch after successful completion.
                # Recover a crash between that operation and our completion marker.
                saved=model.ckpt
                finished=saved.get('epoch') == -1 and saved.get('optimizer') is None
                if not finished: model.train(resume=True,device=self.device)
            else:
                model=YOLO(str(resolve(self.spec['weights'])))
                attach_ultralytics_progress(model)
                model.train(data=str(data_yaml),epochs=self.settings['epochs'],imgsz=self.settings['image_size'],
                    batch=self.settings['batch_size'],workers=self.settings['workers'],
                    project=str(output.parent),name=output.name,exist_ok=True,device=self.device,
                    seed=seed,deterministic=True,optimizer=self.settings['optimizer'],lr0=self.settings['learning_rate'],
                    # Mixed precision is requested from the recipe; Ultralytics itself turns it off
                    # on the CPU and on GPUs where its check finds it unreliable.
                    patience=0,amp=bool(self.settings.get('amp',False)),cache=False,plots=True,save=True,val=True,close_mosaic=0,
                    hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
                    flipud=0,fliplr=0,mosaic=0,mixup=0,cutmix=0,copy_paste=0,multi_scale=0)
            self.model=YOLO(str(checkpoint))
            parameters=sum(p.numel() for p in self.model.model.parameters())
            initial_weights=self.spec['weights']
            init_hash=sha256(resolve(initial_weights)) if resolve(initial_weights).is_file() else None
        elif self.spec['backend']=='yolox':
            checkpoint,parameters,init_hash=self._train_yolox(train_rows,val_rows,output,seed)
        else: raise ValueError('Unsupported detector backend')
        metadata={'train_time_sec':time.perf_counter()-started,'parameters':parameters,
            'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha256(checkpoint),
            'checkpoint_bytes':checkpoint.stat().st_size,'selection':'last completed epoch, fixed epoch budget',
            'initial_weights_sha256':init_hash,'device':self.device,'smoke':smoke,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.startswith('cuda') else None,
            'torch_version':str(torch.__version__),'training_recipe':self.settings,
            'backend':self.spec['backend'],'yolox_commit':self.yolox_settings['commit'] if self.spec['backend']=='yolox' else None}
        write_json(output/'train_complete.json',metadata)
        return metadata

    def _preprocess(self,row):
        image=cv2.imread(row['image_path'])
        if image is None: raise ValueError('Unreadable image')
        h,w=image.shape[:2]; size=self.settings['image_size']; ratio=min(size/h,size/w)
        canvas=np.full((size,size,3),114,dtype=np.uint8)
        resized=cv2.resize(image,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        return torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1))).float(),ratio

    def _train_yolox(self,train_rows,val_rows,output,seed):
        model=load_yolox(self.yolox_settings,len(self.names)).to(self.device)
        self.model=model
        weights=resolve(self.spec['weights'])
        if not weights.is_file(): raise RuntimeError('YOLOX pretrained checkpoint missing; run setup_detection_study.py')
        # Official release is a trusted project dependency. Only matching shapes transfer.
        pretrained=torch.load(weights,map_location='cpu',weights_only=False)['model']
        state=model.state_dict(); matched={k:v for k,v in pretrained.items() if k in state and state[k].shape==v.shape}
        if not matched: raise ValueError('No compatible pretrained YOLOX parameters')
        model.load_state_dict(matched,strict=False)
        write_json(output/'initialization.json',{'transferred_tensors':len(matched),'total_tensors':len(state),
                  'weights_sha256':sha256(weights),'recipe':'Official YOLOX-S model/loss; local fixed-size AdamW trainer, no EMA/mosaic/multiscale'})
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.settings['learning_rate'],weight_decay=.0005)
        checkpoint=output/'last.pt'; start=0; rng=np.random.default_rng(seed)
        if checkpoint.exists():
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
            start=ckpt['epoch']; rng.bit_generator.state=ckpt['numpy_rng']; torch.set_rng_state(ckpt['torch_rng'])
            if self.device.startswith('cuda') and ckpt.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        max_targets=max(len(r['targets']) for r in train_rows)
        history=output/'results.csv'
        batches=math.ceil(len(train_rows)/self.settings['batch_size'])
        progress=Progress('YOLOX training',self.settings['epochs']*batches,initial=start*batches,unit='batches')
        with history.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','loss','val_map50','val_map50_95','epoch_sec'])
            if start==0: writer.writeheader()
            for epoch in range(start,self.settings['epochs']):
                tick=time.perf_counter(); model.train(); losses=[]
                indices=rng.permutation(len(train_rows)); batch=self.settings['batch_size']
                for offset in range(0,len(indices),batch):
                    chosen=indices[offset:offset+batch]
                    if len(chosen)==1: chosen=np.concatenate([chosen,indices[:1]]) # batch norm valid; recorded recipe
                    images=[]; labels=[]
                    for index in chosen:
                        row=train_rows[int(index)]; image,ratio=self._preprocess(row); images.append(image)
                        target=torch.zeros(max_targets,5)
                        for j,(c,x1,y1,x2,y2) in enumerate(row['targets']):
                            target[j]=torch.tensor([c,(x1+x2)/2*row['width']*ratio,(y1+y2)/2*row['height']*ratio,
                                (x2-x1)*row['width']*ratio,(y2-y1)*row['height']*ratio])
                        labels.append(target)
                    optimizer.zero_grad(set_to_none=True)
                    loss=model(torch.stack(images).to(self.device),torch.stack(labels).to(self.device))['total_loss']
                    if not torch.isfinite(loss): raise ValueError('Non-finite YOLOX loss')
                    loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                    progress.update(epoch*batches+offset//batch+1,
                                    f'epoch {epoch+1}/{self.settings["epochs"]} loss={losses[-1]:.4f}')
                print(f'YOLOX epoch {epoch+1}: validating...',flush=True)
                eval_settings={'confidence_floor':.001,'nms_iou':.7,'max_detections':300}
                predictions,_=self.predict(val_rows,eval_settings)
                metrics=coco_metrics(val_rows,predictions,self.names)
                writer.writerow({'epoch':epoch+1,'loss':np.mean(losses),'val_map50':metrics['map50'],
                                 'val_map50_95':metrics['map50_95'],'epoch_sec':time.perf_counter()-tick}); f.flush()
                temp=output/'last.tmp'
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                            'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else None},temp)
                temp.replace(checkpoint)
                progress.update((epoch+1)*batches,'validation/checkpoint complete',force=True)
                print(f'YOLOX epoch {epoch+1}: loss={np.mean(losses):.4f} val mAP50={metrics["map50"]:.4f}',flush=True)
        return checkpoint,sum(p.numel() for p in model.parameters()),sha256(weights)

    def load(self,checkpoint):
        if self.spec['backend']=='ultralytics':
            from ultralytics import YOLO
            self.model=YOLO(str(checkpoint))
            if [self.model.names[i] for i in range(len(self.names))]!=self.names:
                raise ValueError('Checkpoint class mismatch')
        else:
            self.model=load_yolox(self.yolox_settings,len(self.names)).to(self.device)
            self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])

    def predict(self,records,settings):
        if not records: raise ValueError('Empty evaluation split')
        predictions={}; timings=[]
        progress=Progress('Evaluation inference',len(records),unit='images')
        if self.spec['backend']=='ultralytics':
            kwargs={'imgsz':self.settings['image_size'],'device':self.device,'conf':settings['confidence_floor'],
                    'iou':settings['nms_iou'],'max_det':settings['max_detections'],'verbose':False,'save':False}
            # Do not override YOLO26's native head choice; checkpoint/installed version are recorded.
            self.model.predict(records[0]['image_path'],**kwargs)
            for row in records:
                synchronize(self.device); started=time.perf_counter()
                result=self.model.predict(row['image_path'],**kwargs)[0]
                synchronize(self.device); timings.append((time.perf_counter()-started)*1000)
                ps=[[int(c),float(s),*np.clip(box,0,1).tolist()] for c,s,box in zip(
                    result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.xyxyn.cpu().numpy())]
                predictions[row['export_stem']]=ps
                progress.update(len(predictions))
        else:
            from yolox.utils import postprocess
            self.model.eval()
            with torch.no_grad():
                image,_=self._preprocess(records[0]); self.model(image.unsqueeze(0).to(self.device))
                for row in records:
                    synchronize(self.device); started=time.perf_counter()
                    image,ratio=self._preprocess(row)
                    output=self.model(image.unsqueeze(0).to(self.device))
                    output=postprocess(output,len(self.names),settings['confidence_floor'],settings['nms_iou'],class_agnostic=False)[0]
                    ps=[]
                    if output is not None:
                        for x1,y1,x2,y2,obj,cls,c in output.cpu().tolist():
                            box=np.clip([x1/ratio/row['width'],y1/ratio/row['height'],x2/ratio/row['width'],y2/ratio/row['height']],0,1)
                            ps.append([int(c),obj*cls,*box.tolist()])
                    ps=sorted(ps,key=lambda p:-p[1])[:settings['max_detections']]
                    synchronize(self.device); timings.append((time.perf_counter()-started)*1000)
                    predictions[row['export_stem']]=ps
                    progress.update(len(predictions))
        return predictions,timings
