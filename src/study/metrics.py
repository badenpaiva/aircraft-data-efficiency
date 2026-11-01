"""One backend-independent COCO evaluator plus fixed operating point diagnostics."""
from collections import defaultdict
import contextlib
import io

import numpy as np

from detection_pipeline import match_counts


def iou(a,b):
    a,b = np.asarray(a),np.asarray(b)
    area = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union = np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-area
    return float(area/union) if union else 0.0


def coco_metrics(records, predictions, names, max_detections=300):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset = {'info':{},'images':[],'annotations':[],
               'categories':[{'id':i+1,'name':n} for i,n in enumerate(names)]}
    detections = []; aid = 1
    for image_id,r in enumerate(records,1):
        w,h = r['width'],r['height']
        dataset['images'].append({'id':image_id,'width':w,'height':h,'file_name':r['export_stem']})
        for cid,x1,y1,x2,y2 in r['targets']:
            box = [x1*w,y1*h,(x2-x1)*w,(y2-y1)*h]
            dataset['annotations'].append({'id':aid,'image_id':image_id,'category_id':cid+1,
                'bbox':box,'area':box[2]*box[3],'iscrowd':0}); aid += 1
        for cid,confidence,x1,y1,x2,y2 in predictions.get(r['export_stem'],[]):
            detections.append({'image_id':image_id,'category_id':int(cid)+1,'score':float(confidence),
                'bbox':[x1*w,y1*h,max(0,x2-x1)*w,max(0,y2-y1)*h]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = dataset; gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO(); dt.dataset = {**dataset,'annotations':[]}; dt.createIndex()
        ev = COCOeval(gt,dt,'bbox'); ev.params.maxDets = [1,10,max_detections]
        ev.evaluate(); ev.accumulate()
    precision = ev.eval['precision']  # IoU, recall, class, area, maxDets
    recall = ev.eval['recall']
    def mean_valid(x):
        x = x[x >= 0]
        return float(x.mean()) if len(x) else None
    report = {'map50_95':mean_valid(precision[:,:,:,0,-1]),
              'map50':mean_valid(precision[0,:,:,0,-1]),
              'map75':mean_valid(precision[5,:,:,0,-1]),
              'ar1':mean_valid(recall[:,:,0,0]),'ar10':mean_valid(recall[:,:,0,1]),
              'ar_max':mean_valid(recall[:,:,0,-1]),
              **{f'ap_{name}':mean_valid(precision[:,:,:,i,-1]) for i,name in enumerate(['all','small','medium','large'])},
              'per_class':[{'name':name,'ap50':mean_valid(precision[0,:,i,0,-1]),
                            'ap50_95':mean_valid(precision[:,:,i,0,-1]),
                            'ar_max':mean_valid(recall[:,i,0,-1])} for i,name in enumerate(names)],
              'pr_curves':{name:{'recall':ev.params.recThrs.tolist(),
                                  'precision_iou50':precision[0,:,i,0,-1].tolist()} for i,name in enumerate(names)},
              'evaluator':'pycocotools COCOeval bbox; no crowd; area measured at original image resolution',
              'max_detections':max_detections}
    return report


def rates(total):
    tp,fp,fn = [total[k] for k in ('tp','fp','fn')]
    return {**total,'precision':tp/(tp+fp) if tp+fp else 0.0,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def fixed_metrics(records, predictions, names, confidence, matching_iou):
    totals = [{'tp':0,'fp':0,'fn':0} for _ in names]; per_image = []
    confusion = np.zeros((len(names)+1,len(names)+1),dtype=int)
    for r in records:
        preds = sorted([p for p in predictions.get(r['export_stem'],[]) if p[1] >= confidence],key=lambda p:-p[1])
        counts = match_counts(preds,r['targets'],len(names),matching_iou)
        for a,b in zip(totals,counts):
            for key in a: a[key] += b[key]
        per_image.append({'stem':r['export_stem'],'group':r['group'],'counts':counts,
                          'targets':len(r['targets']),'predictions':len(preds)})
        # Class-agnostic spatial matching for a diagnostic class confusion matrix.
        used = set()
        for p in preds:
            options = [(iou(p[2:],t[1:]),j) for j,t in enumerate(r['targets']) if j not in used]
            best = max(options,default=(0,-1))
            if best[0] >= matching_iou:
                used.add(best[1]); confusion[int(r['targets'][best[1]][0]),int(p[0])] += 1
            else: confusion[-1,int(p[0])] += 1
        for j,t in enumerate(r['targets']):
            if j not in used: confusion[int(t[0]),-1] += 1
    per_class = [{'name':n,**rates(t)} for n,t in zip(names,totals)]
    return {'confidence':confidence,'iou':matching_iou,'per_class':per_class,
            'micro':rates({k:sum(t[k] for t in totals) for k in ('tp','fp','fn')}),
            'macro_f1':float(np.mean([x['f1'] for x in per_class])),
            'confusion_matrix':confusion.tolist(),'confusion_labels':names+['background'],
            'confusion_orientation':'rows=true, columns=predicted; class-agnostic spatial matching',
            'per_image':per_image}


def group_bootstrap(per_image,names,repetitions=2000,seed=1234):
    grouped = defaultdict(lambda:np.zeros((len(names),3)))
    for image in per_image:
        grouped[image['group']] += [[c[k] for k in ('tp','fp','fn')] for c in image['counts']]
    if len(grouped) < 2 or repetitions < 1:
        return {'unit':'related-image group','groups':len(grouped),'intervals':None}
    array = np.stack(list(grouped.values())); rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        counts = array[rng.integers(len(array),size=len(array))].sum(0)
        values.append([[t/(t+f) if t+f else 0.0, t/(t+n) if t+n else np.nan,
                        2*t/(2*t+f+n) if 2*t+f+n else 0.0] for t,f,n in counts])
    samples = np.asarray(values)
    intervals = {}
    for i,name in enumerate(names):
        intervals[name] = {}
        for j,metric in enumerate(('precision','recall','f1')):
            valid = samples[:,i,j][np.isfinite(samples[:,i,j])]
            intervals[name][metric] = np.quantile(valid,[.025,.975]).tolist() if len(valid) else None
    return {'unit':'related-image group','groups':len(grouped),'repetitions':repetitions,
            'seed':seed,'confidence_level':.95,'method':'percentile bootstrap of pooled counts',
            'intervals':intervals}


def evaluate(records,predictions,names,settings,timings=None):
    if set(predictions) != {r['export_stem'] for r in records}:
        raise ValueError('Prediction inventory differs from evaluation images')
    for ps in predictions.values():
        for p in ps:
            if len(p)!=6 or not np.isfinite(p).all() or int(p[0])!=p[0] or not 0 <= p[0] < len(names) or not 0<=p[1]<=1 or not (0<=p[2]<=p[4]<=1 and 0<=p[3]<=p[5]<=1):
                raise ValueError('Invalid prediction coordinates/class/confidence')
    report = coco_metrics(records,predictions,names,settings['max_detections'])
    fixed = fixed_metrics(records,predictions,names,settings['operating_confidence'],settings['matching_iou'])
    report.update(images=len(records),groups=len({r['group'] for r in records}),
                  fixed_operating_point=fixed,macro_f1=fixed['macro_f1'],
                  confidence_sweep=[{k:v for k,v in fixed_metrics(records,predictions,names,c,settings['matching_iou']).items() if k not in ('per_image','confusion_matrix')}
                                    for c in settings['confidence_sweep']],
                  group_bootstrap=group_bootstrap(fixed['per_image'],names,settings['bootstrap_samples'],settings['bootstrap_seed']))
    report['crack_recall'] = next((c['recall'] for c in fixed['per_class'] if c['name']=='crack'),None)
    if timings:
        a = np.array(timings)
        report['latency_ms'] = {'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'images_per_second':1000/float(a.mean()) if a.mean()>0 else None,
            'protocol':'batch=1, warmup excluded, wall time including decode/preprocess/network/postprocess; device synchronized'}
    return report
