"""Auditable CSV/JSON tables, confidence intervals, paired tests and figures."""
from collections import defaultdict,Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from detection_pipeline import write_json


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def summary(values):
    a=np.array([v for v in values if v is not None],dtype=float)
    if not len(a): return {'n':0,'mean':None,'sd':None,'ci95_low':None,'ci95_high':None}
    mean=float(a.mean()); sd=float(a.std(ddof=1)) if len(a)>1 else None
    half=float(stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None
    return {'n':len(a),'mean':mean,'sd':sd,'ci95_low':mean-half if half is not None else None,
            'ci95_high':mean+half if half is not None else None}


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); adjusted=[None]*len(values); previous=0
    for rank,index in enumerate(order):
        previous=max(previous,min(1,values[index]*(len(values)-rank))); adjusted[index]=previous
    return adjusted


def paired_comparisons(rows,metrics,reference):
    grouped=defaultdict(dict)
    for row in rows: grouped[(row['split'],row['model'],row['fraction'],row['arm'])][row['seed']]=row
    results=[]
    for (split,model,fraction,arm),runs in grouped.items():
        if arm==reference: continue
        baseline=grouped.get((split,model,fraction,reference),{})
        for metric in metrics:
            seeds=sorted(s for s in runs.keys() & baseline.keys() if runs[s].get(metric) is not None and baseline[s].get(metric) is not None)
            differences=np.array([runs[s][metric]-baseline[s][metric] for s in seeds])
            row={'split':split,'model':model,'fraction':fraction,'arm':arm,'reference':reference,
                 'metric':metric,'paired_seeds':','.join(map(str,seeds)),'n_pairs':len(seeds),
                 'mean_difference':float(differences.mean()) if len(seeds) else None,
                 'paired_t_p':None,'wilcoxon_p':None,'cohens_dz':None,'rank_biserial':None,
                 'caution':'Three seeds provide weak statistical power; descriptive evidence only.'}
            if len(seeds)>=2:
                sd=differences.std(ddof=1)
                if sd>0:
                    row['paired_t_p']=float(stats.ttest_1samp(differences,0).pvalue)
                    row['cohens_dz']=float(differences.mean()/sd)
                elif np.all(differences==0): row['paired_t_p']=1.0
                # Constant nonzero differences make t/dz undefined, not infinite evidence.
                nz=differences[differences!=0]
                row['wilcoxon_p']=float(stats.wilcoxon(differences,zero_method='wilcox',method='auto').pvalue) if len(nz) else 1.0
                if len(nz):
                    ranks=stats.rankdata(np.abs(nz)); row['rank_biserial']=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
                else: row['rank_biserial']=0.0
            results.append(row)
    for field in ('paired_t_p','wilcoxon_p'):
        valid=[r for r in results if r[field] is not None]
        for r,value in zip(valid,holm([r[field] for r in valid])): r[field+'_holm']=value
    return results


def report(directory):
    directory=Path(directory).resolve(); plan=json.loads((directory/'plan.json').read_text())
    status=json.loads((directory/'status.json').read_text()); output=directory/'report'; output.mkdir(exist_ok=True)
    rows=[]; per_class=[]; per_image=[]; counts=[]; full=[]
    for job in plan['jobs']:
        counts.append({**job,**{k:v for k,v in status[job['id']].items() if k!='traceback'}})
        for split in ('val','test'):
            path=directory/'runs'/job['id']/split/'metrics.json'
            if not path.exists(): continue
            m=json.loads(path.read_text())
            if m['plan_identity']!=plan['plan_identity']: raise ValueError('Metric from another study')
            full.append(m)
            row={**job,'split':split,'smoke':m['smoke'],'quality':m['quality'],
                'real_images':m['real_images'],'augmented_images':m['augmented_images'],
                'actual_fraction':m['subset']['actual_fraction'],'train_groups':m['subset']['groups'],
                'eval_images':m['images'],'eval_groups':m['groups'],
                **{k:m.get(k) for k in ('map50','map50_95','map75','ap_small','ap_medium','ap_large','ar1','ar10','ar_max','macro_f1','crack_recall')},
                **{k:m['training'].get(k) for k in ('train_time_sec','parameters','checkpoint_bytes','peak_cuda_allocated_bytes','device')},
                **{'latency_'+k:v for k,v in m.get('latency_ms',{}).items() if k!='protocol'}}
            rows.append(row)
            for ap,operating in zip(m['per_class'],m['fixed_operating_point']['per_class']):
                per_class.append({**job,'split':split,'smoke':m['smoke'],**ap,**operating})
            for image in m['fixed_operating_point']['per_image']:
                for name,c in zip(plan['names'],image['counts']):
                    per_image.append({**job,'split':split,'class':name,'stem':image['stem'],'group':image['group'],**c})
    grouped=defaultdict(list)
    for row in rows: grouped[(row['split'],row['model'],row['arm'],row['fraction'])].append(row)
    summaries=[]
    metrics=plan['config']['statistics']['paired_metrics']+['map75','train_time_sec','latency_mean','parameters']
    for (split,model,arm,fraction),group in grouped.items():
        for metric in metrics:
            summaries.append({'split':split,'model':model,'arm':arm,'fraction':fraction,'metric':metric,
                              'smoke':plan['smoke'],**summary([r.get(metric) for r in group])})
    comparisons=paired_comparisons(rows,plan['config']['statistics']['paired_metrics'],plan['config']['statistics']['reference_arm'])
    write_csv(output/'runs.csv',rows); write_csv(output/'per_class.csv',per_class)
    write_csv(output/'per_image_counts.csv',per_image); write_csv(output/'coverage.csv',counts)
    write_csv(output/'summary.csv',summaries); write_csv(output/'paired_comparisons.csv',comparisons)
    write_json(output/'all_metrics.json',full)
    dataset=[]
    for split in ('train','val','test'):
        selected=[r for r in plan['records'] if r['split']==split]
        for cid,name in enumerate(plan['names']):
            sizes=[[(t[3]-t[1])*r['width'],(t[4]-t[2])*r['height']] for r in selected for t in r['targets'] if t[0]==cid]
            areas=[w*h for w,h in sizes]
            dataset.append({'split':split,'class':name,'images':len(selected),'groups':len({r['group'] for r in selected}),
                'class_images':sum(any(t[0]==cid for t in r['targets']) for r in selected),'instances':len(areas),
                'median_area_px':float(np.median(areas)) if areas else None,
                'quality':plan['counts']['quality']})
    write_csv(output/'dataset_inventory.csv',dataset)
    write_csv(output/'subsets.csv',[{'id':k,**{n:v for n,v in s.items() if n not in ('stems','instances_per_class')},
                                   **{name+'_instances':s['instances_per_class'][i] for i,name in enumerate(plan['names'])}}
                                  for k,s in plan['subsets'].items()])
    _plots(summaries,full,output,plan)
    coverage=Counter(s['status'] for s in status.values())
    lines=['# Detection study results', '',
        '**SMOKE CHECK ONLY: these scores are not research results.**' if plan['smoke'] else '**Provisional dataset and HQGAN reconstruction: not a certified paper reproduction.**',
        '',f"Plan: `{plan['plan_identity']}`",f"Code: `{plan['code_identity']}`",
        f"Planned detector cells: {len(plan['jobs'])}. Status: {dict(coverage)}.",
        f"Completed validation metrics: {sum(r['split']=='val' for r in rows)}; test metrics: {sum(r['split']=='test' for r in rows)}.",
        '', '## Measured results', '', '| Split | Model | Arm | Fraction | Seeds | mAP50 mean | mAP50-95 mean |',
        '|---|---|---|---:|---:|---:|---:|']
    for (split,model,arm,fraction),group in sorted(grouped.items()):
        lines.append(f"| {split} | {model} | {arm} | {fraction:g} | {len(group)} | {np.mean([r['map50'] for r in group]):.4f} | {np.mean([r['map50_95'] for r in group]):.4f} |")
    if not rows: lines.append('\nNo measured detector results are available yet.')
    lines+=['','## Reporting notes','',
        '- AP uses the same pycocotools evaluator and original image area bins for all models. Scores are fractions, not percentages.',
        '- summary.csv uses Student-t intervals over seeds. No interval is claimed from a single seed. Intervals are not clipped to [0,1].',
        '- Per-run fixed-point precision/recall/F1 intervals resample related-image groups. These are not AP confidence intervals.',
        '- Paired tests match model, fraction, split and seed; Holm correction spans all available contrasts per test type. Missing pairs are reported, not imputed.',
        '- Fewer than all planned seeds/cells means an incomplete comparison. Three seeds have weak inferential power.',
        '- Classical augmentation is offline with transformed boxes; its image count and optimizer exposure differ from baseline. Compare cost as well as accuracy.',
        '- YOLOX-S and the Ultralytics nano models are different parameter sizes. This is a family comparison, not a size-matched causal comparison.',
        '- YOLOX uses the official architecture/loss with a local AdamW trainer; the paper used SGD and 300 detector epochs. This grid is not an exact reproduction.',
        '- Latency uses batch one wall time including preprocessing after warmup; compare only on the same device/environment.',
        '- Generated samples must pass semantic, placement and final composite review before HQGAN detector training.',
        '- No CHPDNet, segmentation scores, FID or IS are inferred from detection metrics.',
        '', 'Artifacts: runs.csv, per_class.csv, per_image_counts.csv, summary.csv, paired_comparisons.csv, coverage.csv, dataset_inventory.csv, subsets.csv, all_metrics.json and figures/.']
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output


def _plots(summaries,metrics,output,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    for metric in ('map50','map50_95','crack_recall','macro_f1'):
        for split in ('val','test'):
            rows=[r for r in summaries if r['metric']==metric and r['split']==split and r['mean'] is not None]
            if not rows: continue
            fig,ax=plt.subplots(figsize=(9,5))
            for model,arm in sorted({(r['model'],r['arm']) for r in rows}):
                series=sorted([r for r in rows if r['model']==model and r['arm']==arm],key=lambda r:r['fraction'])
                x=[r['fraction']*100 for r in series]; y=[r['mean'] for r in series]
                ax.plot(x,y,marker='o',label=f'{model}/{arm}')
            ax.set(xlabel='Requested training fraction (%)',ylabel=metric,title=('SMOKE / ' if plan['smoke'] else '')+split)
            ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout()
            fig.savefig(figures/f'{split}_{metric}.png',dpi=160); fig.savefig(figures/f'{split}_{metric}.svg'); plt.close(fig)
    for m in metrics:
        prefix=m['job']['id']+'_'+m['split']
        fig,ax=plt.subplots(figsize=(5,4)); matrix=np.asarray(m['fixed_operating_point']['confusion_matrix'])
        im=ax.imshow(matrix,cmap='Blues'); fig.colorbar(im,ax=ax)
        labels=m['fixed_operating_point']['confusion_labels']; ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right'); ax.set_yticks(range(len(labels)),labels)
        ax.set(xlabel='Predicted',ylabel='True',title='Fixed-threshold confusion'); fig.tight_layout()
        fig.savefig(figures/(prefix+'_confusion.png'),dpi=120); plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,4))
        for name,curve in m['pr_curves'].items():
            ax.plot(curve['recall'],np.maximum(curve['precision_iou50'],0),label=name)
        ax.set(xlabel='Recall',ylabel='Precision',title='PR at IoU 0.5',xlim=(0,1),ylim=(0,1)); ax.legend(); fig.tight_layout()
        fig.savefig(figures/(prefix+'_pr.png'),dpi=120); plt.close(fig)
