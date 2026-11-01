"""Summarize a GAN run without inferring realism or detector benefit from GAN losses."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image,ImageDraw
from detection_pipeline import resolve,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); source=resolve(a.run); output=resolve(a.output); output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'run_manifest.json').read_text()); completion=json.loads((source/'completion.json').read_text())
    rows=[json.loads(line) for line in (source/'losses.jsonl').read_text().splitlines()]
    if not rows: raise ValueError('No GAN training measurements')
    report={'source_run':str(source),'source_manifest_sha256':sha256(source/'run_manifest.json'),
            'completion':completion,'settings':manifest['settings'],'identity':manifest['identity'],
            'steps_logged':len(rows),'image_presentations':rows[-1]['kimg']*1000,
            'paper_budget_image_presentations':10000000,
            'fraction_of_paper_budget':rows[-1]['kimg']/10000,
            'diffusion_horizon_range':[min(r['diffusion_horizon'] for r in rows),max(r['diffusion_horizon'] for r in rows)],
            'last_100_mean_losses':{k:float(np.mean([r[k] for r in rows[-100:]])) for k in ('d_loss','g_loss')},
            'quality_conclusion':'Losses alone do not establish image realism, correct class, diversity or detection utility.',
            'checkpoint_sha256':sha256(source/'last.pt')}
    write_json(output/'inspection.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for k in ('d_loss','g_loss'): axes[0].plot([r['step'] for r in rows],[r[k] for r in rows],label=k,alpha=.75)
    axes[0].legend(); axes[0].set_ylabel('Loss (not image quality)')
    axes[1].plot([r['step'] for r in rows],[r['diffusion_horizon'] for r in rows]); axes[1].set(xlabel='Optimizer step',ylabel='Diffusion horizon')
    fig.tight_layout(); fig.savefig(output/'training_curves.png',dpi=160); plt.close(fig)
    files=sorted(source.glob('sample_*.png'))[:16]
    if files:
        sheet=Image.new('RGB',(256*4,280*((len(files)+3)//4)),'white'); draw=ImageDraw.Draw(sheet)
        for i,file in enumerate(files):
            with Image.open(file) as im: tile=im.convert('RGB').resize((256,256))
            x,y=(i%4)*256,(i//4)*280; sheet.paste(tile,(x,y)); draw.text((x+4,y+258),file.name,fill='black')
        sheet.save(output/'samples.png')
    (output/'inspection.md').write_text(f"# HQGAN run inspection\n\nClass: {manifest['identity']['class_name']}. Completed steps: {completion.get('steps')}.\n\n"
        f"Image presentations: {report['image_presentations']:,.0f}, or {report['fraction_of_paper_budget']:.2%} of the paper's stated budget.\n\n"
        "Use the curves to inspect training stability, not to claim image quality. Review generated patches for defect presence, class correctness and diversity before blending. A full-training-pool GAN cannot be used for reduced-data cells.\n",encoding='utf-8')
    print(output)


if __name__=='__main__': main()
