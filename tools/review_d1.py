"""Reproducible D1 geometry audit. Raw data is never modified.
Run: python tools/review_d1.py
Outputs are candidates, never automatically approved training/evaluation labels.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, math, re, hashlib
import numpy as np
import yaml, imagehash
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_out/d1_review'
COLORS = ['#ff4040','#00ddff','#ffbb00','#44ff44','#ff44ff']

def parse_line(line, nc):
    vals = list(map(float, line.split()))
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError('nonfinite/empty')
    if vals[0] != int(vals[0]) or not 0 <= vals[0] < nc:
        raise ValueError('class')
    cid, xy = int(vals[0]), vals[1:]
    if any(v < 0 or v > 1 for v in xy):
        raise ValueError('range')
    if len(xy) == 4:
        x,y,w,h = xy
        if w <= 0 or h <= 0: raise ValueError('area')
        bounds = [x-w/2,y-h/2,x+w/2,y+h/2]
        if min(bounds) < -1e-6 or max(bounds) > 1+1e-6:
            raise ValueError('box outside image')
        return cid, 'box', xy, bounds
    if len(xy) < 6 or len(xy)%2: raise ValueError('arity')
    pts = list(zip(xy[::2],xy[1::2]))
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(pts,pts[1:]+pts[:1])))/2
    if area <= 1e-12: raise ValueError('polygon area')
    xs,ys = xy[::2],xy[1::2]
    return cid,'polygon',xy,[min(xs),min(ys),max(xs),max(ys)]

def clusters(hashes):
    parent=list(range(len(hashes)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    arr=np.array(hashes,dtype=np.uint64)
    for i,h in enumerate(arr):
        x=arr[i+1:] ^ h
        counts=np.zeros(len(x),dtype=np.uint8)
        for _ in range(64): counts+=(x&1).astype(np.uint8); x>>=1
        for j in np.flatnonzero(counts<=5)+i+1:
            a,b=find(i),find(int(j))
            if a!=b: parent[b]=a
    groups=defaultdict(list)
    for i in range(len(hashes)): groups[find(i)].append(i)
    return list(groups.values())

def sheet(items, name, overlay=True):
    # Two columns per image: unobscured image, and geometry overlay.
    for start in range(0,len(items),8):
        page=Image.new('RGB',(1280,4*365),'#202020'); d=ImageDraw.Draw(page)
        for k,r in enumerate(items[start:start+8]):
            ox=(k%2)*640; oy=(k//2)*365
            im=Image.open(ROOT/r['image']).convert('RGB')
            im.thumbnail((310,310)); iw,ih=im.size
            page.paste(im,(ox,oy+45)); marked=im.copy(); md=ImageDraw.Draw(marked)
            if overlay:
                for a in r['annotations']:
                    color=COLORS[a['class_id']]
                    if a['kind']=='polygon':
                        xy=a['coordinates']; pts=[(x*iw,y*ih) for x,y in zip(xy[::2],xy[1::2])]
                        md.line(pts+[pts[0]],fill=color,width=2)
                    else:
                        x1,y1,x2,y2=a['bounds']; md.rectangle((x1*iw,y1*ih,x2*iw,y2*ih),outline=color,width=2)
            page.paste(marked,(ox+320,oy+45))
            title=f"{r['id']} {Path(r['image']).name[:69]}"
            desc=', '.join(f"{a['class']}:{a['kind']}:{len(a['coordinates'])//2 if a['kind']=='polygon' else 0}" for a in r['annotations'])
            d.text((ox+3,oy+3),title,fill='white'); d.text((ox+3,oy+20),desc[:95],fill='white')
        page.save(OUT/f'{name}_{start//8+1:02}.jpg',quality=95)

def visibility(items):
    for idx,(r,a) in enumerate(items):
        im=Image.open(ROOT/r['image']).convert('RGB'); w,h=im.size
        scale=640/max(w,h); nw,nh=round(w*scale),round(h*scale)
        resized=im.resize((nw,nh),Image.Resampling.BILINEAR)
        x1,y1,x2,y2=a['bounds']; pad=.025
        bounds=(max(0,int((x1-pad)*w)),max(0,int((y1-pad)*h)),min(w,math.ceil((x2+pad)*w)),min(h,math.ceil((y2+pad)*h)))
        before=im.crop(bounds); after=resized.crop(tuple(round(v*scale) for v in bounds))
        canvas=Image.new('RGB',(1200,680),'#202020'); d=ImageDraw.Draw(canvas)
        full=ImageOps.contain(resized,(620,620)); canvas.paste(full,(0,55))
        before.thumbnail((270,550)); canvas.paste(before,(640,55))
        after=ImageOps.contain(after,(270,550),Image.Resampling.NEAREST); canvas.paste(after,(930,55))
        d.text((8,5),f"{r['id']} {a['class']} | original {w}x{h} -> {nw}x{nh} | box {a['width_at_640']:.2f}x{a['height_at_640']:.2f}px",fill='white')
        d.text((8,25),'640 longest side (bilinear) | source crop | resized crop magnified nearest-neighbor',fill='white')
        canvas.save(OUT/f'visibility_{idx+1:02}.jpg',quality=96)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())
    assert cfg['training']['image_size']==640, 'Update visibility renderer for configured size'
    records=[]; errors=[]; raw=Counter(); normalized=Counter(); kinds=Counter(); duplicate_rows=0
    for did,hint in [('D1','Innovation Hangar'),('D2','aircraft_skin_defects.v1')]:
        folder=next(p for p in (ROOT/'data/raw').iterdir() if hint in p.name)
        names=yaml.safe_load((folder/'data.yaml').read_text())['names']
        for p in sorted(folder.glob('*/images/*')):
            if p.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
            label=p.parent.parent/'labels'/f'{p.stem}.txt'
            with Image.open(p) as im:
                w,h=im.size; ph=int(str(imagehash.phash(im)),16) if did=='D1' else None
            r={'id':f'{did}-{sum(x["dataset"]==did for x in records)+1:04}', 'dataset':did,'image':p.relative_to(ROOT).as_posix(), 'label':label.relative_to(ROOT).as_posix(),'width':w,'height':h,'phash':ph,'annotations':[], 'review_status':'unreviewed','segmentation_complete':False}
            r['label_sha256']=hashlib.sha256(label.read_bytes()).hexdigest() if label.exists() else None
            lines=label.read_text().splitlines() if label.exists() else []
            r['empty_label']=label.exists() and not any(s.strip() for s in lines)
            r['missing_label']=not label.exists(); seen=set()
            for lineno,line in enumerate(lines,1):
                if not line.strip(): continue
                try:
                    cid,kind,xy,b=parse_line(line,len(names))
                except ValueError as e:
                    errors.append({'image':r['image'],'line':lineno,'reason':str(e),'raw':line}); continue
                if did=='D1': raw[names[cid]]+=1; kinds[f'{names[cid]}:{kind}']+=1
                key=(cid,kind,tuple(xy))
                if key in seen:
                    if did=='D1': duplicate_rows+=1
                    continue
                seen.add(key)
                x1,y1,x2,y2=b; s=640/max(w,h)
                a={'line':lineno,'class_id':cid,'class':names[cid],'kind':kind,'coordinates':xy,'bounds':b,'candidate_detection_box':[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],'width_at_640':(x2-x1)*w*s,'height_at_640':(y2-y1)*h*s,'approval':'pending'}
                r['annotations'].append(a)
                if did=='D1': normalized[names[cid]]+=1
            r['mask_candidate']=bool(r['annotations']) and all(a['kind']=='polygon' for a in r['annotations'])
            records.append(r)
    d1=[r for r in records if r['dataset']=='D1']; groups=clusters([r['phash'] for r in d1])
    kept=[d1[min(g,key=lambda i:d1[i]['image'])] for g in groups]
    post=Counter(a['class'] for r in kept for a in r['annotations'])
    for gi,g in enumerate(groups):
        for i in g: d1[i]['phash5_group']=gi
    polygons=[r for r in d1 if any(a['kind']=='polygon' for a in r['annotations'])]
    selected=[]
    for cls in ['crack','dent','paint-off','missing-head','scratch']:
        rs=[r for r in polygons if any(a['class']==cls and a['kind']=='polygon' for a in r['annotations'])]
        rs.sort(key=lambda r:max(len(a['coordinates']) for a in r['annotations'] if a['kind']=='polygon'))
        for index in sorted(set([0,len(rs)//4,len(rs)//2,3*len(rs)//4,len(rs)-1])) if rs else []:
            if rs[index] not in selected: selected.append(rs[index])
    # Additional deterministic filename-family and complexity coverage.
    families=defaultdict(list)
    for r in polygons:
        stem=Path(r['image']).name.split('.rf.')[0]
        family=re.split(r'\d',stem)[0] or 'numeric'
        families[family].append(r)
    for fam,rs in sorted(families.items()):
        candidate=max(rs,key=lambda r:max(len(a['coordinates']) for a in r['annotations']))
        if candidate not in selected: selected.append(candidate)
    sheet(selected,'polygons')
    empties=[r for r in records if r['empty_label'] or r['missing_label']]; sheet(empties,'empty',False)
    rng=np.random.default_rng(20261003)
    domain=[d1[int(i)] for i in rng.choice(len(d1),48,replace=False)]; sheet(domain,'domain')
    vis=[]
    for cls in ['crack','paint-off']:
        pairs=[(r,a) for r in d1 for a in r['annotations'] if a['class']==cls]
        pairs.sort(key=lambda ra:min(ra[1]['width_at_640'],ra[1]['height_at_640']))
        for quantile in [0,.05,.25,.5]:
            vis.append(pairs[min(len(pairs)-1,int(quantile*len(pairs)))])
    visibility(vis)
    summary={'raw_valid_geometry_records':dict(raw),'geometry_counts':dict(kinds),'exact_repeated_rows_removed':duplicate_rows,'normalized_pre_image_dedup':dict(normalized),'phash5_representative_images':len(kept),'normalized_phash5_representative_records':dict(post),'invalid_records':errors,'empty_labels':dict(Counter(r['dataset'] for r in empties if r['empty_label'])),'polygon_images':len(polygons),'polygon_only_candidate_images':sum(r['mask_candidate'] for r in d1),'verified_complete_segmentation_images':0,'polygon_sample_ids':[r['id'] for r in selected],'polygon_families':{k:len(v) for k,v in families.items()},'domain_sample_ids':[r['id'] for r in domain],'visibility_samples':[{'id':r['id'],'line':a['line'],'class':a['class'],'width_at_640':a['width_at_640'],'height_at_640':a['height_at_640']} for r,a in vis],'counts_status':'geometry-normalized inventory; NOT final admitted or visually validated counts','resize':'aspect-preserving bilinear to longest side 640; remaining canvas padded; no stretch'}
    for cls in ['crack','paint-off']:
        aa=[a for r in d1 for a in r['annotations'] if a['class']==cls]
        summary[f'{cls}_resize_risk']={'total':len(aa),'box_short_side_below_2px':sum(min(a['width_at_640'],a['height_at_640'])<2 for a in aa),'box_short_side_below_4px':sum(min(a['width_at_640'],a['height_at_640'])<4 for a in aa),'warning':'box extent is not defect thickness or visibility'}
    (OUT/'inventory.json').write_text(json.dumps(summary,indent=2))
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (OUT/'review_index.json').write_text(json.dumps({'polygons':[r['id'] for r in selected],'empty':[r['id'] for r in empties],'domain':[r['id'] for r in domain]},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
