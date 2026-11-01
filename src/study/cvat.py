"""Offline CVAT image-task exchange for screened GAN review queues."""
import copy
import json
import math
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image
from detection_pipeline import resolve, sha256, write_json
from .data import identity
from .runner import load_plan, subset_records
from .gan import review_file, apply_review_decision
from .screening import require_screening, item_id

CONFIRMATIONS = {'candidates': ('defect_present','class_correct'),
                 'placements': ('placement_valid',), 'composites': ('realistic','labels_complete')}


def attributes(stage):
    return [
        {'name':'decision','input_type':'select','mutable':False,'values':['pending','approve','reject'],'default_value':'pending'},
        {'name':'note','input_type':'text','mutable':False,'values':[''],'default_value':''},
        {'name':'screening','input_type':'text','mutable':False,'values':[''],'default_value':''},
        {'name':'context','input_type':'text','mutable':False,'values':[''],'default_value':''},
        {'name':'exchange_id','input_type':'text','mutable':False,'values':[''],'default_value':''},
        *[{'name':k,'input_type':'checkbox','mutable':False,'values':['false'],'default_value':'false'} for k in CONFIRMATIONS[stage]]]


def queue(directory,key,stage):
    directory=resolve(directory); plan=load_plan(directory)
    if stage not in CONFIRMATIONS: raise ValueError('Unsupported CVAT review stage')
    path=review_file(directory,key,stage); bundle=json.loads(path.read_text())
    if bundle['study_subset_identity']!=plan['subsets'][key]['identity']: raise ValueError('Review subset mismatch')
    return directory,plan,path,bundle


def row_id(row,stage):
    return row['id'] if stage=='placements' else item_id(row,stage)


def source_info(row,stage,backgrounds):
    if stage=='candidates': return row['path'],row['sha256']
    if stage=='placements':
        bg=backgrounds[row['background_stem']]
        return bg['image_path'],row['background_sha256']
    return row['record']['image_path'],row['record']['image_sha256']


def reference_boxes(row,stage,names,width,height):
    if stage=='candidates': return []
    if stage=='placements':
        x,y,w,h=row['rectangle_xywh']; return [('placement',x,y,x+w,y+h)]
    return [(names[int(c)],x1*width,y1*height,x2*width,y2*height) for c,x1,y1,x2,y2 in row['record']['targets']]


def add_box(parent,box):
    label,*values=box
    ET.SubElement(parent,'box',label=label,occluded='0',source='manual',z_order='0',
                  **dict(zip(('xtl','ytl','xbr','ybr'),(f'{v:.6f}' for v in values))))


def export_review(directory,key,stage,output,limit=None):
    directory,plan,path,bundle=queue(directory,key,stage)
    if limit is not None and limit<1: raise ValueError('limit must be positive')
    rows=[r for r in bundle[stage] if not r.get('approved') and not r.get('reviewer')
          and r.get('screening',{}).get('outcome')!='reject']
    if limit is not None: rows=rows[:limit]
    if not rows: raise ValueError('No undecided, technically eligible images to export')
    backgrounds={r['export_stem']:r for r in subset_records(plan,key)}
    root=directory/'gan'/key
    dependencies={str(p.resolve()):sha256(p) for p in
                  ([root/'candidates_review.json'] if stage=='placements' else
                   [root/'candidates_review.json',root/'placements_review.json'] if stage=='composites' else [])}
    candidates={r['id']:r for r in json.loads((root/'candidates_review.json').read_text())['candidates']} if stage=='placements' else {}
    exchange_id=identity([plan['plan_identity'],key,stage,sha256(path),[row_id(r,stage) for r in rows]])
    labels=[{'name':name,'type':'rectangle','attributes':[]} for name in plan['names']]
    if stage=='placements': labels.append({'name':'placement','type':'rectangle','attributes':[]})
    labels.append({'name':'review','type':'tag','attributes':attributes(stage)})
    annotations=ET.Element('annotations'); ET.SubElement(annotations,'version').text='1.1'
    task=ET.SubElement(ET.SubElement(annotations,'meta'),'task')
    for k,v in {'id':'0','name':f'{key}_{stage}','size':str(len(rows)),'mode':'annotation','overlap':'0','flipped':'False'}.items():
        ET.SubElement(task,k).text=v
    label_nodes=ET.SubElement(task,'labels')
    for label in labels:
        node=ET.SubElement(label_nodes,'label'); ET.SubElement(node,'name').text=label['name']
        attrs=ET.SubElement(node,'attributes')
        for a in label['attributes']:
            attr=ET.SubElement(attrs,'attribute')
            for k,v in a.items(): ET.SubElement(attr,k).text='\n'.join(v) if isinstance(v,list) else str(v).lower() if isinstance(v,bool) else v
    items=[]
    for index,row in enumerate(rows):
        if stage!='placements':
            probe=copy.deepcopy(row); probe['screening_override_reason']='Export for human inspection'
            require_screening(probe,stage,bundle['study_subset_identity'])
        original,expected=source_info(row,stage,backgrounds)
        if sha256(original)!=expected: raise ValueError('Review image changed before CVAT export')
        with Image.open(original) as image: width,height=image.size
        name=f'{index:06d}_{identity(row_id(row,stage))[:12]}{Path(original).suffix.lower()}'
        node=ET.SubElement(annotations,'image',id=str(index),name=name,width=str(width),height=str(height))
        boxes=reference_boxes(row,stage,plan['names'],width,height)
        for box in boxes: add_box(node,box)
        tag=ET.SubElement(node,'tag',label='review',source='manual')
        receipt=row.get('screening',{})
        values={'decision':'pending','note':'','exchange_id':exchange_id,
                'context':('Expected class: '+row['class_name']) if stage=='candidates' else
                          ('Candidate: '+row['candidate_id']+'; reference: '+identity(row['candidate_id'])[:16]+'.png') if stage=='placements' else 'Inspect all existing boxes; do not change geometry',
                'screening':receipt.get('outcome','placement geometry only')+'; '+'; '.join(receipt.get('reasons',[])),
                **{k:'false' for k in CONFIRMATIONS[stage]}}
        for k,v in values.items(): ET.SubElement(tag,'attribute',name=k).text=v
        items.append({'id':row_id(row,stage),'name':name,'width':width,'height':height,
                      'image_path':str(original),'image_sha256':expected,'reference_boxes':boxes,
                      'class_name':row.get('class_name')})
    output=resolve(output); output.mkdir(parents=True,exist_ok=False)
    with ZipFile(output/'images.zip','x',compression=ZIP_DEFLATED) as archive:
        for item in items:
            archive.write(item['image_path'],item['name'])
            if sha256(item['image_path'])!=item['image_sha256']: raise ValueError('Image changed while exporting')
    if stage=='placements':
        with ZipFile(output/'candidate_references.zip','x',compression=ZIP_DEFLATED) as archive:
            for cid in sorted({r['candidate_id'] for r in rows}):
                candidate=candidates[cid]
                if sha256(candidate['path'])!=candidate['sha256']: raise ValueError('Placement candidate image changed')
                # Encode as PNG so the reference extension always matches its content.
                import io
                buffer=io.BytesIO()
                with Image.open(candidate['path']) as image: image.convert('RGB').save(buffer,format='PNG')
                archive.writestr(identity(cid)[:16]+'.png',buffer.getvalue())
    ET.indent(annotations); ET.ElementTree(annotations).write(output/'annotations.xml',encoding='utf-8',xml_declaration=True)
    write_json(output/'labels.json',labels)
    manifest={'version':1,'exchange_id':exchange_id,'study':str(directory),'plan_identity':plan['plan_identity'],
              'subset':key,'stage':stage,'queue_sha256':sha256(path),'queue_identity':identity(bundle),
              'dependencies':dependencies,'items':items}
    if identity(json.loads(path.read_text()))!=identity(bundle): raise ValueError('Review queue changed while exporting; discard package')
    manifest['manifest_identity']=identity(manifest); write_json(output/'exchange.json',manifest)
    return output


def read_xml(path):
    path=Path(path)
    if path.suffix.lower()=='.zip':
        with ZipFile(path) as archive:
            matches=[n for n in archive.namelist() if PurePosixPath(n).name=='annotations.xml']
            if len(matches)!=1: raise ValueError('Expected exactly one annotations.xml in CVAT archive')
            if archive.getinfo(matches[0]).file_size>50_000_000: raise ValueError('CVAT XML exceeds size limit')
            data=archive.read(matches[0])
    else:
        if path.stat().st_size>50_000_000: raise ValueError('CVAT XML exceeds size limit')
        data=path.read_bytes()
    text=data.decode('utf-8-sig')
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper(): raise ValueError('XML declarations/entities are not allowed')
    root=ET.fromstring(text)
    if root.tag!='annotations' or root.findtext('version')!='1.1' or root.findall('track'):
        raise ValueError('Use CVAT for images 1.1, not video/tracks')
    return root


def parse_boxes(node,width,height):
    result=[]
    for shape in node:
        if shape.tag=='tag': continue
        if shape.tag!='box': raise ValueError('Only axis-aligned rectangles are supported')
        values=[float(shape.get(k,'nan')) for k in ('xtl','ytl','xbr','ybr')]
        x1,y1,x2,y2=values
        if not all(math.isfinite(v) for v in values) or not (0<=x1<x2<=width and 0<=y1<y2<=height):
            raise ValueError('Invalid CVAT box bounds')
        if float(shape.get('rotation','0'))!=0 or shape.get('outside','0')!='0': raise ValueError('Rotated/outside boxes are unsupported')
        result.append((shape.get('label'),*values))
    return result


def same_boxes(actual,expected):
    # CVAT serializes coordinates to two decimals. Allow only that rounding.
    remaining=list(expected)
    for box in actual:
        matches=[i for i,b in enumerate(remaining) if b[0]==box[0] and all(abs(x-y)<=.011 for x,y in zip(b[1:],box[1:]))]
        if not matches: return False
        remaining.pop(matches[0])
    return not remaining


def import_review(directory,key,stage,package,annotations,reviewer,dry_run=False):
    directory,plan,path,bundle=queue(directory,key,stage)
    manifest=json.loads((resolve(package)/'exchange.json').read_text())
    check=dict(manifest); digest=check.pop('manifest_identity')
    if identity(check)!=digest: raise ValueError('CVAT exchange manifest changed')
    if (manifest['plan_identity'],manifest['subset'],manifest['stage'])!=(plan['plan_identity'],key,stage):
        raise ValueError('CVAT package belongs to a different study/subset/stage')
    if manifest['queue_sha256']!=sha256(path) or manifest['queue_identity']!=identity(bundle):
        raise ValueError('Review queue changed since export; create a fresh CVAT package')
    if any(sha256(p)!=h for p,h in manifest['dependencies'].items()): raise ValueError('Upstream reviews changed since CVAT export')
    if not reviewer.strip(): raise ValueError('Reviewer name is required')
    xml=read_xml(resolve(annotations)); expected={item['name']:item for item in manifest['items']}
    nodes={}
    for node in xml.findall('image'):
        name=node.get('name','')
        if name not in expected or name in nodes: raise ValueError('Unknown or duplicate CVAT image name')
        nodes[name]=node
    if set(nodes)!=set(expected): raise ValueError('CVAT export must contain every image in the package, including pending/rejected images')
    lookup={row_id(r,stage):r for r in bundle[stage]}
    outcomes={'approve':0,'reject':0,'pending':0}
    for name,item in expected.items():
        node=nodes[name]; width,height=item['width'],item['height']
        if (int(node.get('width','0')),int(node.get('height','0')))!=(width,height): raise ValueError('CVAT image dimensions changed')
        if sha256(item['image_path'])!=item['image_sha256']: raise ValueError('Original review image changed')
        tags=node.findall('tag')
        if len(tags)!=1 or tags[0].get('label')!='review': raise ValueError('Each image needs exactly one review tag')
        attrs={a.get('name'):a.text or '' for a in tags[0].findall('attribute')}
        if len(attrs)!=len(tags[0].findall('attribute')): raise ValueError('Duplicate review attributes')
        if attrs.get('exchange_id')!=manifest['exchange_id']: raise ValueError('Wrong CVAT exchange_id')
        decision=attrs.get('decision')
        if decision not in outcomes: raise ValueError('Decision must be pending, approve or reject')
        outcomes[decision]+=1
        if decision=='pending': continue
        row=lookup[item['id']]; kwargs={}
        if decision=='approve':
            for k in CONFIRMATIONS[stage]:
                if attrs.get(k)!='true': raise ValueError(f'Approval requires {k}=true')
                kwargs[k]=True
            boxes=parse_boxes(node,width,height)
            if stage=='candidates':
                if len(boxes)!=1 or boxes[0][0]!=row['class_name']: raise ValueError('Candidate needs exactly one rectangle of its intended class')
                _,x1,y1,x2,y2=boxes[0]; kwargs['box']=[x1/width,y1/height,x2/width,y2/height]
            elif not same_boxes(boxes,item['reference_boxes']):
                raise ValueError('Placement/composite geometry changed: reject instead, then regenerate/re-screen')
        apply_review_decision(row,stage,bundle['study_subset_identity'],plan['smoke'],decision,reviewer,attrs.get('note',''),**kwargs)
    if not dry_run and outcomes['approve']+outcomes['reject']:
        audit={'exchange_id':manifest['exchange_id'],'annotations_sha256':sha256(resolve(annotations)),
               'reviewer':reviewer,'outcomes':outcomes,'previous_queue_sha256':manifest['queue_sha256']}
        bundle.setdefault('cvat_imports',[]).append(audit)
        # Keep the exact pre-import queue for audit/recovery. Validate the entire batch
        # before any mutation, then atomically replace the authoritative queue.
        backup=path.with_name(path.stem+'.before_cvat_'+manifest['exchange_id'][:16]+'.json')
        if not backup.exists(): backup.write_bytes(path.read_bytes())
        if sha256(path)!=manifest['queue_sha256']: raise ValueError('Concurrent review update; import cancelled')
        temp=path.with_suffix('.json.tmp'); write_json(temp,bundle); temp.replace(path)
    return {'dry_run':dry_run,'outcomes':outcomes,'queue':str(path)}
