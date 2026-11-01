import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detection_pipeline import sha256,write_json
from study.cvat import export_review,import_review,add_box
from study.screening import scan_bundle
from study.data import target_label


class CvatTests(unittest.TestCase):
    def fixture(self,root,stage='candidates',count=1):
        gan=root/'gan/key'; gan.mkdir(parents=True)
        checkpoint=gan/'model.pt'; checkpoint.write_bytes(b'test')
        candidates=[]
        for i in range(count):
            image=gan/f'patch{i}.png'
            Image.fromarray(np.random.default_rng(i).integers(0,256,(64,80,3),dtype=np.uint8)).save(image)
            candidates.append({'id':f'crack_{i}','class_id':0,'class_name':'crack','path':str(image),
                               'sha256':sha256(image),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                               'approved':False,'reviewer':''})
        candidate_file=gan/'candidates_review.json'
        write_json(candidate_file,{'study_subset_identity':'s','smoke':False,'candidates':candidates})
        scan_bundle(candidate_file,'candidates',names=['crack'])
        label=gan/'real.txt'; targets=[[0,.05,.05,.15,.15]]; label.write_text(target_label(targets))
        record={'export_stem':'real','split':'train','image_path':candidates[0]['path'],
                'image_sha256':candidates[0]['sha256'],'width':80,'height':64,'label_path':str(label),
                'export_label_sha256':sha256(label),'targets':targets}
        placement={'id':'p','candidate_id':'crack_0','background_stem':'real','background_sha256':record['image_sha256'],
                   'rectangle_xywh':[20,20,20,20],'approved':False,'reviewer':''}
        write_json(gan/'placements_review.json',{'study_subset_identity':'s','placements':[placement]})
        file=candidate_file
        if stage=='placements': file=gan/'placements_review.json'
        if stage=='composites':
            out=gan/'composites'; out.mkdir()
            synthetic={**record,'export_stem':'synthetic_a','augmentation':{'placement':placement,'study_subset_identity':'s'}}
            file=out/'review.json'; write_json(file,{'study_subset_identity':'s','composites':[{'record':synthetic,'approved':False,'reviewer':''}]})
            scan_bundle(file,'composites',names=['crack'])
        plan={'plan_identity':'plan','subsets':{'key':{'identity':'s','stems':['real']}},'names':['crack'],'records':[record],'smoke':False}
        return plan,file

    def change(self,package,stage,decision='approve'):
        tree=ET.parse(package/'annotations.xml')
        for node in tree.getroot().findall('image'):
            for attr in node.find('tag').findall('attribute'):
                if attr.get('name')=='decision': attr.text=decision
                if attr.get('name') in {'defect_present','class_correct','placement_valid','realistic','labels_complete'}: attr.text='true'
                if attr.get('name')=='note': attr.text='Human inspected full-size image and confirmed the flagged detail.'
            if stage=='candidates' and decision=='approve': add_box(node,('crack',8,16,40,48))
        target=package/'reviewed.xml'; tree.write(target,encoding='utf-8',xml_declaration=True)
        return target

    def test_candidate_roundtrip_normalizes_box_and_preserves_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange')
                xml=self.change(package,'candidates'); before=file.read_bytes()
                result=import_review(root,'key','candidates',package,xml,'Human',True)
                self.assertEqual(result['outcomes']['approve'],1); self.assertEqual(file.read_bytes(),before)
                import_review(root,'key','candidates',package,xml,'Human')
                data=json.loads(file.read_text()); row=data['candidates'][0]
                self.assertEqual(row['defect_box_xyxy'],[.1,.25,.5,.75]); self.assertTrue(row['approved'])
                self.assertEqual(data['cvat_imports'][0]['reviewer'],'Human')
                self.assertEqual(len(list(file.parent.glob('*.before_cvat_*.json'))),1)
                with self.assertRaisesRegex(ValueError,'changed since export'):
                    import_review(root,'key','candidates',package,xml,'Human')

    def test_invalid_second_row_leaves_entire_queue_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root,count=2)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange'); xml=self.change(package,'candidates')
                tree=ET.parse(xml); tree.getroot().findall('image')[1].find('box').set('label','dent'); tree.write(xml)
                before=file.read_bytes()
                with self.assertRaisesRegex(ValueError,'intended class'): import_review(root,'key','candidates',package,xml,'Human')
                self.assertEqual(file.read_bytes(),before)

    def test_missing_and_duplicate_images_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange'); xml=self.change(package,'candidates')
                tree=ET.parse(xml); node=tree.getroot().find('image'); tree.getroot().remove(node); tree.write(xml)
                with self.assertRaisesRegex(ValueError,'every image'): import_review(root,'key','candidates',package,xml,'Human')
                tree.getroot().extend([node,copy.deepcopy(node)]); tree.write(xml)
                with self.assertRaisesRegex(ValueError,'duplicate'): import_review(root,'key','candidates',package,xml,'Human')

    def test_reject_without_box_and_pending_do_not_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange')
                before=file.read_bytes(); import_review(root,'key','candidates',package,package/'annotations.xml','Human')
                self.assertEqual(file.read_bytes(),before)
                xml=self.change(package,'candidates','reject'); import_review(root,'key','candidates',package,xml,'Human')
                self.assertFalse(json.loads(file.read_text())['candidates'][0]['approved'])

    def test_placement_and_composite_roundtrip_and_geometry_changes(self):
        for stage in ['placements','composites']:
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); plan,file=self.fixture(root,stage)
                with patch('study.cvat.load_plan',return_value=plan):
                    package=export_review(root,'key',stage,root/'exchange'); xml=self.change(package,stage)
                    tree=ET.parse(xml); tree.getroot().find('image/box').set('xtl','1'); tree.write(xml)
                    with self.assertRaisesRegex(ValueError,'geometry changed'):
                        import_review(root,'key',stage,package,xml,'Human')
                    xml=self.change(package,stage); import_review(root,'key',stage,package,xml,'Human')
                    self.assertTrue(json.loads(file.read_text())[stage][0]['approved'])
                    if stage=='placements': self.assertTrue((package/'candidate_references.zip').exists())

    def test_zip_import_and_confirmation_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange'); xml=self.change(package,'candidates')
                tree=ET.parse(xml)
                attr=next(a for a in tree.getroot().find('image/tag') if a.get('name')=='class_correct')
                attr.text='false'; tree.write(xml)
                with self.assertRaisesRegex(ValueError,'class_correct'):
                    import_review(root,'key','candidates',package,xml,'Human')
                xml=self.change(package,'candidates')
                archive=package/'download.zip'
                with ZipFile(archive,'w') as zipfile: zipfile.write(xml,'annotations.xml')
                import_review(root,'key','candidates',package,archive,'Human')
                self.assertTrue(json.loads(file.read_text())['candidates'][0]['approved'])

    def test_screen_rejections_excluded_and_stale_files_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan,file=self.fixture(root,count=2)
            content=json.loads(file.read_text()); content['candidates'][1]['screening']['outcome']='reject'; write_json(file,content)
            with patch('study.cvat.load_plan',return_value=plan):
                package=export_review(root,'key','candidates',root/'exchange')
                self.assertEqual(len(json.loads((package/'exchange.json').read_text())['items']),1)
                xml=self.change(package,'candidates'); Path(content['candidates'][0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'image changed'):
                    import_review(root,'key','candidates',package,xml,'Human')

if __name__=='__main__': unittest.main()
