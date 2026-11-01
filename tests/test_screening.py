import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detection_pipeline import sha256,write_json
from study.screening import scan_bundle,require_screening,settings_for
from study.augmentation import validate_candidates
from study.gan import review_item


class ScreeningTests(unittest.TestCase):
    def fixture(self, root):
        image=root/'patch.png'
        Image.fromarray(np.random.default_rng(42).integers(0,256,(64,64,3),dtype=np.uint8)).save(image)
        ckpt=root/'model.pt'; ckpt.write_bytes(b'fixture')
        row={'id':'crack_0','path':str(image),'sha256':sha256(image),'class_id':0,'class_name':'crack',
             'checkpoint':str(ckpt),'checkpoint_sha256':sha256(ckpt),'gan_identity':{'subset':'s'},
             'approved':False,'reviewer':'','defect_present':False,'class_correct':False,'defect_box_xyxy':None}
        path=root/'candidates_review.json'
        write_json(path,{'study_subset_identity':'s','smoke':False,'candidates':[row]})
        return path,row

    def approve(self,row):
        row.update(approved=True,reviewer='Person',defect_present=True,class_correct=True,defect_box_xyxy=[.1,.1,.8,.8])

    def test_pass_requires_separate_human_approval_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.fixture(Path(tmp)); output=scan_bundle(path,'candidates',names=['crack'])
            bundle=json.loads(path.read_text()); row=bundle['candidates'][0]
            self.assertEqual(row['screening']['outcome'],'pass'); self.assertFalse(row['approved'])
            self.assertTrue((output/'index.csv').is_file()); self.assertTrue((output/'00000.jpg').is_file())
            self.approve(row); self.assertEqual(len(validate_candidates(bundle,'s')),1)

    def test_unscreened_and_stale_metadata_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.fixture(Path(tmp)); self.approve(row)
            with self.assertRaisesRegex(ValueError,'Run screen'): validate_candidates({'study_subset_identity':'s','candidates':[row]},'s')
            scan_bundle(path,'candidates'); bundle=json.loads(path.read_text()); row=bundle['candidates'][0]
            self.approve(row); row['class_id']=1
            with self.assertRaisesRegex(ValueError,'stale'): validate_candidates(bundle,'s')

    def test_blank_flag_requires_explanation_and_rescreen_clears_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.fixture(Path(tmp)); Image.new('RGB',(64,64),'gray').save(row['path']); row['sha256']=sha256(row['path'])
            write_json(path,{'study_subset_identity':'s','candidates':[row]}); scan_bundle(path,'candidates')
            bundle=json.loads(path.read_text()); row=bundle['candidates'][0]
            self.assertEqual(row['screening']['outcome'],'needs_review'); self.approve(row)
            with self.assertRaisesRegex(ValueError,'override_reason'): validate_candidates(bundle,'s')
            row['screening_override_reason']='Inspected at full size; actual visible defect verified'
            self.assertEqual(len(validate_candidates(bundle,'s')),1)
            write_json(path,bundle); scan_bundle(path,'candidates'); row=json.loads(path.read_text())['candidates'][0]
            self.assertFalse(row['approved']); self.assertIsNone(row['defect_box_xyxy'])

    def test_duplicate_rejected_and_cannot_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.fixture(Path(tmp)); duplicate={**row,'id':'crack_1'}
            write_json(path,{'study_subset_identity':'s','candidates':[row,duplicate]}); scan_bundle(path,'candidates')
            rows=json.loads(path.read_text())['candidates']; self.assertEqual(rows[1]['screening']['outcome'],'reject')
            rows[1]['screening_override_reason']='accept anyway'
            with self.assertRaisesRegex(ValueError,'rejected'): require_screening(rows[1],'candidates','s')

    def test_changed_image_and_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.fixture(Path(tmp)); Image.new('RGB',(64,64),'red').save(row['path'])
            scan_bundle(path,'candidates'); result=json.loads(path.read_text())['candidates'][0]
            self.assertIn('image_hash_mismatch',result['screening']['reasons'])
            Path(row['checkpoint']).unlink(); scan_bundle(path,'candidates')
            self.assertEqual(json.loads(path.read_text())['candidates'][0]['screening']['outcome'],'reject')

    def test_perceptual_duplicate_is_flagged_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,row=self.fixture(root)
            pixels=np.array(Image.open(row['path'])); pixels[0,0]=[0,0,0]
            other=root/'other.png'; Image.fromarray(pixels).save(other)
            duplicate={**row,'id':'crack_1','path':str(other),'sha256':sha256(other)}
            write_json(path,{'study_subset_identity':'s','candidates':[row,duplicate]})
            scan_bundle(path,'candidates'); result=json.loads(path.read_text())['candidates'][1]
            self.assertEqual(result['screening']['outcome'],'needs_review')
            self.assertIn('near_duplicate_of:crack_0',result['screening']['reasons'])

    def test_composite_corrupt_labels_and_small_region(self):
        from study.data import target_label
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,candidate=self.fixture(root)
            label=root/'label.txt'; targets=[[0,.1,.1,.2,.2]]; label.write_text(target_label(targets))
            record={'export_stem':'synthetic_1','image_path':candidate['path'],'image_sha256':candidate['sha256'],
                    'label_path':str(label),'export_label_sha256':sha256(label),'targets':targets,
                    'width':64,'height':64,'augmentation':{'placement':{'rectangle_xywh':[10,10,8,8]}}}
            path=root/'review.json'; write_json(path,{'study_subset_identity':'s','composites':[{'record':record,'approved':False}]})
            scan_bundle(path,'composites',names=['crack']); row=json.loads(path.read_text())['composites'][0]
            self.assertIn('small_inserted_region',row['screening']['reasons'])
            label.write_text('0 .5 .5 .2 .2\n'); scan_bundle(path,'composites',names=['crack'])
            row=json.loads(path.read_text())['composites'][0]
            self.assertEqual(row['screening']['outcome'],'reject')
            self.assertIn('label_hash_mismatch',row['screening']['reasons'])

    def test_gan_smoke_sample_keeps_two_usable_rare_class_patches(self):
        from study.data import gan_smoke_sample
        rows=[{'export_stem':str(i).zfill(3),'width':100,'height':100,
               'targets':[[0 if i<12 else 1,.1,.1,.3,.3]]} for i in range(14)]
        selected=gan_smoke_sample(rows,9,2)
        self.assertEqual(len(selected),9)
        self.assertEqual(sum(r['targets'][0][0]==1 for r in selected),2)
        with self.assertRaisesRegex(ValueError,'two usable'):
            gan_smoke_sample(rows[:-1],9,2)

    def test_settings_fail_closed(self):
        for settings in ({'min_side':0},{'near_duplicate_hamming':65},{'min_gray_std':float('nan')},{'typo':1}):
            with self.assertRaises(ValueError): settings_for(settings)

    def test_human_review_cli_records_decision_and_checks_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); directory=root/'gan/key'; directory.mkdir(parents=True)
            path,row=self.fixture(directory); scan_bundle(path,'candidates')
            plan={'subsets':{'key':{'identity':'s'}},'smoke':False}
            with patch('study.gan.load_plan',return_value=plan):
                with self.assertRaisesRegex(ValueError,'Confirm defect'):
                    review_item(root,'key','candidates','crack_0','approve','Human')
                self.assertFalse(json.loads(path.read_text())['candidates'][0]['approved'])
                review_item(root,'key','candidates','crack_0','approve','Human',box=[.1,.1,.8,.8],defect_present=True,class_correct=True)
                self.assertTrue(json.loads(path.read_text())['candidates'][0]['approved'])
                review_item(root,'key','candidates','crack_0','reject','Human',note='Wrong defect')
                self.assertFalse(json.loads(path.read_text())['candidates'][0]['approved'])

if __name__=='__main__': unittest.main()
