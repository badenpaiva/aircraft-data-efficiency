import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detection_pipeline import sha256,write_json
from study.data import nested_subsets,target_label
from study.metrics import evaluate,group_bootstrap
from study.augmentation import classical,blend,approved_composites,ReviewRequired
from study.screening import scan_bundle
from study.report import holm,paired_comparisons,summary


class StudyTests(unittest.TestCase):
    def test_nested_whole_groups_class_coverage_and_determinism(self):
        rows=[{'export_stem':str(i),'group':str(i//2),'split':'train','image_sha256':str(i),
               'export_label_sha256':str(i),'targets':[[i//2%3,.1,.1,.2,.2]]} for i in range(120)]
        rows.append({**rows[0],'export_stem':'heldout','split':'val'})
        result=nested_subsets(rows,[1,.5,.1],42,3,2)
        self.assertEqual(result,nested_subsets(rows,[1,.5,.1],42,3,2))
        small=set(result['0.1']['stems']); medium=set(result['0.5']['stems']); large=set(result['1']['stems'])
        self.assertTrue(small<=medium<=large); self.assertEqual(len(large),120)
        for s in result.values():
            self.assertNotIn('heldout',s['stems']); self.assertGreaterEqual(min(s['instances_per_class']),2)
            for i in s['stems']: self.assertIn(str(int(i)^1),s['stems'])
        self.assertNotEqual(result['0.1']['stems'],nested_subsets(rows,[1,.5,.1],5,3,2)['0.1']['stems'])
        with self.assertRaises(ValueError): nested_subsets(rows,[.01,1],42,3,5)

    def test_coco_metrics_perfect_empty_and_wrong_class(self):
        settings={'max_detections':300,'operating_confidence':.25,'matching_iou':.5,
                  'confidence_sweep':[.1,.5],'bootstrap_samples':20,'bootstrap_seed':1}
        rows=[{'export_stem':str(i),'width':100,'height':100,'group':str(i),
               'targets':[[i,.1,.1,.5,.5]]} for i in range(3)]
        predictions={str(i):[[i,.9,.1,.1,.5,.5]] for i in range(3)}
        perfect=evaluate(rows,predictions,['dent','crack','paint_peeling'],settings)
        self.assertAlmostEqual(perfect['map50_95'],1); self.assertEqual(perfect['crack_recall'],1)
        empty=evaluate(rows,{str(i):[] for i in range(3)},['dent','crack','paint_peeling'],settings)
        self.assertEqual(empty['map50'],0); self.assertEqual(empty['crack_recall'],0)
        predictions['1'][0][0]=0
        wrong=evaluate(rows,predictions,['dent','crack','paint_peeling'],settings)
        self.assertEqual(wrong['fixed_operating_point']['confusion_matrix'][1][0],1)
        with self.assertRaises(ValueError): evaluate(rows,{},['dent','crack','paint_peeling'],settings)

    def test_statistics_handle_missing_pairs_and_constant_effects(self):
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
        self.assertIsNone(summary([.5])['ci95_low'])
        rows=[{'split':'val','model':'m','fraction':1,'arm':arm,'seed':s,'map50':value}
              for arm,value in [('baseline',.5),('hqgan',.5)] for s in (0,1,2)]
        comparison=paired_comparisons(rows,['map50'],'baseline')[0]
        self.assertEqual(comparison['n_pairs'],3); self.assertEqual(comparison['paired_t_p'],1)
        self.assertEqual(comparison['rank_biserial'],0)

    def fixture(self,root):
        image=root/'source.png'; label=root/'source.txt'
        Image.fromarray(np.random.default_rng(0).integers(0,255,(100,100,3),dtype=np.uint8)).save(image)
        targets=[[0,.05,.05,.15,.15]]; label.write_text(target_label(targets))
        return {'export_stem':'real','group':'g','split':'train','width':100,'height':100,
            'image_path':str(image),'label_path':str(label),'image_sha256':sha256(image),
            'export_label_sha256':sha256(label),'targets':targets}

    def test_classical_transforms_boxes_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); row=self.fixture(root)
            result=classical([row],{'copies_per_real':1,'degrees':0,'flip_probability':1,'noise_std':0},root/'aug',0)
            np.testing.assert_allclose(result[0]['targets'][0],[0,.85,.05,.95,.15])
            self.assertEqual(sha256(row['image_path']),row['image_sha256'])

    def test_poisson_review_gates_preserve_labels_and_block_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); row=self.fixture(root)
            sample=root/'sample.png'; Image.new('RGB',(32,32),(100,80,60)).save(sample)
            ckpt=root/'g.pt'; ckpt.write_bytes(b'test checkpoint')
            c={'id':'c','path':str(sample),'sha256':sha256(sample),'checkpoint':str(ckpt),
               'checkpoint_sha256':sha256(ckpt),'class_id':1,'approved':False,'reviewer':'tester',
               'defect_present':True,'class_correct':True,'defect_box_xyxy':[.2,.2,.8,.8]}
            candidates=root/'c.json'; placements=root/'p.json'
            bundle={'study_subset_identity':'s','smoke':True,'candidates':[c]}
            write_json(candidates,bundle)
            p={'id':'p','background_stem':'real','background_sha256':row['image_sha256'],'candidate_id':'c',
               'rectangle_xywh':[40,40,20,20],'approved':True,'reviewer':'tester','placement_valid':True}
            write_json(placements,{'study_subset_identity':'s','placements':[p]})
            with self.assertRaises(ReviewRequired): blend([row],candidates,placements,'s',root/'out',True)
            scan_bundle(candidates,'candidates')
            bundle=json.loads(candidates.read_text()); c=bundle['candidates'][0]
            c.update(approved=True,reviewer='tester',defect_present=True,class_correct=True,
                     defect_box_xyxy=[.2,.2,.8,.8],screening_override_reason='Fixture: deliberately flat test patch')
            write_json(candidates,bundle)
            result=blend([row],candidates,placements,'s',root/'out',True)
            self.assertEqual(result[0]['targets'][0],row['targets'][0])
            np.testing.assert_allclose(result[0]['targets'][1],[1,.44,.44,.56,.56])
            review=root/'out/review.json'
            with self.assertRaises(ReviewRequired): approved_composites(review,'s',[row],True)
            scan_bundle(review,'composites')
            content=json.loads(review.read_text()); content['composites'][0].update(approved=True,reviewer='tester',labels_complete=True,realistic=True,screening_override_reason='Fixture: synthetic blend checked')
            write_json(review,content)
            self.assertEqual(len(approved_composites(review,'s',[row],True)),1)
            with self.assertRaises(ValueError): approved_composites(review,'s',[row],False)
            with self.assertRaises(ValueError): approved_composites(review,'other',[row],True)
            self.assertEqual(sha256(row['image_path']),row['image_sha256'])


if __name__=='__main__': unittest.main()
