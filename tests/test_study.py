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
from study.augmentation import classical,blend,approved_composites,ReviewRequired,propose_placements,paste_patch,turned_box
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

    def test_seeds_choose_different_subsets_when_groups_are_few_and_distinct(self):
        # Capture-unit groups: few, unequal, no ties for the seed's order to break.
        rows=[]
        for g in range(20):
            for i in range(3+g):
                rows.append({'export_stem':f'{g}_{i}','group':str(g),'split':'train','image_sha256':f'{g}_{i}',
                             'export_label_sha256':f'{g}_{i}','targets':[[(g+k)%3,.1,.1,.2,.2] for k in range(1+g%4)]})
        runs={seed:nested_subsets(rows,[1,.5,.25],seed,3,5) for seed in range(6)}
        self.assertEqual(runs[0],nested_subsets(rows,[1,.5,.25],0,3,5))
        self.assertGreater(len({tuple(r['0.25']['stems']) for r in runs.values()}),3)
        self.assertGreater(len({tuple(r['0.5']['stems']) for r in runs.values()}),3)
        total=len(rows)
        for result in runs.values():
            small,medium,large=(set(result[k]['stems']) for k in ('0.25','0.5','1'))
            self.assertTrue(small<=medium<=large); self.assertEqual(len(large),total)
            self.assertGreaterEqual(min(result['0.25']['instances_per_class']),5)
            # Whole groups only, and the image budget is respected.
            self.assertLessEqual(len(small),total//4); self.assertLessEqual(len(medium),total//2)
            for stem in small|medium:
                self.assertTrue({r['export_stem'] for r in rows if r['group']==stem.split('_')[0]}<=medium)

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

    def test_placements_mirror_the_gan_crop_rule(self):
        row={'export_stem':'real','image_sha256':'h','width':200,'height':200,'targets':[[0,.05,.05,.10,.25]]}
        first={'a':{'id':'a','class_id':0,'gan_identity':{}}}
        rule={'margin':.25,'min_box_side':8,'canonical_orientation':True}
        context={'a':{'id':'a','class_id':0,'gan_identity':{'crop':rule}}}
        for p in propose_placements([row],first,4,0):
            # No recorded crop rule: the first study's behaviour, the exemplar box is the whole patch.
            self.assertEqual(p['rectangle_xywh'],p['core_box_xywh']); self.assertEqual(p['rectangle_xywh'][2:],[10,40])
            self.assertFalse(p['quarter_turn'])
        self.assertEqual(propose_placements([row],context,4,0),propose_placements([row],context,4,0))
        for p in propose_placements([row],context,4,0):
            x,y,w,h=p['rectangle_xywh']; cx,cy,cw,ch=p['core_box_xywh']
            # The 10x40 exemplar is the core; a quarter of each side is added around it.
            self.assertEqual((cw,ch,w,h,cx-x,cy-y),(10,40,16,60,3,10))
            self.assertTrue(p['quarter_turn'])  # tall core: the long-side-horizontal patch is turned back
            self.assertTrue(2<=x and x+w<=198 and 2<=y and y+h<=198)
            self.assertTrue(x+w<=10 or x>=20 or y+h<=10 or y>=50)  # clear of the real defect
        upright={'a':{'id':'a','class_id':0,'gan_identity':{'crop':{**rule,'canonical_orientation':False}}}}
        self.assertFalse(any(p['quarter_turn'] for p in propose_placements([row],upright,4,0)))
        # A box as wide as the image still leaves room for its margin and the blend border.
        wide={**row,'targets':[[0,0,.4,1,.5]]}
        for p in propose_placements([wide,{**row,'export_stem':'clear','targets':[[0,.9,.9,.95,.95]]}],context,4,1):
            x,y,w,h=p['rectangle_xywh']
            self.assertTrue(2<=x and x+w<=198 and 2<=y and y+h<=198)

    def test_a_crop_pasted_through_its_mirrored_placement_lands_where_it_came_from(self):
        import cv2
        from PIL import Image
        from augmentation.hqgan.data import context_patch,crop_settings
        noise=np.random.default_rng(0).integers(0,256,(300,300,3),dtype=np.uint8)
        scene=cv2.normalize(cv2.GaussianBlur(noise,(0,0),6),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        rule=crop_settings({'margin':.125,'min_box_side':8,'canonical_orientation':True})
        for box in ((60,100,180,140),(100,50,140,210)):  # wide, tall
            x1,y1,x2,y2=box
            patch,window,turned=context_patch(Image.fromarray(scene),box,rule)
            self.assertEqual(turned,y2-y1>x2-x1)
            placement={'rectangle_xywh':[window[0],window[1],window[2]-window[0],window[3]-window[1]],'quarter_turn':turned}
            pasted=paste_patch(scene.copy(),np.asarray(patch).copy(),placement)
            # Blending a region onto itself changes nothing, so any shift or wrong turn would show.
            self.assertLessEqual(int(np.abs(pasted.astype(int)-scene.astype(int)).max()),3)
            upside_down=paste_patch(scene.copy(),np.ascontiguousarray(np.rot90(np.asarray(patch),2)),placement)
            self.assertGreater(int(np.abs(upside_down.astype(int)-scene.astype(int)).max()),10)

    def test_paste_turns_the_patch_back_and_the_box_with_it(self):
        patch=np.zeros((8,16,3),np.uint8); patch[:,:4]=255  # bright band at the left end, long side horizontal
        background=np.full((100,100,3),100,np.uint8)
        out=paste_patch(background,patch,{'rectangle_xywh':[40,20,16,32],'quarter_turn':True})
        # Clockwise: the left end of the stored patch becomes the top of the tall rectangle.
        self.assertGreater(float(out[23:27,44:52].mean()),float(out[30:34,44:52].mean())+50)
        self.assertEqual(turned_box([0,0,.25,1],True),[0,0,1,.25])
        self.assertEqual(turned_box([.1,.2,.3,.4],False),[.1,.2,.3,.4])
        flat=paste_patch(background,patch,{'rectangle_xywh':[40,20,32,16]})
        self.assertGreater(float(flat[24:32,43:47].mean()),float(flat[24:32,50:54].mean())+50)
        np.testing.assert_array_equal(out[:19],background[:19]); np.testing.assert_array_equal(out[:,:39],background[:,:39])

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
