import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import yaml
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_detection import normalized_annotations, assign_status, group_records, scan, export_reviewed, CHECKS, FIELDS, digest

class PreparationTests(unittest.TestCase):
    def test_polygon_mapping_and_duplicate_rows(self):
        line='0 0.1 0.2 0.9 0.2 0.5 0.8'
        rows,issues,duplicates=normalized_annotations(line+'\n'+line,['crack'],{'crack':'crack'},['dent','crack','paint_peeling'])
        self.assertEqual((len(rows),issues,duplicates),(1,[],1))
        self.assertEqual(rows[0]['target_id'],1)
        self.assertEqual(rows[0]['coordinates'],[.1,.2,.9,.2,.5,.8])
        self.assertAlmostEqual(rows[0]['box'][2],.8)

    def test_invalid_and_unmapped(self):
        rows,issues,_=normalized_annotations('0 nan .5 .2 .2\n1 .5 .5 .2 .2',['crack','scratch'],{'crack':'crack'},['crack'])
        self.assertEqual(len(rows),1)
        self.assertTrue(any(i.startswith('invalid_label') for i in issues))
        self.assertIn('unmapped_class:scratch',issues)
        self.assertEqual(assign_status({'issues':issues},None),'excluded_unmapped')

    def test_review_hash_and_background_gate(self):
        r={'issues':[],'annotations':[],'image_sha256':'image','label_sha256':'label','schema_sha256':'schema'}
        review={'action':'approve','image_sha256':'image','label_sha256':'label','schema_sha256':'schema','reviewer':'person',**{k:'yes' for k in CHECKS}}
        self.assertEqual(assign_status(r,review),'quarantine_incomplete_approval')
        review['background_verified']='yes'
        self.assertEqual(assign_status(r,review),'approved')
        review['label_sha256']='old'
        self.assertEqual(assign_status(r,review),'quarantine_stale_review')
        self.assertEqual(assign_status(r,None),'quarantine_empty')

    def test_transitive_cross_source_and_original_family(self):
        rs=[{'image':'a/1.png','source':'D4','image_sha256':'a','phash':0},
            {'image':'a/1_2.png','source':'D4','image_sha256':'b','phash':255},
            {'image':'b/x.png','source':'D1','image_sha256':'c','phash':254},
            {'image':'b/y.png','source':'D2','image_sha256':'d','phash':2**64-1}]
        groups=group_records(rs,1)
        self.assertEqual(sorted(map(len,groups)),[1,3])
        self.assertEqual(rs[0]['group'],rs[2]['group'])

    def fixture(self, root):
        config={'task':{'core_classes':['dent','crack','paint_peeling']},
                'datasets':{'unmapped_class_policy':'exclude_images','annotation_policy':{'review_ledger':'reviews.json'},
                            'source_ids':{'D1':'one','D2':'two','D4':'four'},'sources':{}},
                'dedup':{'hamming_threshold':0},'split':{'train_pool':.7,'val':.1,'test':.2,'split_seed':1234,'stratify_by_source':True}}
        rng=np.random.default_rng(5)
        for did,folder,key in [('D1','Innovation Hangar','one'),('D2','aircraft_skin_defects.v1','two'),('D4','airscraft-skin-crack-dent','four')]:
            base=root/'data/raw'/folder; (base/'train/images').mkdir(parents=True); (base/'train/labels').mkdir()
            names=['crack','dent'] if did=='D4' else ['crack','dent','paint-off']
            config['datasets']['sources'][key]={'class_map':{'crack':'crack','dent':'dent','paint-off':'paint_peeling'},'source_label_names':names}
            if did!='D4': (base/'data.yaml').write_text(yaml.safe_dump({'names':names}))
            for i in range(20):
                Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(base/'train/images'/f'scene{i}.png')
                (base/'train/labels'/f'scene{i}.txt').write_text(''.join(f'{k} .5 .5 .2 .2\n' for k in range(len(names))))
        return config

    def test_scan_review_export_preserves_raw_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=self.fixture(root)
            before={str(p):digest(p) for p in (root/'data/raw').rglob('*') if p.is_file()}
            records,summary=scan(root,config,root/'first')
            self.assertEqual(len(records),60)
            with self.assertRaisesRegex(ValueError,'No approved'): export_reviewed(root,records,config,root/'first')
            with (root/'approved.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader()
                for r in records:
                    writer.writerow({**{k:r[k] for k in ['image','source','image_sha256','label_sha256','schema_sha256']},'action':'approve','reviewer':'tester',**{k:'yes' for k in CHECKS}})
            records,summary=scan(root,config,root/'second',root/'approved.csv')
            export_reviewed(root,records,config,root/'second')
            exported=json.loads((root/'second/dataset/split_manifest.json').read_text())
            self.assertEqual(len(exported),60)
            membership={}
            for r in exported: membership.setdefault(r['group'],set()).add(r['split'])
            self.assertTrue(all(len(v)==1 for v in membership.values()))
            counts=json.loads((root/'second/dataset/counts.json').read_text())
            self.assertTrue(all(counts['annotations'][sp].get(c,0)>0 for sp in ['train','val','test'] for c in config['task']['core_classes']))
            self.assertEqual(before,{str(p):digest(p) for p in (root/'data/raw').rglob('*') if p.is_file()})
            self.assertEqual(yaml.safe_load((root/'second/dataset/data.yaml').read_text())['names'],config['task']['core_classes'])

    def test_exact_image_conflict_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=self.fixture(root)
            first=root/'data/raw/Innovation Hangar/train/images/scene0.png'
            second=root/'data/raw/aircraft_skin_defects.v1/train/images/scene0.png'
            second.write_bytes(first.read_bytes())
            (second.parent.parent/'labels/scene0.txt').write_text('0 .2 .2 .1 .1\n')
            records,_=scan(root,config,root/'scan')
            conflicted=[r for r in records if r['image_sha256']==digest(first)]
            self.assertEqual(len(conflicted),2)
            self.assertTrue(all(r['status']=='quarantine_conflicting_labels' for r in conflicted))

if __name__=='__main__': unittest.main()
