import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detection_pipeline import check_dataset, match_counts, smoke_subset
import test_prepare_detection
from prepare_detection import scan, export_reviewed


class DetectionTests(unittest.TestCase):
    def fixture(self, root):
        config = test_prepare_detection.PreparationTests().fixture(root)
        config['datasets']['annotation_policy']['require_domain_and_completeness_review'] = False
        records,_ = scan(root,config,root/'prepared')
        export_reviewed(root,records,config,root/'prepared')
        return config,root/'prepared/dataset/data.yaml'

    def test_dataset_integrity_and_strict_reenable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,data=self.fixture(root)
            _,counts,_,runtime=check_dataset(data,config)
            self.assertEqual(counts['quality'],'provisional')
            config['datasets']['annotation_policy']['require_domain_and_completeness_review']=True
            with self.assertRaisesRegex(ValueError,'Manual review'): check_dataset(data,config)
            config['datasets']['annotation_policy']['require_domain_and_completeness_review']=False
            label=next((data.parent/'train/labels').glob('*.txt'))
            label.write_text('0 .1 .1 .1 .1\n')
            with self.assertRaisesRegex(ValueError,'changed'): check_dataset(data,config)

    def test_leakage_detection_and_smoke_split_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,data=self.fixture(root)
            _,_,manifest,runtime=check_dataset(data,config)
            smoke=smoke_subset(runtime,root,per_split=6)
            self.assertNotIn('test',smoke)
            train=set(Path(smoke['train']).read_text().splitlines())
            val=set(Path(smoke['val']).read_text().splitlines())
            self.assertFalse(train & val)
            other=next(r for r in manifest if r['split'] != manifest[0]['split'])
            other['group']=manifest[0]['group']
            (data.parent/'split_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'crosses'): check_dataset(data,config)

    def test_fixed_recall_does_not_double_count_or_match_wrong_classes(self):
        predictions=[[0,.9,0,0,1,1],[0,.8,0,0,1,1],[1,.7,0,0,1,1]]
        targets=[[0,0,0,1,1],[1,2,2,3,3]]
        self.assertEqual(match_counts(predictions,targets,2),
                         [{'tp':1,'fp':1,'fn':0},{'tp':0,'fp':1,'fn':1}])


if __name__=='__main__': unittest.main()
