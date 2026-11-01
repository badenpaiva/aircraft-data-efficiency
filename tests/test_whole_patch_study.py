import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

SPEC = importlib.util.spec_from_file_location('whole_patch_study', Path(__file__).resolve().parents[1] / 'scripts/whole_patch_study.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
from study.screening import fingerprint, settings_for


class WholePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).resolve().parents[1] / 'tmp').mkdir(exist_ok=True)

    def test_pass_only_boxes_provenance_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as tmp:
            root = Path(tmp)
            checkpoint = root / 'weights.pt'
            checkpoint.write_bytes(b'test checkpoint')
            patch_file = root / 'patch.png'
            cv2.imwrite(str(patch_file), np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8))
            row = {'id': 'defect_0', 'path': str(patch_file), 'sha256': w.sha256(patch_file),
                   'checkpoint': str(checkpoint), 'checkpoint_sha256': w.sha256(checkpoint),
                   'class_id': 0, 'class_name': 'defect', 'gan_identity': {},
                   'approved': False, 'reviewer': '', 'defect_box_xyxy': None}
            settings = settings_for()
            row['screening'] = {'version': 1, 'stage': 'candidates', 'settings': settings,
                                'outcome': 'pass', 'fingerprint': fingerprint(row, 'candidates', 'subset', settings)}
            flagged = copy.deepcopy(row)
            flagged['id'] = 'flagged'
            flagged['screening']['outcome'] = 'needs_review'
            bundle = {'study_subset_identity': 'subset', 'smoke': False, 'candidates': [row, flagged]}
            subset = {'identity': 'subset', 'seed': 0, 'stems': ['real']}
            self.assertEqual(list(w.select_candidates(bundle, subset, ['defect'])), ['defect_0'])
            bad = copy.deepcopy(bundle)
            bad['candidates'][0]['class_name'] = 'changed'
            with self.assertRaises(ValueError):
                w.select_candidates(bad, subset, ['defect'])
            source = root / 'source'
            queue = source / 'gan' / 'key' / 'candidates_review.json'
            queue.parent.mkdir(parents=True)
            w.write_json(queue, bundle)
            queue_hash = w.sha256(queue)
            bg = root / 'bg.png'
            label = root / 'bg.txt'
            cv2.imwrite(str(bg), np.full((100, 100, 3), 120, dtype=np.uint8))
            label.write_text('0 0.15 0.15 0.1 0.1\n')
            real = {'export_stem': 'real', 'image_path': str(bg), 'label_path': str(label),
                    'image_sha256': w.sha256(bg), 'export_label_sha256': w.sha256(label),
                    'targets': [[0, .1, .1, .2, .2]], 'width': 100, 'height': 100, 'split': 'train'}
            payload = {'subsets': {'key': subset}, 'records': [real], 'names': ['defect'],
                       'plan_identity': 'plan', 'config': {'arms': {'hqgan_2p0': {'kind': 'hqgan', 'synthetic_ratio': 2}}}}
            output = root / 'out'
            result = w.make_composites(source, output, payload, 'key')
            self.assertEqual(len(result['records']), 2)
            self.assertEqual(w.sha256(queue), queue_hash)
            for r in result['records']:
                self.assertEqual(r['targets'][0], real['targets'][0])
                x, y, width, height = r['augmentation']['placement']['rectangle_xywh']
                self.assertEqual(r['targets'][1], [0, x/100, y/100, (x+width)/100, (y+height)/100])
                self.assertFalse(r['augmentation']['human_reviewed'])
                self.assertFalse(r['augmentation']['candidate']['approved'])
                self.assertEqual(r['augmentation']['defect_box_xyxy'], [0, 0, 1, 1])
            self.assertEqual(w.make_composites(source, output, payload, 'key'), result)
            Path(result['records'][0]['label_path']).write_text('changed')
            with self.assertRaises(ValueError):
                w.make_composites(source, output, payload, 'key')

    def test_nested_ratios_and_insufficient_quota(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as tmp:
            directory = Path(tmp)
            records = [{'export_stem': str(i)} for i in range(4)]
            (directory / 'whole_patch' / 'key').mkdir(parents=True)
            w.write_json(directory / 'whole_patch' / 'key' / 'manifest.json', {'records': records})
            payload = {'config': {'arms': {'half': {'synthetic_ratio': .5}, 'double': {'synthetic_ratio': 2}}}}
            with patch.object(w, 'validate_extension'), patch.object(w, 'verify_records'), patch.object(w.runner, 'subset_records', return_value=[{}, {}]):
                low, nlow = w.whole_patch_records(directory, payload, {'subset': 'key', 'arm': 'half', 'seed': 0})
                high, nhigh = w.whole_patch_records(directory, payload, {'subset': 'key', 'arm': 'double', 'seed': 0})
                self.assertEqual((nlow, nhigh), (1, 4))
                self.assertEqual(low[2:], high[2:3])
                payload['config']['arms']['double']['synthetic_ratio'] = 3
                with self.assertRaises(ValueError):
                    w.whole_patch_records(directory, payload, {'subset': 'key', 'arm': 'double', 'seed': 0})


if __name__ == '__main__':
    unittest.main()
