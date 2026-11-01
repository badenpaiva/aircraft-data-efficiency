import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_detection import scan_coco


class CocoPreparationTests(unittest.TestCase):
    def test_roboflow_split_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = self.fixture(root)
            folder = root / spec['path']
            for split in ('train', 'val', 'test'):
                target = folder / ('valid' if split == 'val' else split)
                target.mkdir()
                (folder/'images'/split/'image.png').rename(target/'image.png')
                (folder/'annotations'/f'instances_{split}.json').rename(target/'_annotations.coco.json')
            rows = scan_coco(root, 'SUTD', spec, ['crack'], {}, False)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['original_split'] for r in rows}, {'train', 'val', 'test'})
            self.assertTrue(any('/valid/' in r['image'] for r in rows))

    def fixture(self, root):
        folder = root / 'data/raw/coco'
        (folder / 'annotations').mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (folder / 'images' / split).mkdir(parents=True)
            Image.new('RGB', (100, 80)).save(folder / 'images' / split / 'image.png')
            data = {'categories': [{'id': 0, 'name': 'parent'}, {'id': 7, 'name': 'crack'}],
                    'images': [{'id': 0, 'file_name': 'image.png', 'width': 100, 'height': 80}],
                    'annotations': [{'id': 0, 'image_id': 0, 'category_id': 7, 'bbox': [10, 20, 40, 20]}]}
            (folder / 'annotations' / f'instances_{split}.json').write_text(json.dumps(data))
        return {'path': 'data/raw/coco', 'class_map': {'crack': 'crack'}}

    def test_noncontiguous_categories_and_split_local_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = self.fixture(root)
            rows = scan_coco(root, 'NEW', spec, ['crack'], {}, False)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['original_split'] for r in rows}, {'train', 'val', 'test'})
            self.assertTrue(all(r['status'] == 'provisional' for r in rows))
            self.assertEqual(rows[0]['annotations'][0]['target_id'], 0)
            self.assertEqual(rows[0]['annotations'][0]['box'], [.3, .375, .4, .25])

    def test_out_of_bounds_and_empty_are_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = self.fixture(root)
            path = root / 'data/raw/coco/annotations/instances_train.json'
            data = json.loads(path.read_text()); data['annotations'][0]['bbox'] = [90, 20, 40, 20]
            path.write_text(json.dumps(data))
            rows = scan_coco(root, 'NEW', spec, ['crack'], {}, False)
            self.assertEqual(rows[0]['status'], 'quarantine_invalid')
            data['annotations'] = []; path.write_text(json.dumps(data))
            self.assertEqual(scan_coco(root, 'NEW', spec, ['crack'], {}, False)[0]['status'], 'quarantine_empty')

    def test_dangling_reference_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = self.fixture(root)
            path = root / 'data/raw/coco/annotations/instances_train.json'
            data = json.loads(path.read_text()); data['annotations'][0]['image_id'] = 99
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'dangling'):
                scan_coco(root, 'NEW', spec, ['crack'], {}, False)
            data['annotations'][0]['image_id'] = 0; data['images'][0]['file_name'] = '../image.png'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                scan_coco(root, 'NEW', spec, ['crack'], {}, False)
