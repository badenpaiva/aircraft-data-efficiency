import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from augmentation.hqgan.data import pixel_box, prepare_patches, selected_rows
from detection_pipeline import sha256
import test_detection_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_hqgan


class HQGANDataTests(unittest.TestCase):
    def test_training_loader_rejects_tampered_patches_and_split_provenance(self):
        from augmentation.hqgan.training import PatchDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = test_detection_pipeline.DetectionTests().fixture(root)
            export = root/'patches'
            prepare_patches(data, config, export, min_side=1)
            records_file = export/'patches.json'
            original = records_file.read_text()
            records = json.loads(original)
            class_name = records[0]['class_name']
            loader = PatchDataset(export,class_name,64,config)
            self.assertEqual(tuple(loader[0].shape),(3,64,64))
            records[0]['source_split'] = 'test'
            records_file.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                PatchDataset(export,class_name,64,config)
            records_file.write_text(original)
            (export/records[0]['patch']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                PatchDataset(export,class_name,64,config)

    def test_null_subset_cannot_silently_use_full_training_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subset = root / 'subset.json'
            subset.write_text('null')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                prepare_hqgan.main(['--subset', str(subset), '--output', str(root / 'out')])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / 'out').exists())

    def test_subset_rejects_leakage_and_ambiguous_membership(self):
        manifest = [{'export_stem': 'a', 'split': 'train'}, {'export_stem': 'b', 'split': 'val'}]
        for subset in ([], ['b'], ['a', 'a'], ['missing'], 'a', [42]):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                selected_rows(manifest, subset)
        self.assertEqual(selected_rows(manifest), [manifest[0]])

    def test_geometry_rejects_invalid_boxes(self):
        self.assertEqual(pixel_box('1 .5 .5 .5 .5', 100, 80, 3), (1, (25, 20, 75, 60)))
        for line in ('0 nan .5 .2 .2', '0 .5 .5 -1 .2', '3 .5 .5 .2 .2', '0 .1 .1 .5 .5'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                pixel_box(line, 100, 80, 3)

    def test_export_is_traceable_deterministic_and_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = test_detection_pipeline.DetectionTests().fixture(root)
            before = sha256(data.parent / 'split_manifest.json')
            first = prepare_patches(data, config, root / 'patches1', min_side=1)
            second = prepare_patches(data, config, root / 'patches2', min_side=1)
            self.assertEqual(first, second)
            self.assertEqual(before, sha256(data.parent / 'split_manifest.json'))
            sources = {r['export_stem'] for r in first['training_sources']}
            frozen = json.loads((data.parent / 'split_manifest.json').read_text())
            self.assertEqual(sources, {r['export_stem'] for r in frozen if r['split'] == 'train'})
            records = json.loads((root / 'patches1/patches.json').read_text())
            self.assertGreater(len(records), 0)
            for record in records:
                self.assertIn(record['source_stem'], sources)
                self.assertEqual(record['source_split'], 'train')
                patch = root / 'patches1' / record['patch']
                self.assertEqual(sha256(patch), record['sha256'])
                with Image.open(patch) as image:
                    x1, y1, x2, y2 = record['pixel_box_xyxy']
                    self.assertEqual(image.size, (x2-x1, y2-y1))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare_patches(data, config, root / 'patches1')
            with self.assertRaisesRegex(ValueError, 'outside'):
                prepare_patches(data, config, data.parent / 'patches')
            label = next((data.parent / 'train/labels').glob('*.txt'))
            label.write_text('0 .5 .5 .2 .2\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare_patches(data, config, root / 'changed')

    def test_unusable_corpus_is_marked_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = test_detection_pipeline.DetectionTests().fixture(root)
            with self.assertRaisesRegex(ValueError, 'lacks usable'):
                prepare_patches(data, config, root / 'patches', min_side=10000)
            self.assertFalse((root / 'patches/manifest.json').exists())
            self.assertEqual(json.loads((root / 'patches/failure.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
