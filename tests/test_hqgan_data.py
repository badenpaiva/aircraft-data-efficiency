import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from augmentation.hqgan.data import context_patch, crop_settings, pixel_box, prepare_patches, selected_rows
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

    def test_crop_rule_margin_mirrored_context_and_quarter_turn(self):
        source = np.random.default_rng(1).integers(0, 256, (40, 30, 3), dtype=np.uint8)  # 30 wide, 40 high
        rgb = Image.fromarray(source)
        rule = crop_settings({'margin': .25, 'min_box_side': 4, 'canonical_orientation': True})
        # Wide box inside the photograph: the window is the box plus a quarter of each side.
        patch, window, turned = context_patch(rgb, (8, 16, 20, 24), rule)
        self.assertEqual((window, turned, patch.size), ([5, 14, 23, 26], False, (18, 12)))
        np.testing.assert_array_equal(np.asarray(patch), source[14:26, 5:23])
        # Tall box: stored long side horizontal; the paste's clockwise turn restores it exactly.
        patch, window, turned = context_patch(rgb, (10, 8, 16, 28), rule)
        self.assertEqual((window, turned, patch.size), ([8, 3, 18, 33], True, (30, 10)))
        np.testing.assert_array_equal(np.rot90(np.asarray(patch), -1), source[3:33, 8:18])
        # Box in the corner: the patch keeps its size and the missing context is a mirror image.
        patch, window, turned = context_patch(rgb, (0, 0, 12, 8), rule)
        self.assertEqual((window, patch.size), ([-3, -2, 15, 10], (18, 12)))
        pixels = np.asarray(patch)
        np.testing.assert_array_equal(pixels[2:, 3:], source[0:10, 0:15])
        np.testing.assert_array_equal(pixels[0, 3:], source[2, 0:15])
        np.testing.assert_array_equal(pixels[2:, 0], source[0:10, 3])
        # The same rule without the orientation step leaves the tall box upright.
        upright = crop_settings({'margin': .25, 'min_box_side': 4})
        self.assertEqual(context_patch(rgb, (10, 8, 16, 28), upright)[0].size, (10, 30))
        # Defaults are the first study's crops: the box exactly as annotated.
        self.assertEqual(crop_settings(), {'margin': 0.0, 'min_box_side': 8, 'canonical_orientation': False})
        patch, window, turned = context_patch(rgb, (10, 8, 16, 28), crop_settings())
        self.assertEqual((window, turned, patch.size), ([10, 8, 16, 28], False, (6, 20)))
        for bad in ({'margin': .6}, {'margin': True}, {'margin': -.1}, {'min_box_side': 0},
                    {'canonical_orientation': 1}, {'unknown': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                crop_settings(bad)

    def test_export_records_the_crop_rule_and_the_loader_enforces_it(self):
        from augmentation.hqgan.training import PatchDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = test_detection_pipeline.DetectionTests().fixture(root)
            rule = {'margin': .25, 'min_box_side': 1, 'canonical_orientation': True}
            report = prepare_patches(data, config, root / 'context', crop=rule)
            self.assertEqual(report['crop'], crop_settings(rule))
            records = json.loads((root / 'context/patches.json').read_text())
            for record in records:
                x1, y1, x2, y2 = record['pixel_box_xyxy']
                wx1, wy1, wx2, wy2 = record['crop_window_xyxy']
                self.assertEqual((wx1, wy1, wx2, wy2), (x1 - 2, y1 - 2, x2 + 2, y2 + 2))  # 8 px boxes
                self.assertFalse(record['quarter_turn'])
                with Image.open(root / 'context' / record['patch']) as image:
                    self.assertEqual(image.size, (wx2 - wx1, wy2 - wy1))
            name = records[0]['class_name']
            loader = PatchDataset(root / 'context', name, 64, config, rule)
            self.assertEqual(loader.identity['crop'], crop_settings(rule))
            with self.assertRaisesRegex(ValueError, 'different crop rule'):
                PatchDataset(root / 'context', name, 64, config, {**rule, 'margin': 0.0})
            # A minimum side larger than the fixture's boxes leaves nothing to train on.
            with self.assertRaisesRegex(ValueError, 'lacks usable'):
                prepare_patches(data, config, root / 'strict', crop={**rule, 'min_box_side': 9})
            # An export from before the rule was recorded holds tight boxes.
            prepare_patches(data, config, root / 'first', min_side=1)
            manifest = json.loads((root / 'first/manifest.json').read_text())
            del manifest['crop']
            (root / 'first/manifest.json').write_text(json.dumps(manifest))
            tight = {'margin': 0.0, 'min_box_side': 1, 'canonical_orientation': False}
            self.assertEqual(PatchDataset(root / 'first', name, 64, config, tight).identity['crop'], tight)
            with self.assertRaisesRegex(ValueError, 'different crop rule'):
                PatchDataset(root / 'first', name, 64, config, rule)

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
