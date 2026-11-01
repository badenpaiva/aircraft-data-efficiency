"""Repackage the raw Innovation Hangar dataset for manual detection review."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'aircraft_cvat_detection.zip'
    if target.exists():
        raise FileExistsError(target)
    records, counts = [], Counter()
    with ZipFile(args.source) as src:
        config = yaml.safe_load(src.read('data.yaml'))
        names = config['names']
        entries = set(src.namelist())
        prepared = []
        for image in sorted(entries):
            p = PurePosixPath(image)
            if len(p.parts) != 3 or p.parts[1] != 'images' or p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            split = {'valid': 'val'}.get(p.parts[0], p.parts[0])
            label = str(p.parent.parent / 'labels' / (p.stem + '.txt'))
            if label not in entries:
                raise ValueError(f'Missing label: {label}')
            output_rows, converted = [], []
            for row_number, line in enumerate(src.read(label).decode().splitlines(), 1):
                if not line.strip():
                    continue
                tokens = line.split()
                cls = int(tokens[0])
                values = list(map(float, tokens[1:]))
                if not 0 <= cls < len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise ValueError(f'Invalid annotation: {label}:{row_number}')
                if len(values) == 4:
                    box = values
                    counts['original_boxes'] += 1
                elif len(values) >= 6 and len(values) % 2 == 0:
                    xs, ys = values[::2], values[1::2]
                    box = [(min(xs)+max(xs))/2, (min(ys)+max(ys))/2, max(xs)-min(xs), max(ys)-min(ys)]
                    converted.append(row_number)
                    counts['polygons_to_boxes'] += 1
                else:
                    raise ValueError(f'Unsupported annotation: {label}:{row_number}')
                if box[2] <= 0 or box[3] <= 0:
                    raise ValueError(f'Degenerate box: {label}:{row_number}')
                output_rows.append(str(cls) + ' ' + ' '.join(f'{v:.10g}' for v in box))
            dst_image = f'images/{split}/{p.name}'
            dst_label = f'labels/{split}/{p.stem}.txt'
            prepared.append((image, dst_image, dst_label, '\n'.join(output_rows)+'\n' if output_rows else '', split))
            records.append({'source_image': image, 'import_image': dst_image, 'source_label': label, 'polygon_rows_converted': converted})
            counts['images'] += 1
        with ZipFile(target, 'x', compression=ZIP_DEFLATED) as dst:
            splits = sorted({r[4] for r in prepared})
            cfg = {'path': '.', **{s: f'{s}.txt' for s in splits}, 'names': dict(enumerate(names))}
            dst.writestr('data.yaml', yaml.safe_dump(cfg, sort_keys=False))
            for split in splits:
                dst.writestr(f'{split}.txt', '\n'.join(r[1] for r in prepared if r[4] == split)+'\n')
            for original, image, label, content, _ in prepared:
                dst.writestr(image, src.read(original))
                dst.writestr(label, content)
    with ZipFile(target) as check:
        assert check.testzip() is None
        for record in records:
            assert record['import_image'] in check.namelist()
    report = {'source': str(args.source.resolve()), 'source_sha256': hashlib.file_digest(args.source.open('rb'), 'sha256').hexdigest(), 'classes': names, 'counts': dict(counts), 'images': records}
    (args.output / 'conversion_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'archive': str(target.resolve()), 'classes': names, 'counts': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
