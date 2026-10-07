"""Evaluate a trained detector; defaults to validation, test must be selected explicitly."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from detection_pipeline import (resolve, load_config, check_dataset, device_name, write_json,
                                metrics_dict, checkpoint_matches, sha256, versions, match_counts, smoke_subset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--data')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--name', default='d1_evaluation')
    parser.add_argument('--device')
    parser.add_argument('--smoke', action='store_true', help='Evaluate nine validation samples at 128px for a pipeline check')
    args = parser.parse_args()
    if args.smoke and args.split != 'val': parser.error('Smoke evaluation uses validation only')
    config = load_config(args.config)
    data, counts, _, runtime = check_dataset(args.data or config['training']['data'], config)
    weights = resolve(args.weights)
    if not weights.is_file(): parser.error(f'Weights not found: {weights}')
    if Path(args.name).name != args.name or args.name in {'.','..'}: parser.error('Invalid run name')
    output = resolve(config['training']['project']) / args.name
    output.mkdir(parents=True, exist_ok=False)
    if args.smoke: runtime = smoke_subset(runtime, output)
    image_size = 128 if args.smoke else config['training']['image_size']
    runtime_path = output / 'runtime_data.yaml'
    runtime_path.write_text(yaml.safe_dump(runtime), encoding='utf-8')
    from ultralytics import YOLO
    model = YOLO(str(weights))
    checkpoint_matches(model, runtime['names'])
    metrics = model.val(data=str(runtime_path), split=args.split,
                        imgsz=image_size, batch=config['training']['batch_size'],
                        workers=config['training']['workers'], device=device_name(args.device or config['training']['device']),
                        project=str(output.parent), name=output.name, exist_ok=True, plots=True, conf=0.001)
    report = {**metrics_dict(metrics, runtime['names']), 'quality':counts['quality'],
              'split':args.split, 'smoke':args.smoke, 'image_size':image_size,
              'weights':str(weights), 'weights_sha256':sha256(weights),
              'split_manifest_sha256':sha256(data.parent/'split_manifest.json'), 'versions':versions()}
    operating = config['metrics']['safety_recall']
    totals = [{'tp':0,'fp':0,'fn':0} for _ in runtime['names']]
    for result in model.predict(source=runtime[args.split], stream=True, verbose=False,
                                imgsz=image_size, conf=operating['confidence_threshold'],
                                device=device_name(args.device or config['training']['device'])):
        predictions = [[int(cid),float(conf),*box] for cid,conf,box in zip(
            result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.boxes.xyxyn.tolist())]
        label = Path(result.path).parent.parent/'labels'/(Path(result.path).stem+'.txt')
        targets = []
        for line in label.read_text().splitlines():
            cid,cx,cy,w,h = map(float,line.split())
            targets.append([int(cid),cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        for total, image_counts in zip(totals,match_counts(predictions,targets,len(totals),operating['iou_threshold'])):
            for key in total: total[key] += image_counts[key]
    report['fixed_operating_point'] = {'confidence':operating['confidence_threshold'],'iou':operating['iou_threshold'],
        'per_class':[{ 'name':name, **total,
            'precision':total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else 0.0,
            'recall':total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else None}
            for name,total in zip(runtime['names'],totals)]}
    write_json(output/'metrics.json', report)
    print(f'Metrics: {output / "metrics.json"}')


if __name__ == '__main__':
    main()
