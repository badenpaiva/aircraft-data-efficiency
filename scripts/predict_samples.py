"""Save annotated predictions and boxes for a bounded set of validation images."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from detection_pipeline import resolve, load_config, check_dataset, device_name, checkpoint_matches, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--data')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--name', default='d1_predictions')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--device')
    args = parser.parse_args()
    if args.limit <= 0: parser.error('limit must be positive')
    if Path(args.name).name != args.name or args.name in {'.','..'}: parser.error('Invalid run name')
    config = load_config(args.config)
    _, counts, _, runtime = check_dataset(args.data or config['training']['data'], config)
    weights = resolve(args.weights)
    if not weights.is_file(): parser.error(f'Weights not found: {weights}')
    output = resolve(config['training']['project']) / args.name
    output.mkdir(parents=True, exist_ok=False)
    from ultralytics import YOLO
    model = YOLO(str(weights)); checkpoint_matches(model, runtime['names'])
    images = sorted(Path(runtime['val']).glob('*'))[:args.limit]
    records = []
    for result in model.predict(source=[str(p) for p in images], stream=True,
                                imgsz=config['training']['image_size'],
                                conf=config['metrics']['safety_recall']['confidence_threshold'],
                                device=device_name(args.device or config['training']['device']),
                                save=False, verbose=False):
        result.save(filename=str(output / Path(result.path).name))
        records.append({'image':result.path,'detections':result.summary()})
    write_json(output/'predictions.json', {'quality':counts['quality'],'split':'val','images':records})
    print(f'Predictions: {output}')


if __name__ == '__main__':
    main()
