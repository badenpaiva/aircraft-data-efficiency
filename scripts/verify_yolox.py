"""Local YOLOX parity and optional training-set memorization diagnostic, not research evaluation."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from detection_pipeline import resolve, sha256, write_json
import study.backends as backends
from study.metrics import evaluate
from study.runner import save_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--subset', default='seed0_f00500')
    parser.add_argument('--output', required=True)
    parser.add_argument('--overfit-epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted diagnostic in the same output directory')
    args = parser.parse_args()
    if args.overfit_epochs < 0: parser.error('overfit-epochs must be nonnegative')
    root = resolve(args.study)
    plan = json.loads((root/'plan.json').read_text())
    out = resolve(args.output); out.mkdir(parents=True, exist_ok=args.resume)
    cfg = plan['config']; names = plan['names']
    stems = set(plan['subsets'][args.subset]['stems'])
    pool = [r for r in plan['records'] if r['split']=='train' and r['export_stem'] in stems]
    selected = []; groups = set()
    for c in range(len(names)):
        candidates = [r for r in pool if r['group'] not in groups and any(t[0]==c for t in r['targets'])]
        row = max(candidates, key=lambda r: max((t[3]-t[1])*(t[4]-t[2]) for t in r['targets'] if t[0]==c)/len(r['targets']))
        selected.append(row); groups.add(row['group'])
    for row in selected:
        assert sha256(row['image_path']) == row['image_sha256']
        assert sha256(row['label_path']) == row['export_label_sha256']
    if args.resume:
        assert json.loads((out/'selected_training_images.json').read_text()) == selected
    else:
        write_json(out/'selected_training_images.json', selected)
    settings = copy.deepcopy(cfg['training'])
    settings.update(image_size=320, batch_size=len(selected), epochs=args.overfit_epochs)
    backends.seed_all(0, settings['cpu_threads'])
    detector = backends.Detector(cfg['models']['yolox'], names, settings, cfg['yolox'], 'cpu')
    model = backends.load_yolox(cfg['yolox'], len(names))
    from yolox.data.data_augment import preproc, TrainTransform
    from yolox.utils import postprocess
    checks = {}
    # Independent official preprocessing reference, including non-square fixtures.
    rng = np.random.default_rng(42)
    for h,w in [(80,160),(160,80),(97,133),(640,640)]:
        array = rng.integers(0,256,(h,w,3),dtype=np.uint8)
        file = out/f'fixture_{h}_{w}.png'; cv2.imwrite(str(file), array)
        actual, ratio = detector._preprocess({'image_path':str(file)})
        expected, reference_ratio = preproc(array,(320,320))
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert ratio == reference_ratio
    checks['official_preprocessing_exact_four_shapes'] = True
    pretrained = torch.load(resolve(cfg['models']['yolox']['weights']),map_location='cpu',weights_only=False)['model']
    state = model.state_dict()
    excluded = [k for k in state if k not in pretrained or state[k].shape!=pretrained[k].shape]
    assert len(excluded)==6 and all(k.startswith('head.cls_preds.') for k in excluded), excluded
    checks['pretrained_exclusions_only_three_class_heads'] = excluded
    job = f'{args.subset}__baseline__yolox'
    checkpoint = root/'runs'/job/'training/last.pt'
    detector.load(checkpoint)
    evaluation = cfg['evaluation']
    actual,_ = detector.predict(selected,evaluation)
    reference = {}
    # Separate model, official image preprocessing/postprocess, independent box conversion.
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']); model.eval()
    with torch.no_grad():
        for row in selected:
            tensor,ratio = preproc(cv2.imread(row['image_path']),(320,320))
            prediction = postprocess(model(torch.from_numpy(tensor).unsqueeze(0)),len(names),
                evaluation['confidence_floor'],evaluation['nms_iou'],class_agnostic=False)[0]
            results = []
            if prediction is not None:
                boxes = prediction[:,:4].numpy()/ratio
                boxes /= np.array([row['width'],row['height'],row['width'],row['height']])
                for p,b in zip(prediction.numpy(),np.clip(boxes,0,1)):
                    results.append([int(p[6]),float(p[4])*float(p[5]),*b.tolist()])
            reference[row['export_stem']] = sorted(results,key=lambda p:-p[1])[:evaluation['max_detections']]
            np.testing.assert_allclose(actual[row['export_stem']],reference[row['export_stem']],rtol=1e-5,atol=1e-6)
    checks['checkpoint_official_inference_parity'] = True
    # Feed perfect boxes through the actual evaluator to verify class/box conventions.
    perfect = {r['export_stem']:[[t[0],1.0,*t[1:]] for t in r['targets']] for r in selected}
    metrics = evaluate(selected,perfect,names,evaluation)
    assert abs(metrics['map50_95']-1)<1e-8
    checks['perfect_prediction_evaluator_ap'] = metrics['map50_95']
    write_json(out/'checks.json',checks)
    print('Preprocessing, pretrained transfer, inference and evaluator checks passed.',flush=True)
    if args.overfit_epochs:
        checkpoint_path = out/'memorization/last.pt'
        if args.resume:
            saved_epoch = torch.load(checkpoint_path,map_location='cpu',weights_only=False)['epoch']
            assert saved_epoch <= args.overfit_epochs
            history = out/'memorization/results.csv'
            original_history = history.read_text()
            backup = out/'memorization'/f'history_before_resume_epoch{saved_epoch}.csv'
            if not backup.exists(): backup.write_text(original_history)
            with history.open(newline='') as file:
                reader = csv.DictReader(file); fields = reader.fieldnames
                rows = [r for r in reader if int(r['epoch']) <= saved_epoch]
            with history.open('w',newline='') as file:
                writer = csv.DictWriter(file,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        # Compare the production trainer's actual tensors and labels with official TrainTransform.
        expected = []
        max_labels = max(len(r['targets']) for r in selected)
        for row in selected:
            labels = np.array([[x1*row['width'],y1*row['height'],x2*row['width'],y2*row['height'],c]
                               for c,x1,y1,x2,y2 in row['targets']],dtype=np.float32)
            expected.append(TrainTransform(max_labels,flip_prob=0,hsv_prob=0)(cv2.imread(row['image_path']),labels,(320,320)))
        calls = [0]
        def inspect_batch(module, inputs):
            if not module.training: return
            for image,label in zip(inputs[0],inputs[1]):
                matches = [target for pixels,target in expected if np.array_equal(image.cpu().numpy(),pixels)]
                assert len(matches)==1
                np.testing.assert_allclose(label.cpu().numpy(),matches[0],atol=1e-4,rtol=1e-6)
            calls[0] += 1
        original = backends.load_yolox
        def instrumented(*a,**kw):
            m = original(*a,**kw); m.register_forward_pre_hook(inspect_batch); return m
        backends.load_yolox = instrumented
        try:
            print('Diagnostic: training and evaluating on the SAME three training images. No validation/test data.',flush=True)
            detector.train(selected,selected,None,out/'memorization',0,smoke=True)
        finally:
            backends.load_yolox = original
        predictions,times = detector.predict(selected,evaluation)
        final = evaluate(selected,predictions,names,evaluation,times)
        final['diagnostic_only'] = True
        final['evaluation_is_training_data'] = True
        write_json(out/'memorization_metrics.json',final)
        save_preview(selected,predictions,out/'previews',.25,len(selected))
        checks['production_training_batches_match_official_transform_this_invocation'] = calls[0]
        checks['memorization_map50'] = final['map50']
        checks['memorization_gate_map50_at_least_0p9'] = final['map50'] >= .9
        write_json(out/'checks.json',checks)
        print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
