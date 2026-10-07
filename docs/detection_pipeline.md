# D1 detection development pipeline

The first milestone uses YOLOv8n with COCO pretrained weights to detect
`0=dent`, `1=crack`, `2=paint_peeling`. D1 is the only selected source.
Manual review is temporarily optional. Exported data and resulting metrics are
marked provisional; image provenance and semantic annotation quality remain
unverified. The full GAN/few-shot/architecture grid is not implemented yet.

## Verified development run (2026-10-07)

The complete smoke workflow passed locally on CPU (Intel Core i5-10310U):

- Training: `results/detection/d1_baseline_smoke_20261007`, one epoch using
  nine training and nine validation images at 128px; `weights/best.pt` saved.
- Evaluation: `results/detection/d1_smoke_evaluation_20261007/metrics.json`,
  nine validation images, including per-class AP and fixed-threshold recall.
- Prediction: `results/detection/d1_smoke_predictions_20261007`, six saved
  validation previews and `predictions.json`.

These verify execution only. The smoke model returned no detections at confidence
0.25 on the evaluation sample; a useful detector still needs full training.
No test-split evaluation or full 30-epoch baseline has been run.
The earlier `d1_baseline_smoke` directory records the failed weight-download
attempt and is preserved separately. Official pretrained weights are now local
at `yolov8n.pt`. Local library/plot caches live under `.ultralytics/`.

To repeat the successful checks, choose fresh output names:

```powershell
.\.venv\Scripts\python.exe scripts/train_detector.py --smoke --name d1_smoke_repeat
.\.venv\Scripts\python.exe scripts/evaluate_detector.py --weights results/detection/d1_smoke_repeat/weights/best.pt --smoke --name d1_eval_repeat
.\.venv\Scripts\python.exe scripts/predict_samples.py --weights results/detection/d1_smoke_repeat/weights/best.pt --limit 6 --name d1_predictions_repeat
```

## Setup and data

Run from the repository root. On this machine use `.\.venv\Scripts\python.exe`.
On Linux/Colab use the active environment's `python` instead.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B tools/prepare_detection.py --output data/processed/d1_provisional_v1 --export
```

Preparation never overwrites a run. If that dataset already exists, reuse it.
The raw photos and labels stay unchanged. Invalid/missing/empty labels,
unmapped classes and recorded unresolved review findings remain excluded or
quarantined. Related images remain in a single split. The exporter writes
counts, review status, content hashes and split membership alongside `data.yaml`.

The initial `d1_provisional_v1` export contains 2,787 training images, 398
validation images and 803 test images (3,988 total). It leaves out 276 images
with unmapped classes and 24 empty-label images. These are exported counts,
not manually certified independent-example counts.

## Check and train

```powershell
# Validate the dataset and print the plan (no training).
.\.venv\Scripts\python.exe scripts/train_detector.py --dry-run

# One epoch, at most nine images from each of train/val, 128px, batch 2.
# Checks pipeline mechanics only; never uses the test split for model selection.
.\.venv\Scripts\python.exe scripts/train_detector.py --smoke

# Full development baseline: 30 epochs maximum, 640px, batch 8.
.\.venv\Scripts\python.exe scripts/train_detector.py
```

The default device is CUDA if available, otherwise CPU. Full training on CPU
can be slow. Set `--device 0` for a configured NVIDIA GPU or `--device cpu` for
CPU; GPU use needs a compatible PyTorch build. The first training run downloads
the pretrained checkpoint. The scripts perform local training and save local
outputs. Existing run directories are rejected; choose a fresh `--name` for a
rerun. `--epochs`, `--batch`, `--imgsz` and `--seed` override pilot settings and
are recorded in the run manifest.

Stochastic color, geometric, mosaic and mixing augmentations are explicitly
disabled for this baseline. Source images may themselves contain inherited
transformations; disabling model augmentation does not establish clean originals.
No additional Albumentations package is required.

Outputs under `results/detection/<name>/` include `weights/best.pt`, training
curves, `results.csv`, resolved arguments, a config/version/split-hash manifest,
an environment package list and completion status. Smoke outputs are explicitly
marked as smoke runs and do not establish useful detection performance.

## Evaluate and inspect predictions

After the full baseline completes:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_detector.py --weights results/detection/d1_baseline/weights/best.pt
.\.venv\Scripts\python.exe scripts/predict_samples.py --weights results/detection/d1_baseline/weights/best.pt
```

Evaluation defaults to validation. It saves mAP50, mAP50–95, per-class AP,
library-selected precision/recall, inference timing, and separately calculated
per-class precision/recall at confidence 0.25 and matching IoU 0.5. Matching is
class-aware, confidence-ordered and one-to-one. Prediction output includes up
to 12 validation images with boxes/class names and machine-readable detections.
Use the smoke checkpoint instead only for testing these commands.

The test split is selected only with `--split test`; reserve it for a later,
fixed evaluation. Current test labels are provisional too. Do not use test
results to tune training settings.

## Restore manual approval later

Set `datasets.annotation_policy.require_domain_and_completeness_review: true`
and `unknown_review_status: quarantine` in `config/experiment.yaml`. Complete
the review CSV and export a **new** dataset version with `--reviews ... --export`.
Update `training.data` to that version. The training/evaluation scripts reject
old provisional exports when the review flag is true. Do not compare results
across changed test-set versions as if the benchmark were unchanged.

The earlier research plan's reproduction gate is not applied to this development
baseline. Method reproduction and the full research grid are separate later work.

API references: [Ultralytics training](https://docs.ultralytics.com/modes/train/)
and [validation](https://docs.ultralytics.com/modes/val/).
