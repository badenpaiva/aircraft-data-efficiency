# Aircraft data efficiency

This project measures aircraft-defect detection performance on **SUTD aircraft_AI1 v4** as the amount of real training data and HQGAN-generated augmentation changes. The classes are `missing-rivet`, `rust`, and `scratch`.

The pipeline is: prepare SUTD data, freeze training subsets, train a GAN for each class and subset, generate and screen defect patches, blend patches into training images, train detectors, and evaluate on fixed real validation images.

The current run uses screening-pass patches, automatic placement, and whole-patch bounding boxes. These are approximate labels; review and generated-defect box refinement are the planned changes to this pipeline.

| Document | Purpose |
|---|---|
| [Pipeline](docs/pipeline.md) | Setup, generation, training, resuming, results and evaluation |
| [Dataset preparation](docs/prepare_detection.md) | SUTD source, class mapping, splits and provenance |
| [HQGAN design](docs/hqgan.md) | Model settings, reconstruction choices and limitations |
| [Image review](docs/review.md) | Current label policy and the supported review/box-refinement workflow |
| [Project map](PROJECT_MAP.md) | What each folder contains and what must be retained |

## Current experiment

- Source study: `results/studies/sutd_full_v1`.
- Current detector study: `results/studies/sutd_whole_patch_seed0_v1`.
- Seed 0, five real-data fractions (5%, 10%, 25%, 50%, 100%), four detectors (YOLOX-S, YOLOv8n, YOLO11n, YOLO26n), and three synthetic-to-real ratios (0.5, 1, 2): **60 detector runs**.
- Each detector runs for 30 epochs. The source study plans seeds 0, 1 and 2; those additional seeds are not part of the current detector queue.
- Configuration: `config/experiment.yaml`, `config/detection_study.yaml`, and `config/hqgan_sutd.yaml`.

From the repository root in PowerShell, inspect progress:

```powershell
Get-Content results/studies/sutd_whole_patch_seed0_v1/training.stdout.log -Tail 20 -Wait
```

To resume after the existing training process has stopped:

```powershell
.\.venv\Scripts\python.exe scripts/whole_patch_study.py run --study results/studies/sutd_whole_patch_seed0_v1 --device cpu
```

Do not start a second trainer on the same study directory. Completed runs are reused. See the pipeline guide for the full sequence and recovery options.
