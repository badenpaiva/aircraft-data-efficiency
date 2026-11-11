# Aircraft data efficiency

This project measures aircraft-defect detection performance on **SUTD aircraft_AI1 v4** as the amount of real training data and HQGAN-generated augmentation changes. The classes are `missing-rivet`, `rust`, and `scratch`.

The pipeline is: prepare SUTD data, freeze training subsets, train a GAN for each class and subset, generate and screen defect patches, blend patches into training images, train detectors, and evaluate on fixed real validation images.

The completed first run used screening-pass patches, automatic placement, and whole-patch bounding boxes. The pipeline has since changed: the data is re-split by capture unit, the GAN is method `v2`, and pasted patches are labelled by their core. No run has used the new pipeline yet; [next steps](docs/next_steps.md) has the status. Labels remain approximate until generated images are reviewed.

| Document | Purpose |
|---|---|
| [Pipeline](docs/pipeline.md) | Setup, generation, training, resuming, results and evaluation |
| [Dataset preparation](docs/prepare_detection.md) | SUTD source, class mapping, splits and provenance |
| [HQGAN design](docs/hqgan.md) | Model settings, reconstruction choices and limitations |
| [Image review](docs/review.md) | Current label policy and the supported review/box-refinement workflow |
| [Project map](PROJECT_MAP.md) | What each folder contains and what must be retained |
| [Next steps](docs/next_steps.md) | Audit findings, what has changed since the first run, and what to do next |

## Planned experiment

- Three real-data fractions (25%, 50%, 100%), three seeds, two detectors from one family (YOLO26n as the light model, YOLO26m as the heavy one).
- Six arms: real-only, two classical augmentation controls, and three synthetic-to-real ratios (0.5, 1, 2). **108 detector runs** and 27 GANs.
- Each detector runs for 30 epochs.
- Configuration: `config/experiment.yaml`, `config/detection_study.yaml`, and `config/hqgan_sutd.yaml`.
- Runs use a CUDA GPU automatically when PyTorch can see one. The [pipeline guide](docs/pipeline.md) covers setup on the training machine and every command.

## Completed first experiment

- Source study: `results/studies/sutd_full_v1`. Detector study: `results/studies/sutd_whole_patch_seed0_v1`.
- Seed 0, five real-data fractions (5%, 10%, 25%, 50%, 100%), four detectors (YOLOX-S, YOLOv8n, YOLO11n, YOLO26n), and three synthetic-to-real ratios (0.5, 1, 2): 60 detector runs, plus 36 real-only baseline runs (seeds 0 and 1).
- These folders are records. They used the first split and GAN method `v1`, and the current code cannot resume or re-report them; their reports and metrics are in the folders.
