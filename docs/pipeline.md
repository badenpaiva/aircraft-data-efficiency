# SUTD experiment pipeline

This is the pipeline used by the current run. Dataset definitions are in [dataset preparation](prepare_detection.md); label policies are in [image review](review.md).

## Settings and scope

| Setting | Current value |
|---|---|
| Dataset | SUTD aircraft_AI1 v4 |
| Classes | missing-rivet, rust, scratch |
| Real training fractions | 5%, 10%, 25%, 50%, 100% |
| Source study seeds | 0, 1, 2 |
| Current detector queue | Seed 0 only |
| Detectors | YOLOX-S, YOLOv8n, YOLO11n, YOLO26n |
| HQGAN ratios | 0.5, 1, 2 synthetic composites per real image |
| GAN budget | 20,000 steps per class and subset; 400 candidates per class |
| Detector budget | 30 epochs, 640px, batch 8, AdamW, learning rate 0.001 |
| Evaluation | Fixed real validation split; last-epoch checkpoint |

Each fraction/seed has its own class-specific GANs. Within that subset, models and ratios share the generated pool. Never use a full-pool GAN to supply a reduced-data subset. Real subsets are nested within each seed; group boundaries can make the achieved image count smaller than the requested fraction.

The source configuration also defines real-only and classical augmentation controls. Its complete grid has 360 detector cells. The current derived study selects only the three HQGAN arms for seed 0, giving 60 cells. Controls are comparisons on the same SUTD data, not a separate dataset pipeline.

## Setup and planning

Run from `D:\Documents\aircraft-data-efficiency` in PowerShell. The prepared data and existing plans already exist. Setup and planning below are for a new installation or a new experiment, not for resuming the current run.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-study.txt
& $python scripts/setup_detection_study.py

# Use a new output directory for each new plan.
& $python scripts/detection_study.py plan --config config/detection_study.yaml --output results/studies/sutd_new
```

Setup obtains detector weights and the pinned YOLOX source. Plans freeze settings, source identities, subsets, model-weight hashes and implementation identity. Do not edit a saved `plan.json`. Use a new plan when changing data or methods.

## Train GANs and generate candidates

For the existing source study:

```powershell
$source = 'results/studies/sutd_full_v1'
& $python scripts/detection_study.py gan --study $source --device cpu
```

This trains all planned class/subset GANs, generates candidates and screens them. Add `--subset seed0_f00500` to work on one subset. Omit it to process all planned seeds and fractions. Use `cuda:0` only with a supported GPU and CUDA-enabled PyTorch.

Rerun the same command after an interruption. Completed GANs and candidate queues are reused; interrupted GANs resume from the saved checkpoint. Keep settings, total steps and device unchanged. Progress lost since the last checkpoint must be repeated. Standalone generation commands are diagnostic utilities; the study command provides the subset-bound candidates used here.

## Prepare synthetic training images

```powershell
$study = 'results/studies/sutd_whole_patch_seed0_v1'
& $python scripts/whole_patch_study.py prepare --source $source --output $study --seed 0
```

Preparation verifies candidate and checkpoint identities, selects only technical-pass candidates, requires at least one passing candidate for every class, and creates enough composites for ratio 2. It preserves real-image labels and adds the inserted patch rectangle as an approximate defect box. Patches may be reused at different placements; finished composites are not duplicated to fill quotas. Lower ratios use nested deterministic prefixes of the same composite pool.

The source study and its review decisions remain unchanged. The derived plan records source identity, script hash, manifest hashes and approximate-label status. Completed preparation manifests are verified and reused. Use a new output directory when inputs or label policies change.

## Train and resume detectors

```powershell
& $python scripts/whole_patch_study.py run --study $study --device cpu
```

This is the entry point for the current run. It trains all four models at all three ratios and five fractions for the selected seed. It also evaluates each completed model on validation data. It supports `--models`, `--arms`, `--limit-jobs` and `--retry-failed`.

```powershell
# Retry failed cells only after resolving the cause; completed cells are skipped.
& $python scripts/whole_patch_study.py run --study $study --device cpu --retry-failed
```

After interruption, repeat the run command with the same study. Saved detector checkpoints are resumed where supported; if only evaluation failed, trained weights are reused. Do not change the frozen training implementation or `whole_patch_study.py` mid-study. Keep the machine awake during CPU training. ETA is an estimate for the stage shown, not necessarily the remaining experiment.

For matched SUTD controls, the source study runner can train the baseline/classical arms:

```powershell
& $python scripts/detection_study.py run --study $source --arms baseline classical_light classical_heavy --device cpu
```

That command covers all seeds and fractions in the source plan; it has no seed filter. Avoid competing CPU training processes unless that resource use is intentional.

## Results and interpretation

Under the detector study directory:

| Path | Contents |
|---|---|
| `status.json` | Pending, running, complete or failed job states |
| `training.stdout.log`, `training.stderr.log` | Logs from the background launch; a foreground CLI invocation prints to the terminal |
| `runs/<job>/job.json` | Model, subset, ratio and training-input provenance |
| `runs/<job>/training/` | Checkpoints, training history and completion/timing records |
| `runs/<job>/val/metrics.json` | Validation mAP50, mAP50-95, per-class AP, precision, recall and F1 |
| `runs/<job>/val/predictions.json` | Saved predictions |
| `runs/<job>/val/previews/` | Green annotation boxes and red model predictions |
| `report/` | Combined tables, figures and results draft |

A job such as `seed0_f00500__hqgan_0p5__yolox` means seed 0, 5% real training data, ratio 0.5, and YOLOX. Scores in JSON are fractions: 0.4124 means 41.24%.

The run command generates a report at the end of its invocation. To refresh a partial report without launching training:

```powershell
& $python scripts/detection_study.py report --study $study
```

Compare each augmented run with a matching real-only model, fraction, seed and split. A larger synthetic ratio is not automatically better. Current whole-patch results must retain their approximate-label description. One seed does not establish repeatability. mAP50 is not image-classification accuracy; inspect recall and stricter localization metrics too.

## Evaluation and verification boundaries

The current whole-patch entry point exposes validation evaluation only. The reserved test split is not used to select methods, thresholds or labels. Reviewed-image runs can use `detection_study.py finalize-test` only after the recipe is frozen; do not use that as a shortcut to finalize the whole-patch study.

Regression coverage includes split/group isolation, COCO conversion, GAN resume, screening receipts, review exchange, label transforms, nested synthetic quotas and detector evaluation. Run relevant tests when changing those components. A passing test verifies code behavior, not image realism or detection usefulness.
