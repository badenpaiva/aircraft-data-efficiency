# SUTD experiment pipeline

This is the pipeline for the planned experiment. No study has been run with it yet. The two completed studies (`results/studies/sutd_full_v1` and `sutd_whole_patch_seed0_v1`) used an earlier split, GAN method and detector list; the current code cannot load them, and they stay as records. Dataset definitions are in [dataset preparation](prepare_detection.md); label policies are in [image review](review.md).

## Settings and scope

| Setting | Current value |
|---|---|
| Dataset | SUTD aircraft_AI1 v4, split by capture unit |
| Classes | missing-rivet, rust, scratch |
| Real training fractions | 25%, 50%, 100% |
| Seeds | 0, 1, 2 |
| Detectors | YOLO26n (light, 2.6M parameters) and YOLO26m (heavy, 21.9M) |
| HQGAN ratios | 0.5, 1, 2 synthetic composites per real image |
| GAN budget | 20,000 steps per class and subset; 400 candidates per class |
| Detector budget | 30 epochs, 640px, batch 8, AdamW, learning rate 0.001, mixed precision on GPUs that support it |
| Evaluation | Fixed real validation split; last-epoch checkpoint |

Each fraction/seed has its own class-specific GANs. Within that subset, models and ratios share the generated pool. Never use a full-pool GAN to supply a reduced-data subset. Real subsets are nested within each seed and are made of whole capture groups, so the achieved image count can be smaller than the requested fraction.

The configuration defines six arms: real-only, two classical augmentation controls and three HQGAN ratios. The source plan therefore holds 108 detector cells (3 fractions, 3 seeds, 6 arms, 2 models) and 27 GANs. Controls are comparisons on the same SUTD data, not a separate dataset pipeline.

## Setup on the machine that will run it

Plans store absolute paths, so create the plan on the machine that trains. From the repository root in PowerShell:

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-study.txt

# NVIDIA GPU: replace the CPU-only PyTorch with a CUDA build of the same version.
& $python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

& $python scripts/setup_detection_study.py
```

- **Data is not in git.** Copy `data/processed/sutd_v4_units` from the machine that prepared it; the export holds its own images and works from any location. The raw download under `data/raw/` is only needed to rebuild the export.
- **CUDA build.** `cu126` is PyTorch's CUDA 12.6 build; its [previous-versions page](https://pytorch.org/get-started/previous-versions/) lists the builds available for 2.14.1. Pick one your driver supports.
- **Setup** downloads the detector weights and ends with one line saying whether PyTorch can use a GPU. If it says no, runs fall back to the CPU and take many times longer.
- **Device.** Every command takes `--device`; the default `auto` uses the first CUDA GPU when there is one. A resumed run must use the device it started on. GPU runs are not bit-for-bit repeatable.
- **Memory.** YOLO26m at batch 8 and 640px has not been run on a GPU here. If it runs out of memory, lower `training.batch_size` in `config/detection_study.yaml` before planning; the value applies to both detectors. Set `training.workers` to 0 if data loading misbehaves.

```powershell
# Use a new output directory for each new plan.
$source = 'results/studies/sutd_units_v2'
& $python scripts/detection_study.py plan --config config/detection_study.yaml --output $source
```

Plans freeze settings, source identities, subsets, model-weight hashes and implementation identity. Do not edit a saved `plan.json`. Use a new plan when changing data, code or methods.

## Train GANs and generate candidates

```powershell
& $python scripts/detection_study.py gan --study $source
```

This trains all planned class/subset GANs, generates candidates and screens them. Add `--subset seed0_f02500` to work on one subset. Omit it to process all planned seeds and fractions. Each GAN run writes `sheets/step_NNNNNN.png`; look at them before going further (see [HQGAN design](hqgan.md)).

Rerun the same command after an interruption. Completed GANs and candidate queues are reused; interrupted GANs resume from the saved checkpoint. Keep settings, total steps and device unchanged. Progress lost since the last checkpoint must be repeated. Standalone generation commands are diagnostic utilities; the study command provides the subset-bound candidates used here.

For a single-class pilot outside a study:

```powershell
& $python scripts/prepare_hqgan.py --output data/interim/sutd_v4_units_gan_patches
& $python scripts/train_hqgan.py --class-name rust --steps 20000 --name pilot_rust
```

## Prepare synthetic training images

```powershell
$study = 'results/studies/sutd_units_v2_core_seed0'
& $python scripts/whole_patch_study.py prepare --source $source --output $study --seed 0
```

Preparation verifies candidate and checkpoint identities, selects only technical-pass candidates, requires at least one passing candidate for every class, and creates enough composites for ratio 2. It preserves real-image labels and adds the core of each inserted patch as an approximate defect box; the core follows from the GAN's crop rule (see [HQGAN design](hqgan.md#crops-placement-and-labels)). Patches may be reused at different placements; finished composites are not duplicated to fill quotas. Lower ratios use nested deterministic prefixes of the same composite pool.

The source study and its review decisions remain unchanged. The derived plan records source identity, script hash, manifest hashes and approximate-label status. Completed preparation manifests are verified and reused. Use a new output directory when inputs or label policies change.

## Train and resume detectors

```powershell
& $python scripts/whole_patch_study.py run --study $study
```

This is the entry point for the automatically labelled HQGAN arms. It trains both models at all three ratios and three fractions for the selected seed: 18 cells. It also evaluates each completed model on validation data. It supports `--models`, `--arms`, `--limit-jobs` and `--retry-failed`. Repeat `prepare` and `run` with `--seed 1` and `--seed 2` and their own output directories for the other seeds.

```powershell
# Retry failed cells only after resolving the cause; completed cells are skipped.
& $python scripts/whole_patch_study.py run --study $study --retry-failed
```

After interruption, repeat the run command with the same study. Saved detector checkpoints are resumed where supported; if only evaluation failed, trained weights are reused. Do not change the frozen training implementation or `whole_patch_study.py` mid-study. Keep the machine awake during training. ETA is an estimate for the stage shown, not necessarily the remaining experiment.

For matched SUTD controls, the source study runner trains the baseline and classical arms:

```powershell
& $python scripts/detection_study.py run --study $source --arms baseline classical_light classical_heavy
```

That command covers all seeds and fractions in the source plan, 54 cells; it has no seed filter. Avoid competing training processes on one device unless that resource use is intentional.

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

A job such as `seed0_f02500__hqgan_0p5__yolo26m` means seed 0, 25% real training data, ratio 0.5, and YOLO26m. Scores in JSON are fractions: 0.4124 means 41.24%.

The run command generates a report at the end of its invocation. To refresh a partial report without launching training:

```powershell
& $python scripts/detection_study.py report --study $study
```

Compare each augmented run with a matching real-only model, fraction, seed and split. A larger synthetic ratio is not automatically better. Automatically labelled results must retain their approximate-label description. One seed does not establish repeatability. mAP50 is not image-classification accuracy; inspect recall and stricter localization metrics too.

## Evaluation and verification boundaries

The automatic-label entry point (`whole_patch_study.py`) exposes validation evaluation only. The reserved test split is not used to select methods, thresholds or labels. Reviewed-image runs can use `detection_study.py finalize-test` only after the recipe is frozen; do not use that as a shortcut to finalize the whole-patch study.

Regression coverage includes split/group isolation, COCO conversion, GAN resume, screening receipts, review exchange, label transforms, nested synthetic quotas and detector evaluation. Run relevant tests when changing those components. A passing test verifies code behavior, not image realism or detection usefulness.
