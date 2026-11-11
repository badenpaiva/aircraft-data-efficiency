# Project map

The project uses SUTD aircraft_AI1 v4. Start with [README](README.md) for the experiment overview and [pipeline](docs/pipeline.md) for commands.

| Folder or file | What it is meant to contain |
|---|---|
| `docs/` | Project documents explaining the pipeline, dataset preparation, HQGAN design, and image review decisions |
| `config/` | Active YAML settings: `experiment.yaml` for SUTD data, `detection_study.yaml` for the experiment grid and evaluation, `hqgan_sutd.yaml` for GAN architecture and optimization |
| `data/raw/` | Original downloaded dataset exports; preserve unchanged |
| `data/interim/` | Derived working data such as cropped training patches |
| `data/processed/` | Prepared detector datasets, labels, split/group manifests and preparation receipts |
| `results/studies/` | Frozen experiment plans, subset definitions, GAN outputs, synthetic datasets, detector runs, metrics and reports |
| `results/hqgan/` | Standalone GAN diagnostics: checkpoints, losses, generated candidates and inspections |
| `results/detection/` | Standalone detector diagnostics, evaluations and predictions |
| `results/dataset_audits/` | Generated dataset inventory and preparation audit artifacts |
| `scripts/` | Runnable entry points for planning, GAN generation, detector training, evaluation, reporting and inspection |
| `src/study/` | Study planning, data subsets, augmentation, screening, review exchange, detector backends, metrics and reporting |
| `src/augmentation/hqgan/` | GAN data handling, generator/discriminator architecture and training |
| `src/detection_pipeline.py` | Shared configuration, provenance and detector-data helpers |
| `src/training_progress.py` | Training progress and ETA reporting |
| `tools/` | Dataset preparation, annotation geometry and audit utilities; use the SUTD commands documented in `docs/` |
| `tests/` | Automated regression tests for preparation, training, resume, labels, review and metrics |
| `notebooks/` | Space for analysis notebooks; not a required runtime dependency |
| `tmp/` | Disposable previews, rendered reference pages and temporary local test files |
| `.ultralytics/` | Local library settings, downloaded weights and pinned YOLOX dependency source |
| `.venv/` | Project Python environment and installed packages |
| `.git/` | Version-control metadata |
| `requirements.txt`, `requirements-study.txt` | Dependency lists for the project and study workflow |
| Root `*.pt` files | Downloaded pretrained detector weights; source plans bind to their hashes |
| `.gitignore` | Rules excluding local datasets, large results, environments and scratch files from Git |

## Active experiment folders

The two folders below are the completed first experiment. They are records: the current code cannot resume or re-report them. A new study made with the [pipeline guide](docs/pipeline.md) has the same layout.

`results/studies/sutd_full_v1/` is the source study:

- `plan.json`: frozen settings, dataset identities, subsets and planned jobs.
- `subsets/<subset>/`: real training-image selections and manifests.
- `gan/<subset>/patches/`: subset-bound real training crops.
- `gan/<subset>/<class>/training/`: GAN checkpoint and training records. Runs made with GAN method `v2` also hold `sheets/`, one image of samples per checkpoint.
- `gan/<subset>/<class>/candidates/`: generated patches.
- `gan/<subset>/candidates_review.json` and `candidates_screening/`: screening outcomes and image-review queue.
- `cvat/` and placement/composite review files: created when the review workflow is used.

`results/studies/sutd_whole_patch_seed0_v1/` is its detector study:

- `plan.json`: derived seed-0 experiment and whole-patch labeling provenance.
- `whole_patch/<subset>/images/`, `labels/`, `manifest.json`: generated composite training data and approximate boxes.
- `runs/<job>/data/`: runtime detector dataset files.
- `runs/<job>/training/`: model checkpoints and training histories.
- `runs/<job>/val/`: metrics, predictions and visual previews.
- `status.json`: execution state of each job.
- `report/`: combined analysis exports, written when reporting runs.
- `training.stdout.log`, `training.stderr.log`, `training.pid`: background-launch logs and recorded process ID; a saved PID alone does not establish that a process is still running.

## What can be removed?

The top-level `tmp/` folder is disposable when no preview-generation or test command is using it. Its current contents are JPEG previews and rendered PDF pages, not training inputs. Tests that need it recreate it. Deleting this folder does not remove GAN checkpoints, detector results or datasets.

Do not confuse `tmp/` with `checkpoint.tmp`, `last.tmp` or `*.json.tmp` inside active result folders: those are atomic-write intermediates and must not be removed while training or review is writing them.

Preserve raw data, prepared datasets, active study folders, pretrained weights and the pinned YOLOX source. Plans and manifests refer to their paths and hashes. Deleting or moving them can prevent resume or verification. Generated study receipts are experiment records, not editable project documentation.
