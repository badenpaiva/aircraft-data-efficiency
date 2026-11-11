# SUTD dataset preparation

The project uses **SUTD aircraft_AI1 v4**, exported in COCO format from [aircraft_ai1](https://universe.roboflow.com/sutd-4mhea/aircraft_ai1). Export metadata identifies CC BY 4.0 licensing, a 640x640 stretch and no export augmentation.

| Item | Location or value |
|---|---|
| Raw source | `data/raw/aircraft_AI1.v4-983-images-70-20-10-ratio-without-augmentation.coco` |
| Prepared export | `data/processed/sutd_v4_units` (capture-unit split, active). `data/processed/sutd_v4` is the earlier similarity-only split used by the completed studies; keep it. |
| Detector dataset YAML | `data/processed/sutd_v4_units/dataset/data.yaml` |
| Full-pool diagnostic GAN patches | `data/interim/sutd_v4_gan_patches` |
| Dataset configuration | `config/experiment.yaml` |
| Classes | `missing-rivet`, `rust`, `scratch` |
| Included data | 983 images and 5,515 boxes, provisionally accepted |
| Grouping | 41 capture units (34 drone videos, 7 photo series), 38 groups after similarity merging |
| Prepared split | 683 training, 106 validation, 194 test images |

The source spelling `scractch` is normalized to `scratch`. The prepared split is regenerated at approximately 70/10/20; it does not retain the source export's memberships. Source stretching cannot be undone by detector letterboxing.

## Capture units

The images are frames sampled every 20 frames from drone videos, plus numbered photo series. Consecutive frames show the same physical defects, and perceptual hashing alone kept only 21% of them together, so the earlier split put near-identical views on both sides. `capture_unit_patterns` in `config/experiment.yaml` now names each image's video or photo series from its file name, and a whole unit stays in one split. Similarity merging still applies on top: four photo series share near-identical images and form one 233-image group.

- `scene7` and `sceneadd7` are different videos and are separate units.
- A configured source fails on a file name that matches no pattern, so a new naming scheme cannot slip in ungrouped.
- In the active export no capture unit crosses splits, and no cross-split image pair is within the similarity threshold.

All footage is of one parked aircraft, and different videos pass over the same regions. A unit split removes near-identical frames from evaluation; it cannot make validation and test defects physically new. Class balance is uneven because videos differ in content: test has 135 missing-rivet boxes against about 450 for each other class.

Training fractions are drawn as whole groups: 25% is 6 to 8 groups and 50% is 15 to 17, depending on the seed. The sampler picks one of the three best-balancing groups at each step by seed, so each seed gets its own subset with similar class counts. The 233-image photo group is too large for the smaller budgets and only enters at 100%.

## Import and export

Preparation is already complete. To rebuild for a deliberate data revision, use a new output directory:

```powershell
.\.venv\Scripts\python.exe tools/prepare_detection.py --config config/experiment.yaml --output data/processed/sutd_v4_revision --export
```

Raw files are read-only. The importer checks split-local IDs, image dimensions, box bounds, coverage and path containment. COCO pixel `xywh` boxes become normalized YOLO center `xywh` labels through an explicit class map. Invalid/crowd annotations quarantine an image; unmapped categories exclude the whole image. Conflicting exact duplicates are quarantined.

Preparation writes manifests, counts, grouping, hashes, candidate labels and a review queue. The export contains the dataset YAML, split image/label folders and split manifest. Training verifies hashes, class order, quality status and group isolation. Freeze one prepared export for an experiment; changing annotations or grouping requires a new export and study plan.

## Real-image label review

The active dataset labels are provisional. Manual certification uses the preparation `review.csv`, recording reviewer, action, domain, label completeness, visible targets, grouping and unaugmented-source checks. Empty images also need verified-background status. COCO corrections belong in a separately versioned source export.

After review, enable `require_domain_and_completeness_review` in a new dataset configuration and pass the completed CSV with `--reviews` when preparing a new output. Approvals bind to source and schema hashes. Do not silently change the data beneath an existing experiment.

## GAN crops

The study's `gan` command extracts patches from real training boxes within each selected subset. It writes them under `results/studies/<source-study>/gan/<subset>/patches/`. Validation and test images never supply GAN crops or synthetic-image backgrounds.

The full-pool patches under `data/interim/` support standalone diagnostics only. They must not supply reduced-data study cells. See [pipeline](pipeline.md) for study commands and [image review](review.md) for generated-image labels.
