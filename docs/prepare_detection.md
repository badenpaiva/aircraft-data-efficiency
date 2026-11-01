# SUTD dataset preparation

The project uses **SUTD aircraft_AI1 v4**, exported in COCO format from [aircraft_ai1](https://universe.roboflow.com/sutd-4mhea/aircraft_ai1). Export metadata identifies CC BY 4.0 licensing, a 640x640 stretch and no export augmentation.

| Item | Location or value |
|---|---|
| Raw source | `data/raw/aircraft_AI1.v4-983-images-70-20-10-ratio-without-augmentation.coco` |
| Prepared export | `data/processed/sutd_v4` |
| Detector dataset YAML | `data/processed/sutd_v4/dataset/data.yaml` |
| Full-pool diagnostic GAN patches | `data/interim/sutd_v4_gan_patches` |
| Dataset configuration | `config/experiment.yaml` |
| Classes | `missing-rivet`, `rust`, `scratch` |
| Included data | 983 images and 5,515 boxes, provisionally accepted |
| Grouping | 825 similarity groups before splitting |
| Prepared split | 688 training, 98 validation, 197 test images |

The source spelling `scractch` is normalized to `scratch`. The prepared split is regenerated at approximately 70/10/20; it does not retain the source export's memberships. Similarity groups stay together, but image similarity is not a guarantee that every shared physical defect has been identified. Source stretching cannot be undone by detector letterboxing.

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
