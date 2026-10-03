# Preparing the combined D1/D2/D4 detection dataset

Implemented script: `tools/prepare_detection.py`. Python dependencies: Pillow, numpy, PyYAML, ImageHash. These are available in the current environment. Run commands from the repository root. No GPU is needed.

## What it does

- Reads D1, D2 and D4 without modifying raw images or labels.
- Reads target classes, mapping, seed, split fractions and duplicate threshold from `config/experiment.yaml`.
- Validates box/polygon geometry, preserves original polygons and source line numbers in a JSONL manifest, and creates enclosing boxes for detection candidates.
- Removes identical repeated annotation rows, excludes whole images with unmapped defects, and quarantines corrupt images, missing/invalid labels and prior unresolved review findings.
- Keeps empty labels out unless an explicit background review approves them.
- Removes redundant byte-identical image copies only when annotations agree; conflicting identical-image labels are quarantined.
- Groups pHash candidates across sources, related numeric filename families, matching Roboflow base names within each source, and reviewer-supplied groups. Near-duplicates are grouped, not silently deleted.
- Generates batches of 25 overlay thumbnails tied to CSV row numbers. Check full-resolution images before approval; thumbnails are a screening aid.
- Exports only approved images. Related groups stay together in approximately 70/10/20 train/val/test splits, balancing class presence and source counts. Splits must include every core class, otherwise export fails with an explanation.

This cannot automatically establish aircraft provenance, remove unknown source augmentation, find every missing annotation or distinguish every scratch from a crack. Candidate labels are not certified training data. The script does not crop, resize or fabricate annotations, and does not create segmentation masks.

## First run (already completed here)

```powershell
python -B tools/prepare_detection.py --output data/interim/d124_review --contact-sheets
```

Use a NEW output directory for each run. Existing runs are never overwritten. Current outputs:

- `data/interim/d124_review/summary.json`: per-source counts and exclusion/quarantine status.
- `data/interim/d124_review/review.csv`: decisions to complete.
- `data/interim/d124_review/contact_sheets/`: 25-image batches with CSV row numbers.
- `data/interim/d124_review/manifest.jsonl`: image/label hashes, preserved polygon coordinates, original and mapped classes, derived boxes and reasons.
- `data/interim/d124_review/groups.json`: related-image groups, including cross-source groups.
- `data/interim/d124_review/candidate_labels/`: normalized YOLO labels for review only.
- `data/interim/d124_review/settings.json` and `run_info.json`: config and script fingerprint.

Current source counts: D1 4,288; D2 372; D4 1,558. There are 578 unmapped-class exclusions, one redundant exact copy, 131 prior-review quarantines and 5,508 pending images. D1 has 24 empty labels and D2 eight; these occur within the quarantines. No image is automatically approved. Thirteen groups span sources. Grouping is conservative and does not establish that all group members are duplicate photos.

D4 is one image below the previous inventory and `train/labels/272_4.txt` has no corresponding image. The report lists this orphan; the script does not invent its missing image or delete the label.

## Review and approval

Copy review.csv to a working review file before editing. Keep `image`, `image_sha256`, `label_sha256` and `schema_sha256` unchanged. Each row identifies the full raw image path, relative to the repository.

Set `action` to `approve`, `exclude`, `repair` or `quarantine`. Blank means pending. Approval requires a reviewer name plus `yes` in all of:

| Column | Required confirmation |
|---|---|
| domain_ok | Image fits the aircraft external-skin domain and provenance rules. |
| labels_complete | Every visible dent, crack and paint-peeling target is labeled correctly, including accurate converted boxes. |
| unmapped_absent | No visible out-of-scope defect is being silently treated as background. |
| visible_at_640 | Target details are adequate at training resolution. |
| grouping_checked | Related views/derivatives have been considered; add shared groups where the automatic grouping misses relationships. |
| unaugmented_real | Real image with acceptable source history; no pre-applied augmentation or compositing in the core dataset. |

Empty-label approval additionally requires `background_verified=yes`. An annotation that is simply missing or invalid cannot be approved away. For related images missed by automatic grouping, give them the same `additional_group` value; use a globally meaningful value such as an aircraft/scene ID rather than generic numbers. This merges groups; it cannot split an automatic conservative group.

Changed image contents, label contents or class mapping invalidate approval. Re-review the newly generated row rather than updating hashes to bypass the check. Prior contact-sheet notes are not acceptance decisions. Known unmapped annotations always exclude the image even if the action says approve.

## Correcting labels without touching raw data

Use `--label-overrides data/interim/label_fixes`. Place corrected label files underneath that directory mirroring the original repository-relative label path, for example:

`data/interim/label_fixes/data/raw/<source-folder>/train/labels/<image-stem>.txt`

Overrides use the SOURCE class order and source YOLO box/polygon format, not the combined output IDs. Rescan after corrections and re-approve the changed hashes. D4 lacks a paint-peeling source class: exclude affected images unless the source schema is deliberately extended and reviewed; do not delete visible peeling from the task. Candidate labels use combined target IDs and must not be pasted into source-format overrides.

## Export after review

```powershell
python -B tools/prepare_detection.py --output data/processed/d124_v1 --reviews data/interim/d124_review/review.csv --export
```

Add the same `--label-overrides` argument when applicable. Approved output lives under `data/processed/d124_v1/dataset/`, with `data.yaml`, image/label split directories, `split_manifest.json` and `counts.json`. The combined IDs follow `task.core_classes`: **0=dent, 1=crack, 2=paint_peeling**. This is different from D4's SOURCE order.

No approved images means no dataset export. Insufficient class coverage in any split also blocks export. Continue reviewing independent groups rather than forcing related images into different splits. The current scan has zero approved images; only the test fixture has exercised a successful export so far.

Freeze the completed output and split manifest for the experiment. Rerunning after dataset membership or grouping changes may change splits; do not regenerate test sets independently for each training fraction or model. Approximate source/class balance is not exact stratification, especially for large groups. Similarity misses and transitive over-grouping still need review.

## D4 evidence

See [D4 verification](d4_verification.md): the paper confirms the class names; local overlays support source **0=crack, 1=dent**. This inference is recorded in config; `--d4-class-order` allows an explicit correction if author metadata is recovered. Approval hashes include the mapping, so changing it invalidates previous approvals. The paper also documents pre-augmentation, so the full D4 archive is not automatically admissible as independent unaugmented imagery.

## Tests

```powershell
python -B -m unittest discover -s tests -p test_prepare_detection.py -v
```

The tests cover polygon mapping, invalid/unmapped labels, stale review/background gates, transitive cross-source/family grouping, conflicting labels on identical images, and a complete reviewed export with raw-file preservation and class coverage.
