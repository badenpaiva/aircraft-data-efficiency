# Detection dataset cleaning and pooling plan

Active sources: D1 = Innovation Hangar v2 (primary); D2 = DDIISc; D3 = SUTD; D4 = Xiong/Zenodo. See config/experiment.yaml source_ids. Removed or unavailable sources are documented separately in source_archive.yaml and are not expected by the audit script.

## Manageable cleaning workflow

1. Keep raw exports unchanged. Build a reviewed manifest with keep/fix/exclude/uncertain decisions and reasons.
2. Automate inventory, label parsing, missing/empty-file flags, duplicate candidates and source-family grouping. Similarity is a review aid, not permission to delete images or copy labels.
3. Screen contact sheets in batches of about 50 images or related-image groups. Review at full resolution for acceptance and whenever thumbnails are uncertain. Prioritize empty labels, polygons, visible transformations, uncertain provenance and inconsistent duplicate labels.
4. For detection, retain valid existing boxes; derive candidate enclosing boxes from polygons while preserving their coordinates. Correct only defective annotations, including missing targets and repeated labels for one instance.
5. Fully review all final validation/test images. Give the training pool a visual screening pass, with detailed checks for flagged images. A random sample alone cannot certify all labels or domain membership.
6. Start with a reviewed subset, train a basic detector, and expand the reviewed training pool in batches. Keep evaluation groups fixed. Do not use test predictions to tune labels or selection iteratively; version and document any genuine test-label correction.

D1 currently has 4,288 images and 2,891 pHash representative candidates. The latter is not a guaranteed unique-image count. There are already review sheets and a ledger in audit_out/d1_review and docs/d1_visual_review.json. The known 24 D1 and eight D2 empty-label images remain quarantined, and no negatives have been certified. Data-quality flags are not automatic corrections.

## Combining sources

Combining is possible after cleaning, common class definitions, conversion, cross-source duplicate checks and joint grouped splitting. Keep each image's source ID and report metrics by source as well as pooled metrics. Never concatenate original train/test folders.

For the core crack/dent/paint-peeling study, consider D1 + eligible D2 + eligible D4; D4 class names are paper-confirmed and source IDs 0=crack, 1=dent are visually supported (see d4_verification.md). Pre-augmentation admission still needs review. D3 primarily contributes corrosion and belongs to the separate corrosion track unless the project explicitly broadens the core task. Under the current unmapped-class policy, images with scratches or missing fasteners are excluded, which can substantially reduce supplementary contributions.

Class absence in a label schema does not establish visual absence. For example, D4 has no paint-peeling class: any visible peeling must be annotated or the affected image excluded before pooling into a three-class detector. Otherwise it becomes an incorrect negative. Likewise, D3's rust/coating/weathering semantics need review before treating rust as structural corrosion.

Pooling is not enabled by this naming change. Confirm domain compatibility, completeness for all modeled classes, cross-source overlap (especially D4), and final eligible counts first. Compare a D1-only baseline with the pooled experiment on a fixed evaluation design to determine whether pooling actually helps.

The preparation workflow is implemented in tools/prepare_detection.py; see [usage](prepare_detection.md).
