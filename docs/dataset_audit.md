# D1 annotation and admission review — 2026-10-03

**Development exception (2026-10-06):** the user temporarily disabled mandatory
manual approval to begin the detection pipeline. The preparation script may
export geometrically valid, nonempty unreviewed candidates as `provisional`,
including provisional validation/test splits. The findings and final-study
review standards below still apply; this exception does not certify the labels
or provenance. Restore the config review flag and rebuild a reviewed version
before claiming reviewed-data results. See `docs/detection_pipeline.md`.

## Evidence and reproducibility

Run `python tools/review_d1.py`. Generated evidence is in `audit_out/d1_review/`: `inventory.json`, `manifest.jsonl`, `review_index.json`, eight polygon overlay sheets, four empty-label sheets, six domain sheets and eight resolution comparisons. Left/right pairs show unobscured imagery and overlays; red = crack, cyan = dent, green = paint-off, orange = missing-head, magenta = scratch. Each sample has an ID mapped to its complete relative filename and source line in the manifest. The tracked `docs/d1_visual_review.json` records visual decisions and label hashes; rerunning the generator does not overwrite these decisions.

Reviewed all 58 selected polygon-image pairs (complexity quantiles for both polygon classes plus filename-family examples), all 32 empty-label images, 48 seeded domain samples, and eight visibility examples. This targeted sample does not estimate a population-wide annotation accuracy rate. Some samples overlap or are duplicates.

## Geometry and counts

The 803 formerly discarded rows are 522 crack polygons and 281 dent polygons. There are 298 polygon-bearing images: 231 polygon-only candidates and 67 mixed box/polygon images. Vertices range from 4 to 229. Paint-off, scratch and missing-head have boxes only. Names are confirmed from the local D1 v2 `data.yaml`.

| Raw class | Boxes | Polygons | Normalized pre-image-dedup records | pHash<=5 representative records |
|---|---:|---:|---:|---:|
| crack | 3,673 | 522 | 4,195 | 2,548 |
| dent | 3,545 | 281 | 3,826 | 2,283 |
| paint-off | 294 | 0 | 294 | 294 |
| missing-head | 370 | 0 | 370 | 370 |
| scratch | 83 | 0 | 83 | 83 |

No exact repeated rows were removed. All rows pass finite-coordinate, class, arity, range and positive-area checks; this is not a polygon-topology or semantic validation. Overlapping box/polygon annotations can still describe the same instance (D1-1041); do not automatically merge by IoU. The representative comparison uses transitive pHash<=5 connected components and the lexicographically first image, reproducing 2,891 representatives. pHash similarity is not proof of duplication, and transitive components can join different views. These numbers precede domain, completeness, unmapped-class and semantic-instance review: **they are not final validated counts**. Earlier 2,208 crack / 2,089 dent totals omitted polygons and must not drive training-fraction estimates.

Final count order: validate image provenance/domain and full annotation coverage; adjudicate instance duplicates; exclude unmapped-defect images; review duplicate groups and select the highest-quality complete representative; normalize; freeze manifest; count; grouped split. Do not choose a poorer representative merely because its filename sorts first. Group related source photographs and transformed derivatives before splitting even if pHash misses them. Final admitted counts remain unavailable while these checks are open.

## What the overlays establish

The contours generally follow visible cracks, tears or dents rather than arbitrary image regions. Examples include D1-2722 (114-vertex crack beside rivets), D1-2734 (120-vertex crack on metal), D1-1678 (five-vertex dent on inlet), and D1-4045 (complex branching damage). However, they are not uniformly accurate segmentation masks: margins are generous, fine branches can be omitted, dents have subjective boundaries, and tears/coating cracks are mixed with structural cracks.

D1-2723 traces only part of a branching crack. D1-2731 omits lower damage. D1-1041 has potentially redundant box and polygon annotations. D1-2314 and D1-0323 contain faint marks whose crack-versus-scratch meaning cannot be certified from appearance. Similar images D1-3614 / D1-2728 use different instance granularity for the same dented region. No blanket polygon approval is justified.

**Detection policy:** preserve raw text, original polygons, source line, class and label checksum. Derive candidate enclosing axis-aligned boxes as min/max of polygon vertices; export only after per-image domain, class, instance and completeness approval. Conversion itself does not validate an annotation. Do not clip out-of-range coordinates silently. Keep original source classes alongside mapped classes.

**Segmentation policy:** the manifest explicitly identifies 231 polygon-only candidates, but **zero images currently have verified complete evaluation masks**. Mixed images need masks for box-only defects; all images need an exhaustive pass for unlabeled target defects, contour corrections, class adjudication and overlap/ignore handling. Paint peeling has no native polygons and needs new masks. Polygon-only does not mean complete.

**Evaluation-mask decision:** manually draw or correct genuine masks on the frozen real validation/test pool at native resolution, then have a second independent reviewer verify every target class, omitted defect, boundary and ignore region. Record reviewer, source image hash, mask hash, version and disagreement resolution. Block segmentation evaluation until both approvals exist. Box-filled masks and model/box-derived proposals are weak training labels only; never score segmentation against them as ground truth. Existing polygons may be tracing proposals, not automatically accepted masks. This pass did not create or certify evaluation masks.

## Empty labels and completeness

All 24 D1 and eight D2 empty files were visually screened. **No image was certified as background.** Fourteen D1 images visibly show damage and are assigned missing-annotation status: D1-0327, 1788, 2172, 2542, 2543, 2658, 3269, 3270, 3280, 3281, 3620, 3922, 4003, 4216. Examples include a clear cracked metal component (0327), cracked latch surround (2172/4003), peeled coating (3269/3270/4216) and a long split edge (3620).

The remaining ten D1 and all eight D2 files are ambiguous, not verified negatives: weathering, faint lines, shadowed detail, possible coating defects or uncertain dents require full-resolution expert review. Exclude all 32 from training/validation/test until repaired or explicitly certified. This conservative decision avoids silently turning unknowns into negatives. D2 is supplementary and must undergo its own full admission review if pooled. The nonempty sample also contains omitted defects (D1-1231, D1-3128), so empty-label repair alone does not establish completeness.

## Unmapped classes and background

Choose `exclude_images`: if any scratch, missing-head or missing-rivet annotation is present, exclude the entire image from the core experiment, retaining it in the source archive. D1 has 276 such images before dedup. Excluding these and the 24 empty files leaves 3,988 images for further review, not admission. Stage counts are in `audit_out/d1_review/policy_counts.json`. Apply the same rule when visual review discovers an unlabeled out-of-scope defect. Do not merely delete its annotation. No `other` class is introduced in this study.

An admitted background image means a reviewer found no visible core target and no excluded defect at native and training resolution within the observable aircraft-skin region. It does not mean defect-free material, structural safety or absence of subsurface damage. Images with visible scratches/missing fasteners are excluded, not called healthy/background. Record unobservable regions explicitly; do not use ambiguous negatives for evaluation.

## Domain admission

The core domain is real aircraft external skin with surface-visible dent, crack or paint loss. It is not a certified in-service/MRO dataset.

- Intact aircraft skin in inspection/maintenance context: eligible only after identity/provenance and annotation review.
- Preserved/parked aircraft: eligible as an explicitly tagged preserved-aircraft stratum if real and sufficiently documented. Group by aircraft/location; report separately and avoid claims of in-service generalization. This retains the study's stated surface-defect scope without concealing the source bias.
- Severe accident damage, crash wreckage, catastrophic tears and detached wreckage: exclude from the core; retain in a separate incident-domain archive, with no automatic training admission.
- Loose components: only eligible with documented aircraft identity/part provenance and a surface compatible with the skin task; engines, industrial castings, glass and unrelated structural specimens are outside the core skin task.
- Unknown provenance: quarantine; appearance or a filename alone cannot positively establish aircraft origin. Car, marine and industrial context is exclusion evidence requiring resolution, not permission to pool.
- Synthetic, composited and pre-augmented images: exclude from the real core. Generated augmentation belongs only to the declared training arm and current training subset. Keep source/derivative groups together.
- Watermarks: retain unchanged in archive. Quarantine pending source/license verification and evaluation for confounding or occlusion; a watermark does not by itself prove synthetic content. Do not remove it to conceal provenance.

Examples: D1-1084 / 1426 are watermarked severe aircraft damage; D1-3611 is a damaged corrugated-sheet close-up with unclear aircraft origin; D1-4118 is a car body panel; D1-3621 has mast/deck context; D1-3624 shows a heat-exchanger/nozzle context; D1-2668 is an uncertain casting. They cannot all be admitted under an aircraft-skin claim.

**Additional blocker:** every local D1 image is already 640x640, contradicting the repository's original-resolution claim. Although the export README says no augmentation, D1-1397/2546/1182/1320 show posterized colors, D1-0259/2507/0761 show repeated translucent light spots, D1-4105 shows tiled rectangular content, and multiple files have rotated black corners. These establish transformed-looking source content, not which software created it or when. An unaugmented export can still inherit altered source images. Stock-version metadata is not sufficient admission evidence. Verify upstream originals and source families before calling this a clean classical-augmentation control dataset.

## Visibility at the planned resolution

Choose aspect-preserving 640-long-side bilinear resize plus centered padding (no stretch); masks use nearest-neighbor after genuine mask approval. All local D1 images are already 640x640, so the planned resize is an identity operation: **100% of the decoded local pixel grid survives**, with no additional downsampling. This says nothing about detail lost before acquisition. Larger training sizes cannot recover missing source information.

The eight comparisons sample crack/paint box short-side quantiles 0/5/25/50 percent. Crack extents are 7x32, 73x17, 83x31 and 91.5x52.5 pixels. Paint extents are 22.5x20.5, 42.5x40.5, 112x73.5 and 128x102.5 pixels. Crack D1-0519 is discernible but thin/quantized; D1-0252 is visibly blurred; D1-0525 has small tears amid scratches; D1-3130 has a clear opening but questionable crack/coating semantics. Paint D1-2953's smallest patch survives as a visible light spot with coarse boundary; D1-3210 is discernible but mixed with holes/coating wear; D1-3420 has low-contrast ambiguous peeling; D1-4146 has a clearly visible patch.

No crack or paint bounding-box short side is below four pixels, but **box extent does not measure crack width**, and this is not proof every defect is resolvable. Thin branches may occupy only a few pixels even inside large boxes. Keep 640 as the baseline with an explicit low-visibility review flag; obtain native originals for ambiguous details, or quarantine them. Any future change in preprocessing requires rerunning the visibility review before training.

## Remaining release gates

The inventory, review evidence and policies are complete for this screening pass. Dataset certification is not: full-image/instance review, original-image provenance and transformation reconciliation, duplicate-group adjudication, repaired labels and independently verified masks remain required. Do not publish final counts or segmentation metrics until those gates pass. Config policies are declarative; no training/export pipeline exists yet, so future loaders must enforce them and fail closed on missing reviews.

## Evidence links

- [Polygon overlays, first sheet](../audit_out/d1_review/polygons_01.jpg)
- [Empty labels, first sheet](../audit_out/d1_review/empty_01.jpg)
- [Domain examples, first sheet](../audit_out/d1_review/domain_01.jpg)
- [Smallest crack visibility comparison](../audit_out/d1_review/visibility_01.jpg)
- [Smallest paint patch visibility comparison](../audit_out/d1_review/visibility_05.jpg)
- [Recalculated inventory](../audit_out/d1_review/inventory.json)
- [Image and annotation manifest](../audit_out/d1_review/manifest.jsonl)
- [Visual review decisions](d1_visual_review.json)

Validation: box/polygon normalization, six invalid-input cases, complete D1 count reconciliation through the repaired legacy parser, YAML parsing/policy assertions, and coverage of all 32 empty labels passed. Raw label hashes in the review ledger tie decisions to the reviewed label versions.
