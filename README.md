# Data-Efficiency in Aircraft Skin Defect Detection

**Generative Augmentation vs. Few-Shot Segmentation Under Data Scarcity**

Senior Design Project, B.Sc. Computer Science and Engineering
American University of Sharjah, Fall 2026
Advisor: Dr. Mustafa Hammad

| Team | ID |
|---|---|
| Baden Albert Dominic Paiva | b00100562 |
| Mazin Rizvi | b00096249 |
| Sanjana Sara Siju | g00101441 |
| Sidhanth Menon | b00098743 |

> **Project status: scaffolding only.** The directory skeleton exists; no
> experiment code has been written and **no results exist yet**. Public
> datasets have been collected (see [Section 3.1](#31-data-sources)) and
> partly audited (see Section 3.1). Every command in the "Usage" section is marked
> *planned*. This README will be updated file by file as each module lands
> (see [Status](#status)).

---

## 1. What this project is

Labeled aircraft-defect imagery is scarce: cracks and corrosion are rare on
well-maintained fleets and costly to photograph. Two published techniques
attack this scarcity from opposite directions:

- **Generative augmentation** (HQGAN, Li et al. 2024 [7]): synthesize more
  realistic defect images from a small real set. Reported +5.27 pp mAP for
  YOLOX (to 78.55%).
- **Few-shot segmentation** (CHPDNet, Li et al. 2026 [6]): need fewer labels
  per class using CLIP-guided priors. Reported 60.13% mIoU at 1-shot.

Each was validated at **one data volume** and **one detector generation**.
Nobody has tested whether the gains persist as data shrinks, when the two
methods are combined, or when the detector is replaced by a newer one
(YOLOv8, YOLO11, YOLO26).

This project is a controlled **verification study** that closes that gap.
It is not a proposal for a new detector or a new generative model.

## 2. Research questions

1. **RQ1 (scarcity).** How do generative augmentation and few-shot
   segmentation compare, individually and combined, as the amount of labeled
   aircraft-defect data decreases?
2. **RQ2 (architecture).** Do the gains reported for each method persist when
   the detector or segmentation backbone is updated to a newer architecture?

## 3. Experimental design

Every result is one cell of this grid:

| Axis | Levels |
|---|---|
| **Training-data fraction** | e.g. 100%, 50%, 25%, 10%, 5% (final values set in `config/experiment.yaml`) |
| **Data-efficiency arm** | `baseline`, `classical_aug`, `hqgan_aug`, `fewshot`, `hqgan_aug + fewshot` |
| **Detector / backbone** | YOLOX (reproduction anchor), YOLOv8, YOLO11, YOLO26 |
| **Seed** | 3-5 seeds per cell |

Design rules (non-negotiable, enforced by tests where possible):

- **Fixed val/test sets.** Only the training pool shrinks. Test images never change.
- **Nested subsets.** The 10% subset is contained in the 25% subset, and so on.
- **No leakage.** Perceptual-hash dedup, then grouped splits so near-duplicates
  never straddle train/test. The GAN is trained **only** on the current run's
  training subset. Synthetic images never enter validation or test.
- **Real-only evaluation.** All metrics are computed on real images.
- **Controls.** A no-augmentation baseline and a classical-augmentation
  baseline (flip/rotate/noise) are included, since [7] found overused classical
  augmentation *hurts*.
- **Reproduction gate.** Before any new-detector result is trusted, the
  published YOLOX baseline (with and without HQGAN) must be roughly reproduced.
  If it cannot be, results are reported with that caveat.

### 3.1 Data sources

AIRSD, the labeled base named in the project report, is currently
**unavailable**: the Google Drive link in the J-DDL paper [3] has expired. The
**primary dataset is Innovation Hangar v2 (D1)**. Smaller sets
(D2-D4) are kept as supplementary data. Per-source settings (versions, class
maps, known issues) are recorded in `config/experiment.yaml` under
`datasets.sources`.

| ID | Source (version used) | Images | Raw classes | License | Status |
|---|---|---|---|---|---|
| **D1** | **Innovation Hangar, Roboflow `innovation-hangar-v2` (v2, stock): PRIMARY**. Appears to be the second public set used in [12] (to confirm) | 4,288 | crack, dent, missing-head, paint-off, scratch | CC BY 4.0 | screened; admission blocked (see D1 review) |
| D2 | DDIISc, Roboflow `aircraft_skin_defects` (v1, stock) | 372 | crack, dent, missing-head, paint-peel-off, scratch | CC BY 4.0 | audited |
| D3 | SUTD, Roboflow `aircraft-ai-dataset` (v4, no augmentation); the dataset of report ref [12] | 983 | rust, scractch (sic), missing-rivet | CC BY 4.0 | audited |
| D4 | Xiong, Zenodo `airscraft-skin-crack-dent` (sic), v1, DOI 10.5281/zenodo.16792216; companion to the INN-YOLO paper | 1,559 | crack, dent (paper names; numeric order visually checked) | CC BY 4.0 | partly audited |

**Which sources feed the experiments.** The main grid runs on D1 alone. D2-D4
are supplementary; whether they are pooled into the main grid
(`datasets.pool_supplementary`) is undecided (TBD). **Dibya is excluded** because
all of its images are contained in D1. Excluded and unavailable source details
are kept in `docs/source_archive.yaml`; they have no active dataset IDs.

**Admission rules.**

- **Un-augmented exports only.** Roboflow-side flips, rotations, grayscale,
  contrast equalization and class remapping are rejected, because they
  confound the `classical_aug` arm and spread copies of one photo across
  splits. Two linked versions were rejected for this reason:
  - D2 **v23**: 3x augmentation, grayscale, remapped classes. Use stock **v1**.
  - D1 **v1 ("Aug", 10,722 images)**: its training split is 3x-augmented
    (9,651 = 3 x 3,217). Use stock **v2** (4,288 images; the export reports no added augmentation, but
    altered-looking source imagery still needs review).
- **Pinned versions.** Changing one requires a commit message saying why.
- **Roboflow's own splits are discarded.** Sources are deduped across sources,
  then re-split with the fixed seed in `experiment.yaml` (split stratified by
  class and, when sources are pooled, by source).
- **D1 mixes boxes and polygons.** Preserve polygons and derive candidate
  enclosing boxes after review. No complete segmentation evaluation masks
  are certified. See [the D1 review](docs/dataset_audit.md).

**Unified class mapping** (`class_map.py` target: dent / crack / corrosion /
paint_peeling):

| Unified class | D1 (primary) | D2 | D3 | D4 |
|---|---|---|---|---|
| `crack` | crack | crack | none | crack |
| `dent` | dent | dent | none | dent |
| `corrosion` | none | none | rust | none |
| `paint_peeling` | paint-off | paint-peel-off | none | none |
| unmapped | missing-head, scratch | missing-head, scratch | missing-rivet, scractch | none listed |

Unmapped defects now **exclude the entire image** from the core study. They
are not relabeled as healthy background. All 32 D1/D2 empty-label images
remain quarantined: 14 show missing annotations, 18 are ambiguous, and none
is certified background. Policies and filename-level findings are in
[the D1 review](docs/dataset_audit.md) and [review ledger](docs/d1_visual_review.json).

**Audit results** (`tools/audit_datasets.py`; D1 is now stock v2, replacing the
wrong v1-aug copy used in the first run).

- **D1 v2 metadata is confirmed, but admission is blocked.** Its 803 apparent
  malformed rows are polygons (522 crack, 281 dent), generally following
  damage in reviewed overlays but not certified complete masks. Normalized
  pre-dedup records: crack 4,195, dent 3,826, paint-off 294. The same pHash<=5
  representative rule retains 2,891 images with 2,548 crack, 2,283 dent and
  294 paint records **before admission review**, not final counts.
- **Source-content discrepancy:** all local images are already 640x640;
  sampled images show posterization, light-spot effects, rotation and tiled
  content despite export metadata saying no augmentation. Non-aircraft and
  severe incident imagery also occur. Upstream provenance and annotation
  completeness must be resolved before claiming a clean core dataset.
- **Dibya (excluded) is contained in D1.** All 1,115 Dibya (excluded) images have a phash <= 5 match in D1,
  with the same `IMG_2023...` base filenames. Dibya is excluded.
- D2 and D3 are stock, un-augmented exports. D2, D3 and D1 share no images with
  each other (0 matches at phash <= 5). D4 was not in the overlap matrix.
- **Near-duplicates**, images in phash <= 5 clusters: Dibya (excluded) 14 (1%), D2 2, D3 230
  (23%), D4 1,235 (79%), D1 v1-aug 2,919 (27%). Clusters that straddle
  Roboflow's own train/valid/test: Dibya (excluded) 3, D2 0, D3 46, D4 177, D1 157. This
  confirms that Roboflow's splits cannot be used.
- **D3:** classes rust, scractch (sic), missing-rivet. Raw boxes: rust 2,039
  (in 639 images), scractch 2,068, missing-rivet 1,408. Filename groups:
  `sceneadd*` 595 images (72% of the rust boxes), `scene*` 127, `image-*` 261.
- **D4:** 1,559 images (provided split 1,338/192/29), YOLO labels. The class
  names are now confirmed by the paper; local overlays support 0=crack,
  1=dent. The paper documents augmentation from 247 originals. See
  [D4 evidence](docs/d4_verification.md). The current cleaner finds 1,558
  images and one orphan label; 1,559 is the historical inventory.
- **Exact-byte check (second pass, 15,547 images in six folders):** only 2
  byte-identical pairs inside D4 (one straddles its train/valid) and 47
  byte-identical Dibya (excluded)/D1 images. A byte-level check understates overlap, since
  the perceptual check shows all of Dibya (excluded) is in D1. The augmented D2 export (v23i)
  has already been removed from `data/raw/`; the augmented D1 v1 export has
  been replaced by v2; source transformations remain an admission issue.
- **Source-predictability test: inconclusive.** Its 0.82 accuracy matches the
  share of D1 in the pool (81%); balanced accuracy was about 34% vs. 25%
  chance.

**Consequences for the design.**

- **Corrosion is a stretch goal** (team decision); the core study covers dent,
  crack and paint_peeling, with crack as the safety-critical class. If it is
  attempted, it is viable by count but fragile: D3 has 2,039 rust boxes, but
  all come from one source, D3 is 23-47% near-duplicate images, and 72% of the
  rust boxes sit in `sceneadd*` images. Their contact sheets look like real
  photographs, but all D3 images show a single parked Boeing 747, and the
  `rust` labels appear to include weathering streaks. The corrosion result
  must be stated as a single-source, single-aircraft limitation.
- **D4's effective size is much smaller than 1,559** (79% near-duplicates), and
  its own splits leak. Grouped splitting is mandatory.
- **Paint peeling is thin in D1:** 294 geometry-normalized records before
  admission. Scarcity and split estimates must be recalculated from the
  final reviewed manifest; historical box-only planning estimates are invalid.
- **Another audit run is needed:** the latest report still has no D4 in the
  overlap matrix and no per-class counts.

## 4. Metrics

- **Detection:** mAP50, mAP50-95, per-class AP.
- **Segmentation:** mIoU, per-class IoU.
- **Safety-critical recall:** recall on **cracks and corrosion**, tracked
  *separately* from overall mAP. A single blended metric can hide lopsided
  performance [5].
- **Cost:** training time, inference ms/image, parameter count.
- **Statistics:** paired tests across seeds, Holm-Bonferroni correction,
  bootstrap 95% CIs, effect sizes.

## 5. Scope and limitations

- **Surface-visible defects only** (dent, crack, corrosion, paint peeling).
  Barely visible impact damage and sub-surface damage are out of scope [4].
- **Decision-support framing.** The pipeline is a proof of concept to inform
  human inspectors, not a replacement for them.
- **Dataset caveats.** The report names AIRSD as the labeled base, but its
  download link has expired, so the study currently runs on the public
  Innovation Hangar v2 dataset (Section 3.1), with smaller supplementary
  sets, and its primary source has no corrosion labels. D1 also contains polygons,
  incomplete annotations and transformed-looking source images. The full audit (duplicates, real vs.
  synthetic images, licenses) is **only partly done** (see
  [Open decisions](#8-open-decisions)). Results on these data cannot be
  compared directly with the AIRSD numbers in [3].
- **Re-implementation risk.** If official HQGAN / CHPDNet code or the ASD-2^i
  dataset are not public, the methods will be re-implemented from the papers.
  That would be documented as a deviation and a limitation.

## 6. Repository layout

`[ ]` = not started, `[x]` = implemented and tested. Everything is `[ ]` today.

```
aircraft-data-efficiency/
├── README.md                          [x] this file
├── requirements.txt                   [ ]
├── config/
│   ├── experiment.yaml                [x] fractions, arms, detectors, seeds, thresholds
│   ├── datasets.yaml                  [ ] paths, class maps, licenses
│   └── detectors/                     [ ] one yaml per detector generation
├── data/                              gitignored: raw/ interim/ processed/ (raw/ has one subfolder per source ID)
├── src/
│   ├── datasets/
│   │   ├── airsd.py                   [ ] loader
│   │   ├── class_map.py               [ ] unify labels -> dent/crack/corrosion/paint_peeling
│   │   ├── dedup.py                   [ ] perceptual-hash grouping
│   │   ├── splits.py                  [ ] fixed val/test, grouped splits
│   │   └── yolo_export.py             [ ] write YOLO-format dirs for a subset
│   ├── subsampling.py                 [ ] seeded, stratified, nested fractions
│   ├── augmentation/
│   │   ├── classical.py               [ ] control arm
│   │   ├── hqgan/                     [ ] model, train, generate, blend
│   │   └── synth_registry.py          [ ] which synthetic set belongs to which (fraction, seed)
│   ├── fewshot/
│   │   ├── chpdnet/                   [ ] CLIP prior + purification decoder
│   │   ├── episodes.py                [ ] k-shot support/query sampling
│   │   └── pseudo_masks.py            [ ] box -> mask conversion if needed
│   ├── detectors/
│   │   ├── base.py                    [ ] common train/predict interface
│   │   └── yolo_wrappers.py           [ ] per-generation adapters
│   ├── evaluation/
│   │   ├── detection.py               [ ] mAP, per-class AP
│   │   ├── segmentation.py            [ ] mIoU
│   │   └── safety_recall.py           [ ] crack/corrosion recall
│   ├── statistics.py                  [ ]
│   └── visualization.py               [ ]
├── scripts/
│   ├── 01_prepare_data.py             [ ]
│   ├── 02_train_gan.py                [ ]
│   ├── 03_run_detection_grid.py       [ ]
│   ├── 04_run_fewshot_grid.py         [ ]
│   ├── 05_run_combined.py             [ ]
│   ├── 06_aggregate_and_stats.py      [ ]
│   ├── 07_make_figures_tables.py      [ ]
│   └── smoke_test.py                  [ ]
├── notebooks/                         [ ] Colab/Kaggle launchers
├── tests/                             [ ]
├── results/                           raw/ aggregated/ tables/ figures/  (empty)
└── docs/                              [ ] protocol, dataset audit, deviations log
```

**Data flow** (every number in the report must come from this chain, with no
hand-typed values):

```
config/*.yaml -> scripts/0x_*.py -> results/raw/*.csv
              -> 06_aggregate_and_stats.py -> results/aggregated/*.csv
              -> 07_make_figures_tables.py -> results/{tables,figures}/
```

## 7. Usage (*planned, none of this works yet*)

```bash
# 0. environment (GPU required for GAN / YOLO / CLIP stages)
pip install -r requirements.txt

# 1. prepare data: dedup, class map, fixed splits, nested subsets
python scripts/01_prepare_data.py --config config/experiment.yaml

# 2. train one GAN per (fraction, seed); cached
python scripts/02_train_gan.py --config config/experiment.yaml

# 3-5. run the experiment arms
python scripts/03_run_detection_grid.py --config config/experiment.yaml
python scripts/04_run_fewshot_grid.py   --config config/experiment.yaml
python scripts/05_run_combined.py       --config config/experiment.yaml

# 6-7. aggregate, test, and produce tables/figures
python scripts/06_aggregate_and_stats.py
python scripts/07_make_figures_tables.py

# sanity check of the pipeline mechanics on tiny fake data (no GPU)
python scripts/smoke_test.py
```

## 8. Open decisions

These must be resolved and recorded in `docs/` before the matching code is
written.

1. **Shared task.** HQGAN is evidenced for *detection* (boxes, YOLO); CHPDNet
   for *segmentation* (masks, mIoU). Combining them requires choosing one of:
   evaluate everything as detection; evaluate everything as segmentation
   (YOLO-seg); or keep two tracks with a combined third arm. This depends on
   what AIRSD annotations actually contain.
2. **Dataset audit.** Two runs done (Section 3.1). Still to do: finish D1 provenance,
   completeness and mask certification (see docs/dataset_audit.md), resolve D4's augmented-source admission and review cross-source similarity groups, confirm D3's real-vs-synthetic
   status beyond the visual contact-sheet review (looks real), decide whether supplementary sets are pooled,
   and run the real-vs-synthetic checks. AIRSD (and ASD-2^i if obtainable) is still
   pending; write to the J-DDL authors for a working link and a per-image
   real/synthetic flag. Record the outcome in `docs/dataset_audit.md`.
3. **Code availability.** Check whether official HQGAN and CHPDNet
   implementations exist; otherwise budget for re-implementation.
4. **Compute budget.** Grid size = arms x fractions x detectors x seeds, each
   cell a real training run. Fix the seed count and fractions accordingly.
5. **Gulf / desert objectives.** Either (a) drop them from this study's
   claims, or (b) add a separate, clearly-scoped robustness experiment. As
   written, no experiment tests them.

## 9. Development workflow

Files are built one at a time, in dependency order:

1. `config/` -> `src/datasets/` -> `src/subsampling.py` (+ tests)
2. `src/evaluation/` and `src/statistics.py` (+ tests)
3. `src/detectors/` baseline arm, then the **reproduction gate**
4. `src/augmentation/` (classical, then HQGAN)
5. `src/fewshot/`
6. Combined arm, aggregation, figures

Each file is added with its tests and a status flip in Section 6.

## 10. References

Key sources (full list in the project report):

- [3] Huang et al., "J-DDL: Surface damage detection and localization system
  for fighter aircraft" (introduces AIRSD), arXiv:2506.10505, 2025.
- [4] Kucukkalfa et al., *Composite Structures*, 392, 2026.
- [5] Kurniawan et al., *Technologies*, 14(1), 2026.
- [6] Li et al., "Few-shot aircraft skin defect segmentation via CLIP-guided
  prior masks and hierarchical purification decoder," *Aerosp. Sci. Technol.*,
  168, 2026.
- [7] Li, Wang, Liu, "A high-quality GAN with data augmentation for aircraft
  skin defect detection under limited data," *IEEE TIM*, 73, 2024.
- [9] Plastropoulos et al., *Technologies*, 12(9), 2024.
- [12] Suvittawat et al., *Aerospace*, 12(4), 2025.
- [14] Wang et al., YOLO26-SimAM, ICCS 2026.
- [15] Molleda et al., YOLOv12/YOLOv26 steel-defect study, *J. Real-Time Image
  Process.*, 23, 2026.

Datasets (Section 3.1; all Roboflow Universe entries are CC BY 4.0 and require
attribution):

- D1 (primary): Innovation Hangar, *innovation-hangar-v2*, v2, Roboflow
  Universe, 2023. https://universe.roboflow.com/innovation-hangar/innovation-hangar-v2

- Dibya (excluded): Dibya Dillip, *aircraft-skin-defects-new-dataset*, v1, Roboflow
  Universe, 2023. https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset
- D2: DDIISc, *aircraft_skin_defects*, v1, Roboflow Universe, 2023.
  https://universe.roboflow.com/ddiisc/aircraft_skin_defects
- D3: SUTD, *aircraft-ai-dataset*, v4, Roboflow Universe, 2024 (dataset of
  [12]). https://universe.roboflow.com/sutd-4mhea/aircraft-ai-dataset
- D4: Xiong, J., *airscraft-skin-crack-dent* (sic), v1, Zenodo, 2025.
  https://doi.org/10.5281/zenodo.16792216. Companion paper: Xiong et al., "An
  aircraft skin defect detection method with UAV based on GB-CPP and
  INN-YOLO," *Drones*, 9(9), 594, 2025. https://doi.org/10.3390/drones9090594

## 11. License

Project code: TBD. Third-party datasets keep their own licenses (CC BY 4.0
for D1-D4) and must be attributed.

## Detection preparation (implemented)

Use [the preparation guide](docs/prepare_detection.md) for the D1/D2/D4
cleaner: `python -B tools/prepare_detection.py --output data/interim/d124_review --contact-sheets`.
Use a fresh output path if that run already exists. The script creates review
candidates and exports approved images with `--reviews ... --export`.
Raw datasets remain unchanged; masks are not required for this detection step.
