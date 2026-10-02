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
> experiment code has been written and **no results exist yet**. Every command
> in the "Usage" section is marked *planned*. This README will be updated file
> by file as each module lands (see [Status](#status)).

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
- **Dataset caveats.** The report names AIRSD as the labeled base. Its label
  format (boxes vs. masks), class coverage and license have **not yet been
  audited** (see [Open decisions](#8-open-decisions)).
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
│   ├── experiment.yaml                [ ] fractions, arms, detectors, seeds, thresholds
│   ├── datasets.yaml                  [ ] paths, class maps, licenses
│   └── detectors/                     [ ] one yaml per detector generation
├── data/                              gitignored: raw/ interim/ processed/
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
2. **Dataset audit.** Confirm AIRSD (and ASD-2^i if obtainable): label format,
   classes, size, license, and duplicates.
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

## 11. License

TBD. 