# Next steps

Status as of 11 November 2026. Based on an audit of the HQGAN code against Li, Wang and Liu (IEEE TIM 2024, [DOI 10.1109/TIM.2024.3420361](https://doi.org/10.1109/TIM.2024.3420361)) and on the outputs of the first run. Design background is in [HQGAN design](hqgan.md); commands are in [pipeline](pipeline.md).

## Changed since the audit

The audit below describes the first study, which used the first split and GAN method `v1`. Since then:

- **Split:** re-cut by capture unit (`data/processed/sutd_v4_units`); see step 2a.
- **Subsets:** the sampler now gives each seed its own subset; see step 2a.
- **GAN (`v2`):** train-mode fakes, diffusion noise on the input image, averaged weights for sampling, a sample sheet per checkpoint, and crops with a margin, a size floor and a fixed orientation; see step 3 and [HQGAN design](hqgan.md).
- **Planned grid:** 100%, 50% and 25% of the training data, three seeds, and two detectors from one family: YOLO26n (light, 2.6M parameters) and YOLO26m (heavy, 21.9M). YOLOX-S, YOLOv8n and YOLO11n are out.
- **GPU:** every entry point now defaults to a CUDA GPU when PyTorch can see one, detectors use mixed precision where the card supports it, and setup reports the device. Runs are planned for a different, GPU machine; [pipeline](pipeline.md) has the setup. None of the GPU paths has been executed yet, because PyTorch on the development PC is the CPU-only build.

No detector or full GAN run has used any of this yet. The first study's folders cannot be loaded by the new code and stay as records.

## Where things stand

- **Complete:** seed-0 baselines (20 runs), seed-0 HQGAN whole-patch study (60 runs), 15 seed-0 GANs, and 16 seed-1 baselines (5% to 50%).
- **Not run:** classical augmentation arms, seed 2, seed-1 100% baselines, reserved-test evaluation, any FID/IS measurement.
- **Stopped:** the seed-1 5% rust GAN is marked `failed` with an empty error, which is what an interrupted run leaves behind. Seed-1 scratch was never started.

**The first HQGAN run does not show a benefit.** Compared with the matching seed-0 baselines, the three Ultralytics models improve in 18 of 45 cells and change by about -0.8 mAP50 points on average (computed from the two report tables). That is inside single-seed noise. YOLOX-S is too erratic to read: its own baseline drops from 14.4% at 25% data to 8.9% at 50%.

**The reason is upstream of the detectors: the GANs did not learn usable defects.**

| Subset | What the samples look like | What the logs show |
|---|---|---|
| 5% (53 to 75 patches per class) | Two or three near-identical images per class. Scratch samples are about five times less varied than the real patches. | Diffusion horizon T climbs to its ceiling of 500 while discriminator loss keeps falling (0.09 to 0.49). |
| 100% (1,005 to 1,452 patches per class) | Blurry grey texture with checkerboard artefacts. No recognisable rivet hole, rust colour or scratch. | Discriminator loss sits at 1.386 (chance) for nearly all 20,000 steps. T never leaves 1 for rust and scratch. |

So the 60 detector runs measured the effect of pasting blurred or duplicated patches with whole-patch boxes. They say nothing yet about HQGAN augmentation.

**The dataset itself needs work before any further training.** The files are technically clean: every box is valid, and there are no empty images or exact duplicates. The problems are in how the data was captured and labelled.

- **The split leaks.** The 983 images are 722 frames sampled every 20 frames from 34 drone videos, plus 261 stills in 7 photo series. In the first split, every validation and test video frame has training frames from the same video; 46% of validation frames and 41% of test frames sit within one sampling step of a training frame. The perceptual-hash grouping keeps only 21% of consecutive sampled frames together, so near-identical views of the same physical defect are on both sides of that split. The completed studies used it.
- **Labels are inconsistent between consecutive frames.** Of 4,569 boxes in consecutive frame pairs, 32% have no same-class box in the neighbouring frame even at a loose overlap of 0.1 (some of this is camera motion, so treat it as an upper bound). Inspected pairs show the same marks labelled scratch in one frame and missing-rivet in the next, and rivet rows boxed in one frame and unlabelled in the next.
- **Class definitions are loose.** Missing-rivet boxes usually enclose a whole row of fasteners or a seam (39% are longer than 4:1). Scratch covers stains, dirt streaks and faded paint. Boxes mark regions, not individual defects.
- **Images are low-detail.** Frames are stretched to 640x640 from their original shape, and 22 to 32% of boxes have a short side under 32px.

Leakage should inflate validation scores, yet the best model reaches only 49% mAP50 at 100% data. Inconsistent labels are the likely ceiling. Loose region boxes are also why GAN crops contain little that looks like a single defect.

## Audit summary

The code is an honest, reduced, paper-guided reconstruction. It is not a faithful implementation, and `run_manifest.json` already records `paper_reproduction: false`.

**Matches the paper**

- Haar DWT and inverse (Eqs. 1 to 3), including the fix for the duplicated index in Eq. 3.
- Generator layout at `width: 1.0`: 256-d noise, 2048/1024/512/256/256/128 channels, batch norm and GLU, HSC wired from layer i to layer i+3 (Fig. 2, Eq. 4).
- HFFB at the 16px and 8px discriminator layers; four branches giving 64 logits (Fig. 4).
- AHD noise, overfitting statistic, T update and t sampling (Eqs. 7 to 11); beta 0.0001 to 0.01, T ceiling 500, threshold 0.6, update every four minibatches.
- Logistic discriminator loss (Eq. 12), Adam at 0.0002, Poisson blending through `cv2.seamlessClone`.

**Deviations, most consequential first**

| # | Paper | This codebase | Documented in hqgan.md? |
|---|---|---|---|
| 1 | 10,000 kimg at batch 16 | 20,000 steps at batch 8 = 160 kimg, 1.6% of the budget | Partly (budget stated, ratio not) |
| 2 | Discriminator built on the Projected-GAN baseline. Fig. 4's channel and resolution layout matches Projected-GAN's frozen pretrained feature network with its mixing layers. | Small convolutional pyramid trained from scratch, no normalisation | Yes, as an assumption |
| 3 | AHD noise is added to those fixed features | Noise (sigma 1) is added to features the discriminator learns itself, so it can outgrow the noise. Measured: feature RMS is 3 to 8 at 5% data, and accuracy with and without AHD is almost identical at T = 500. | No |
| 4 | Full-size model, 64px for small defects and 128px for skin damage | `width: 0.25`, latent 128, 64px for every class | Yes |
| 5 | 200 fixed-size 64px or 128px crops per class from 4000x3000 photos | Every bounding-box crop, stretched to a 64px square. Over 70% of crops have a short side under 64px, 22 to 45% under 32px, and 41 to 76% are more than 2:1 in aspect. Blur and distortion are in the training data. | Partly (resize stated, effect not) |
| 6 | Not stated; standard practice is one generator mode for both updates | Fakes for the discriminator step come from the generator in eval mode (running batch-norm statistics); the generator step uses train mode. The two networks see different fake distributions. | No |
| 7 | HSC pools signed high-frequency detail | Pools the absolute value by default, because the literal operation cancels to zero at 8px and 16px | Yes, with a test |
| 8 | Minimax generator loss | `non_saturating` | Yes |
| 9 | FID and IS reported per class | No generation-quality metric at all | Yes |
| 10 | 400 composites, one defect of each of four types per image, locations chosen by hand, labelled by hand | One patch per composite, random non-overlapping placement, whole-patch box, no human review | Yes |
| 11 | FCOS, YOLOX, YOLOv8; SGD; 300 epochs | YOLOX-S, YOLOv8n, YOLO11n, YOLO26n; AdamW; 30 epochs | Yes |
| 12 | Classical control: flip, mirror, noise, scale 80 to 120%, rotation | Rotation, flip and noise only | No |

Items 2 and 3 rest on reading Fig. 4 against Projected-GAN from memory, and the AHD equations closely follow Diffusion-GAN (Wang et al., 2022), which has a Projected-GAN variant. Check both public repositories before acting on them; they are the nearest thing to author code.

## What to do next

### 1. Decide what the project claims

Either "we reproduce HQGAN" or "we test an HQGAN-style augmentation at reduced scale". The first needs items 1 to 5 closed. The second is achievable sooner, but still needs a GAN that produces recognisable defects. Everything below assumes the second. A GPU makes parts of the first reachable, starting with the training budget.

### 2. Stop spending compute on the current GANs

- Do not resume seed 1 or start seed 2 GAN training with the present model. It would reproduce the same failure.
- Do not launch more HQGAN detector cells from the current candidates.
- Keep `sutd_full_v1` and `sutd_whole_patch_seed0_v1` frozen as the record of the exploratory run.

### 2a. Clean the dataset first

Everything downstream inherits the split and the labels, so this comes before GAN or detector work. It needs a new prepared export and a new study plan; existing results stay frozen as they are.

1. **Re-split by capture unit. Done on 10 November 2026.** The active export is `data/processed/sutd_v4_units` (683/106/194 images); see [dataset preparation](prepare_detection.md). `scene N` and `sceneadd N` turned out to be different videos, giving 41 units. The subset sampler now picks one of the three best-balancing groups at each step by seed, so seeds 0 to 2 get three different 25% subsets (6 to 8 groups each, sharing 3) with similar class counts. The 50% subsets differ less (15 to 17 groups, sharing 12 to 14), because the 233-image photo group never fits and 50% is then three quarters of the video groups.
2. **Write a labelling convention.** One page: what counts as each class, whether a rivet row is one box or several, how to treat stains and paint, and the minimum visible size.
3. **Relabel validation and test to that convention.** Evaluation labels matter most. Use the existing `review.csv` and CVAT workflow. The consecutive-frame disagreement check can rank which training images to review next.
4. **Curate GAN crops separately.** Pick tight, clearly visible single defects per class. Do not feed every box crop to the GAN.
5. **Ask for the unstretched originals.** The 640x640 stretch was applied at export. If the dataset owners can provide original frames, both detection and generation gain detail.

Until step 1 is done, do not use absolute validation scores as evidence of generalisation. Comparisons between arms on the current split are still like-for-like.

### 3. Fix the GAN, in this order

Items 1 to 3 are done as method `v2` (11 November 2026). Two of them ended up different from the first plan, because measurements contradicted it; [HQGAN design](hqgan.md) has the numbers.

1. **Training data. Done, but not with square crops.** Crops are the box plus a 12.5% margin, boxes under 32px are skipped, and tall boxes are turned long side horizontal; pasting mirrors the same rule, so the label is exact by construction. Square crops were dropped: SUTD boxes are long and thin, and a square crop would leave 40 to 61% of generated labels under IoU 0.5.
2. **Batch-norm mismatch. Done.** Discriminator fakes come from the generator in train mode.
3. **AHD. Done, but not by normalising features.** A memorisation test showed feature noise does not restrain the discriminator even when normalised. The noise now goes on the input image, which slowed memorisation at least threefold at the ceiling. It is a slowdown, not a stop.
4. **Sampling. Done.** Samples use averaged generator weights with freshly measured BatchNorm statistics, and every checkpoint writes a sample sheet.
5. **Pilot. Not done.** Train one class at 100% data for the full 20,000 steps and look at `sheets/`. A 1,000-step run only showed that the new code works; the discriminator was still at chance, as at the same point in `v1`. Watch `diffusion_horizon`: pinned at 500 means the regulariser has run out.
6. **Discriminator backbone. Implemented, pilot pending (11 November 2026).** `discriminator: projected` (see `config/hqgan_sutd_projected.yaml`) uses frozen ImageNet EfficientNet-B0 features, fixed random channel and scale mixing, and trained heads with HFFB at 16 and 8px, 64 logits. AHD noise stays on the input image. B0 stands in for Projected-GAN's EfficientNet-lite0, which torchvision does not ship. Context: the `v2` rust pilot with the learned discriminator plateaued from about step 10,000 and grew grid artefacts by step 42,000 (`results/hqgan/pilot_rust_100k`).
7. **Budget.** The paper's 10,000 kimg is 62 times the current 160 kimg. On the GPU machine, time the pilot first, then raise `gan.steps` in `config/detection_study.yaml` to what the 27 GANs can afford. If the horizon then pins at 500, raise the diffusion ceiling as described in [HQGAN design](hqgan.md).
8. **HSC.** Ablate `hsc_magnitude` true against false once the model trains at all. Low priority until then.
9. **Exact labels without stretching.** Stretching remains: a patch pasted at a different aspect ratio than its training crop is distorted along one axis. Removing it needs the generator to output its own box alongside the image.

### 4. Add a way to tell whether a GAN is good

- FID and IS per class against held-out real crops, to compare with the paper's Table I. With under 100 real patches at 5%, also report KID, which is less biased at small sample sizes.
- A nearest-real-neighbour check per generated sample, to catch memorisation at the small fractions.
- A contact sheet saved with every checkpoint. Done: `sheets/step_NNNNNN.png` in each GAN run. The first study's failure was visible in ten seconds of looking and invisible in the loss curves alone.
- A minimum-diversity gate in screening. The present near-duplicate check passed 96 scratch candidates from a collapsed 5% model.

Set a pass criterion before training, and do not send candidates to blending until it is met.

### 5. Complete the comparisons that do not depend on the GAN

These are needed whatever the GAN does, and none of the first study's runs can stand in for them: the split and the detector list have both changed.

- Run the real-only and classical arms of the new plan: 54 cells (3 fractions, 3 seeds, 3 arms, 2 detectors). One command does it; see [pipeline](pipeline.md).
- Decide whether to add scaling to the classical arms so they match the paper's recipe.
- Check that YOLO26m fits the GPU at batch 8 before planning. The batch size is shared by both detectors, so change it for both or not at all.
- Decide about online detector augmentation. It is all switched off in `src/study/backends.py`, which is why the nano models overfitted within 10 to 20 epochs in the first study. A model eight times larger will overfit sooner.

### 6. Rerun HQGAN detection only after step 4 passes

- Use the reviewed workflow in [image review](review.md) for at least one subset, so labels are tight boxes instead of whole patches.
- Consider multiple defects per composite and location-aware placement, both closer to the paper.
- Compare against baseline and classical arms at the same model, fraction and seed.

### 7. Housekeeping

- Commit the work since the first study; nothing from 10 or 11 November is committed.
- Copy `data/processed/sutd_v4_units` to the GPU machine; `data/` is not in git.
- The development PC has an unused GPU (GTX 1660 SUPER, 6 GB): its PyTorch is the CPU-only build. Installing the CUDA build there would allow GPU paths to be tested before long runs elsewhere.
- Row 12 of the deviations table (the reduced classical recipe) is still not in [hqgan.md](hqgan.md) or [pipeline](pipeline.md).
- Build the Graphify graph with `/graphify .` when wanted; finish the Ponytail plugin install.
