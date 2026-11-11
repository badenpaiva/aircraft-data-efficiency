# HQGAN design and implementation

The SUTD pipeline uses a paper-guided HQGAN reconstruction to generate defect patches. The current configuration is `config/hqgan_sutd.yaml`: 64px output, latent dimension 128, width multiplier 0.25, batch 8, Adam learning rate 0.0002, non-saturating generator loss and checkpoints every 500 steps. The study budget is 20,000 steps per class/subset, with 400 generated candidates per class. Training crops are the annotated box plus a 12.5% margin, with boxes under 32px skipped and tall boxes turned long side horizontal. Samples come from a moving average of the generator weights.

This is method `hqgan_paper_guided_reconstruction_v2`. The first study used `v1`; its checkpoints and study folders cannot be loaded by this code and remain as records. The changes from `v1` are the three sections after the mapping table: where the diffusion noise is added, how fakes are generated and sampled, and how crops are cut and pasted.

The generator produces image pixels, not bounding boxes. The automatic pipeline labels the core of each inserted patch, which follows from the crop rule; [image review](review.md) describes human refinement of that policy. [Pipeline](pipeline.md) contains all run and resume commands.

The architecture below describes the reference reconstruction; the active SUTD model reduces resolution, latent dimension and channel widths. These differences must accompany reported results.

## Paper mapping and reconstruction choices

Target: Li, Wang and Liu, *A High-Quality GAN With Data Augmentation for Aircraft
Skin Defect Detection Under Limited Data*, IEEE TIM 73 (2024),
[DOI 10.1109/TIM.2024.3420361](https://doi.org/10.1109/TIM.2024.3420361).

Pages 3-7 and Figs. 2/4 drive the model implementation; pages 8-10 describe
synthesis. No author source code was obtained.
The inspected PDF SHA-256 is
`08d6ea7dea59ee816b2eafc21cc3f7ed17e54b036b543a2b147a58c6877fce4e`.
The licensed PDF and rendered review pages are not added to version control.

| Paper component | Implementation / fidelity |
|---|---|
| Eqs. 1-3, unnormalized Haar DWT and inverse | Exact arithmetic; correct the duplicated third inverse-coordinate index to (2i, 2j-1). Roundtrip and gradient tested. |
| Fig. 2 generator | z=256; channels 2048,1024,512,256,256,128 at 4,8,16,32,64,128px; batch normalization and GLU; HSC 4-to-32, 8-to-64, 16-to-128. |
| HSC, Eq. 4 | High-pass reconstruction, 4x4 pooling, 4x4/1x1 convolutions, sigmoid residual gate. Alpha=1 is a choice. **Default magnitude pooling is a deliberate deviation**, explained below. |
| Fig. 4 discriminator | Four FPN scales 16/32/64/128, independent heads and 64 concatenated logits. HFFB sums LH+HL+HH before a learned channel projection at 16/8px. |
| AHD, Eqs. 7-11 | Beta endpoints .0001/.01, 500-step ceiling, t sampling proportional to t, target .6, controller updated every four D minibatches from their average sign statistic. **The noise is added to the discriminator's input image, not to the pyramid features: a deliberate deviation**, explained below. |
| Eq. 12 | Stable logistic discriminator loss; the paper specifies minimax. The active SUTD configuration uses non_saturating, a documented deviation. |
| Section IV-B | Adam learning rate .0002 and batch 16. Paper budget is 10,000 kimg (10 million image presentations), not 10,000 iterations. |

The text does not fully specify the FPN backbone/pretraining, convolutions and
activation details, Adam betas, diffusion schedule shape, controller gain C,
initial T, or explicit timestep conditioning mechanism. This reconstruction uses
a jointly learned convolutional FPN (NOT the pretrained Projected-GAN feature
extractor), nearest upsampling, bilinear 64-to-128 discriminator input resize,
3x3 generator convolutions, 4x4 stride-two discriminator convolutions,
LeakyReLU(.2), a tanh RGB output, Adam betas (.5,.999), a linear beta schedule,
C=1, initial T=1, unit Gaussian variance and no explicit timestep embedding.
AHD draws one timestep per image. Fractional T is retained by the
controller and floored when sampling. These choices require comparison with
author code before calling this a reproduction. No pretrained feature network,
spectral normalization or additional regularizers are implied; the generator
average used for sampling is described below.

**HSC cancellation:** removing LL yields zero-mean signed details in each 2x2
block. Pooling those aligned details from 8/16px to 4px cancels the feature
signal, so the gate becomes driven by biases rather than source frequencies.
The test suite demonstrates this. `hsc_magnitude: true` takes the absolute
detail before pooling to retain a signal. Set it to false for the literal
signed operation, and report which variant is used. This is not attributed to
the authors and needs an ablation or clarification from them.

## Where the diffusion noise is added

The paper adds the diffusion noise to the feature maps of the pyramid, and its pyramid is a fixed feature network. Ours is trained together with the discriminator heads. It can therefore settle real-versus-fake before the noise is added and repeat the answer across thousands of feature values, where independent noise averages out. The first study showed the result: on the 5% subsets the horizon reached its ceiling of 500 while the discriminator kept winning.

A direct test confirms it. A discriminator was trained to separate two halves of the same 72 real rust patches, which is only possible by memorising them, with the horizon held at its maximum of 500. The score is the paper's overfitting statistic: 0 is chance, 1 is complete memorisation, and the controller's target is 0.6.

| Where the noise is added | Score at step 1,000 | Score later |
|---|---:|---|
| Nowhere (control) | 1.00 | |
| Pyramid features (`v1`) | 0.55 | 0.88 at step 2,000; 0.98 at step 3,000 |
| Pyramid features, normalised per sample | 1.00 | |
| Input image (`v2`) | 0.17 | 0.43 at step 2,000; 0.53 at step 3,000 |
| Input image, horizon held at 100 | 0.82 | 0.86 at step 1,500 |

Normalising the features, the obvious repair, does nothing. With feature noise the score passed the target between steps 1,000 and 1,250. With noise on the input image it was still under the target at step 3,000, and it rises freely at a low horizon, so the controller has a working range. That is what `v2` uses.

The protection is a slowdown, not a stop: the image-noise score was still creeping up at step 3,000 on this 72-image set, where each image had been shown about 670 times. If `diffusion_horizon` in `losses.jsonl` sits at 500, the regulariser has run out. The remedy is a higher ceiling (`max_steps`, with `beta_end` raised to keep the same slope), which departs from the paper's 500. It has not been needed or tested yet.

## How fakes are generated and sampled

- **Train-mode fakes for the discriminator.** The discriminator step judges fakes produced in train mode, the same batch-statistics output that the generator step is optimised through. `v1` produced them in eval mode, so the two networks were trained on different fake distributions.
- **Averaged weights for sampling.** Samples come from an exponential moving average of the generator weights (`ema_decay`, 0.999; 0 samples the current weights). The paper does not mention one; FastGAN and Projected-GAN, its baseline, use one. Averaged weights never produced the BatchNorm statistics stored beside them, so the statistics are measured afresh over 512 noise samples at every checkpoint, and sampling runs in eval mode.
- **A sample sheet at every checkpoint.** `sheets/step_NNNNNN.png` holds 32 samples from fixed noise. Look at these before trusting a loss curve.

## Crops, placement and labels

Each training crop is the annotated box plus 12.5% of its width and of its height on every side, stretched to the model's square input.

- **Small boxes are skipped, not enlarged.** A box with a side under 32px is left out; that is 22% of the boxes in the full training pool.
- **Context beyond the photograph is mirrored.** About a third of crops touch an image edge.
- **Tall boxes are turned a quarter turn**, so the long side of every crop is horizontal.

Placement mirrors the crop. A real box of the same class from the subset supplies the core, the same margin is added around it, and a tall core turns the patch back. The annotated box fills the central 80% of every training crop, so the central 80% of every pasted patch is its label. No guess is involved.

In `v1` there was no orientation step, so a wide patch could be pasted into a tall box and be squeezed along both axes. From the share of tall boxes, about half of all placements had that mismatch.

Square, unstretched crops were considered and rejected for this dataset. SUTD boxes are long and thin (median short-to-long ratio 0.33 for missing-rivet, 0.50 for rust, 0.55 for scratch), so the crop rule would no longer say where the defect lies inside the patch. The best fixed label would overlap the true box with a mean IoU of 0.44 to 0.58, and 40 to 61% of generated labels would fall under IoU 0.5. Dense annotation is a second problem: a square window around a thin box takes in neighbouring defects, which would be pasted without labels.

Stretching therefore remains. A generated patch pasted at a different aspect ratio than its training crop is distorted along one axis. Removing that needs the generator to produce its own box, which is not implemented.

Set `crop` to margin 0, `min_box_side` 8 and `canonical_orientation` false to reproduce the `v1` crops.

## Dataset adaptation

The dataset adaptation trains one unconditional model per class on all usable crops of
that class in the selected training subset. Crops are resized to square inputs
(bilinear) as described above, unlike the paper's selected 200 fixed-size patches
per class. The SUTD classes are missing-rivet, rust and scratch. Results are not
an exact reproduction of the paper's dataset or training protocol.


## Interpretation and model checks

Haar transforms, HSC, HFFB and adaptive high-dimensional diffusion are implemented, with regression tests for model behavior and training resume. Passing those checks does not establish photorealism or detection benefit. The pipeline must be assessed through candidate/composite inspection and matched real-only detector comparisons.

Poisson blending uses a rectangular interior mask, not a segmentation mask. The paper describes manual placement (page 9) and manual labeling (page 11); it does not specify our exact review exchange or crop-rule box policy. The current outputs retain approximate-label provenance. Changes to review and boxes must create separate training inputs so completed results remain interpretable.
