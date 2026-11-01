# HQGAN design and implementation

The SUTD pipeline uses a paper-guided HQGAN reconstruction to generate defect patches. The current configuration is `config/hqgan_sutd.yaml`: 64px output, latent dimension 128, width multiplier 0.25, batch 8, Adam learning rate 0.0002, non-saturating generator loss and checkpoints every 500 steps. The study budget is 20,000 steps per class/subset, with 400 generated candidates per class.

The generator produces image pixels, not bounding boxes. The current pipeline labels the whole inserted patch; [image review](review.md) describes refinement of that policy. [Pipeline](pipeline.md) contains all run and resume commands.

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
| AHD, Eqs. 7-11 | Beta endpoints .0001/.01, 500-step ceiling, t sampling proportional to t, target .6, controller updated every four D minibatches from their average sign statistic. |
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
AHD draws independently per image and scale. Fractional T is retained by the
controller and floored when sampling. These choices require comparison with
author code before calling this a reproduction. No pretrained feature network,
spectral normalization, EMA or additional regularizers are implied.

**HSC cancellation:** removing LL yields zero-mean signed details in each 2x2
block. Pooling those aligned details from 8/16px to 4px cancels the feature
signal, so the gate becomes driven by biases rather than source frequencies.
The test suite demonstrates this. `hsc_magnitude: true` takes the absolute
detail before pooling to retain a signal. Set it to false for the literal
signed operation, and report which variant is used. This is not attributed to
the authors and needs an ablation or clarification from them.

The dataset adaptation trains one unconditional model per class on all patches of
that class in the selected training subset. Native rectangular crops are resized
to square inputs (bilinear), unlike the paper's selected 200 fixed-size patches
per class. The SUTD classes are missing-rivet, rust and scratch. Results are not
an exact reproduction of the paper's dataset or training protocol.


## Interpretation and model checks

Haar transforms, HSC, HFFB and adaptive high-dimensional diffusion are implemented, with regression tests for model behavior and training resume. Passing those checks does not establish photorealism or detection benefit. The pipeline must be assessed through candidate/composite inspection and matched real-only detector comparisons.

Poisson blending uses a rectangular interior mask, not a segmentation mask. The paper describes manual placement (page 9) and manual labeling (page 11); it does not specify our exact review exchange or whole-patch box policy. The current outputs retain approximate-label provenance. Changes to review and boxes must create separate training inputs so completed results remain interpretable.
