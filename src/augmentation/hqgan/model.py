"""Paper-guided HQGAN reconstruction; see docs/hqgan.md for unresolved details."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def haar(x):
    """Unnormalized Haar bands, paper Eqs. 1-2 (LL, LH, HL, HH)."""
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError('Haar input dimensions must be even')
    a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
    c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return a+b+c+d, -a-b+c+d, -a+b-c+d, a-b-c+d


def inverse_haar(bands):
    """Eq. 3; correct its duplicated (2i-1,2j) index to (2i,2j-1)."""
    ll, lh, hl, hh = bands
    out = ll.new_empty(*ll.shape[:-2], ll.shape[-2]*2, ll.shape[-1]*2)
    out[..., ::2, ::2] = (ll-lh-hl+hh)/4
    out[..., ::2, 1::2] = (ll-lh+hl-hh)/4
    out[..., 1::2, ::2] = (ll+lh-hl-hh)/4
    out[..., 1::2, 1::2] = (ll+lh+hl+hh)/4
    return out


class HSC(nn.Module):
    def __init__(self, source_channels, target_channels, alpha=1.0, magnitude=True):
        super().__init__()
        self.alpha = alpha
        self.magnitude = magnitude
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  nn.Conv2d(source_channels, target_channels, 4),
                                  nn.LeakyReLU(0.2), nn.Conv2d(target_channels, target_channels, 1),
                                  nn.Sigmoid())

    def forward(self, low, high):
        ll, lh, hl, hh = haar(low)
        detail = inverse_haar((torch.zeros_like(ll), lh, hl, hh))
        # Signed 2x2 Haar details have zero mean: aligned average pooling kills
        # this path at 8/16px. Magnitude pooling is an explicit local deviation.
        return high + self.alpha * high * self.gate(detail.abs() if self.magnitude else detail)


class Down(nn.Module):
    def __init__(self, cin, cout, frequency=False):
        super().__init__()
        self.spatial = nn.Conv2d(cin, cout, 4, 2, 1)
        self.frequency = nn.Conv2d(cin, cout, 1) if frequency else None

    def forward(self, x):
        y = self.spatial(x)
        if self.frequency is not None:
            _, lh, hl, hh = haar(x)
            y = y + self.frequency(lh+hl+hh)
        return F.leaky_relu(y, 0.2)


class AHD(nn.Module):
    """Eqs. 7-11. Schedule shape and controller gain are documented choices."""
    def __init__(self, beta_start=0.0001, beta_end=0.01, max_steps=500,
                 initial_steps=1.0, target=0.6, gain=1.0, interval=4):
        super().__init__()
        if not (0 < beta_start <= beta_end < 1 and max_steps >= 1 and interval >= 1
                and 1 <= initial_steps <= max_steps and 0 <= target <= 1 and gain > 0):
            raise ValueError('Invalid AHD settings')
        self.max_steps, self.target, self.gain, self.interval = max_steps, target, gain, interval
        self.register_buffer('alpha_bar', torch.cumprod(1-torch.linspace(beta_start, beta_end, max_steps), 0))
        self.register_buffer('horizon', torch.tensor(float(initial_steps)))
        self.register_buffer('stat_sum', torch.tensor(0.0))
        self.register_buffer('batches', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        horizon = max(1, min(self.max_steps, int(self.horizon.item())))
        weights = torch.arange(1, horizon+1, dtype=x.dtype, device=x.device)
        steps = torch.multinomial(weights, x.shape[0], replacement=True)
        a = self.alpha_bar[steps].view(-1, 1, 1, 1)
        return a.sqrt()*x + (1-a).sqrt()*torch.randn_like(x)

    @torch.no_grad()
    def update(self, real_logits, fake_logits):
        self.stat_sum.add_((real_logits.sign().mean()-fake_logits.sign().mean())/2)
        self.batches.add_(1)
        if self.batches.item() == self.interval:
            self.horizon.add_((self.stat_sum/self.interval-self.target)*self.gain).clamp_(1, self.max_steps)
            self.stat_sum.zero_()
            self.batches.zero_()


def channels(width, base):
    if not math.isfinite(width) or not 0 < width <= 1:
        raise ValueError('width must be in (0, 1]')
    return max(4, round(base*width))


class Generator(nn.Module):
    def __init__(self, resolution=128, latent_dim=256, width=1.0, hsc_alpha=1.0, hsc_magnitude=True):
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError('Only 64 and 128 pixel generators are supported')
        self.latent_dim, self.resolution = latent_dim, resolution
        sizes = {4:2048, 8:1024, 16:512, 32:256, 64:256, 128:128}
        c = {r:channels(width, v) for r,v in sizes.items()}
        self.initial = nn.Sequential(nn.ConvTranspose2d(latent_dim, c[4]*2, 4),
                                     nn.BatchNorm2d(c[4]*2), nn.GLU(dim=1))
        self.blocks = nn.ModuleDict({str(r):nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(c[r//2], c[r]*2, 3, padding=1), nn.BatchNorm2d(c[r]*2), nn.GLU(dim=1))
                    for r in (8,16,32,64,128) if r <= resolution})
        self.skips = nn.ModuleDict({str(r):HSC(c[r//8], c[r], hsc_alpha, hsc_magnitude)
                                   for r in (32,64,128) if r <= resolution})
        self.rgb = nn.Conv2d(c[resolution], 3, 1)

    def forward(self, z):
        features = {4:self.initial(z.reshape(-1, self.latent_dim, 1, 1))}
        for name, block in self.blocks.items():
            r = int(name)
            features[r] = block(features[r//2])
            if name in self.skips:
                features[r] = self.skips[name](features[r//8], features[r])
        return self.rgb(features[self.resolution]).tanh()


class Discriminator(nn.Module):
    """Four FPN branches and 64 logits (Fig. 4); learned FPN is an explicit assumption."""
    def __init__(self, width=1.0, ahd=None):
        super().__init__()
        c = {128:channels(width,64), 64:channels(width,64), 32:channels(width,128),
             16:channels(width,256), 8:channels(width,512), 4:channels(width,512)}
        self.stem = nn.Conv2d(3, c[64], 4, 2, 1)
        self.down = nn.ModuleDict({str(r):Down(c[r*2], c[r]) for r in (32,16,8)})
        self.lateral = nn.ModuleDict({str(r):nn.Conv2d(c[r//2], c[r], 1) for r in (16,32,64)})
        self.full = nn.Conv2d(c[64], c[128], 1)
        self.heads = nn.ModuleDict()
        for start in (16,32,64,128):
            layers = []
            r = start
            while r > 4:
                layers.append(Down(c[r], c[r//2], frequency=r in (16,8)))
                r //= 2
            layers.append(nn.Conv2d(c[4], 1, 1))
            self.heads[str(start)] = nn.Sequential(*layers)
        self.ahd = AHD(**(ahd or {}))

    def forward(self, images, diffuse=True):
        # Fig. 4 uses a 128px pyramid even when the generator produces 64px crops.
        x = F.interpolate(images, size=(128,128), mode='bilinear', align_corners=False)
        bottom = {64:F.leaky_relu(self.stem(x),0.2)}
        for r in (32,16,8):
            bottom[r] = self.down[str(r)](bottom[r*2])
        top = {8:bottom[8]}
        for r in (16,32,64):
            top[r] = bottom[r] + F.interpolate(self.lateral[str(r)](top[r//2]), size=(r,r), mode='nearest')
        top[128] = self.full(F.interpolate(top[64], size=(128,128), mode='nearest'))
        return torch.cat([head(self.ahd(top[int(r)]) if diffuse else top[int(r)]).flatten(1)
                          for r,head in self.heads.items()], dim=1)


def discriminator_loss(real, fake):
    return F.softplus(-real).mean()+F.softplus(fake).mean()


def generator_loss(fake, objective='minimax'):
    if objective == 'minimax':
        return -F.softplus(fake).mean()  # literal minimizing log(1-D) in Eq. 12
    if objective == 'non_saturating':
        return F.softplus(-fake).mean()
    raise ValueError('Unknown generator objective')
