"""Bounded pilot training and sampling, with frozen training-subset provenance."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from detection_pipeline import check_dataset, sha256, write_json
from training_progress import Progress
from .data import crop_settings, selected_rows
from .model import Generator, Discriminator, discriminator_loss, generator_loss

METHOD = 'hqgan_paper_guided_reconstruction_v2'


class PatchDataset(Dataset):
    def __init__(self, directory, class_name, resolution, experiment, crop=None):
        self.directory = Path(directory).resolve()
        self.resolution = resolution
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        m = self.manifest
        if m.get('status') != 'complete' or (self.directory/'failure.json').exists():
            raise ValueError('Patch export is not complete')
        # Exports from before the crop rule was recorded were tight boxes, as annotated.
        exported = crop_settings(m.get('crop'), m.get('min_side', 8))
        if crop is not None and crop_settings(crop) != exported:
            raise ValueError('Patch export was prepared with a different crop rule; prepare it again')
        data, _, frozen, runtime = check_dataset(m['source_data'], experiment)
        if sha256(data.parent/'split_manifest.json') != m['split_manifest_sha256']:
            raise ValueError('Frozen dataset identity changed')
        sources = m['training_sources']
        rows = selected_rows(frozen, [s['export_stem'] for s in sources])
        expected = [{k:r[k] for k in ('export_stem','group','image_sha256','export_label_sha256')} for r in rows]
        if sources != expected or hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest() != m['training_subset_sha256']:
            raise ValueError('Training subset provenance changed')
        if class_name not in runtime['names']:
            raise ValueError('Unknown class')
        self.records = [r for r in json.loads((self.directory/'patches.json').read_text()) if r['class_name'] == class_name]
        if len(self.records) < 2:
            raise ValueError('At least two patches are required for the selected class')
        index = {r['export_stem']:r for r in rows}
        for record in self.records:
            source = index.get(record['source_stem'])
            if (source is None or record['source_split'] != 'train' or record['group'] != source['group']
                    or record['source_image_sha256'] != source['image_sha256']
                    or record['source_label_sha256'] != source['export_label_sha256']
                    or record['class_id'] != runtime['names'].index(class_name)):
                raise ValueError('Patch provenance is inconsistent with training subset')
            path = (self.directory/record['patch']).resolve()
            if not path.is_relative_to(self.directory) or sha256(path) != record['sha256']:
                raise ValueError('Patch content changed or escaped export')
        self.identity = {'training_subset_sha256':m['training_subset_sha256'],
                         'split_manifest_sha256':m['split_manifest_sha256'],
                         'patch_records_sha256':sha256(self.directory/'patches.json'),
                         'class_name':class_name, 'quality':m['quality'], 'crop':exported}
        self.loaded = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Each patch is verified and decoded once, then served from memory. On a GPU the disk
        # read and hash would otherwise take longer than the training step.
        if index not in self.loaded:
            record = self.records[index]
            path = self.directory/record['patch']
            if sha256(path) != record['sha256']:
                raise ValueError('Patch changed during training')
            with Image.open(path) as image:
                # Local dataset adaptation: rectangular crops are resized to the model's square input.
                array = np.array(image.convert('RGB').resize((self.resolution, self.resolution), Image.Resampling.BILINEAR), copy=True)
            self.loaded[index] = torch.from_numpy(array).permute(2,0,1).float()/127.5-1
        return self.loaded[index]


def finite_step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise ValueError('Non-finite GAN loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Non-finite GAN gradient')
    optimizer.step()


@torch.no_grad()
def standing_statistics(generator, seed, device, batches=32, batch_size=16):
    """Measure BatchNorm statistics for the weights about to be sampled; returns it in eval mode.

    Averaged weights never produced the running statistics stored beside them, so they are
    measured over fresh noise batches (BigGAN's "standing statistics") before eval-mode sampling.
    """
    norms = [m for m in generator.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats(); m.momentum = None  # None: plain average over the batches below
    rng = torch.Generator(device=device).manual_seed(seed)
    generator.train()
    for _ in range(batches):
        generator(torch.randn(batch_size, generator.latent_dim, generator=rng, device=device))
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    return generator.eval()


def sampling_generator(checkpoint, device):
    """The generator to sample from: averaged weights with their own BatchNorm statistics."""
    generator = Generator(**checkpoint['architecture']).to(device)
    generator.load_state_dict(checkpoint['generator_ema'])
    return generator.eval()


def to_pixels(image):
    return ((image.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()


def save_samples(generator, output, count, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    was_training = generator.training
    generator.eval()
    paths = []
    try:
        with torch.no_grad():
            for i in range(count):
                image = generator(torch.randn(1, generator.latent_dim, generator=rng, device=device))[0]
                path = output/f'sample_{i:04d}.png'
                Image.fromarray(to_pixels(image)).save(path)
                paths.append({'image':path.name, 'sha256':sha256(path)})
    finally:
        generator.train(was_training)
    return paths


@torch.no_grad()
def save_sheet(generator, path, noise, columns=8):
    """One image of fixed-noise samples: the quickest check of what a checkpoint produces."""
    images = [to_pixels(generator(z[None])[0]) for z in noise]
    side = images[0].shape[0]; rows = math.ceil(len(images)/columns)
    sheet = Image.new('RGB', (columns*(side+2), rows*(side+2)), 'white')
    for i, pixels in enumerate(images):
        sheet.paste(Image.fromarray(pixels), ((i % columns)*(side+2), (i//columns)*(side+2)))
    path.parent.mkdir(exist_ok=True)
    sheet.save(path)


def train(dataset, settings, output, steps, device, smoke=False, resume=False):
    if steps < 1 or settings['batch_size'] < 2 or settings['checkpoint_every'] < 1:
        raise ValueError('Steps/checkpoint interval must be positive and batch must be at least two')
    if not math.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0:
        raise ValueError('Learning rate must be positive and finite')
    decay = settings['ema_decay']
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay < 1:
        raise ValueError('ema_decay must be in [0, 1); 0 samples the current weights')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    torch.manual_seed(settings['seed'])
    torch.set_num_threads(settings['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    architecture = {k:settings[k] for k in ('resolution','latent_dim','width','hsc_alpha','hsc_magnitude')}
    manifest = {'method':METHOD, 'smoke':smoke,
                'settings':settings, 'steps_requested':steps, 'identity':dataset.identity,
                'torch_version':str(torch.__version__), 'device':str(device),
                'preprocessing':'RGB, bilinear square resize, [-1,1]',
                'sampling':'averaged generator weights, standing BatchNorm statistics, eval mode',
                'paper_reproduction':False}
    if resume:
        old=json.loads((output/'run_manifest.json').read_text())
        if old!=manifest: raise ValueError('Resume requires identical settings, dataset, device and total steps')
    else:
        write_json(output/'run_manifest.json', manifest)
    try:
        g = Generator(**architecture).to(device)
        d = Discriminator(width=settings['width'], ahd=settings['ahd']).to(device)
        # Sampled weights: an average of the generator's recent states, smoother than any single step.
        ema = copy.deepcopy(g).requires_grad_(False)
        og = torch.optim.Adam(g.parameters(), lr=settings['learning_rate'], betas=tuple(settings['adam_betas']))
        od = torch.optim.Adam(d.parameters(), lr=settings['learning_rate'], betas=tuple(settings['adam_betas']))
        sampling = torch.Generator().manual_seed(settings['seed'])
        sheet_noise = torch.randn(32, settings['latent_dim'], generator=torch.Generator().manual_seed(settings['seed']+7)).to(device)
        start=0; sampler=None
        if resume:
            ckpt=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if ckpt['identity']!=dataset.identity or ckpt['settings']!=settings:
                raise ValueError('Checkpoint identity differs from resumed run')
            g.load_state_dict(ckpt['generator']); d.load_state_dict(ckpt['discriminator'])
            ema.load_state_dict(ckpt['generator_ema'])
            og.load_state_dict(ckpt['generator_optimizer']); od.load_state_dict(ckpt['discriminator_optimizer'])
            torch.set_rng_state(ckpt['torch_rng']); sampling.set_state(ckpt['sampling_rng']); start=ckpt['step']
            if str(device).startswith('cuda') and ckpt.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
            # Trim logs written after the last durable checkpoint.
            existing=[json.loads(line) for line in (output/'losses.jsonl').read_text().splitlines()]
            (output/'losses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in existing if r['step']<=start))
        progress = Progress('HQGAN '+str(dataset.identity.get('class_name', 'training')), steps, initial=start)
        with (output/'losses.jsonl').open('a' if resume else 'w', encoding='utf-8') as log:
            for step in range(start+1, steps+1):
                ids = torch.randint(len(dataset), (settings['batch_size'],), generator=sampling)
                real = torch.stack([dataset[int(i)] for i in ids]).to(device)
                d.requires_grad_(True)
                od.zero_grad(set_to_none=True)
                # Train mode on purpose: the discriminator must judge the same batch-statistics
                # output that the generator step below is optimised through.
                with torch.no_grad():
                    fake = g(torch.randn(len(real), settings['latent_dim'], device=device))
                real_logits, fake_logits = d(real), d(fake)
                dl = discriminator_loss(real_logits, fake_logits)
                finite_step(dl, d, od)
                d.ahd.update(real_logits.detach(), fake_logits.detach())
                d.requires_grad_(False)
                og.zero_grad(set_to_none=True)
                fake = g(torch.randn(len(real), settings['latent_dim'], device=device))
                gl = generator_loss(d(fake), settings['generator_objective'])
                finite_step(gl, g, og)
                with torch.no_grad():
                    # Early steps follow the generator closely; the average lengthens to ema_decay.
                    keep = min(decay, (1+step)/(10+step))
                    for average, current in zip(ema.parameters(), g.parameters()):
                        average.lerp_(current, 1-keep)
                row = {'step':step, 'd_loss':float(dl.detach()), 'g_loss':float(gl.detach()),
                       'diffusion_horizon':float(d.ahd.horizon), 'kimg':step*len(real)/1000}
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                if step == 1 or step % settings['checkpoint_every'] == 0 or step == steps:
                    # A copy, so the averaged weights keep training untouched; its statistics
                    # depend only on those weights and the step, which keeps resume exact.
                    sampler = standing_statistics(copy.deepcopy(ema), 1_000_003*(settings['seed']+1)+step, device)
                    checkpoint = {**manifest, 'architecture':architecture, 'step':step,
                                  'generator':g.state_dict(), 'discriminator':d.state_dict(),
                                  'generator_ema':sampler.state_dict(),
                                  'generator_optimizer':og.state_dict(), 'discriminator_optimizer':od.state_dict(),
                                  'torch_rng':torch.get_rng_state(), 'sampling_rng':sampling.get_state(),
                                  'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
                    temp = output/'checkpoint.tmp'
                    torch.save(checkpoint, temp)
                    temp.replace(output/'last.pt')
                    save_sheet(sampler, output/'sheets'/f'step_{step:06d}.png', sheet_noise)
                progress.update(step, f'D={row["d_loss"]:.4f} G={row["g_loss"]:.4f} diffusion={row["diffusion_horizon"]:.2f}',
                                force=step % settings['checkpoint_every'] == 0)
        print('HQGAN training finished; saving sample previews...', flush=True)
        if sampler is None:  # resumed at the final step: rebuild from the durable checkpoint
            sampler = sampling_generator(torch.load(output/'last.pt',map_location='cpu',weights_only=True), device)
        previews = save_samples(sampler, output, 4, settings['seed']+1, device)
        write_json(output/'samples.json', {'synthetic':True, 'smoke':smoke,
                   'approved_for_detector_training':False, 'identity':dataset.identity,
                   'checkpoint_sha256':sha256(output/'last.pt'), 'images':previews})
        write_json(output/'completion.json', {'status':'complete', 'steps':steps, 'smoke':smoke})
    except BaseException as exc:
        write_json(output/'completion.json', {'status':'failed', 'error':str(exc)})
        raise
