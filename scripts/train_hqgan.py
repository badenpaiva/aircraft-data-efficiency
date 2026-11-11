"""Train one class of the paper-guided HQGAN reconstruction; see docs/hqgan.md."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import yaml
from detection_pipeline import resolve, load_config, torch_device
from augmentation.hqgan.training import PatchDataset, train


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches', default='data/interim/sutd_v4_units_gan_patches')
    p.add_argument('--config', default='config/hqgan_sutd.yaml')
    p.add_argument('--experiment', default='config/experiment.yaml')
    p.add_argument('--class-name', required=True)
    p.add_argument('--steps', type=int, help='Explicit bounded number of optimizer steps')
    p.add_argument('--smoke', action='store_true', help='Two steps, batch two, 64px, 1/32 channel widths')
    p.add_argument('--device', default='auto', help='auto (a CUDA GPU if PyTorch can use one, else the CPU), cpu, or cuda:0; a resumed run must use the device it started on')
    p.add_argument('--name')
    p.add_argument('--resume', action='store_true', help='Resume the named run with identical total steps and settings')
    args = p.parse_args(argv)
    if (args.smoke and args.steps is not None) or (not args.smoke and (args.steps is None or args.steps < 1)):
        p.error('Choose --smoke or an explicit positive --steps')
    settings = yaml.safe_load(resolve(args.config).read_text())
    if args.smoke:
        settings.update(resolution=64, width=1/32, batch_size=2, checkpoint_every=1)
    name = args.name or 'hqgan_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    if Path(name).name != name or name in {'.','..'}:
        p.error('name must be a directory name')
    output = resolve('results/hqgan')/name
    if args.resume and not args.name: p.error('--resume requires --name')
    if output.exists() and not args.resume:
        p.error(f'Output already exists: {output}')
    print('Checking training patches and dataset provenance...', flush=True)
    dataset = PatchDataset(resolve(args.patches), args.class_name, settings['resolution'], load_config(args.experiment),
                           settings['crop'])
    device = torch_device(args.device)
    print(f'Training on {device}.', flush=True)
    train(dataset, settings, output, 2 if args.smoke else args.steps, device, args.smoke,args.resume)
    print(f'Completed: {output}')


if __name__ == '__main__':
    main()
