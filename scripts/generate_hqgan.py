"""Generate unapproved defect patch candidates from a local HQGAN checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from detection_pipeline import resolve, sha256, torch_device, write_json
from augmentation.hqgan.training import METHOD, sampling_generator, save_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='auto', help='auto (a CUDA GPU if PyTorch can use one, else the CPU), cpu, or cuda:0')
    a = p.parse_args()
    if a.count < 1:
        p.error('count must be positive')
    a.device = torch_device(a.device)
    output, weights = resolve(a.output), resolve(a.weights)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('method') != METHOD:
        p.error('Not a supported HQGAN reconstruction checkpoint (older checkpoints need the code that made them)')
    torch.set_num_threads(2)
    g = sampling_generator(checkpoint, a.device)
    output.mkdir(parents=True, exist_ok=False)
    images = save_samples(g, output, a.count, a.seed, a.device)
    write_json(output/'samples.json', {'synthetic':True, 'smoke':checkpoint['smoke'],
               'approved_for_detector_training':False, 'identity':checkpoint['identity'],
               'checkpoint_sha256':sha256(weights), 'seed':a.seed, 'images':images})
    print(f'Generated {len(images)} unapproved patch candidates: {output}')


if __name__ == '__main__':
    main()
