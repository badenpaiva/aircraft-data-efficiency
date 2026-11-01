"""Generate unapproved defect patch candidates from a local HQGAN checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from detection_pipeline import resolve, sha256, write_json
from augmentation.hqgan.model import Generator
from augmentation.hqgan.training import save_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.count < 1:
        p.error('count must be positive')
    output, weights = resolve(a.output), resolve(a.weights)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('method') != 'hqgan_paper_guided_reconstruction_v1':
        p.error('Not a supported HQGAN reconstruction checkpoint')
    torch.set_num_threads(2)
    g = Generator(**checkpoint['architecture']).to(a.device)
    g.load_state_dict(checkpoint['generator'])
    output.mkdir(parents=True, exist_ok=False)
    images = save_samples(g, output, a.count, a.seed, a.device)
    write_json(output/'samples.json', {'synthetic':True, 'smoke':checkpoint['smoke'],
               'approved_for_detector_training':False, 'identity':checkpoint['identity'],
               'checkpoint_sha256':sha256(weights), 'seed':a.seed, 'images':images})
    print(f'Generated {len(images)} unapproved patch candidates: {output}')


if __name__ == '__main__':
    main()
