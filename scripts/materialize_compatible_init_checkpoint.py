#!/usr/bin/env python3
"""Materialize a deterministic iteration-zero checkpoint for baseline evaluation."""

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disorderflow.models import get_model  # noqa: E402
from disorderflow.utils.misc import load_config, seed_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite baseline: {args.output}')

    config, _ = load_config(str(args.config))
    seed = int(config.train.seed)
    seed_all(seed)
    model = get_model(config.model)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    target_state = model.state_dict()
    compatible = {
        key: value for key, value in source['model'].items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'config': config,
        'model': model.state_dict(),
        'iteration': 0,
        'weights_kind': 'compatible_initialization',
        'initialization_seed': seed,
        'source_checkpoint': str(args.source),
        'compatible_tensors_loaded': len(compatible),
        'total_tensors': len(target_state),
    }, args.output)
    print({
        'output': str(args.output),
        'seed': seed,
        'compatible_tensors_loaded': len(compatible),
        'total_tensors': len(target_state),
    })


if __name__ == '__main__':
    main()
