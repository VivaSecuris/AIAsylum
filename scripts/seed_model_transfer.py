#!/usr/bin/env python3
"""Build private CPU-only rsync bases; these are NEVER published as originals.

Use remote_models.sh --basis-root to correct any numerical/serialization
changes and verify the original file SHA-256 before publishing a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--directions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    output = args.output.resolve()
    if not output.is_relative_to(project / '.model-transfers'):
        parser.error('Private output must be inside this project/.model-transfers/')
    if args.output.is_symlink():
        parser.error('Private output must not be a symlink')
    manifest = json.loads(args.manifest.read_text())
    metadata = json.loads(args.metadata.read_text())
    checkpoints = {item['name']: item for item in manifest['checkpoints']}
    selected = list(checkpoints) if args.models == ['all'] else args.models
    for name in selected:
        if name not in checkpoints or Path(name).name != name:
            parser.error(f'Unknown checkpoint {name}')
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    from vivasecuris.aiasylum.weights.direction import RefusalDirection
    from vivasecuris.aiasylum.weights.surgery import edit_and_save
    for name in selected:
        checkpoint = checkpoints[name]
        destination = output / name
        receipt = output / f'{name}.basis.json'
        if receipt.is_file() and destination.is_dir():
            previous = json.loads(receipt.read_text())
            if previous['source_sha256'] == checkpoint['sha256']:
                print(f'{name}: private basis already built', flush=True)
                continue
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f'Incomplete basis exists: {destination}; inspect it before retrying')
        weights = {item['path']: item for item in checkpoint['files'] if item['path'].endswith(('.safetensors', '.bin'))}
        surgery = metadata[name]['surgery']
        config = metadata[name]['config']
        source = surgery['source_model']
        direction_id = surgery.get('extra', {}).get('direction_run_id')
        matching = None
        # Historical CLI edits may lack a run id. An identical weight manifest
        # from a named sibling supplies its exact derivation and reusable basis.
        for other_name, other in checkpoints.items():
            other_weights = {item['path']: item for item in other['files'] if item['path'].endswith(('.safetensors', '.bin'))}
            if weights == other_weights:
                sibling_id = metadata[other_name]['surgery'].get('extra', {}).get('direction_run_id')
                if direction_id is None and sibling_id is not None:
                    direction_id = sibling_id
                sibling = output / other_name
                if other_name != name and sibling.is_dir() and (output / f'{other_name}.basis.json').is_file():
                    matching = sibling
        if matching:
            print(f'{name}: copying private basis from byte-identical original sibling {matching.name}', flush=True)
            shutil.copytree(matching, destination, dirs_exist_ok=True)
        else:
            if direction_id is None:
                raise RuntimeError(f'{name}: no saved direction or identical sibling; use a normal upload')
            direction_path = args.directions / str(direction_id)
            direction = RefusalDirection.load(direction_path)
            if direction.model_id != source or direction.split_hash != surgery.get('split_hash'):
                raise RuntimeError(f'{name}: saved direction does not match checkpoint provenance')
            rank = surgery.get('extra', {}).get('subspace_rank')
            if rank is not None:
                if direction.basis is None or int(direction.basis.shape[0]) < rank:
                    raise RuntimeError(f'{name}: saved subspace is too small')
                direction.basis = direction.basis[:rank].contiguous()
                direction.basis_layers = direction.basis_layers[:rank]
            dtype = config.get('dtype') or config.get('torch_dtype')
            if dtype not in ('float32', 'float16', 'bfloat16'):
                raise RuntimeError(f'{name}: unsupported original dtype {dtype}')
            print(f'{name}: rebuilding private {dtype} CPU basis from direction {direction_id}', flush=True)
            edit_and_save(source_model=source, direction=direction, out_dir=str(destination),
                          beta=surgery.get('beta') if surgery.get('beta') is not None else 0.0,
                          device='cpu', dtype=dtype, include_embeddings=surgery.get('embeddings_edited', True),
                          notes='Private transfer basis; must not be published without original hash verification.',
                          use_subspace=surgery['method'] == 'direction_subspace',
                          k=surgery.get('extra', {}).get('k'))
        checks = []
        for filename, expected in weights.items():
            actual = destination / filename
            checks.append({'file': filename, 'matches_original_file': actual.is_file() and actual.stat().st_size == expected['bytes'] and sha256(actual) == expected['sha256']})
        result = {'name': name, 'status': 'private_basis_only', 'source_sha256': checkpoint['sha256'], 'direction_run_id': direction_id, 'weight_file_checks': checks}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
