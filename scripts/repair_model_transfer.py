#!/usr/bin/env python3
"""Repair private transfer bases using hierarchical hashes, then exact SHA-256.

This never publishes a model. remote_models.sh performs the final full-checkpoint
verification and atomic publication after this bandwidth-saving preparation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HASH_BYTES = 16
MAGIC = b'ASYLUMP1'
BLOCK_SIZES = (1024 * 1024, 64 * 1024, 4096, 256, 16)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            value.update(data)
    return value.hexdigest()


def blocks(ranges: list[list[int]], block_size: int, file_size: int):
    if block_size <= 0:
        raise ValueError('Block size must be positive')
    previous_end = 0
    for offset, length in ranges:
        if offset < previous_end or length <= 0 or offset + length > file_size:
            raise ValueError('Invalid, overlapping, or unordered block range')
        end = offset + length
        for start in range(offset, end, block_size):
            yield start, min(block_size, end - start)
        previous_end = end


def block_hashes(path: Path, ranges: list[list[int]], block_size: int, expected_size: int) -> bytes:
    if path.stat().st_size != expected_size:
        raise ValueError('Basis length differs from original')
    result = bytearray()
    with path.open('rb') as stream:
        for offset, length in blocks(ranges, block_size, expected_size):
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError('File changed during block hashing')
            result.extend(hashlib.sha256(data).digest()[:HASH_BYTES])
    return bytes(result)


def changed_blocks(original: Path, ranges: list[list[int]], block_size: int, remote_hashes: bytes) -> list[list[int]]:
    local_hashes = block_hashes(original, ranges, block_size, original.stat().st_size)
    if len(local_hashes) != len(remote_hashes):
        raise ValueError('Incomplete remote hash response')
    return [[offset, length] for index, (offset, length) in enumerate(blocks(ranges, block_size, original.stat().st_size))
            if local_hashes[index * HASH_BYTES:(index + 1) * HASH_BYTES] != remote_hashes[index * HASH_BYTES:(index + 1) * HASH_BYTES]]


def make_patch(original: Path, ranges: list[list[int]], max_bytes: int = 128 * 1024 * 1024) -> bytes:
    if sum(length for _, length in ranges) > max_bytes:
        raise ValueError('Correction exceeds transfer budget; basis is too different')
    output = io.BytesIO()
    output.write(MAGIC)
    output.write(struct.pack('<QQ', original.stat().st_size, len(ranges)))
    with original.open('rb') as stream:
        for offset, length in blocks(ranges, original.stat().st_size or 1, original.stat().st_size):
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError('Original changed while making patch')
            output.write(struct.pack('<QI', offset, length))
            output.write(data)
    return gzip.compress(output.getvalue(), compresslevel=6)


def apply_verified_patch(basis: Path, output: Path, compressed_patch: bytes, expected_size: int, expected_sha256: str) -> dict:
    if basis.stat().st_size != expected_size:
        raise ValueError('Basis length differs from original')
    if output.exists():
        if output.stat().st_size == expected_size and sha256(output) == expected_sha256:
            return {'status': 'already_verified', 'bytes': expected_size, 'sha256': expected_sha256}
        raise ValueError('Different corrected output already exists; refusing overwrite')
    patch = io.BytesIO(gzip.decompress(compressed_patch))
    if patch.read(8) != MAGIC:
        raise ValueError('Invalid correction header')
    size, count = struct.unpack('<QQ', patch.read(16))
    if size != expected_size:
        raise ValueError('Correction file length differs from original')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.repair-', dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(basis, temporary)
        previous_end = 0
        with temporary.open('r+b') as stream:
            for _ in range(count):
                header = patch.read(12)
                if len(header) != 12:
                    raise ValueError('Truncated correction record')
                offset, length = struct.unpack('<QI', header)
                if length <= 0 or offset < previous_end or offset + length > size:
                    raise ValueError('Invalid correction range')
                data = patch.read(length)
                if len(data) != length:
                    raise ValueError('Truncated correction data')
                stream.seek(offset)
                stream.write(data)
                previous_end = offset + length
        if patch.read(1):
            raise ValueError('Trailing correction data')
        if sha256(temporary) != expected_sha256:
            raise ValueError('Final SHA-256 differs from original; private output not published')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'status': 'verified_private_basis', 'bytes': size, 'blocks_corrected': count, 'sha256': expected_sha256}


def private_path(project: str, relative: str) -> Path:
    base = (Path.home() / project).resolve()
    result = (base / relative).resolve()
    if not base.is_relative_to(Path.home().resolve()) or not result.is_relative_to(base / '.model-transfers'):
        raise ValueError('Repair paths must remain in the remote project/.model-transfers')
    return result


def remote(action: str, project: str, basis: str, output: str | None = None) -> None:
    request = json.loads(sys.stdin.buffer.read()) if action != 'apply' else None
    path = private_path(project, basis)
    if action == 'hashes':
        sys.stdout.buffer.write(block_hashes(path, request['ranges'], request['block_size'], request['bytes']))
    elif action == 'status':
        target = private_path(project, output)
        if not target.exists():
            print(json.dumps({'status': 'missing'}))
        elif target.stat().st_size == request['bytes'] and sha256(target) == request['sha256']:
            print(json.dumps({'status': 'already_verified', 'bytes': request['bytes'], 'sha256': request['sha256']}))
        else:
            raise ValueError('Existing private correction differs from original; refusing overwrite')
    elif action == 'prepare':
        destination = private_path(project, output)
        destination.mkdir(parents=True, exist_ok=True)
        for source in path.rglob('*'):
            if source.is_symlink():
                raise ValueError('Symlinks are not valid private bases')
            if source.is_file() and source.suffix not in ('.safetensors', '.bin'):
                target = destination / source.relative_to(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
        print(json.dumps({'status': 'prepared'}))
    elif action == 'apply':
        expected_size = int(sys.argv[6])
        expected_sha = sys.argv[7]
        result = apply_verified_patch(path, private_path(project, output), sys.stdin.buffer.read(), expected_size, expected_sha)
        print(json.dumps(result))
    else:
        raise ValueError('Unknown remote operation')


def transfer(args) -> None:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._@-]*', args.host):
        raise ValueError('Invalid SSH alias')
    for value in (args.remote_dir, args.basis_root, args.output_root):
        if not re.fullmatch(r'[A-Za-z0-9.][A-Za-z0-9._/-]*', value) or '..' in Path(value).parts:
            raise ValueError('Expected plain relative project paths')
    for value in (args.basis_root, args.output_root):
        if not value.startswith('.model-transfers/'):
            raise ValueError('Bases and corrections must remain private')
    manifest = json.loads(args.manifest.read_text())
    checkpoints = {item['name']: item for item in manifest['checkpoints']}
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', args.host]
    helper = f'{args.remote_dir}/.model-transfers/repair_model_transfer.py'
    command = f'mkdir -p {shlex.quote(str(Path(helper).parent))} && cat > {shlex.quote(helper)}'
    subprocess.run(ssh + [command], input=Path(__file__).read_bytes(), check=True)
    report = {'status': 'running', 'models': [], 'started_at_unix': time.time()}
    def run(action, basis, output=None, payload=b'', extra=()):
        argv = ['python3', helper, 'remote', action, args.remote_dir, basis]
        if output is not None:
            argv.append(output)
        argv.extend(str(value) for value in extra)
        result = subprocess.run(ssh + [shlex.join(argv)], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        return result.stdout
    for name in args.models:
        if name not in checkpoints or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
            raise ValueError(f'Unknown checkpoint: {name}')
        model = checkpoints[name]
        run('prepare', f'{args.basis_root}/{name}', f'{args.output_root}/{name}', payload=b'{}')
        entry = {'name': name, 'files': []}
        for record in model['files']:
            filename = record['path']
            if not filename.endswith(('.safetensors', '.bin')):
                continue
            if Path(filename).is_absolute() or '..' in Path(filename).parts:
                raise ValueError('Invalid weight path')
            original = Path('models') / name / filename
            if original.stat().st_size != record['bytes']:
                raise ValueError('Original length changed since manifest')
            basis = f'{args.basis_root}/{name}/{filename}'
            output = f'{args.output_root}/{name}/{filename}'
            status = json.loads(run('status', basis, output, payload=json.dumps({'bytes': record['bytes'], 'sha256': record['sha256']}).encode()))
            if status['status'] == 'already_verified':
                entry['files'].append({'file': filename, 'rounds': [], 'patch_bytes_sent': 0, **status})
                print(f'{name}/{filename}: existing private original SHA-256 reverified', flush=True)
                continue
            ranges = [[0, record['bytes']]]
            rounds = []
            for size in BLOCK_SIZES:
                if not ranges:
                    break
                request = json.dumps({'ranges': ranges, 'block_size': size, 'bytes': record['bytes']}).encode()
                hashes = run('hashes', basis, payload=request)
                ranges = changed_blocks(original, ranges, size, hashes)
                rounds.append({'block_size': size, 'hash_bytes_received': len(hashes), 'mismatching_blocks': len(ranges)})
                print(f'{name}/{filename}: {size}-byte blocks: {len(ranges)} differ; {len(hashes)} hash bytes', flush=True)
            patch = make_patch(original, ranges)
            print(f'{name}/{filename}: sending {len(patch)} correction bytes', flush=True)
            result = json.loads(run('apply', basis, output, payload=patch, extra=(record['bytes'], record['sha256'])))
            entry['files'].append({'file': filename, 'rounds': rounds, 'patch_bytes_sent': len(patch), **result})
            print(f'{name}/{filename}: exact original SHA-256 verified privately', flush=True)
        report['models'].append(entry)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'passed'
    report['finished_at_unix'] = time.time()
    args.report.write_text(json.dumps(report, indent=2) + '\n')


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        # argv: file remote action project basis [output] [bytes sha256]
        action, project, basis = sys.argv[2:5]
        output = sys.argv[5] if len(sys.argv) > 5 else None
        remote(action, project, basis, output)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='aiasylum-gpu')
    parser.add_argument('--remote-dir', default='aiasylum')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--basis-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    transfer(parser.parse_args())


if __name__ == '__main__':
    main()
