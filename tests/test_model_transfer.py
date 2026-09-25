"""Private delta correction must never publish content without its final hash."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('repair_model_transfer', Path(__file__).parents[1] / 'scripts' / 'repair_model_transfer.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_repairs_boundaries_tail_and_preserves_basis(tmp_path):
    original = tmp_path / 'original'
    basis = tmp_path / 'basis'
    output = tmp_path / 'corrected'
    data = bytes(range(256)) * 5000 + b'tail!'
    changed = bytearray(data)
    for index in (0, 255, 256, 4095, 4096, 65535, 65536, 1048575, 1048576, len(data) - 1):
        changed[index] ^= 1
    original.write_bytes(data)
    basis.write_bytes(changed)
    ranges = [[0, len(data)]]
    for size in repair.BLOCK_SIZES:
        ranges = repair.changed_blocks(original, ranges, size, repair.block_hashes(basis, ranges, size, len(data)))
    result = repair.apply_verified_patch(basis, output, repair.make_patch(original, ranges), len(data), hashlib.sha256(data).hexdigest())
    assert result['status'] == 'verified_private_basis'
    assert output.read_bytes() == data
    assert basis.read_bytes() == changed


def test_wrong_basis_length_is_rejected(tmp_path):
    path = tmp_path / 'basis'
    path.write_bytes(b'abc')
    with pytest.raises(ValueError, match='length'):
        repair.block_hashes(path, [[0, 4]], 256, 4)


def test_final_hash_failure_does_not_publish(tmp_path):
    basis = tmp_path / 'basis'
    original = tmp_path / 'original'
    output = tmp_path / 'corrected'
    basis.write_bytes(b'wrong')
    original.write_bytes(b'right')
    incomplete_patch = repair.make_patch(original, [])
    with pytest.raises(ValueError, match='SHA-256'):
        repair.apply_verified_patch(basis, output, incomplete_patch, 5, hashlib.sha256(b'right').hexdigest())
    assert not output.exists()
    assert basis.read_bytes() == b'wrong'
    assert not list(tmp_path.glob('.repair-*'))


def test_rejects_bad_ranges_and_incomplete_hash_response(tmp_path):
    original = tmp_path / 'original'
    original.write_bytes(b'abcdefgh')
    with pytest.raises(ValueError, match='overlapping'):
        repair.block_hashes(original, [[0, 4], [3, 2]], 2, 8)
    with pytest.raises(ValueError, match='Incomplete'):
        repair.changed_blocks(original, [[0, 8]], 2, b'')


def test_refuses_different_existing_output(tmp_path):
    original = tmp_path / 'original'
    output = tmp_path / 'existing'
    original.write_bytes(b'right')
    output.write_bytes(b'older')
    with pytest.raises(ValueError, match='refusing overwrite'):
        repair.apply_verified_patch(original, output, repair.make_patch(original, []), 5, hashlib.sha256(b'right').hexdigest())
    assert output.read_bytes() == b'older'


def test_disk_budget_reuses_only_exact_staged_files(tmp_path):
    script = (Path(__file__).parents[1] / 'scripts' / 'remote_models.sh').read_text()
    helper_source = script.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    scope = {'__name__': 'transfer_helper_test'}
    exec(compile(helper_source, 'checkpoint_transfer.py', 'exec'), scope)
    target = tmp_path / 'models'
    audit = tmp_path / 'audit'
    stage = audit / 'staging' / 'fingerprint' / 'edited'
    stage.mkdir(parents=True)
    exact = b'exact'
    (stage / 'first').write_bytes(exact)
    (stage / 'second').write_bytes(b'wrong')
    partial = stage / '.rsync-partial'
    partial.mkdir()
    (partial / 'third').write_bytes(b'ex')
    records = [{'path': name, 'bytes': len(exact), 'sha256': hashlib.sha256(exact).hexdigest()} for name in ('first', 'second', 'third')]
    manifest = {'checkpoints': [{'name': 'edited', 'sha256': 'fingerprint', 'bytes': 15, 'files': records}]}
    assert scope['required_new_bytes'](manifest, target, audit) == (10, 5)
