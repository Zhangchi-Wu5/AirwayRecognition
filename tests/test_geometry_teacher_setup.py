"""Offline safety and repeatability tests for the public teacher setup helper."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/setup_geometry_teacher.sh'
SOURCE = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
NAMESPACE = {'__name__': 'setup_helpers'}
exec(compile(SOURCE, str(SCRIPT), 'exec'), NAMESPACE)


def write_zip(path, entries):
    with zipfile.ZipFile(path, 'w') as out:
        for name, data in entries:
            out.writestr(name, data)


@pytest.mark.parametrize('bad', ['../escape.pth', '/absolute.pth', r'dir\escape.pth', 'C:/escape.pth'])
def test_archive_paths_rejected_before_checkpoint_copy(tmp_path, bad):
    archive, target = tmp_path / 'weights.zip', tmp_path / 'depth_model.pth'
    write_zip(archive, [('weights/depth_model.pth', b'checkpoint'), (bad, b'bad')])
    with pytest.raises(ValueError, match='Unsafe'):
        NAMESPACE['safe_depth_checkpoint'](archive, target)
    assert not target.exists()


def test_symlink_and_multiple_checkpoints_rejected(tmp_path):
    archive, target = tmp_path / 'weights.zip', tmp_path / 'depth_model.pth'
    link = zipfile.ZipInfo('link')
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    write_zip(archive, [('depth_model.pth', b'checkpoint'), (link, b'../outside')])
    with pytest.raises(ValueError, match='Unsafe'):
        NAMESPACE['safe_depth_checkpoint'](archive, target)
    write_zip(archive, [('one/depth_model.pth', b'one'), ('two/depth_model.pth', b'two')])
    with pytest.raises(ValueError, match='exactly one'):
        NAMESPACE['safe_depth_checkpoint'](archive, target)


def test_extraction_is_idempotent_and_preserves_existing_file(tmp_path):
    archive, target = tmp_path / 'weights.zip', tmp_path / 'depth_model.pth'
    write_zip(archive, [('official/depth_model.pth', b'checkpoint')])
    extract = NAMESPACE['safe_depth_checkpoint']
    extract(archive, target)
    extract(archive, target)
    assert target.read_bytes() == b'checkpoint'
    target.write_bytes(b'user file')
    with pytest.raises(ValueError, match='differs'):
        extract(archive, target)
    assert target.read_bytes() == b'user file'
    assert not list(tmp_path.glob('.depth-*'))


def test_full_offline_setup_rejects_later_asset_tampering(tmp_path):
    repo, weights = tmp_path / 'repo', tmp_path / 'weights'
    package = repo / 'models/endodac'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'endodac.py').write_text('# offline import fixture\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'models'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    weights.mkdir()
    write_zip(weights / 'EndoDAC_fullmodel.zip', [('full/depth_model.pth', b'depth')])
    backbone = weights / 'depth_anything_vitb14.pth'
    backbone.write_bytes(b'backbone')
    command = ['bash', str(SCRIPT), '--teacher-root', str(repo), '--weights-root', str(weights),
               '--teacher-revision', revision, '--skip-download', '--with-backbone']
    environment = dict(os.environ, PYTHON_BIN=sys.executable)
    first = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    manifest_path = weights / 'setup_manifest.json'
    original = manifest_path.read_bytes()
    assert json.loads(original)['revision'] == revision
    second = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert manifest_path.read_bytes() == original
    backbone.write_bytes(b'changed')
    third = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert third.returncode != 0
    assert 'changed or disappeared' in third.stderr
    assert manifest_path.read_bytes() == original
