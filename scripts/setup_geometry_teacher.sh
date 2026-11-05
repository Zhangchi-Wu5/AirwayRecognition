#!/usr/bin/env bash
# Download public teacher assets without replacing the existing Torch stack.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON_BIN:-python}" - "$@" <<'PY'
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

REPOSITORY = 'https://github.com/BeileiCui/EndoDAC.git'
# Verified with the public repository's git ls-remote on 2026-11-05.
REVISION = '22ec911b270d056fdd7f430474e4be407cce8287'
FULL_ID = '1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70'
BACKBONE_ID = '163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(*args, capture=False):
    return subprocess.run(args, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def safe_depth_checkpoint(archive, target):
    """Validate every archive name, then copy only its unique depth checkpoint."""
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if len(infos) > 10000 or sum(i.file_size for i in infos) > 8 * 1024**3:
            raise ValueError('Teacher ZIP exceeds the supported extraction limits')
        candidates = []
        names = set()
        for info in infos:
            name = info.filename
            p = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (not name or p.is_absolute() or '..' in p.parts or '\\' in name
                    or ':' in name or stat.S_ISLNK(mode) or name in names):
                raise ValueError(f'Unsafe or duplicate teacher ZIP member: {name!r}')
            names.add(name)
            if p.name == 'depth_model.pth' and not info.is_dir():
                candidates.append(info)
        if len(candidates) != 1:
            raise ValueError('Expected exactly one depth_model.pth in the official ZIP; '
                             f'found {len(candidates)}. Do not choose an epoch automatically.')
        fd, temp_name = tempfile.mkstemp(prefix='.depth-', dir=target.parent)
        os.close(fd)
        temporary = Path(temp_name)
        try:
            with source.open(candidates[0]) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if not temporary.stat().st_size:
                raise ValueError('Empty depth checkpoint')
            if target.exists():
                if target.is_symlink() or sha256(target) != sha256(temporary):
                    raise ValueError('Existing depth_model.pth differs from archive; use a new weights directory')
            else:
                temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def download(file_id, destination, skip):
    if destination.is_symlink():
        raise ValueError(f'Refusing symlink asset: {destination}')
    if destination.is_file() and destination.stat().st_size:
        return
    url = f'https://drive.google.com/file/d/{file_id}/view'
    if skip:
        raise ValueError(f'Missing asset: {destination}\nDownload manually from {url}')
    if importlib.util.find_spec('gdown') is None:
        raise ValueError('gdown is missing. Re-run with --install-deps, or download '
                         f'{url} manually to {destination} and use --skip-download.')
    import gdown
    partial = destination.with_name(destination.name + '.part')
    try:
        result = gdown.download(id=file_id, output=str(partial), quiet=False)
        if result is None or not partial.is_file() or not partial.stat().st_size:
            raise ValueError('The download did not produce a non-empty file')
        with partial.open('rb') as handle:
            if b'<html' in handle.read(512).lower():
                raise ValueError('Download returned an HTML page rather than model data')
        partial.replace(destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise ValueError(f'Download failed: {exc}\nDownload {url} manually to {destination}, '
                         'then re-run with --skip-download.') from exc


def main():
    parser = argparse.ArgumentParser(description='Prepare pinned EndoDAC code and official model files; no training.')
    parser.add_argument('--teacher-root', default='external/EndoDAC')
    parser.add_argument('--weights-root', default='pretrained/EndoDAC')
    parser.add_argument('--teacher-revision', default=REVISION,
                        help='Explicit 40-character revision; defaults to verified source revision')
    parser.add_argument('--skip-download', action='store_true',
                        help='Use locally supplied official assets; existing checkout required')
    parser.add_argument('--with-backbone', action='store_true',
                        help='Also archive the optional upstream initialization weights (not needed by our strict full-checkpoint loader)')
    parser.add_argument('--install-deps', action='store_true',
                        help='Install only fvcore and gdown with their small dependencies, never Torch/xformers')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.teacher_revision):
        raise ValueError('--teacher-revision must be a full lowercase Git commit SHA')
    repo = Path(args.teacher_root).expanduser().resolve()
    weights = Path(args.weights_root).expanduser().resolve()
    if args.install_deps:
        run(sys.executable, '-m', 'pip', 'install', 'fvcore==0.1.5.post20221221', 'gdown==5.2.0')
    if repo.exists():
        actual = run('git', '-C', str(repo), 'rev-parse', 'HEAD', capture=True).strip()
        dirty = run('git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no', capture=True).strip()
        if actual != args.teacher_revision or dirty:
            raise ValueError('Existing teacher checkout has a different revision or tracked changes; '
                             'use a new --teacher-root. It will not be reset automatically.')
    else:
        if args.skip_download:
            raise ValueError(f'Missing teacher checkout {repo}; run without --skip-download first')
        repo.parent.mkdir(parents=True, exist_ok=True)
        run('git', 'clone', '--no-checkout', REPOSITORY, str(repo))
        run('git', '-C', str(repo), 'checkout', '--detach', args.teacher_revision)
        actual = run('git', '-C', str(repo), 'rev-parse', 'HEAD', capture=True).strip()
        if actual != args.teacher_revision:
            raise ValueError('Teacher revision verification failed')
    if not (repo / 'models/endodac/endodac.py').is_file():
        raise ValueError('Teacher checkout lacks models/endodac/endodac.py')
    weights.mkdir(parents=True, exist_ok=True)
    archive = weights / 'EndoDAC_fullmodel.zip'
    backbone = weights / 'depth_anything_vitb14.pth' if args.with_backbone else None
    depth = weights / 'depth_model.pth'
    manifest_path = weights / 'setup_manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:
        for name, expected in old['asset_sha256'].items():
            path = weights / name
            if not path.is_file() or path.is_symlink() or sha256(path) != expected:
                raise ValueError(f'Existing pinned teacher asset changed or disappeared: {name}')
    download(FULL_ID, archive, args.skip_download)
    if backbone is not None:
        download(BACKBONE_ID, backbone, args.skip_download)
    safe_depth_checkpoint(archive, depth)
    manifest = {
        'version': 'endodac_setup_v1', 'repository': REPOSITORY,
        'revision': actual, 'teacher_repo': str(repo), 'teacher_weights': str(depth),
        'backbone_weights': str(backbone) if backbone is not None else None,
        'official_file_ids': {'full_model': FULL_ID, **({'backbone': BACKBONE_ID} if backbone else {})},
        'asset_sha256': {p.name: sha256(p) for p in (archive, depth) + ((backbone,) if backbone else ())},
        'source_sha256': {str(p.relative_to(repo)): sha256(p) for p in sorted((repo / 'models').rglob('*.py'))},
        'checksum_scope': 'Locally recorded downloaded-file hashes, not author-published checksums',
    }
    if old is not None and old != manifest:
        raise ValueError('Setup identity differs from saved manifest; use a new weights directory')
    if old is None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    # Only paths controlled by the CLI are written; shell quoting handles spaces safely.
    env_path = weights / 'teacher_paths.env'
    env_path.write_text('export XFORMERS_DISABLED=1\n' + ''.join(
        f'export {key}={shlex.quote(value)}\n' for key, value in {
            'GEOMETRY_TEACHER_REPO': str(repo), 'GEOMETRY_TEACHER_WEIGHTS': str(depth),
            'GEOMETRY_BACKBONE_WEIGHTS': str(backbone) if backbone else ''}.items()))
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(repo))
    try:
        import torch
        import models.endodac
    except ImportError as exc:
        raise ValueError(f'Files prepared but teacher import failed: {exc}. '
                         'Use the existing airway environment and --install-deps; '
                         'do not install EndoDAC requirements.txt because it pins an older Torch stack.') from exc
    print(f'Prepared revision: {actual}\nManifest: {manifest_path}\n'
          'Source import verified; checkpoint loading and GPU inference are verified by the cache QA step.\n'
          f'Next: source {shlex.quote(str(env_path))}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
PY
