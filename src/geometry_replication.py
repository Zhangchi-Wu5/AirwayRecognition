"""Independent training-seed replication of the frozen 45-fit geometry pilot.

Only the 40 few-shot fits are repeated.  The historical source files are reused
without mutation; the reference evidence is copied as weight-free JSON/CSV so
that an exported result remains independently auditable offline.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import tarfile
import time

from src import geometry_experiment as base
from src.baseline_reproduction import output_lock, read_json, read_csv, require
from src.fewshot import (PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES,
    classification_metrics, sha256_file, state_fingerprint, write_csv)
from src.fewshot_finetune import (_atomic_json, _digest, seed_training,
    save_reload_final, evaluate, ImageRows)
from src.renew_data import audit_dataset, EXPECTED_FINGERPRINT, get_transforms
from src.renew_experiment import portable, configure_torch, environment, runtime_key

VERSION = 'renew_r18_geometry_seed43_v1'
METHODS = base.METHODS
SOURCE_FILES = (*base.SOURCE_FILES, 'src/geometry_replication.py',
    'scripts/run_geometry_replication.py', 'scripts/run_geometry_replication_server.sh')
RUN_FILES = base.RUN_FILES
REFERENCE_DIR = 'reference_seed42'
now = base.now
train_updates = base.train_updates
export_student = base.export_student
cache_binding = base.cache_binding
check_dependencies = base.check_dependencies


def fixed_config(smoke=False):
    config = base.fixed_config(smoke)
    config.update(training_seed=43, shuffled_target_seed=42)
    return config


def make_plan(rows):
    folds, original = base.make_plan(rows)
    jobs = [dict(job, run_id=job['run_id'] + '_seed43')
            for job in original if job['budget'] in ('5', '10')]
    require(len(jobs) == 40, 'Expected exactly 40 replication jobs')
    return folds, jobs


def guard_roots(root, image_root=None, cache_root=None, reference_root=None):
    """Resolve aliases before even creating the output lock or its directory."""
    root = Path(root).expanduser().resolve()
    for label, value in (('image', image_root), ('cache', cache_root),
                         ('reference', reference_root)):
        if value is not None:
            other = Path(value).expanduser().resolve()
            require(not root.is_relative_to(other) and not other.is_relative_to(root),
                    f'Output and {label} roots must be disjoint')
    return root


def safe_file(root, name):
    root, name = Path(root), Path(name)
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe artifact path')
    path = root / name
    require(path.is_file(), 'Missing artifact: ' + str(name))
    require(all(not (root / Path(*name.parts[:i])).is_symlink()
                for i in range(1, len(name.parts) + 1)), 'Unsafe symlink artifact: ' + str(name))
    require(path.resolve().is_relative_to(root.resolve()), 'Artifact escapes root')
    return path


def check_output_target(root):
    """Never create even a lock file in another experiment's result tree."""
    root = Path(root)
    require(not (root / '.reproduction.lock').is_symlink(), 'Unsafe output lock symlink')
    if (root / 'suite.json').exists() or (root / 'suite.json').is_symlink():
        suite = read_json(safe_file(root, 'suite.json'))
        require(suite.get('identity', {}).get('version') == VERSION,
                'Wrong geometry replication suite version; refusing to modify output')
    elif root.exists():
        require(root.is_dir() and not any(p.name != '.reproduction.lock' for p in root.iterdir()),
                'Refusing nonempty output without suite.json')


def verify_binding(binding, rows):
    """Validate saved provenance without needing the teacher maps on disk."""
    from src.geometry_distillation import VERSION as CACHE_VERSION, VARIANTS, digest
    require(isinstance(binding, dict) and set(binding) == {
        'identity_sha256', 'version', 'teacher_sha256', 'variants', 'entries'}, 'Malformed cache binding')
    require(binding['version'] == CACHE_VERSION and binding['variants'] == [list(v) for v in VARIANTS],
            'Cache protocol differs')
    require(re.fullmatch('[0-9a-f]{64}', str(binding['teacher_sha256'])) is not None,
            'Malformed teacher checksum')
    require(binding['identity_sha256'] == digest({k: v for k, v in binding.items() if k != 'identity_sha256'}),
            'Cache binding digest differs')
    require(set(binding['entries']) == {r['image_sha256'] for r in rows}, 'Cache support differs')
    for row in rows:
        key, entry = row['image_sha256'], binding['entries'][row['image_sha256']]
        require(isinstance(entry, dict) and entry.get('image_name') == row['image_name']
                and entry.get('file') == key + '.npz', 'Cache image identity differs')
        require(re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256'))) is not None,
                'Malformed cache entry checksum')
        shape, size = entry.get('shape'), entry.get('original_size')
        require(isinstance(shape, list) and len(shape) == 3 and shape[0] == len(VARIANTS)
                and all(type(v) is int and v >= 14 for v in shape[1:]), 'Malformed cache shape')
        require(isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size),
                'Malformed cache original size')


def verify_reference(root, expected_source=None):
    """Read-only audit of all 45 historical fits; checkpoints are not needed."""
    root = Path(root).expanduser().resolve()
    suite = read_json(safe_file(root, 'suite.json'))
    base.verify_identity(suite['identity'])
    require(suite['identity']['config']['mode'] == 'formal', 'Reference must be formal')
    source = expected_source if expected_source is not None else {
        name: sha256_file(PROJECT_ROOT / name) for name in base.SOURCE_FILES}
    require(suite['identity']['source_sha256'] == source, 'Reference executed source differs from current frozen source')
    runtime = suite.get('runtime_identity')
    require(isinstance(runtime, dict) and set(runtime) == {'python', 'packages', 'cuda', 'cudnn', 'gpu'}
            and all(runtime[k] is not None for k in runtime) and isinstance(runtime['packages'], dict),
            'Missing formal reference runtime evidence')
    names = ['suite.json', *base.artifacts(suite['identity'])]
    for job in suite['identity']['jobs']:
        names.extend(f"runs/{job['run_id']}/{name}" for name in RUN_FILES)
    for name in names:
        safe_file(root, name)
    base.verify_artifacts(root, suite)
    data = base.aggregate(root, suite)
    require(len(data['results']) == 45 and len(data['oof']) == 9,
            'Reference must contain all 45 completed runs and nine OOF groups')
    by_name = {r['image_name']: r for r in suite['identity']['data']}
    by_id = {r['run_id']: r for r in data['results']}
    teachers = set()
    for job in suite['identity']['jobs']:
        if job['method'] != 'ce':
            binding = by_id[job['run_id']]['cache_binding']
            verify_binding(binding, [by_name[n] for n in job['support']])
            teachers.add(binding['teacher_sha256'])
    require(len(teachers) == 1 and suite.get('teacher_sha256') in (None, next(iter(teachers))),
            'Reference mixes frozen teachers')
    # Identical supports must bind identical map files for every geometry method.
    for fold in range(5):
        for budget in ('5', '10'):
            bindings = [by_id[j['run_id']]['cache_binding'] for j in suite['identity']['jobs']
                        if j['fold'] == fold and j['budget'] == budget and j['method'] != 'ce']
            require(all(b == bindings[0] for b in bindings), 'Reference support cache bindings differ')
    return {'identity_sha256': suite['identity_sha256'], 'runtime_identity': runtime,
            'teacher_sha256': next(iter(teachers)),
            'artifact_sha256': {n: sha256_file(root / n) for n in sorted(names)}}


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong geometry replication suite version')
    config = identity.get('config', {})
    require(config == fixed_config(config.get('mode') == 'smoke'), 'Changed fixed replication protocol')
    require(set(identity.get('source_sha256', {})) == set(SOURCE_FILES), 'Source manifest incomplete')
    rows = identity['data']
    require(len(rows) == 93 and len({r['patient_key'] for r in rows}) == 31, 'Wrong cohort')
    require([r['image_name'] for r in rows] == sorted(r['image_name'] for r in rows), 'Data order changed')
    for index, row in enumerate(rows):
        match = re.fullmatch(r'([0-9]+)\s*(lt|yz|zz)\.png', row['image_name'], flags=re.IGNORECASE)
        require(match is not None and row['source_index'] == index, 'Invalid source image identity')
        require(row['label'] == match.group(2).lower() and row['label_id'] == LABEL_TO_ID[row['label']], 'Image label identity changed')
        require(row['patient_id'] == match.group(1) and row['patient_key'] == str(int(match.group(1))), 'Patient identity changed')
    payload = ''.join(r['image_name'] + '\t' + r['image_sha256'] + '\n' for r in rows)
    require(hashlib.sha256(payload.encode()).hexdigest() == EXPECTED_FINGERPRINT, 'Wrong source fingerprint')
    folds, jobs = make_plan(rows)
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Changed support/fold plan')
    require(config['mode'] == 'smoke' or isinstance(identity.get('reference'), dict),
            'Formal suite requires frozen seed42 reference')


def verify_artifacts(root, suite):
    root = Path(root)
    for name in base.artifacts(suite['identity']):
        safe_file(root, name)
    base.verify_artifacts(root, suite)
    reference = suite['identity'].get('reference')
    if reference is not None:
        expected = {n: suite['identity']['source_sha256'][n] for n in base.SOURCE_FILES}
        require(verify_reference(root / REFERENCE_DIR, expected) == reference,
                'Frozen reference snapshot changed')
        old = read_json(root / REFERENCE_DIR / 'suite.json')['identity']
        require(old['data'] == suite['identity']['data'] and old['folds'] == suite['identity']['folds'],
                'Reference cohort/folds differ')


def prepare(root, image_root, reference_root=None, *, cache_root=None, smoke=False):
    root = guard_roots(root, image_root, cache_root, reference_root)
    check_output_target(root)
    try:
        rows, metadata = audit_dataset(Path(image_root).expanduser().resolve())
    except ValueError as error:
        if 'No first-level PNG' in str(error):
            raise ValueError(str(error) + '; set --image-root to the directory directly containing the audited 93 PNGs (for example dataset/renew/气管镜)') from error
        raise
    reference = None
    if not smoke or reference_root is not None:
        require(reference_root is not None, 'Formal suite requires --reference-root')
        reference = verify_reference(reference_root)
        old = read_json(Path(reference_root) / 'suite.json')['identity']
        require(old['data'] == portable(rows), 'Reference cohort differs')
    folds, jobs = make_plan(portable(rows))
    identity = {'version': VERSION, 'config': fixed_config(smoke), 'data': portable(rows),
                'folds': folds, 'jobs': jobs, 'reference': reference,
                'source_sha256': {n: sha256_file(PROJECT_ROOT / n) for n in SOURCE_FILES}}
    verify_identity(identity)
    if (root / 'suite.json').exists():
        suite = read_json(safe_file(root, 'suite.json'))
        require(suite['identity'] == identity, 'Resume rejected: source/config/data/reference changed; use a new output root')
        verify_artifacts(root, suite)
        return rows, suite
    require(not root.exists() or not any(p.name != '.reproduction.lock' for p in root.iterdir()),
            'Refusing nonempty output without suite.json')
    root.mkdir(parents=True, exist_ok=True)
    if reference is not None:
        for name, digest in reference['artifact_sha256'].items():
            target = root / REFERENCE_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(Path(reference_root), name), target)
            require(sha256_file(target) == digest, 'Reference changed while copying')
    for name, values in base.artifacts(identity).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        write_csv(root / name, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
             'runtime_identity': None, 'status': 'prepared_not_run', 'completed_runs': 0,
             'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
             'protected_roots': {k: str(Path(v).expanduser().resolve()) for k, v in
                 (('image', image_root), ('cache', cache_root), ('reference', reference_root)) if v is not None},
             'plan_artifact_sha256': {n: sha256_file(root / n) for n in base.artifacts(identity)}}
    _atomic_json(root / 'suite.json', suite)
    return rows, suite


def selected_jobs(suite, args):
    jobs = [j for j in suite['identity']['jobs']
            if (not args.methods or j['method'] in args.methods)
            and (not args.budgets or j['budget'] in args.budgets)
            and (args.folds is None or j['fold'] in args.folds)]
    require(bool(jobs), 'Run filters select no jobs')
    return jobs


def reference_result(root, suite, job):
    if suite['identity'].get('reference') is None:
        return None
    old_id = job['run_id'].removesuffix('_seed43')
    name = f'runs/{old_id}/result.json'
    expected = suite['identity']['reference']['artifact_sha256'][name]
    path = safe_file(Path(root) / REFERENCE_DIR, name)
    require(sha256_file(path) == expected, 'Frozen reference result changed')
    return read_json(path)


def reference_binding(root, suite, job):
    old = reference_result(root, suite, job)
    return None if old is None else old['cache_binding']


def preflight_cache(root, suite, jobs, by_name, cache_root):
    for job in jobs:
        if job['method'] != 'ce':
            support = [by_name[n] for n in job['support']]
            binding = cache_binding(cache_root, support)
            verify_binding(binding, support)
            expected = reference_binding(root, suite, job)
            require(expected is None or binding == expected, 'Teacher/cache differs from seed42 reference support')


def validate_run(root, suite, job, require_model=True):
    directory = Path(root) / 'runs' / job['run_id']
    if (directory / 'complete.json').exists():
        for name in (*RUN_FILES, *(('model.pt', 'student.pt') if require_model else ())):
            safe_file(Path(root), f"runs/{job['run_id']}/{name}")
    result = base.validate_run(Path(root), suite, job, require_model=require_model)
    if result is not None and job['method'] != 'ce':
        by_name = {r['image_name']: r for r in suite['identity']['data']}
        verify_binding(result['cache_binding'], [by_name[n] for n in job['support']])
        expected = reference_binding(root, suite, job)
        require(expected is None or result['cache_binding'] == expected, 'Run cache differs from seed42 reference')
    if result is not None and suite['identity'].get('reference') is not None:
        require(suite['runtime_identity'] == suite['identity']['reference']['runtime_identity'],
                'Replication runtime differs from seed42 reference')
        require(result['initial_state_sha256'] != reference_result(root, suite, job)['initial_state_sha256'],
                'Replication initialization must differ from seed42 reference')
    return result


def train_one(root, suite, job, by_name, device, cache_root=None, model_factory=None):
    """Base training lifecycle with an independently frozen shuffled-target seed."""
    from src.geometry_distillation import GeometryImageDataset, build_resnet18_student
    root = Path(root)
    c = suite['identity']['config']
    support = [by_name[n] for n in job['support']]
    preflight_cache(root, suite, [job], by_name, cache_root)
    binding = cache_binding(cache_root, support) if job['method'] != 'ce' else None
    if binding is not None:
        require(suite.get('teacher_sha256') in (None, binding['teacher_sha256']), 'Frozen teacher changed across jobs')
        suite['teacher_sha256'] = binding['teacher_sha256']
        _atomic_json(root / 'suite.json', suite)
    completed = validate_run(root, suite, job)
    if completed:
        require(completed['cache_binding'] == binding, 'Resume teacher/cache changed')
        return completed
    directory = root / 'runs' / job['run_id']
    require(not directory.is_symlink() and not directory.parent.is_symlink(), 'Unsafe run directory')
    if directory.exists():
        archived = root / 'incomplete_attempts'
        require(not archived.is_symlink(), 'Unsafe incomplete archive directory')
        archived.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archived / (job['run_id'] + '_' + str(time.time_ns()))))
    directory.mkdir(parents=True)
    factory = model_factory or (lambda pretrained: build_resnet18_student(pretrained=pretrained, dropout=c['dropout']))
    seed_training(c['training_seed'])
    model = factory(c['mode'] == 'formal').to(device)
    initial = state_fingerprint(model)
    _atomic_json(directory / 'config.json', {'job': job, 'config': c, 'identity_sha256': suite['identity_sha256'],
        'initial_state_sha256': initial, 'cache_binding': binding, 'execution_runtime': suite['runtime_identity']})
    # RGB randomness is controlled by seed_training(43). This dataset argument
    # affects only sample_seed = image_hash XOR 42 for the shuffled control.
    dataset = GeometryImageDataset(support, cache_root=cache_root if binding is not None else None,
                                   train=True, seed=c['shuffled_target_seed'])
    hist, evidence = train_updates(model, dataset, c, device, job['method'])
    write_csv(directory / 'history.csv', hist)
    reloaded, checkpoint = save_reload_final(model, directory / 'model.pt', lambda: factory(False), device)
    exported = export_student(reloaded, directory / 'student.pt', factory, device)
    del model
    eval_rows = [by_name[n] for n in job['eval']]
    probs = evaluate(reloaded, ImageRows(eval_rows, get_transforms('full_frame')), c, device)
    predictions = [{'patient_id': r['patient_id'], 'image_name': r['image_name'], 'label_id': r['label_id'],
                    'predicted_id': int(p.argmax()), **{f'prob_{label}': float(p[i]) for label, i in LABEL_TO_ID.items()}}
                   for r, p in zip(eval_rows, probs)]
    write_csv(directory / 'predictions.csv', predictions)
    result = {**{k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method')},
              'n_train_images': len(support), 'n_train_patients': len(job['patient_order_prefix']),
              'initial_state_sha256': initial, 'cache_binding': binding,
              **classification_metrics([r['label_id'] for r in eval_rows], probs), **evidence, **checkpoint, **exported}
    _atomic_json(directory / 'result.json', result)
    _atomic_json(directory / 'complete.json', {'identity_sha256': suite['identity_sha256'], 'run_id': job['run_id'],
        'artifact_sha256': {n: sha256_file(directory / n) for n in (*RUN_FILES[:-1], 'model.pt')}})
    validate_run(root, suite, job)
    return result


def aggregate(root, suite):
    for job in suite['identity']['jobs']:
        validate_run(root, suite, job, require_model=False)
    return base.aggregate(Path(root), suite)


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root, suite)
    outputs = {
        'per_run_metrics.csv': ([{k: r[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method',
            'n_train_images', 'n_train_patients', *METRIC_NAMES)} for r in data['results']],
            ['run_id', 'fold', 'budget', 'repeat', 'method', *METRIC_NAMES]),
        'oof_predictions.csv': (data['predictions'], ['run_id', 'fold', 'budget', 'method', 'repeat',
            'patient_id', 'image_name', 'label_id', 'predicted_id', *['prob_' + l for l in LABEL_TO_ID]]),
        'oof_metrics.csv': (data['oof'], ['method', 'budget', 'repeat', *METRIC_NAMES]),
        'oof_comparison.csv': (data['paired'], ['method', 'budget', *[k + '_delta_vs_ce' for k in METRIC_NAMES]])}
    for name, (rows, fields) in outputs.items():
        require(not (root / name).is_symlink(), 'Unsafe summary path')
        write_csv(root / name, rows, fieldnames=None if rows else fields)
    lines = ['# ResNet-18 几何训练 seed43 独立复跑', '',
             f"已完成 {len(data['results'])}/40 次；只有完整五折才产生 OOF，最多 8 组。", '',
             '固定原支持集 repeat 0；训练 seed43；打乱目标 seed42；辅助头 seed20261105。',
             'seed42/43 仅提供训练随机性复核，两个种子不能证明稳健性，也不能估计支持抽样波动。',
             '不是外部验证；稳定视差不是深度真值；单图推理只使用 RGB 学生。', '',
             '| 方法 | 每类预算 | OOF准确率 | 宏F1 |', '|---|---|---:|---:|']
    for r in data['oof']:
        lines.append(f"| {r['method']} | {r['budget']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} |")
    for r in data['results']:
        if r['method'] == 'shuffled':
            hist = read_csv(root / 'runs' / r['run_id'] / 'history.csv')
            if sum(float(h.get('shuffled_changed_pairs', 0)) for h in hist) == 0:
                lines.append('警告：' + r['run_id'] + ' 的打乱目标未改变任何关系，此负对照无辨别力。')
    require(not (root / 'REPORT.md').is_symlink(), 'Unsafe report path')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    suite.update(completed_runs=len(data['results']), status='completed' if len(data['results']) == 40 else 'partial',
                 aggregate_sha256={n: sha256_file(root / n) for n in (*outputs, 'REPORT.md')})
    _atomic_json(root / 'suite.json', suite)
    return data


def load_suite(root):
    """Validate before acquiring a lock, so an old root is never modified."""
    root = Path(root)
    check_output_target(root)
    suite = read_json(safe_file(root, 'suite.json'))
    verify_identity(suite['identity'])
    protected = suite.get('protected_roots', {})
    guard_roots(root, protected.get('image'), protected.get('cache'), protected.get('reference'))
    verify_artifacts(root, suite)
    return suite


def run_suite(args):
    guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    check_output_target(args.output_root)
    check_dependencies()
    configure_torch()
    import torch
    runtime = environment(require_cuda=not args.smoke)
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(args.output_root):
        rows, suite = prepare(args.output_root, args.image_root,
            None if args.smoke else args.reference_root, cache_root=args.cache_root, smoke=args.smoke)
        key = runtime_key(runtime)
        require(suite['runtime_identity'] in (None, key), 'Runtime changed; use a new output root')
        if not args.smoke:
            require(key == suite['identity']['reference']['runtime_identity'], 'Runtime differs from seed42 reference')
        suite['runtime_identity'] = key
        jobs, by_name = selected_jobs(suite, args), {r['image_name']: r for r in rows}
        preflight_cache(args.output_root, suite, jobs, by_name, args.cache_root)
        suite.setdefault('attempts', []).append({'at_utc': now(), 'methods': args.methods,
                                                'budgets': args.budgets, 'folds': args.folds})
        _atomic_json(args.output_root / 'suite.json', suite)
        try:
            for i, job in enumerate(jobs, 1):
                print(f"[{i}/{len(jobs)}] {job['run_id']}", flush=True)
                train_one(args.output_root, suite, job, by_name, device, args.cache_root)
                write_aggregates(args.output_root, suite)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as error:
            suite['last_failure'] = {'at_utc': now(), 'message': str(error)}
            _atomic_json(args.output_root / 'suite.json', suite)
            raise


def pack_suite(root, args):
    root = guard_roots(root, args.image_root, args.cache_root, args.reference_root)
    load_suite(root)
    with output_lock(root):
        suite = load_suite(root)
        require(suite['identity']['config']['mode'] == 'formal', 'Smoke results cannot be packaged as formal results')
        jobs = selected_jobs(suite, args)
        require(all(validate_run(root, suite, j, require_model=False) for j in jobs), 'Selected packing scope is incomplete')
        write_aggregates(root, suite)
        manifest = {'version': VERSION, 'created_utc': now(), 'selected_run_ids': [j['run_id'] for j in jobs],
                    'planned_runs': 40, 'selected_completed_runs': len(jobs), 'scope_complete': True,
                    'entire_suite_complete': suite['completed_runs'] == 40,
                    'reference_identity_sha256': suite['identity']['reference']['identity_sha256'],
                    'aggregate_scope': 'all completed jobs in suite; selected runs listed explicitly',
                    'excluded': ['images', 'weights', 'teacher maps', 'teacher code']}
        _atomic_json(root / 'package_scope.json', manifest)
        archive = root.parent / f'{root.name}_seed43_{time.time_ns()}_results.tar.gz'
        names = ['suite.json', 'package_scope.json', *suite['plan_artifact_sha256'], *suite['aggregate_sha256']]
        names.extend(f'{REFERENCE_DIR}/{n}' for n in suite['identity']['reference']['artifact_sha256'])
        for j in suite['identity']['jobs']:
            if validate_run(root, suite, j, require_model=False):
                names.extend(f"runs/{j['run_id']}/{n}" for n in RUN_FILES)
        with tarfile.open(archive, 'x:gz') as out:
            for name in sorted(set(names)):
                out.add(safe_file(root, name), arcname=name, recursive=False)
        Path(str(archive) + '.sha256').write_text(sha256_file(archive) + '  ' + archive.name + '\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='Independent seed43 geometry replication: 40 fits, frozen seed42 reference.')
    p.add_argument('mode', choices=['check', 'plan', 'run', 'status', 'pack'])
    p.add_argument('--image-root', type=Path, default=Path('dataset/renew'))
    p.add_argument('--cache-root', type=Path, default=Path('outputs/renew_geometry_teacher_cache'))
    p.add_argument('--reference-root', type=Path, default=Path('outputs/renew_geometry_fewshot'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/renew_geometry_seed43'))
    p.add_argument('--methods', nargs='+', choices=METHODS)
    p.add_argument('--budgets', nargs='+', choices=['5', '10'])
    p.add_argument('--folds', nargs='+', type=int, choices=range(5))
    p.add_argument('--smoke', action='store_true', help='CPU, random weights, 2 updates; never formal pack')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    check_output_target(args.output_root)
    if args.mode == 'run':
        run_suite(args)
    elif args.mode == 'pack':
        print(pack_suite(args.output_root, args))
    elif args.mode == 'status':
        load_suite(args.output_root)
        with output_lock(args.output_root):
            suite = load_suite(args.output_root)
            data = write_aggregates(args.output_root, suite)
            print(f"Completed {len(data['results'])}/40; OOF groups={len(data['oof'])}/8")
    else:
        if args.mode == 'check':
            check_dependencies()
        with output_lock(args.output_root):
            rows, suite = prepare(args.output_root, args.image_root,
                None if args.smoke else args.reference_root, cache_root=args.cache_root, smoke=args.smoke)
            jobs = selected_jobs(suite, args)
            if args.mode == 'check':
                preflight_cache(args.output_root, suite, jobs, {r['image_name']: r for r in rows}, args.cache_root)
            print(f'PASS: {len(rows)} images, 31 patient IDs, 40 planned fits, training seed43, shuffled target seed42')
            print(f"Selected {len(jobs)} fits; mode={suite['identity']['config']['mode']}; reference snapshot={REFERENCE_DIR}")
