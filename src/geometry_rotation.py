"""Complete within-support B rotation, preserving the original ten diagnostics.

The frozen numerical implementation is reused unchanged. Ten existing units
remain under their original identities in a read-only reference snapshot; only
the other 65 units prepare fresh models. This is not an accuracy experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import shutil
import statistics
import tarfile
import time

from src import geometry_diagnostic as diagnostic
from src import geometry_experiment as base
from src import geometry_replication as reference
from src.baseline_reproduction import output_lock, read_json, read_csv, require
from src.fewshot import PROJECT_ROOT, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest
from src.renew_data import audit_dataset
from src.renew_experiment import portable, configure_torch, environment, runtime_key

VERSION = 'renew_r18_geometry_rotation_v1'
SOURCE_FILES = ('src/geometry_rotation.py', 'scripts/run_geometry_rotation.py',
                'scripts/run_geometry_rotation_server.sh')
REFERENCE_DIR = 'reference_diagnostic'
RUN_FILES = diagnostic.RUN_FILES
AGGREGATES = ('per_unit_probes.csv', 'unit_comparisons.csv',
              'fold_budget_summary.csv', 'REPORT.md')
PENDING_FILE = 'aggregate_pending.json'
COMPARISONS = ('field_vs_ce', 'field_vs_legacy', 'field_vs_random_mean',
               'field_vs_shuffled_mean')
fixed_config = diagnostic.fixed_config
safe_file = diagnostic.safe_file
now = diagnostic.now


def source_hashes():
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def guard_roots(root, image_root=None, cache_root=None, reference_root=None):
    root = diagnostic.guard_roots(root, image_root, cache_root, reference_root)
    old = (PROJECT_ROOT / 'outputs/renew_geometry_diagnostic').resolve()
    require(not root.is_relative_to(old) and not old.is_relative_to(root),
            'Rotation output overlaps the original diagnostic')
    return root


def check_output_target(root):
    diagnostic._no_symlink(root)
    root = Path(root)
    if root.exists():
        require(root.is_dir() and not any(p.is_symlink() for p in root.rglob('*')),
                'Unsafe rotation output directory or artifact symlink')
        if (root / 'suite.json').exists():
            require(read_json(safe_file(root, 'suite.json')).get('identity', {}).get('version') == VERSION,
                    'Wrong rotation suite version; refusing to modify output')
        else:
            require(not any(p.name != '.reproduction.lock' for p in root.iterdir()),
                    'Refusing nonempty output without rotation suite.json')


def make_plan(rows):
    folds, originals = diagnostic.make_plan(rows)
    by_name = {r['image_name']: r for r in rows}
    jobs = []
    for old in originals:
        for index, patient in enumerate(old['patient_order_prefix']):
            if index == len(old['patient_order_prefix']) - 1:
                jobs.append(dict(old))
                continue
            a = [n for n in old['support'] if by_name[n]['patient_key'] != patient]
            b = [n for n in old['support'] if by_name[n]['patient_key'] == patient]
            require(len(b) == 3 and len(a) == 3 * (int(old['budget']) - 1), 'Wrong rotation A/B size')
            jobs.append({**old, 'run_id': old['run_id'].replace(
                'geometry_diagnostic', f'geometry_rotation_b{index:02d}'),
                'A': a, 'B': b, 'A_patients': [p for p in old['patient_order_prefix'] if p != patient],
                'B_patient': patient})
    require(len(jobs) == 75 and len({j['run_id'] for j in jobs}) == 75, 'Expected 75 unique rotations')
    return folds, jobs


def imported_ids(identity):
    return [j['run_id'] for j in diagnostic.make_plan(identity['data'])[1]]


def origin(identity, job):
    return 'imported' if job['run_id'] in identity['imported_run_ids'] else 'new'


def artifacts(identity):
    values = diagnostic.artifacts(identity)
    values['unit_plan.csv'] = [{**{k: j[k] for k in
        ('run_id', 'fold', 'budget', 'repeat', 'reference_run_id', 'B_patient')},
        'B_index': j['patient_order_prefix'].index(j['B_patient']),
        'origin': origin(identity, j)} for j in identity['jobs']]
    return values


def reference_names(suite):
    require(suite['identity'].get('reference') is not None,
            'Original diagnostic must retain its complete seed42 reference, including smoke fixtures')
    names = ['suite.json', *suite['plan_artifact_sha256'], *diagnostic.AGGREGATES]
    names.extend(f'{diagnostic.REFERENCE_DIR}/{n}' for n in
                 suite['identity']['reference']['artifact_sha256'])
    for job in suite['identity']['jobs']:
        names.extend(f"runs/{job['run_id']}/{n}" for n in RUN_FILES)
    require(len(names) == len(set(names)) == 339, 'Expected 339 original diagnostic evidence files')
    return sorted(names)


def verify_reference(root, expected_source=None):
    root = Path(root)
    old = diagnostic.load_suite(root)
    source = diagnostic.source_hashes() if expected_source is None else expected_source
    require(old['identity']['source_sha256'] == source, 'Original diagnostic source differs')
    require(all(diagnostic.verify_complete_job(root, old, j) for j in old['identity']['jobs']),
            'Original diagnostic must contain all ten completed units')
    require(old.get('completed_runs') == 10 and old.get('status') == 'completed',
            'Original diagnostic summary is not complete')
    runtime = old.get('runtime_identity')
    require(isinstance(runtime, dict), 'Original diagnostic runtime is missing')
    return {'identity_sha256': old['identity_sha256'], 'runtime_identity': runtime,
            'artifact_sha256': {n: sha256_file(safe_file(root, n)) for n in reference_names(old)}}


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong rotation version')
    config = identity.get('config', {})
    require(config == fixed_config(config.get('mode') == 'smoke'), 'Changed frozen numerical protocol')
    for key, names in (('source_sha256', diagnostic.SOURCE_FILES),
                       ('orchestration_source_sha256', SOURCE_FILES)):
        hashes = identity.get(key, {})
        require(set(hashes) == set(names) and all(re.fullmatch('[0-9a-f]{64}', str(v))
                for v in hashes.values()), 'Malformed source manifest: ' + key)
    # Preserve the strict historical cohort validator; only the B plan changes.
    folds, old_jobs = base.make_plan(identity['data'])
    base.verify_identity({'version': base.VERSION, 'config': base.fixed_config(),
        'source_sha256': {n: identity['source_sha256'][n] for n in base.SOURCE_FILES},
        'data': identity['data'], 'folds': folds, 'jobs': old_jobs})
    folds, jobs = make_plan(identity['data'])
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Changed rotation plan')
    require(identity.get('imported_run_ids') == imported_ids(identity), 'Changed imported unit identities')
    require(isinstance(identity.get('reference'), dict), 'Missing original diagnostic snapshot identity')
    bindings = identity.get('cache_bindings', {})
    require(set(bindings) == {j['run_id'] for j in jobs}, 'Missing per-A cache bindings')
    by_name = {r['image_name']: r for r in identity['data']}
    for job in jobs:
        reference.verify_binding(bindings[job['run_id']], [by_name[n] for n in job['A']])


def _validate_reference_match(identity, root, old):
    require(old['identity']['data'] == identity['data']
            and old['identity']['config'] == identity['config'], 'Original cohort/config differs')
    expected = {j['run_id']: j for j in old['identity']['jobs']}
    for job in identity['jobs']:
        if origin(identity, job) == 'imported':
            require(job == expected[job['run_id']], 'Imported job was rewritten')
            require(identity['cache_bindings'][job['run_id']] ==
                    old['identity']['cache_bindings'][job['run_id']], 'Imported A cache differs')
    diagnostic._reference_bindings(Path(root) / diagnostic.REFERENCE_DIR, identity['jobs'],
        identity['cache_bindings'], {r['image_name']: r for r in identity['data']})


def verify_artifacts(root, suite):
    root = Path(root)
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Rotation identity corrupted')
    expected = artifacts(suite['identity'])
    require(set(suite.get('plan_artifact_sha256', {})) == set(expected), 'Plan artifact list differs')
    for name, rows in expected.items():
        path = safe_file(root, name)
        require(sha256_file(path) == suite['plan_artifact_sha256'][name]
                and read_csv(path) == [{k: str(v) for k, v in r.items()} for r in rows],
                'Plan artifact changed: ' + name)
    reference_root = root / REFERENCE_DIR
    expected_ref = suite['identity']['reference']
    require(verify_reference(reference_root, suite['identity']['source_sha256']) == expected_ref,
            'Frozen diagnostic snapshot changed')
    require(suite.get('runtime_identity') == expected_ref['runtime_identity'],
            'Runtime differs from the imported diagnostics')
    old = read_json(safe_file(reference_root, 'suite.json'))
    _validate_reference_match(suite['identity'], reference_root, old)
    pending = verify_pending_aggregate(root, suite)
    hashes = suite.get('aggregate_sha256', {})
    require(not hashes or set(hashes) == set(AGGREGATES), 'Aggregate manifest incomplete')
    if not pending:
        for name, digest in hashes.items():
            require(sha256_file(safe_file(root, name)) == digest, 'Aggregate artifact changed: ' + name)


def _check_current_source(suite):
    require(diagnostic.source_hashes() == suite['identity']['source_sha256'], 'Frozen numerical source changed')
    require(source_hashes() == suite['identity']['orchestration_source_sha256'], 'Rotation source changed')


def prepare(root, image_root, reference_root, *, cache_root, smoke=False):
    root = guard_roots(root, image_root, cache_root, reference_root)
    check_output_target(root)
    if (root / 'suite.json').exists():
        _check_current_source(load_suite(root))
    rows, metadata = audit_dataset(Path(image_root).expanduser().resolve())
    data = portable(rows)
    folds, jobs = make_plan(data)
    ref = verify_reference(reference_root)
    old = read_json(safe_file(reference_root, 'suite.json'))
    require(old['identity']['config'] == fixed_config(smoke), 'Original diagnostic mode/config differs')
    by_name = {r['image_name']: r for r in rows}
    bindings = {j['run_id']: base.cache_binding(cache_root, [by_name[n] for n in j['A']]) for j in jobs}
    identity = {'version': VERSION, 'config': fixed_config(smoke), 'data': data,
        'folds': folds, 'jobs': jobs, 'imported_run_ids': imported_ids({'data': data}),
        'cache_bindings': bindings, 'reference': ref,
        'source_sha256': diagnostic.source_hashes(), 'orchestration_source_sha256': source_hashes()}
    verify_identity(identity)
    _validate_reference_match(identity, reference_root, old)
    if (root / 'suite.json').exists():
        suite = load_suite(root)
        require(suite['identity'] == identity, 'Resume rejected: plan/data/cache/reference/source changed')
        return rows, suite
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in ref['artifact_sha256'].items():
        target = root / REFERENCE_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(reference_root, name), target)
        require(sha256_file(target) == digest, 'Original reference changed while copying')
    for name, values in artifacts(identity).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_csv(target, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
        'runtime_identity': ref['runtime_identity'], 'completed_runs': 10,
        'imported_completed_runs': 10, 'new_completed_runs': 0, 'status': 'partial',
        'protected_roots': {k: str(Path(v).expanduser().resolve()) for k, v in
            (('image', image_root), ('cache', cache_root), ('reference', reference_root))},
        'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
        'plan_artifact_sha256': {n: sha256_file(root / n) for n in artifacts(identity)}}
    _atomic_json(root / 'suite.json', suite)
    verify_artifacts(root, suite)
    return rows, suite


selected_jobs = diagnostic.selected_jobs


def verify_complete_job(root, suite, job):
    root = Path(root)
    if origin(suite['identity'], job) == 'imported':
        require(not (root / 'runs' / job['run_id']).exists(), 'Imported unit must not be rewritten as a new run')
        old_root = root / REFERENCE_DIR
        old = read_json(safe_file(old_root, 'suite.json'))
        result = diagnostic.verify_complete_job(old_root, old, job)
        require(result is not None, 'Imported diagnostic is incomplete')
        return result
    return diagnostic.verify_complete_job(root, suite, job)


def verify_pending_aggregate(root, suite):
    """Accept only a bounded, identity-bound interruption of derived writes.

    The marker never excuses corrupt raw units, plans or frozen references. It
    only allows regenerating the four summaries from verified completed units.
    """
    root = Path(root)
    if not (root / PENDING_FILE).exists():
        return False
    marker = read_json(safe_file(root, PENDING_FILE))
    require(isinstance(marker, dict) and set(marker) == {'identity_sha256', 'aggregate_run_ids'}
            and marker['identity_sha256'] == suite['identity_sha256'], 'Invalid pending aggregate identity/schema')
    ids = marker['aggregate_run_ids']
    jobs = suite['identity']['jobs']
    require(isinstance(ids, list) and all(isinstance(n, str) for n in ids)
            and len(ids) == len(set(ids)) and set(suite['identity']['imported_run_ids']).issubset(ids)
            and ids == [j['run_id'] for j in jobs if j['run_id'] in ids],
            'Invalid pending aggregate completion scope')
    require(all(verify_complete_job(root, suite, j) is not None for j in jobs if j['run_id'] in ids),
            'Pending aggregate refers to an incomplete unit')
    return True


def load_suite(root):
    root = Path(root)
    check_output_target(root)
    suite = read_json(safe_file(root, 'suite.json'))
    protected = suite.get('protected_roots', {})
    guard_roots(root, protected.get('image'), protected.get('cache'), protected.get('reference'))
    verify_identity(suite['identity'])
    verify_artifacts(root, suite)
    for job in suite['identity']['jobs']:
        verify_complete_job(root, suite, job)
    if suite.get('aggregate_sha256') and not (root / PENDING_FILE).exists():
        ids = suite.get('aggregate_run_ids', [])
        require(isinstance(ids, list) and len(ids) == len(set(ids))
                and set(suite['identity']['imported_run_ids']).issubset(ids)
                and set(ids).issubset(j['run_id'] for j in suite['identity']['jobs']),
                'Aggregate completion scope differs')
        # A unit can commit immediately before a process interruption. Its old
        # summary is honest but stale; validate the saved scope, then status/run
        # can discover the extra completion without forcing a second training.
        data = aggregate(root, suite, completed_ids=set(ids))
        require([r['run_id'] for r in data['results']] == ids
                and suite['completed_runs'] == len(ids), 'Aggregate completion scope is not complete')
        for name, rows in (('per_unit_probes.csv', data['probes']),
                           ('unit_comparisons.csv', data['comparisons']),
                           ('fold_budget_summary.csv', data['groups'])):
            require(read_csv(root / name) == [
                {k: '' if v is None else str(v) for k, v in row.items()} for row in rows],
                'Aggregate values differ from completed unit evidence: ' + name)
    return suite


def train_one(root, suite, job, by_name, device, cache_root=None, model_factory=None):
    _check_current_source(suite)
    result = verify_complete_job(root, suite, job)
    if result is not None:
        return result
    try:
        result = diagnostic.train_one(root, suite, job, by_name, device, cache_root, model_factory)
        _check_current_source(suite)
        return result
    except Exception:
        # A failed wrapper/source check must not leave a newly committed unit.
        marker = Path(root) / 'runs' / job['run_id'] / 'complete.json'
        if marker.exists():
            marker.unlink()
        raise


def aggregate(root, suite, completed_ids=None):
    root = Path(root)
    results, probes, comparisons = [], [], []
    for job in suite['identity']['jobs']:
        if completed_ids is not None and job['run_id'] not in completed_ids:
            continue
        result = verify_complete_job(root, suite, job)
        if result is None:
            continue
        kind = origin(suite['identity'], job)
        info = {k: job[k] for k in ('run_id', 'fold', 'budget', 'B_patient')}
        info.update(B_index=job['patient_order_prefix'].index(job['B_patient']), origin=kind)
        results.append({**result, 'origin': kind})
        directory = root / REFERENCE_DIR if kind == 'imported' else root
        unit = read_csv(directory / 'runs' / job['run_id'] / 'probes.csv')
        probes.extend({**info, **row} for row in unit)
        by_variant = {r['variant']: r for r in unit}
        def improvement(variant):
            row = by_variant[variant]
            return float(row['relative_improvement']) if row['valid'] == 'True' else None
        field, legacy = improvement('field_plain'), improvement('legacy_plain')
        random = [improvement(f'random_matched_{s}') for s in (42, 43, 44)]
        shuffled = [improvement(f'field_shuffled_{s}') for s in (42, 43, 44)]
        comparison = dict(info, field_vs_ce=field,
            field_vs_legacy=field - legacy if field is not None and legacy is not None else None,
            field_vs_random_mean=field - statistics.mean(random) if field is not None and None not in random else None,
            field_vs_shuffled_mean=field - statistics.mean(shuffled) if field is not None and None not in shuffled else None)
        comparisons.append(comparison)
    groups = []
    for fold in range(5):
        for budget in ('5', '10'):
            unit_rows = [r for r in comparisons if r['fold'] == fold and r['budget'] == budget]
            joint_valid = [r for r in unit_rows if r['field_vs_ce'] is not None
                           and r['field_vs_shuffled_mean'] is not None]
            joint_positive = sum(r['field_vs_ce'] > 0 and r['field_vs_shuffled_mean'] > 0
                                 for r in joint_valid)
            for key in COMPARISONS:
                values = [r[key] for r in unit_rows if r[key] is not None]
                groups.append({'fold': fold, 'budget': budget, 'comparison': key,
                    'planned_B': int(budget), 'completed_B': len(unit_rows), 'valid_B': len(values),
                    'invalid_B': len(unit_rows) - len(values), 'group_complete': len(unit_rows) == int(budget),
                    'joint_ce_shuffle_valid_B': len(joint_valid),
                    'joint_ce_shuffle_positive_B': joint_positive,
                    'joint_ce_shuffle_planned_B': int(budget),
                    'imported_B': sum(r['origin'] == 'imported' for r in unit_rows),
                    'new_B': sum(r['origin'] == 'new' for r in unit_rows),
                    'mean': statistics.mean(values) if values else None,
                    'median': statistics.median(values) if values else None,
                    'sd': statistics.stdev(values) if len(values) > 1 else None,
                    'minimum': min(values) if values else None, 'maximum': max(values) if values else None,
                    'positive': sum(v > 0 for v in values), 'zero': sum(v == 0 for v in values),
                    'negative': sum(v < 0 for v in values)})
    return {'results': results, 'probes': probes, 'comparisons': comparisons, 'groups': groups}


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root, suite)
    # The summaries are derived files, not one atomic filesystem object. Record
    # a narrow recovery scope before touching them and clear it only after the
    # updated suite hashes commit. Raw unit validation always remains mandatory.
    _atomic_json(root / PENDING_FILE, {'identity_sha256': suite['identity_sha256'],
                                     'aggregate_run_ids': [r['run_id'] for r in data['results']]})
    for name, rows in (('per_unit_probes.csv', data['probes']),
                       ('unit_comparisons.csv', data['comparisons']),
                       ('fold_budget_summary.csv', data['groups'])):
        diagnostic._write_table(root / name, rows)
    imported = sum(r['origin'] == 'imported' for r in data['results'])
    new = len(data['results']) - imported
    lines = ['# Complete within-support B rotation diagnostic', '',
        f'Completed units: {imported + new}/75; imported original units: {imported}/10; new units: {new}/65.', '',
        'The ten original units retain their original files, identities and runtime under reference_diagnostic.',
        'Each new B starts a fresh seed44 model using only its other K-1 support patients as A.',
        'RGB still contains the original frame; the field heuristic only excludes auxiliary geometric relationships.',
        'The numerical protocol, 140+140 head preparation steps and eight auxiliary branches are unchanged.',
        'There is no best-B selection, learned gate, threshold search, or outer-evaluation-image use.',
        'Each fold/budget reports every B, including invalid branches. Three control seeds are averaged within a unit.',
        'Units share patients and A images; 75 units are not 75 independent patients or trials.',
        'Differences are relative virtual B classification losses, not accuracy gains or full AdamW training effects.',
        'The legacy calibration target can favor legacy geometry. A positive difference need not improve over the unperturbed model.',
        'This is a bounded mechanistic diagnostic; it does not establish depth accuracy, algorithm effectiveness or patent novelty.', '',
        'See unit_comparisons.csv and fold_budget_summary.csv for the complete B distributions.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    suite.update(completed_runs=imported + new, imported_completed_runs=imported,
        new_completed_runs=new, status='completed' if imported + new == 75 else 'partial',
        aggregate_run_ids=[r['run_id'] for r in data['results']],
        aggregate_sha256={n: sha256_file(root / n) for n in AGGREGATES})
    _atomic_json(root / 'suite.json', suite)
    (root / PENDING_FILE).unlink()
    return data


def run_suite(args):
    root = guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    check_output_target(root)
    if (root / 'suite.json').exists():
        _check_current_source(load_suite(root))
    base.check_dependencies()
    configure_torch()
    import torch
    runtime = runtime_key(environment(require_cuda=not args.smoke))
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(root):
        rows, suite = prepare(root, args.image_root, args.reference_root,
                              cache_root=args.cache_root, smoke=args.smoke)
        require(runtime == suite['runtime_identity'], 'Runtime differs from imported units; restore the original environment')
        jobs = selected_jobs(suite, args)
        by_name = {r['image_name']: r for r in rows}
        diagnostic.preflight_cache(root, suite, jobs, by_name, args.cache_root)
        suite.setdefault('attempts', []).append({'at_utc': now(), 'folds': args.folds, 'budgets': args.budgets})
        _atomic_json(root / 'suite.json', suite)
        try:
            for index, job in enumerate(jobs, 1):
                kind = origin(suite['identity'], job)
                print(f"[{index}/{len(jobs)}] {job['run_id']} ({kind})", flush=True)
                train_one(root, suite, job, by_name, device, args.cache_root)
                write_aggregates(root, suite)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as error:
            suite['last_failure'] = {'at_utc': now(), 'message': str(error)}
            _atomic_json(root / 'suite.json', suite)
            raise
    return suite


def pack_suite(root, args):
    root = guard_roots(root, args.image_root, args.cache_root, args.reference_root)
    load_suite(root)
    with output_lock(root):
        suite = load_suite(root)
        require(suite['identity']['config']['mode'] == 'formal', 'Smoke rotation cannot be packaged as formal')
        jobs = selected_jobs(suite, args)
        require(all(verify_complete_job(root, suite, j) for j in jobs), 'Selected packing scope is incomplete')
        data = write_aggregates(root, suite)
        imported = sum(origin(suite['identity'], j) == 'imported' for j in jobs)
        package = {'version': VERSION, 'created_utc': now(), 'planned_runs': 75,
            'planned_imported_runs': 10, 'planned_new_runs': 65,
            'selected_run_ids': [j['run_id'] for j in jobs], 'selected_completed_runs': len(jobs),
            'selected_imported_runs': imported, 'selected_new_runs': len(jobs) - imported,
            'scope_complete': True, 'entire_suite_complete': len(data['results']) == 75,
            'included_completed_run_ids': [r['run_id'] for r in data['results']],
            'included_imported_runs': suite['imported_completed_runs'],
            'included_new_runs': suite['new_completed_runs'],
            'aggregate_scope': 'all completed units, with imported/new origin explicit',
            'reference_identity_sha256': suite['identity']['reference']['identity_sha256'],
            'interpretation': 'within-support mechanistic diagnostic; not accuracy/OOF',
            'excluded': ['images', 'NPZ teacher maps', 'model weights', 'third-party code']}
        _atomic_json(root / 'package_scope.json', package)
        names = ['suite.json', 'package_scope.json', *suite['plan_artifact_sha256'], *AGGREGATES]
        names.extend(f'{REFERENCE_DIR}/{n}' for n in suite['identity']['reference']['artifact_sha256'])
        for result in data['results']:
            if result['origin'] == 'new':
                names.extend(f"runs/{result['run_id']}/{n}" for n in RUN_FILES)
        require(all(Path(n).suffix in ('.json', '.csv', '.md') for n in names), 'Non-text packing artifact')
        archive = root.parent / f'{root.name}_{time.time_ns()}_results.tar.gz'
        with tarfile.open(archive, 'x:gz') as out:
            for name in sorted(set(names)):
                out.add(safe_file(root, name), arcname=name, recursive=False)
        Path(str(archive) + '.sha256').write_text(sha256_file(archive) + '  ' + archive.name + '\n', encoding='utf-8')
        return archive


def parser():
    p = argparse.ArgumentParser(description='All-B rotation: ten immutable imports plus 65 new diagnostics; no accuracy claims.')
    p.add_argument('mode', choices=['check', 'run', 'status', 'pack'])
    p.add_argument('--image-root', type=Path, default=Path('dataset/renew'))
    p.add_argument('--cache-root', type=Path, default=Path('outputs/renew_geometry_teacher_cache'))
    p.add_argument('--reference-root', type=Path, default=Path('outputs/renew_geometry_diagnostic'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/renew_geometry_rotation'))
    p.add_argument('--budgets', nargs='+', choices=['5', '10'])
    p.add_argument('--folds', nargs='+', type=int, choices=range(5))
    p.add_argument('--smoke', action='store_true', help='Requires a complete original smoke suite; CPU/random weights; no formal pack')
    return p


def next_command(args, mode):
    parts = ['bash', 'scripts/run_geometry_rotation_server.sh', mode, '--output-root', str(args.output_root)]
    if mode in ('check', 'run'):
        for key in ('image_root', 'cache_root', 'reference_root'):
            parts.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
        if args.smoke:
            parts.append('--smoke')
    for key in ('folds', 'budgets'):
        if getattr(args, key):
            parts.extend(['--' + key, *map(str, getattr(args, key))])
    return shlex.join(parts)


def main(argv=None):
    args = parser().parse_args(argv)
    root = guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    args.output_root = root
    check_output_target(root)
    if args.mode == 'run':
        run_suite(args)
        print('Next: ' + next_command(args, 'status'))
    elif args.mode == 'pack':
        print(pack_suite(root, args))
        print('Checksum sidecar written; retain the complete output tree.')
    elif args.mode == 'status':
        suite = load_suite(root)
        data = aggregate(root, suite)
        selected = selected_jobs(suite, args)
        completed = sum(verify_complete_job(root, suite, j) is not None for j in selected)
        new = sum(r['origin'] == 'new' for r in data['results'])
        print(f"Completed {len(data['results'])}/75; imported 10/10; new {new}/65; selected {completed}/{len(selected)}.")
        if (root / PENDING_FILE).exists():
            print('Derived summary write was interrupted; verified unit results are intact. Run or pack will rebuild summaries.')
        args.smoke = suite['identity']['config']['mode'] == 'smoke'
        for label, path in suite['protected_roots'].items():
            setattr(args, label + '_root', Path(path))
        if args.smoke and completed == len(selected):
            print('Smoke scope complete; formal packing is disabled.')
        else:
            print('Next: ' + next_command(args, 'pack' if completed == len(selected) else 'run'))
    else:
        base.check_dependencies()
        if (root / 'suite.json').exists():
            _check_current_source(load_suite(root))
        with output_lock(root):
            rows, suite = prepare(root, args.image_root, args.reference_root,
                                  cache_root=args.cache_root, smoke=args.smoke)
        print(f"PASS: {len(rows)} images / 31 filename patient IDs; 75 planned units; 10 immutable imports + 65 new units.")
        print('Next: ' + next_command(args, 'run'))


if __name__ == '__main__':
    main()
