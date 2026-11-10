"""Frozen ten-unit mechanistic geometry diagnostic, separate from accuracy pilots.

Only the predetermined A images supply geometry targets. B supplies RGB labels
for a virtual, held-out-patient probe; outer evaluation images are never used.
Status and packing validate copied text evidence without external datasets,
teacher caches, pretrained weights, or historical output directories.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shlex
import shutil
import tarfile
import time

from src import geometry_experiment as base
from src import geometry_replication as reference
from src.baseline_reproduction import output_lock, read_json, read_csv, require
from src.fewshot import PROJECT_ROOT, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest, seed_training
from src.renew_data import audit_dataset
from src.renew_experiment import portable, configure_torch, environment, runtime_key

VERSION = 'renew_r18_geometry_diagnostic_v1'
SOURCE_FILES = (*reference.SOURCE_FILES, 'src/geometry_diagnostic_core.py',
                'src/geometry_diagnostic.py', 'scripts/run_geometry_diagnostic.py',
                'scripts/run_geometry_diagnostic_server.sh')
RUN_FILES = ('config.json', 'history.csv', 'pair_audit.csv', 'probes.csv',
             'result.json', 'complete.json')
REFERENCE_DIR = 'reference_seed42'
AUDIT_ANGLES = (-15, 0, 15)
AGGREGATES = ('per_unit_probes.csv', 'control_comparisons.csv', 'REPORT.md')
now = base.now
safe_file = reference.safe_file


def fixed_config(smoke=False):
    from src.geometry_diagnostic_core import fixed_config as core_config
    return core_config(smoke)


def source_hashes():
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def _no_symlink(path):
    path = Path(path).expanduser().absolute()
    require(all(not part.is_symlink() for part in (path, *path.parents)),
            'Symlink output path is forbidden')


def guard_roots(root, image_root=None, cache_root=None, reference_root=None):
    """Reject aliases and overlap before any directory or lock is created."""
    _no_symlink(root)
    root = Path(root).expanduser().resolve()
    protected = [('image', image_root), ('cache', cache_root), ('reference', reference_root)]
    protected.extend((name, PROJECT_ROOT / 'outputs' / name) for name in
                     ('renew_geometry_fewshot', 'renew_geometry_seed43',
                      'renew_geometry_teacher_cache', 'renew_fewshot', 'renew_baseline'))
    for label, value in protected:
        if value is not None:
            other = Path(value).expanduser().resolve()
            require(not root.is_relative_to(other) and not other.is_relative_to(root),
                    f'Output and {label} roots must be disjoint')
    return root


def check_output_target(root):
    _no_symlink(root)
    root = Path(root)
    if root.exists():
        require(root.is_dir(), 'Output must be a directory')
        # Output trees contain text evidence only. Reject hidden write aliases as
        # well as aliases on named artifacts before taking the output lock.
        require(not any(p.is_symlink() for p in root.rglob('*')), 'Unsafe output artifact symlink')
    if (root / 'suite.json').exists():
        suite = read_json(safe_file(root, 'suite.json'))
        require(suite.get('identity', {}).get('version') == VERSION,
                'Wrong diagnostic suite version; refusing to modify output')
    elif root.exists():
        require(not any(p.name != '.reproduction.lock' for p in root.iterdir()),
                'Refusing nonempty output without suite.json')


validate_output_root = guard_roots


def split_support(job, rows):
    by_name = {row['image_name']: row for row in rows}
    b_patient = job['patient_order_prefix'][-1]
    a = [name for name in job['support'] if by_name[name]['patient_key'] != b_patient]
    b = [name for name in job['support'] if by_name[name]['patient_key'] == b_patient]
    require(len(b) == 3 and len(a) == 3 * (int(job['budget']) - 1), 'Wrong A/B image counts')
    require(not set(a) & set(b) and set(a + b) == set(job['support']), 'A/B split overlaps or loses support')
    require(not set(a + b) & set(job['eval']), 'A/B includes outer evaluation images')
    return a, b


def make_plan(rows):
    folds, originals = base.make_plan(rows)
    jobs = []
    for old in originals:
        if old['budget'] not in ('5', '10') or old['method'] != 'plain':
            continue
        a, b = split_support(old, rows)
        run_id = f"fold{old['fold']}_k{old['budget']}_repeat0_geometry_diagnostic_seed44"
        jobs.append({**{key: old[key] for key in ('fold', 'budget', 'repeat',
                     'support', 'eval', 'patient_order_prefix')},
                     'run_id': run_id, 'reference_run_id': old['run_id'],
                     'A': a, 'B': b, 'A_patients': old['patient_order_prefix'][:-1],
                     'B_patient': old['patient_order_prefix'][-1]})
    require(len(jobs) == 10 and len({j['run_id'] for j in jobs}) == 10,
            'Expected exactly ten diagnostic units')
    return folds, jobs


def artifacts(identity):
    by_name = {r['image_name']: r for r in identity['data']}
    out = {'dataset.csv': identity['data'], 'unit_plan.csv': [
        {key: j[key] for key in ('run_id', 'fold', 'budget', 'repeat', 'reference_run_id', 'B_patient')}
        for j in identity['jobs']]}
    for job in identity['jobs']:
        for part in ('support', 'A', 'B'):
            out[f"splits/{job['run_id']}_{part}.csv"] = [by_name[n] for n in job[part]]
    return out


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong diagnostic suite version')
    c = identity.get('config', {})
    require(c == fixed_config(c.get('mode') == 'smoke'), 'Changed fixed diagnostic protocol')
    source = identity.get('source_sha256', {})
    require(set(source) == set(SOURCE_FILES) and all(re.fullmatch('[0-9a-f]{64}', str(v))
            for v in source.values()), 'Incomplete/malformed source manifest')
    # Reuse the historical row/fold validator without mutating its module state.
    folds, old_jobs = base.make_plan(identity['data'])
    base.verify_identity({'version': base.VERSION, 'config': base.fixed_config(),
                         'source_sha256': {n: source[n] for n in base.SOURCE_FILES},
                         'data': identity['data'], 'folds': folds, 'jobs': old_jobs})
    expected_folds, expected_jobs = make_plan(identity['data'])
    require(identity['folds'] == expected_folds and identity['jobs'] == expected_jobs,
            'Changed diagnostic A/B/support plan')
    require(c['mode'] == 'smoke' or isinstance(identity.get('reference'), dict),
            'Formal diagnostic needs a frozen complete reference')
    bindings = identity.get('cache_bindings', {})
    require(set(bindings) == {j['run_id'] for j in expected_jobs}, 'Missing per-A cache bindings')
    by_name = {r['image_name']: r for r in identity['data']}
    for job in expected_jobs:
        reference.verify_binding(bindings[job['run_id']], [by_name[n] for n in job['A']])


validate_plan = verify_identity


def _reference_bindings(reference_root, jobs, bindings, by_name):
    old = read_json(safe_file(reference_root, 'suite.json'))['identity']
    require(old['data'] == list(by_name.values()), 'Reference cohort differs')
    original = {j['run_id']: j for j in old['jobs']}
    for job in jobs:
        prior = original[job['reference_run_id']]
        require(all(job[k] == prior[k] for k in ('fold', 'budget', 'repeat', 'support',
                    'eval', 'patient_order_prefix')), 'Reference support plan differs')
        old_binding = read_json(safe_file(reference_root,
            f"runs/{job['reference_run_id']}/result.json"))['cache_binding']
        new_binding = bindings[job['run_id']]
        require(all(new_binding[k] == old_binding[k] for k in ('version', 'teacher_sha256', 'variants'))
                and all(old_binding['entries'].get(key) == value for key, value in new_binding['entries'].items()),
                'A teacher/cache differs from the exact reference entries')


def verify_artifacts(root, suite):
    root = Path(root)
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Suite identity corrupted')
    expected = artifacts(suite['identity'])
    require(set(expected) == set(suite.get('plan_artifact_sha256', {})), 'Plan manifest incomplete')
    for name, rows in expected.items():
        path = safe_file(root, name)
        require(sha256_file(path) == suite['plan_artifact_sha256'][name], 'Plan artifact changed: ' + name)
        require(read_csv(path) == [{k: str(v) for k, v in row.items()} for row in rows],
                'Plan artifact values differ: ' + name)
    ref = suite['identity'].get('reference')
    if ref is not None:
        expected_source = {n: suite['identity']['source_sha256'][n] for n in base.SOURCE_FILES}
        require(reference.verify_reference(root / REFERENCE_DIR, expected_source) == ref,
                'Frozen reference snapshot changed')
        require(len(ref['artifact_sha256']) == 243, 'Expected 243 frozen reference artifacts')
        _reference_bindings(root / REFERENCE_DIR, suite['identity']['jobs'],
                            suite['identity']['cache_bindings'],
                            {r['image_name']: r for r in suite['identity']['data']})
    for name, digest in suite.get('aggregate_sha256', {}).items():
        require(name in AGGREGATES and sha256_file(safe_file(root, name)) == digest,
                'Aggregate artifact changed: ' + name)


def prepare(root, image_root, reference_root=None, *, cache_root=None, smoke=False):
    root = guard_roots(root, image_root, cache_root, reference_root)
    check_output_target(root)
    if (root / 'suite.json').exists():
        load_suite(root)
    try:
        rows, metadata = audit_dataset(Path(image_root).expanduser().resolve())
    except ValueError as error:
        if 'No first-level PNG' in str(error):
            raise ValueError(str(error) + '; --image-root must directly contain the audited 93 PNGs') from error
        raise
    data = portable(rows)
    folds, jobs = make_plan(data)
    ref = None
    if not smoke or reference_root is not None:
        require(reference_root is not None, 'Formal diagnostic requires --reference-root')
        ref = reference.verify_reference(reference_root)
        require(len(ref['artifact_sha256']) == 243, 'Expected 243 frozen reference artifacts')
    by_name = {r['image_name']: r for r in rows}
    # Only A target files are opened, including when freezing all ten units.
    bindings = {j['run_id']: base.cache_binding(cache_root, [by_name[n] for n in j['A']]) for j in jobs}
    if ref is not None:
        _reference_bindings(reference_root, jobs, bindings, {r['image_name']: r for r in data})
    identity = {'version': VERSION, 'config': fixed_config(smoke), 'data': data,
                'folds': folds, 'jobs': jobs, 'reference': ref, 'cache_bindings': bindings,
                'source_sha256': source_hashes()}
    verify_identity(identity)
    if (root / 'suite.json').exists():
        suite = load_suite(root)
        require(suite['identity'] == identity, 'Resume rejected: source/config/data/cache/reference changed')
        return rows, suite
    root.mkdir(parents=True, exist_ok=True)
    if ref is not None:
        for name, digest in ref['artifact_sha256'].items():
            target = root / REFERENCE_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(reference_root, name), target)
            require(sha256_file(target) == digest, 'Reference changed while copying')
    for name, values in artifacts(identity).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        write_csv(root / name, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
             'runtime_identity': None, 'status': 'prepared_not_run', 'completed_runs': 0,
             'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
             'protected_roots': {k: str(Path(v).expanduser().resolve()) for k, v in
                 (('image', image_root), ('cache', cache_root), ('reference', reference_root)) if v is not None},
             'plan_artifact_sha256': {n: sha256_file(root / n) for n in artifacts(identity)}}
    _atomic_json(root / 'suite.json', suite)
    verify_artifacts(root, suite)
    return rows, suite


def selected_jobs(suite, args):
    jobs = [j for j in suite['identity']['jobs']
            if (not args.budgets or j['budget'] in args.budgets)
            and (args.folds is None or j['fold'] in args.folds)]
    require(bool(jobs), 'Run filters select no units')
    return jobs


def preflight_cache(root, suite, jobs, by_name, cache_root):
    for job in jobs:
        actual = base.cache_binding(cache_root, [by_name[n] for n in job['A']])
        require(actual == suite['identity']['cache_bindings'][job['run_id']],
                'Resume rejected: frozen A cache changed')


def verify_complete_job(root, suite, job):
    """Verify completion atomically and semantically without model checkpoints."""
    from src.geometry_diagnostic_core import VARIANTS
    directory = Path(root) / 'runs' / job['run_id']
    if not (directory / 'complete.json').exists():
        return None
    complete = read_json(safe_file(directory, 'complete.json'))
    require(complete.get('identity_sha256') == suite['identity_sha256']
            and complete.get('run_id') == job['run_id'], 'Wrong completion identity')
    hashes = complete.get('artifact_sha256', {})
    require(set(hashes) == set(RUN_FILES[:-1]), 'Incomplete completion artifact manifest')
    for name, digest in hashes.items():
        require(sha256_file(safe_file(directory, name)) == digest, 'Completed artifact changed: ' + name)
    saved, result = read_json(directory / 'config.json'), read_json(directory / 'result.json')
    require(saved.get('job') == job and saved.get('config') == suite['identity']['config']
            and saved.get('identity_sha256') == suite['identity_sha256'], 'Completed config differs')
    require(saved.get('execution_runtime') == suite['runtime_identity'] and suite['runtime_identity'] is not None,
            'Missing/mismatched execution runtime')
    if suite['identity']['config']['mode'] == 'formal':
        runtime = suite['runtime_identity']
        require(set(runtime) == {'python', 'packages', 'cuda', 'cudnn', 'gpu'}
                and all(v is not None for v in runtime.values()), 'Missing formal CUDA runtime evidence')
    binding = suite['identity']['cache_bindings'][job['run_id']]
    require(saved.get('cache_binding') == binding and result.get('cache_binding') == binding,
            'Completed cache evidence differs')
    require(all(result.get(k) == job[k] for k in ('run_id', 'fold', 'budget', 'repeat')),
            'Completed unit identity differs')
    require(result.get('n_A_images') == len(job['A']) and result.get('n_B_images') == len(job['B']),
            'Completed support counts differ')
    tables = {name: read_csv(directory / filename) for name, filename in
              (('history', 'history.csv'), ('pair_audit', 'pair_audit.csv'), ('probe', 'probes.csv'))}
    for name, rows in tables.items():
        require(result.get(name + '_rows') == len(rows) and bool(rows), 'Missing/wrong evidence rows: ' + name)
    probes = tables['probe']
    require(len(probes) == len(VARIANTS) and {r['variant'] for r in probes} == set(VARIANTS),
            'Missing/duplicate diagnostic branches')
    require(isinstance(result.get('evidence', {}).get('preparation'), dict)
            and isinstance(result.get('evidence', {}).get('probe'), dict), 'Missing stage evidence')
    _verify_numerical_evidence(job, suite['identity']['config'], tables, result['evidence'])
    return result


validate_run = verify_complete_job


def _verify_numerical_evidence(job, config, tables, evidence):
    """Additional core-specific checks are intentionally strict on saved evidence."""
    from src.geometry_diagnostic_core import VARIANTS
    preparation, probe = evidence['preparation'], evidence['probe']
    for key in ('backbone_unchanged', 'calibration_fc_unchanged', 'calibration_bn_unchanged'):
        require(preparation.get(key) is True, 'Missing frozen-state evidence: ' + key)
    for left, right in (('initial_backbone_sha256', 'final_backbone_sha256'),
                        ('calibration_fc_before_sha256', 'calibration_fc_after_sha256'),
                        ('calibration_bn_before_sha256', 'calibration_bn_after_sha256')):
        require(bool(preparation.get(left)) and preparation[left] == preparation.get(right),
                'Frozen-state fingerprints differ: ' + left)
    require(preparation.get('A_image_count') == len(job['A'])
            and preparation.get('A_patient_count') == len(job['A_patients'])
            and preparation.get('preparation_image_names') == job['A']
            and preparation.get('calibration_pair_count', 0) > 0, 'Preparation A-only evidence differs')
    rng = preparation.get('rng', {})
    if config['mode'] == 'formal' or not rng.get('injected_test_model'):
        require(rng.get('cuda_rng_restored') is True
                and rng.get('cuda_before_sha256') == rng.get('cuda_after_sha256'), 'CUDA RNG evidence differs')
        if config['mode'] == 'formal':
            require(rng.get('cuda_device_count', 0) > 0
                    and len(rng['cuda_before_sha256']) == rng['cuda_device_count'], 'Formal CUDA RNG evidence missing')
    expected = [(stage, update) for stage, count in
                (('head', config['head_steps']), ('calibration', config['calibration_steps']))
                for update in range(1, count + 1)]
    require([(r['stage'], int(r['update'])) for r in tables['history']] == expected,
            'Preparation schedule differs')
    require(preparation.get('actual_head_steps') == config['head_steps']
            and preparation.get('actual_calibration_steps') == config['calibration_steps'], 'Preparation update count differs')
    for row in tables['history']:
        head = row['stage'] == 'head'
        require(int(row['image_occurrences']) == config['batch_size']
                and math.isclose(float(row['lr']), config['head_lr'] if head else config['geometry_lr'])
                and all(math.isfinite(float(row[k])) for k in ('loss', 'ce_loss', 'geometry_loss'))
                and float(row['geometry_loss'] if head else row['ce_loss']) == 0, 'Invalid preparation history')
    for key in ('state_restored', 'patient_disjoint', 'all_branches_share_basepoint', 'geometry_head_frozen'):
        require(probe.get(key) is True, 'Missing probe evidence: ' + key)
    require(bool(probe.get('state_before_sha256')) and probe['state_before_sha256'] == probe.get('state_after_sha256')
            and probe.get('A_image_count') == len(job['A']) and probe.get('B_image_count') == len(job['B'])
            and probe.get('shared_parameter_count', 0) > 0, 'Probe restoration/support evidence differs')
    require(math.isclose(probe['ce_displacement_norm'], config['ce_relative_displacement'] * probe['shared_parameter_norm'])
            and math.isclose(probe['auxiliary_displacement_norm'], config['auxiliary_relative_displacement'] * probe['ce_displacement_norm']),
            'Virtual displacement norms differ')
    audit = tables['pair_audit']
    expected_audits = {(name, angle, variant) for name in job['A'] for angle in AUDIT_ANGLES for variant in VARIANTS}
    require(len(audit) == len(expected_audits)
            and {(r['image_name'], int(r['angle']), r['variant']) for r in audit} == expected_audits,
            'Pair audit coverage differs')
    for row in audit:
        count = int(row['selected_pairs'])
        require(int(row['candidate_pairs']) == config['pair_count']
                and int(row['batch_image_denominator']) == len(job['A'])
                and int(row['image_pair_denominator']) == count
                and int(row['positive_pairs']) + int(row['negative_pairs']) == count
                and int(row['near_pairs_selected']) + int(row['far_pairs_selected']) == count,
                'Pair count/normalization evidence differs')
        if row['variant'] != 'legacy_plain':
            require(count == int(row['field_pairs']), 'Matched control pair count differs')
        if row['variant'].startswith('field_'):
            require(int(row['selected_touch_border_pairs']) == 0, 'Field pairs touch exterior')
    # Core records are checked here as well as by their file digests.
    require(len({r['ce_control_B_loss'] for r in tables['probe']}) == 1
            and len({r['ce_B_base_loss'] for r in tables['probe']}) == 1,
            'Branches do not share the same basepoint and CE control')
    for row in tables['probe']:
        for key, value in row.items():
            if key in ('variant', 'reason', 'status', 'valid', 'skip_reason') or value == '':
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            require(math.isfinite(number), 'Nonfinite probe evidence: ' + key)
        valid = row['valid'] == 'True'
        require(row['valid'] in ('True', 'False'), 'Invalid probe validity flag')
        selected = sum(int(r['selected_pairs']) for r in audit
                       if int(r['angle']) == 0 and r['variant'] == row['variant'])
        require(int(row['selected_pairs']) == selected
                and math.isclose(float(row['ce_displacement_norm']), probe['ce_displacement_norm']),
                'Probe pair/displacement evidence differs')
        if valid:
            require(not row['skip_reason'] and int(row['selected_pairs']) > 0
                    and float(row['ce_A_gradient_norm']) > config['minimum_gradient_norm']
                    and float(row['auxiliary_gradient_norm']) > config['minimum_gradient_norm'],
                    'Invalid gradient promoted to valid probe')
            require(math.isclose(float(row['relative_improvement']),
                    float(row['ce_control_B_loss']) - float(row['variant_B_loss']), abs_tol=1e-12)
                    and math.isclose(float(row['auxiliary_displacement_norm']), probe['auxiliary_displacement_norm']),
                    'Virtual delta/control norm evidence differs')
        else:
            require(bool(row['skip_reason']) and row['relative_improvement'] == ''
                    and row['variant_B_loss'] == '', 'Skipped probe contains a positive result')


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
    return suite


def _write_table(path, rows):
    require(bool(rows), 'Cannot commit empty diagnostic table')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, fieldnames=fields)


def train_one(root, suite, job, by_name, device, cache_root=None, model_factory=None):
    from src import geometry_diagnostic_core as core
    root = Path(root)
    require(source_hashes() == suite['identity']['source_sha256'], 'Source changed during execution')
    preflight_cache(root, suite, [job], by_name, cache_root)
    completed = verify_complete_job(root, suite, job)
    if completed:
        return completed
    def verify_images():
        require(all(sha256_file(by_name[n]['path']) == by_name[n]['image_sha256']
                    for n in job['A'] + job['B']), 'A/B source image changed during execution')
    verify_images()
    directory = root / 'runs' / job['run_id']
    if directory.exists():
        archived = root / 'incomplete_attempts'
        archived.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archived / (job['run_id'] + '_' + str(time.time_ns()))))
    directory.mkdir(parents=True)
    c = suite['identity']['config']
    binding = suite['identity']['cache_bindings'][job['run_id']]
    _atomic_json(directory / 'config.json', {'identity_sha256': suite['identity_sha256'],
        'job': job, 'config': c, 'cache_binding': binding, 'execution_runtime': suite['runtime_identity']})
    seed_training(c['training_seed'])
    factory = model_factory or (lambda pretrained: core.build_student(pretrained=pretrained, dropout=c['head_dropout']))
    model = factory(c['mode'] == 'formal').to(device)
    a_rows, b_rows = [by_name[n] for n in job['A']], [by_name[n] for n in job['B']]
    preparation = core.prepare_model(model, a_rows, cache_root, c, device)
    a_batch = core.make_batch(a_rows, cache_root, angle=0)
    b_batch = core.make_batch(b_rows, None, angle=0, need_geometry=False)
    probes = core.probe(model, a_batch, b_batch, c)
    pairs = []
    for angle in AUDIT_ANGLES:
        batch = a_batch if angle == 0 else core.make_batch(a_rows, cache_root, angle=angle)
        pairs.extend(dict(row, angle=angle) for row in core.audit_pairs(batch))
    _write_table(directory / 'history.csv', preparation['history'])
    _write_table(directory / 'pair_audit.csv', pairs)
    _write_table(directory / 'probes.csv', probes['records'])
    result = {**{k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat')},
              'cache_binding': binding, 'history_rows': len(preparation['history']),
              'pair_audit_rows': len(pairs), 'probe_rows': len(probes['records']),
              'n_A_images': len(a_rows), 'n_B_images': len(b_rows),
              'evidence': {'preparation': preparation['evidence'], 'probe': probes['evidence']},
              'interpretation': 'mechanistic diagnostic; not classification performance or OOF'}
    _atomic_json(directory / 'result.json', result)
    verify_images()
    preflight_cache(root, suite, [job], by_name, cache_root)
    require(source_hashes() == suite['identity']['source_sha256'], 'Source changed during execution')
    _atomic_json(directory / 'complete.json', {'identity_sha256': suite['identity_sha256'],
        'run_id': job['run_id'], 'artifact_sha256': {n: sha256_file(directory / n) for n in RUN_FILES[:-1]}})
    verify_complete_job(root, suite, job)
    return result


def aggregate(root, suite):
    results, probes, controls = [], [], []
    for job in suite['identity']['jobs']:
        result = verify_complete_job(root, suite, job)
        if result is None:
            continue
        results.append(result)
        unit = read_csv(Path(root) / 'runs' / job['run_id'] / 'probes.csv')
        probes.extend({**{k: job[k] for k in ('run_id', 'fold', 'budget')}, **row} for row in unit)
        by_variant = {r['variant']: r for r in unit}
        field = by_variant['field_plain']
        for variant, row in by_variant.items():
            if variant.startswith(('field_shuffled_', 'random_matched_')):
                item = {**{k: job[k] for k in ('run_id', 'fold', 'budget')}, 'control': variant}
                # The core provides C0-Cr; use that same quantity for paired controls.
                key = next((k for k in ('relative_improvement', 'C0_minus_Cr', 'delta_vs_ce') if k in row), None)
                if key and row[key] != '' and field.get(key, '') != '':
                    item['field_minus_control_improvement'] = float(field[key]) - float(row[key])
                else:
                    item['field_minus_control_improvement'] = ''
                controls.append(item)
    return {'results': results, 'probes': probes, 'controls': controls}


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root, suite)
    for name, rows, columns in (('per_unit_probes.csv', data['probes'], ['run_id', 'fold', 'budget', 'variant']),
                               ('control_comparisons.csv', data['controls'], ['run_id', 'fold', 'budget', 'control', 'field_minus_control_improvement'])):
        if rows:
            _write_table(root / name, rows)
        else:
            write_csv(root / name, rows, fieldnames=columns)
    lines = ['# Geometry mechanistic diagnostic', '',
             f"Completed units: {len(data['results'])}/10 (five folds × two fixed budgets).", '',
             'This is a mechanistic pilot, not classification performance, accuracy evaluation, or OOF.',
             'B is the predetermined final support patient; A contains the other K−1 patients. There is no B rotation.',
             'Only A supplies warmup/calibration data and geometry targets. Outer evaluation images are never probed.',
             'The heuristic excludes boundary-connected near-gray fill, retains internal dark holes and blue marks, and is not tissue ground truth.',
             'The geometry head is calibrated using legacy_plain, which can bias the basepoint toward that branch.',
             'Virtual displacements are fixed SGD-style direction probes, not reproductions of AdamW updates.',
             'C0−Cr measures improvement over the common CE-only displacement, not decrease from the unperturbed basepoint.',
             'Zero-pair or near-zero/nonfinite gradients are invalid and skipped; no threshold tuning or deployment gate is inferred.',
             'No trained classifier or image/teacher-map/weight payload is exported.', '',
             'See per_unit_probes.csv for all branch deltas and control_comparisons.csv for field-versus-control differences.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    suite.update(completed_runs=len(data['results']), status='completed' if len(data['results']) == 10 else 'partial',
                 aggregate_sha256={n: sha256_file(root / n) for n in AGGREGATES})
    _atomic_json(root / 'suite.json', suite)
    return data


def _check_current_source(suite):
    require(source_hashes() == suite['identity']['source_sha256'], 'Resume rejected: source changed')


def run_suite(args):
    root = guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    check_output_target(root)
    if (root / 'suite.json').exists():
        _check_current_source(load_suite(root))
    base.check_dependencies()
    configure_torch()
    import torch
    runtime = environment(require_cuda=not args.smoke)
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(root):
        rows, suite = prepare(root, args.image_root, None if args.smoke else args.reference_root,
                              cache_root=args.cache_root, smoke=args.smoke)
        key = runtime_key(runtime)
        require(suite['runtime_identity'] in (None, key), 'Runtime changed; use a new output root')
        suite['runtime_identity'] = key
        jobs = selected_jobs(suite, args)
        by_name = {r['image_name']: r for r in rows}
        preflight_cache(root, suite, jobs, by_name, args.cache_root)
        suite.setdefault('attempts', []).append({'at_utc': now(), 'folds': args.folds, 'budgets': args.budgets})
        _atomic_json(root / 'suite.json', suite)
        try:
            for i, job in enumerate(jobs, 1):
                print(f"[{i}/{len(jobs)}] {job['run_id']}", flush=True)
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
        require(suite['identity']['config']['mode'] == 'formal', 'Smoke results cannot be packaged as formal results')
        jobs = selected_jobs(suite, args)
        require(all(verify_complete_job(root, suite, j) for j in jobs), 'Selected packing scope is incomplete')
        data = write_aggregates(root, suite)
        manifest = {'version': VERSION, 'created_utc': now(), 'selected_run_ids': [j['run_id'] for j in jobs],
                    'planned_runs': 10, 'selected_completed_runs': len(jobs), 'scope_complete': True,
                    'entire_suite_complete': len(data['results']) == 10,
                    'included_completed_run_ids': [r['run_id'] for r in data['results']],
                    'reference_identity_sha256': suite['identity']['reference']['identity_sha256'],
                    'aggregate_scope': 'all completed units; selection explicitly listed',
                    'interpretation': 'mechanistic diagnostic; not accuracy/OOF',
                    'excluded': ['images', 'NPZ teacher maps', 'model weights', 'third-party code']}
        _atomic_json(root / 'package_scope.json', manifest)
        names = ['suite.json', 'package_scope.json', *suite['plan_artifact_sha256'], *AGGREGATES]
        names.extend(f'{REFERENCE_DIR}/{n}' for n in suite['identity']['reference']['artifact_sha256'])
        for result in data['results']:
            names.extend(f"runs/{result['run_id']}/{n}" for n in RUN_FILES)
        require(all(Path(n).suffix in ('.json', '.csv', '.md') for n in names), 'Non-text packing artifact')
        archive = root.parent / f'{root.name}_diagnostic_{time.time_ns()}_results.tar.gz'
        with tarfile.open(archive, 'x:gz') as out:
            for name in sorted(set(names)):
                out.add(safe_file(root, name), arcname=name, recursive=False)
        Path(str(archive) + '.sha256').write_text(sha256_file(archive) + '  ' + archive.name + '\n', encoding='utf-8')
        return archive


def parser():
    p = argparse.ArgumentParser(description='Ten-unit frozen geometry mechanistic diagnostic; no accuracy/OOF claims.')
    p.add_argument('mode', choices=['check', 'run', 'status', 'pack'])
    p.add_argument('--image-root', type=Path, default=Path('dataset/renew'))
    p.add_argument('--cache-root', type=Path, default=Path('outputs/renew_geometry_teacher_cache'))
    p.add_argument('--reference-root', type=Path, default=Path('outputs/renew_geometry_fewshot'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/renew_geometry_diagnostic'))
    p.add_argument('--budgets', nargs='+', choices=['5', '10'])
    p.add_argument('--folds', nargs='+', type=int, choices=range(5))
    p.add_argument('--smoke', action='store_true', help='CPU, random weights, two updates per stage; cannot formal-pack')
    return p


def next_command(args, mode):
    parts = ['bash', 'scripts/run_geometry_diagnostic_server.sh', mode, '--output-root', str(args.output_root)]
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
    args.output_root = guard_roots(args.output_root, args.image_root, args.cache_root, args.reference_root)
    check_output_target(args.output_root)
    if args.mode == 'run':
        run_suite(args)
        print('Next: ' + next_command(args, 'status'))
    elif args.mode == 'pack':
        print(pack_suite(args.output_root, args))
        print('Checksum sidecar written; retain the original complete output tree.')
    elif args.mode == 'status':
        suite = load_suite(args.output_root)
        data = aggregate(args.output_root, suite)
        jobs = selected_jobs(suite, args)
        done = sum(verify_complete_job(args.output_root, suite, j) is not None for j in jobs)
        print(f"Completed {len(data['results'])}/10 units; selected {done}/{len(jobs)}. Diagnostic only; not accuracy/OOF.")
        args.smoke = suite['identity']['config']['mode'] == 'smoke'
        for label in ('image', 'cache', 'reference'):
            if label in suite.get('protected_roots', {}):
                setattr(args, label + '_root', Path(suite['protected_roots'][label]))
        if args.smoke and done == len(jobs):
            print('Smoke scope complete; formal packing is disabled. Use a new output root for formal runs.')
        else:
            print('Next: ' + next_command(args, 'pack' if done == len(jobs) else 'run'))
    else:
        base.check_dependencies()
        # Existing corrupt/foreign output is rejected before acquiring a lock.
        if (args.output_root / 'suite.json').exists():
            _check_current_source(load_suite(args.output_root))
        with output_lock(args.output_root):
            rows, suite = prepare(args.output_root, args.image_root,
                None if args.smoke else args.reference_root, cache_root=args.cache_root, smoke=args.smoke)
        jobs = selected_jobs(suite, args)
        print(f"PASS: {len(rows)} images / 31 patients; ten planned units; {len(jobs)} selected; mode={suite['identity']['config']['mode']}.")
        print('Next: ' + next_command(args, 'run'))


if __name__ == '__main__':
    main()
