"""Patient-rotation orchestration tests using explicitly synthetic evidence.

The formal archive fixture simulates schedule/CUDA records and relabels tiny
CPU numerical templates. It is not a formal training result. A separate smoke
test really prepares and probes a fresh TinyStudent for each new B.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import statistics
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import geometry_diagnostic as diagnostic
from src import geometry_diagnostic_core as core
from src import geometry_distillation as geometry
from src import geometry_experiment as legacy
from src import geometry_replication as replication
from src import geometry_rotation as rotation
from src.renew_data import audit_dataset
from tests import test_geometry_diagnostic as fixtures
from tests import test_geometry_replication as historical_fixtures


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def install_synthetic_audit(monkeypatch, rows, metadata):
    for module in (legacy, replication, diagnostic, rotation):
        if hasattr(module, 'audit_dataset'):
            monkeypatch.setattr(module, 'audit_dataset', lambda *a, **k: (rows, metadata))
        if hasattr(module, 'EXPECTED_FINGERPRINT'):
            monkeypatch.setattr(module, 'EXPECTED_FINGERPRINT', metadata['dataset_fingerprint'])


def synthetic_job(target_root, suite, job, template_directory):
    """Rebind an explicitly fabricated same-budget fixture, never train a model.

    All template images have equal dimensions and the same synthetic teacher.
    Only provenance semantics are tested by these rows; their scalar values do
    not represent the rebound patient/model. Real numerical behavior is covered
    separately by actual TinyStudent smoke runs.
    """
    target = Path(target_root) / 'runs' / job['run_id']
    target.mkdir(parents=True)
    original = legacy.read_json(template_directory / 'config.json')['job']
    assert len(original['A']) == len(job['A'])
    rename = dict(zip(original['A'], job['A']))
    by_name = {r['image_name']: r for r in suite['identity']['data']}
    binding = suite['identity']['cache_bindings'][job['run_id']]
    config = suite['identity']['config']
    legacy._atomic_json(target / 'config.json', {
        'job': job, 'config': config, 'identity_sha256': suite['identity_sha256'],
        'cache_binding': binding, 'execution_runtime': suite['runtime_identity'],
    })
    history = legacy.read_csv(template_directory / 'history.csv')
    history = [r for r in history if int(r['update']) <= config['head_steps' if r['stage'] == 'head' else 'calibration_steps']]
    pairs = legacy.read_csv(template_directory / 'pair_audit.csv')
    for r in pairs:
        r['image_name'] = rename[r['image_name']]
        r['patient_key'] = by_name[r['image_name']]['patient_key']
    probes = legacy.read_csv(template_directory / 'probes.csv')
    for name, values in [('history.csv', history), ('pair_audit.csv', pairs), ('probes.csv', probes)]:
        diagnostic._write_table(target / name, values)
    result = legacy.read_json(template_directory / 'result.json')
    result.update({k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat')})
    result.update(cache_binding=binding, history_rows=len(history))
    prep = result['evidence']['preparation']
    prep.update(preparation_image_names=job['A'], actual_head_steps=config['head_steps'],
                actual_calibration_steps=config['calibration_steps'])
    legacy._atomic_json(target / 'result.json', result)
    fixtures.remark_completed(Path(target_root), suite, job)
    diagnostic.verify_complete_job(target_root, suite, job)
    return target


@pytest.fixture(scope='module')
def rotation_template(tmp_path_factory):
    factory = historical_fixtures.GeometryReplicationTests
    previous_threads = torch.get_num_threads()
    factory.setUpClass()
    root = tmp_path_factory.mktemp('rotation_synthetic_template')
    try:
        for name in ('images', 'cache', 'reference'):
            shutil.copytree(factory.template_root / name, root / name)
        rows, metadata = audit_dataset(root / 'images', strict=False)
        with pytest.MonkeyPatch.context() as monkeypatch:
            install_synthetic_audit(monkeypatch, rows, metadata)
            f = SimpleNamespace(root=root, images=root / 'images', cache=root / 'cache',
                                reference=root / 'reference', output=root / 'diagnostic',
                                rows=rows, metadata=metadata, by_name={r['image_name']: r for r in rows})
            suite = fixtures.prepare_fixture(f)
            fixtures.save_runtime(f, suite)
            for job in suite['identity']['jobs']:
                fixtures.write_synthetic_formal_job(f, suite, job)
            diagnostic.write_aggregates(f.output, suite)
            smoke = root / 'diagnostic_smoke'
            _, smoke_suite = diagnostic.prepare(smoke, f.images, f.reference, cache_root=f.cache, smoke=True)
            smoke_suite['runtime_identity'] = suite['runtime_identity']
            legacy._atomic_json(smoke / 'suite.json', smoke_suite)
            for job in smoke_suite['identity']['jobs']:
                synthetic_job(smoke, smoke_suite, job, f.output / 'runs' / job['run_id'])
            diagnostic.write_aggregates(smoke, smoke_suite)
        yield root
    finally:
        factory.tearDownClass()
        torch.set_num_threads(previous_threads)


@pytest.fixture
def data(tmp_path, monkeypatch, rotation_template):
    for name in ('images', 'cache', 'reference', 'diagnostic', 'diagnostic_smoke'):
        shutil.copytree(rotation_template / name, tmp_path / name)
    rows, metadata = audit_dataset(tmp_path / 'images', strict=False)
    install_synthetic_audit(monkeypatch, rows, metadata)
    return SimpleNamespace(root=tmp_path, images=tmp_path / 'images', cache=tmp_path / 'cache',
                           historical=tmp_path / 'reference', reference=tmp_path / 'diagnostic',
                           smoke_reference=tmp_path / 'diagnostic_smoke', output=tmp_path / 'rotation',
                           rows=rows, metadata=metadata, by_name={r['image_name']: r for r in rows})


def prepare(data, *, smoke=False):
    return rotation.prepare(data.output, data.images,
                            data.smoke_reference if smoke else data.reference,
                            cache_root=data.cache, smoke=smoke)[1]


def old_ids(data):
    return {j['run_id'] for j in legacy.read_json(data.reference / 'suite.json')['identity']['jobs']}


def new_jobs(data, suite):
    return [j for j in suite['identity']['jobs'] if j['run_id'] not in old_ids(data)]


def args_for(data, mode, *extra):
    return rotation.parser().parse_args([
        mode, '--output-root', str(data.output), '--image-root', str(data.images),
        '--cache-root', str(data.cache), '--reference-root', str(data.reference), *extra])


def fabricate_new(data, suite, jobs):
    templates = legacy.read_json(data.reference / 'suite.json')['identity']['jobs']
    for job in jobs:
        original = next(j for j in templates if j['fold'] == job['fold'] and j['budget'] == job['budget'])
        synthetic_job(data.output, suite, job, data.reference / 'runs' / original['run_id'])


def test_plan_rotates_every_patient_once_preserving_exact_ten_imported_jobs(data):
    folds, jobs = rotation.make_plan(data.rows)
    old_folds, originals = diagnostic.make_plan(data.rows)
    assert folds == old_folds
    assert len(jobs) == len({j['run_id'] for j in jobs}) == 75
    assert sum(j['run_id'] in old_ids(data) for j in jobs) == 10
    assert sum(j['run_id'] not in old_ids(data) for j in jobs) == 65
    for old in originals:
        group = [j for j in jobs if j['fold'] == old['fold'] and j['budget'] == old['budget']]
        assert len(group) == int(old['budget'])
        assert [j['B_patient'] for j in group] == old['patient_order_prefix']
        assert group[-1] == old, 'The imported source job must remain byte-for-byte equivalent as data'
        for job in group:
            assert job['support'] == old['support'] and job['eval'] == old['eval']
            assert len(job['B']) == 3 and len(job['A']) == 3 * (int(job['budget']) - 1)
            assert set(job['A']) | set(job['B']) == set(job['support'])
            assert not set(job['A']) & set(job['B'])
            assert not set(job['support']) & set(job['eval'])
            assert job['A'] == [n for n in old['support'] if data.by_name[n]['patient_key'] != job['B_patient']]
            assert job['A_patients'] == [p for p in old['patient_order_prefix'] if p != job['B_patient']]


def test_preparation_imports_ten_as_immutable_old_identity_not_new_runs(data):
    before = fixtures.tree_hashes(data.reference)
    suite = prepare(data)
    assert len(suite['identity']['jobs']) == 75
    assert suite['identity']['config'] == core.fixed_config()
    assert suite['identity']['source_sha256'] == diagnostic.source_hashes()
    assert len(suite['identity']['source_sha256']) == 17
    assert len(suite['identity']['orchestration_source_sha256']) == 3
    assert suite['runtime_identity'] == legacy.read_json(data.reference / 'suite.json')['runtime_identity']
    copied = data.output / rotation.REFERENCE_DIR
    assert len(fixtures.tree_hashes(copied)) == 339
    for name, digest in fixtures.tree_hashes(copied).items():
        assert before[name] == digest
    original = legacy.read_json(data.reference / 'suite.json')
    assert original['identity_sha256'] != suite['identity_sha256']
    for job in suite['identity']['jobs']:
        result = rotation.verify_complete_job(data.output, suite, job)
        if job['run_id'] in old_ids(data):
            assert result == diagnostic.verify_complete_job(data.reference, original, job)
            marker = legacy.read_json(copied / 'runs' / job['run_id'] / 'complete.json')
            assert marker['identity_sha256'] == original['identity_sha256']
            assert not (data.output / 'runs' / job['run_id']).exists()
        else:
            assert result is None
    assert before == fixtures.tree_hashes(data.reference)
    assert prepare(data)['identity_sha256'] == suite['identity_sha256']


def test_bad_output_paths_are_rejected_before_any_write(data):
    for protected in (data.images, data.cache, data.reference):
        before = fixtures.tree_hashes(protected)
        for root in (protected, protected / 'child', data.root):
            with pytest.raises(ValueError):
                rotation.prepare(root, data.images, data.reference, cache_root=data.cache)
        assert before == fixtures.tree_hashes(protected)
    data.output.mkdir()
    (data.output / 'unrelated.txt').write_text('preserve')
    before = fixtures.tree_hashes(data.output)
    with pytest.raises(ValueError):
        prepare(data)
    assert before == fixtures.tree_hashes(data.output)
    alias = data.root / 'alias'
    alias.symlink_to(data.reference, target_is_directory=True)
    with pytest.raises(ValueError):
        rotation.prepare(alias, data.images, data.reference, cache_root=data.cache)
    alternative = data.root / 'another_reference'
    shutil.copytree(data.reference, alternative)
    before = fixtures.tree_hashes(data.reference)
    with pytest.raises(ValueError):
        rotation.prepare(data.reference, data.images, alternative, cache_root=data.cache)
    assert before == fixtures.tree_hashes(data.reference)


def test_missing_old_unit_or_corrupt_snapshot_and_plan_are_rejected(data):
    old = next((data.reference / 'runs').glob('*/complete.json'))
    original = old.read_bytes()
    old.unlink()
    with pytest.raises(ValueError):
        prepare(data)
    old.write_bytes(original)
    suite = prepare(data)
    plan = data.output / 'unit_plan.csv'
    original = plan.read_bytes()
    plan.write_bytes(original + b'corrupt\n')
    with pytest.raises(ValueError):
        rotation.load_suite(data.output)
    plan.write_bytes(original)
    copied = next((data.output / rotation.REFERENCE_DIR / 'runs').glob('*/probes.csv'))
    copied.write_bytes(copied.read_bytes() + b'corrupt\n')
    with pytest.raises(ValueError):
        rotation.load_suite(data.output)


def test_resume_rejects_rehashed_changed_teacher_and_live_data(data):
    suite = prepare(data)
    job = new_jobs(data, suite)[0]
    row = data.by_name[job['A'][0]]
    manifest = geometry.load_cache_manifest(data.cache)
    entry = manifest['entries'][row['image_sha256']]
    path = data.cache / entry['file']
    original, original_manifest = path.read_bytes(), (data.cache / 'manifest.json').read_bytes()
    with np.load(path, allow_pickle=False) as saved:
        maps, size = saved['disparity'].copy(), saved['original_size'].copy()
    np.savez_compressed(path, disparity=maps[..., ::-1], original_size=size)
    entry['sha256'] = geometry.file_hash(path)
    geometry.atomic_json(data.cache / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        prepare(data)
    path.write_bytes(original)
    (data.cache / 'manifest.json').write_bytes(original_manifest)
    image = Path(row['path'])
    image.write_bytes(image.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='image|Image|source|Source'):
        rotation.train_one(data.output, suite, job, data.by_name, torch.device('cpu'),
                           data.cache, model_factory=lambda pretrained: pytest.fail('must not construct model'))


@pytest.mark.parametrize('module', [rotation, diagnostic], ids=['new_orchestration', 'frozen_numerics'])
def test_resume_checks_both_source_manifests(data, monkeypatch, module):
    prepare(data)
    changed = dict(module.source_hashes())
    changed[next(iter(changed))] = '0' * 64
    monkeypatch.setattr(module, 'source_hashes', lambda: changed)
    with pytest.raises(ValueError, match='[Ss]ource'):
        prepare(data)


def test_runtime_change_is_rejected_before_any_new_model(data, monkeypatch):
    suite = prepare(data)
    monkeypatch.setattr(rotation, 'environment', lambda require_cuda=False: dict(suite['runtime_identity'], gpu='different GPU'))
    monkeypatch.setattr(rotation, 'configure_torch', lambda: None)
    monkeypatch.setattr(legacy, 'check_dependencies', lambda: None)
    with patch.object(rotation, 'train_one') as train:
        with pytest.raises(ValueError, match='[Rr]untime'):
            rotation.run_suite(args_for(data, 'run', '--folds', '0', '--budgets', '5'))
        train.assert_not_called()


def test_complete_unit_before_summary_write_can_resume_but_rehashed_wrong_summary_cannot(data):
    suite = prepare(data)
    rotation.write_aggregates(data.output, suite)
    assert suite['completed_runs'] == 10
    job = new_jobs(data, suite)[0]
    fabricate_new(data, suite, [job])
    # Simulate a crash after complete.json but before publishing a new summary.
    loaded = rotation.load_suite(data.output)
    assert loaded['completed_runs'] == 10
    assert len(rotation.aggregate(data.output, loaded)['results']) == 11
    rotation.write_aggregates(data.output, loaded)
    assert rotation.load_suite(data.output)['completed_runs'] == 11
    path = data.output / 'fold_budget_summary.csv'
    rows = legacy.read_csv(path)
    rows[0]['mean'] = str(float(rows[0]['mean']) + .1)
    diagnostic._write_table(path, rows)
    loaded['aggregate_sha256'][path.name] = legacy.sha256_file(path)
    legacy._atomic_json(data.output / 'suite.json', loaded)
    with pytest.raises(ValueError, match='Aggregate values'):
        rotation.load_suite(data.output)


def test_interrupted_summary_write_has_scoped_readonly_recovery_and_wrong_identity_rejection(data, monkeypatch, capsys):
    suite = prepare(data)
    rotation.write_aggregates(data.output, suite)
    original_hash = suite['aggregate_sha256']['per_unit_probes.csv']
    selected = [j for j in new_jobs(data, suite) if j['fold'] == 0 and j['budget'] == '5']
    fabricate_new(data, suite, selected)
    actual_write = diagnostic._write_table

    def fail_after_first_csv(path, rows):
        actual_write(path, rows)
        if Path(path).name == 'per_unit_probes.csv':
            raise OSError('simulated interruption after first summary CSV')

    with monkeypatch.context() as scoped:
        scoped.setattr(diagnostic, '_write_table', fail_after_first_csv)
        with pytest.raises(OSError, match='simulated interruption'):
            rotation.write_aggregates(data.output, suite)
    assert legacy.sha256_file(data.output / 'per_unit_probes.csv') != original_hash
    marker = data.output / rotation.PENDING_FILE
    original_marker = marker.read_bytes()
    saved = legacy.read_json(marker)
    assert saved['identity_sha256'] == suite['identity_sha256']
    assert len(saved['aggregate_run_ids']) == 14
    saved['identity_sha256'] = '0' * 64
    legacy._atomic_json(marker, saved)
    with pytest.raises(ValueError):
        rotation.load_suite(data.output)
    marker.write_bytes(original_marker)
    before = fixtures.tree_hashes(data.output)
    loaded = rotation.load_suite(data.output)
    assert len(rotation.aggregate(data.output, loaded)['results']) == 14
    rotation.main(['status', '--output-root', str(data.output)])
    assert '14/75' in capsys.readouterr().out
    assert fixtures.tree_hashes(data.output) == before, 'Status must not repair or otherwise mutate evidence'
    archive = rotation.pack_suite(data.output, args_for(data, 'pack', '--folds', '0', '--budgets', '5'))
    assert not marker.exists()
    restored = rotation.load_suite(data.output)
    assert restored['completed_runs'] == 14
    with tarfile.open(archive) as packed:
        assert rotation.PENDING_FILE not in packed.getnames()


@pytest.mark.parametrize('invalid_variant', ['field_plain', 'field_shuffled_42'])
def test_invalid_branches_keep_planned_B_denominator_and_require_all_shuffled_controls(data, invalid_variant):
    suite = prepare(data)
    job = new_jobs(data, suite)[0]
    fabricate_new(data, suite, [job])
    directory = data.output / 'runs' / job['run_id']
    probes = legacy.read_csv(directory / 'probes.csv')
    row = next(r for r in probes if r['variant'] == invalid_variant)
    row.update(valid='False', skip_reason='invalid_auxiliary_gradient',
               auxiliary_gradient_norm='0', weighted_auxiliary_gradient_norm='0',
               weighted_auxiliary_to_ce_ratio='0', auxiliary_displacement_norm='',
               variant_B_loss='', relative_improvement='')
    diagnostic._write_table(directory / 'probes.csv', probes)
    fixtures.remark_completed(data.output, suite, job)
    rotation.verify_complete_job(data.output, suite, job)
    aggregated = rotation.aggregate(data.output, suite)
    unit = next(r for r in aggregated['comparisons'] if r['run_id'] == job['run_id'])
    assert unit['field_vs_shuffled_mean'] is None
    matching = [r for r in aggregated['comparisons'] if r['fold'] == job['fold'] and r['budget'] == job['budget']]
    valid_joint = [r for r in matching if r['field_vs_ce'] is not None and r['field_vs_shuffled_mean'] is not None]
    expected_positive = sum(r['field_vs_ce'] > 0 and r['field_vs_shuffled_mean'] > 0 for r in valid_joint)
    for group in aggregated['groups']:
        if group['fold'] != job['fold'] or group['budget'] != job['budget']:
            continue
        assert group['planned_B'] == group['joint_ce_shuffle_planned_B'] == 5
        assert group['completed_B'] == 2 and not group['group_complete']
        assert group['joint_ce_shuffle_valid_B'] == len(valid_joint) == 1
        assert group['joint_ce_shuffle_positive_B'] == expected_positive
        if group['comparison'] == 'field_vs_shuffled_mean':
            assert group['valid_B'] == 1 and group['invalid_B'] == 1


def test_real_tiny_smoke_uses_fresh_models_and_A_only_for_each_new_B(data, monkeypatch):
    suite = prepare(data, smoke=True)
    jobs = [j for j in new_jobs(data, suite) if j['fold'] == 0 and j['budget'] == '5'][:2]
    built, prepared, target_reads = [], [], []
    actual_prepare, actual_read = core.prepare_model, core._read_entry

    def factory(pretrained):
        assert pretrained is False
        model = fixtures.TinyStudent()
        built.append(model)
        return model

    def preparing(model, rows, *args, **kwargs):
        prepared.append([r['image_name'] for r in rows])
        return actual_prepare(model, rows, *args, **kwargs)

    def read(cache, row, *args, **kwargs):
        target_reads.append(row['image_name'])
        return actual_read(cache, row, *args, **kwargs)

    monkeypatch.setattr(core, 'prepare_model', preparing)
    monkeypatch.setattr(core, '_read_entry', read)
    for job in jobs:
        directory = data.output / 'runs' / job['run_id']
        directory.mkdir(parents=True)
        (directory / 'interrupted.txt').write_text('preserve incomplete')
        start = len(target_reads)
        result = rotation.train_one(data.output, suite, job, data.by_name, torch.device('cpu'), data.cache, model_factory=factory)
        assert result['history_rows'] == 4 and result['probe_rows'] == 8
        assert prepared[-1] == job['A']
        assert set(target_reads[start:]) == set(job['A'])
        assert not set(target_reads[start:]) & set(job['B'])
        assert rotation.verify_complete_job(data.output, suite, job) == result
        before = fixtures.tree_hashes(directory)
        assert rotation.train_one(data.output, suite, job, data.by_name, torch.device('cpu'), data.cache, model_factory=factory) == result
        assert fixtures.tree_hashes(directory) == before
    assert len(built) == 2 and built[0] is not built[1]
    assert len(list((data.output / 'incomplete_attempts').glob('*/interrupted.txt'))) == 2
    imported = next(j for j in suite['identity']['jobs'] if j['run_id'] in old_ids(data))
    rotation.train_one(data.output, suite, imported, data.by_name, torch.device('cpu'), data.cache,
                       model_factory=lambda p: pytest.fail('Imported old unit must never train'))
    assert not list(data.output.glob('runs/*/*.pt'))
    with pytest.raises(ValueError, match='[Ss]moke'):
        rotation.pack_suite(data.output, args_for(data, 'pack'))


def test_selected_then_full_offline_pack_keeps_import_provenance_and_excludes_raw_payloads(data, monkeypatch, capsys):
    suite = prepare(data)
    jobs = new_jobs(data, suite)
    selected = [j for j in jobs if j['fold'] == 0 and j['budget'] == '5']
    with pytest.raises(ValueError, match='incomplete|Incomplete'):
        rotation.pack_suite(data.output, args_for(data, 'pack', '--folds', '0', '--budgets', '5'))
    fabricate_new(data, suite, selected)
    rotation.pack_suite(data.output, args_for(data, 'pack', '--folds', '0', '--budgets', '5'))
    scope = legacy.read_json(data.output / 'package_scope.json')
    assert scope['planned_runs'] == 75 and scope['selected_completed_runs'] == 5
    assert scope['scope_complete'] and not scope['entire_suite_complete']
    assert scope['selected_imported_runs'] == 1 and scope['selected_new_runs'] == 4
    assert scope['included_imported_runs'] == 10 and scope['included_new_runs'] == 4
    fabricate_new(data, suite, [j for j in jobs if j not in selected])
    for root in (data.images, data.cache, data.reference, data.historical, data.smoke_reference):
        shutil.rmtree(root)
    for name in ('patient.png', 'teacher.npz', 'model.pt', 'external.py'):
        (data.output / name).write_bytes(b'excluded payload')

    def forbidden(*a, **k):
        raise AssertionError('Offline operations must not inspect live data, weights, or source hashes')

    monkeypatch.setattr(core, 'make_batch', forbidden)
    monkeypatch.setattr(core, 'build_student', forbidden)
    monkeypatch.setattr(diagnostic, 'source_hashes', forbidden)
    archive = rotation.pack_suite(data.output, args_for(data, 'pack'))
    assert Path(str(archive) + '.sha256').read_text().split()[0] == legacy.sha256_file(archive)
    extracted = data.root / 'extracted'
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        assert all(Path(n).suffix in ('.json', '.csv', '.md') for n in names)
        assert len([n for n in names if n.startswith('runs/') and n.endswith('/complete.json')]) == 65
        assert len([n for n in names if n.startswith(rotation.REFERENCE_DIR + '/runs/') and n.endswith('/complete.json')]) == 10
        assert len([n for n in names if n.startswith(rotation.REFERENCE_DIR + '/' + diagnostic.REFERENCE_DIR + '/')]) == 243
        packed.extractall(extracted, filter='data')
    saved = legacy.read_json(extracted / 'suite.json')
    scope = legacy.read_json(extracted / 'package_scope.json')
    assert saved['completed_runs'] == 75 and scope['entire_suite_complete']
    assert scope['selected_completed_runs'] == 75
    assert scope['included_imported_runs'] == scope['selected_imported_runs'] == 10
    assert scope['included_new_runs'] == scope['selected_new_runs'] == 65
    comparisons = legacy.read_csv(extracted / 'unit_comparisons.csv')
    groups = legacy.read_csv(extracted / 'fold_budget_summary.csv')
    assert len(comparisons) == 75
    for group in groups:
        selected_rows = [r for r in comparisons if r['fold'] == group['fold'] and r['budget'] == group['budget']]
        assert int(group['planned_B']) == int(group['completed_B']) == len(selected_rows) == int(group['budget'])
        assert group['group_complete'] == 'True'
        assert int(group['imported_B']) == 1 and int(group['new_B']) == int(group['budget']) - 1
        if group['comparison'] in rotation.COMPARISONS:
            values = [float(r[group['comparison']]) for r in selected_rows if r[group['comparison']] != '']
            assert int(group['valid_B']) == len(values)
            assert float(group['mean']) == pytest.approx(statistics.mean(values))
            assert float(group['median']) == pytest.approx(statistics.median(values))
            assert int(group['positive']) == sum(v > 0 for v in values)
            assert int(group['negative']) == sum(v < 0 for v in values)
    rotation.main(['status', '--output-root', str(extracted)])
    assert '75/75' in capsys.readouterr().out
    rotation.pack_suite(extracted, args_for(data, 'pack'))
    copied = next((extracted / rotation.REFERENCE_DIR / 'runs').glob('*/result.json'))
    copied.write_text('{}\n')
    with pytest.raises(ValueError):
        rotation.load_suite(extracted)
