"""Synthetic, bounded CPU tests for the independent mechanistic diagnostic.

No private cohort, trained classifier, external teacher or network access is used.
The historical reference fixture is fabricated and its dataset audit is patched
only inside these tests; production validation remains strict.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src import geometry_diagnostic_core as core
from src import geometry_diagnostic as runner
from src import geometry_distillation as geometry
from src import geometry_experiment as legacy
from src import geometry_replication as replication
from src.renew_data import audit_dataset
from tests import test_geometry_replication as reference_fixtures


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyStudent(nn.Module):
    """Same parameter boundaries as the student, without costly ResNet fitting."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Tanh())
        self.layer4 = nn.Conv2d(4, 4, 1)
        self.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(4, 3))
        self.geometry_head = nn.Conv2d(4, 1, 1)

    def forward(self, image, return_geometry=False):
        features = torch.nn.functional.adaptive_avg_pool2d(image, (14, 14))
        features = self.layer3(self.layer2(self.layer1(self.bn1(self.conv1(features)))))
        logits = self.fc(self.layer4(features).mean((2, 3)))
        return (logits, self.geometry_head(features)) if return_geometry else logits


def model_state(model):
    return {
        'tensors': {k: v.detach().clone() for k, v in model.state_dict().items()},
        'modes': {k: v.training for k, v in model.named_modules()},
        'requires_grad': {k: v.requires_grad for k, v in model.named_parameters()},
    }


def assert_state_equal(expected, model):
    actual = model_state(model)
    assert actual['modes'] == expected['modes']
    assert actual['requires_grad'] == expected['requires_grad']
    assert set(actual['tensors']) == set(expected['tensors'])
    assert all(torch.equal(expected['tensors'][k], v) for k, v in actual['tensors'].items())


def bordered_image():
    rgb = np.full((80, 120, 3), (160, 70, 45), dtype=np.uint8)
    rgb[:12] = 15
    rgb[-12:] = 15
    rgb[:, :16] = 15
    rgb[:, -16:] = 15
    rgb[32:48, 48:64] = 15  # enclosed dark hole, never exterior
    rgb[10:17, 30:40] = (0, 0, 255)  # blue marker remains eligible
    return rgb


def test_exterior_is_boundary_connected_near_15_with_four_connectivity():
    rgb = bordered_image()
    original = rgb.copy()
    exterior = core.exterior_mask(rgb)
    assert exterior.dtype == np.bool_
    assert exterior.shape == rgb.shape[:2]
    assert exterior[0, 0] and exterior[5, 50] and exterior[40, 1]
    assert not exterior[40, 56], 'An enclosed dark airway is not background'
    assert not exterior[12, 35], 'Blue marks are not near-(15,15,15)'
    assert not exterior[30, 30]
    assert np.array_equal(rgb, original)
    tiny = np.full((5, 5, 3), 100, dtype=np.uint8)
    tiny[0, 0] = 15
    tiny[1, 1] = 15
    tiny[0, 3] = (13, 17, 15)
    tiny[0, 4] = (18, 15, 15)
    tiny[4, 0] = 0
    mask = core.exterior_mask(tiny)
    assert mask[0, 0] and mask[0, 3]
    assert not mask[1, 1], 'Diagonal adjacency must not flood an internal pixel'
    assert not mask[0, 4] and not mask[4, 0], 'Dark alone is insufficient'


@pytest.mark.parametrize('angle', [-15, 0, 15])
def test_rgb_is_exact_legacy_transform_and_field_mask_is_coregistered(angle):
    image = Image.fromarray(bordered_image())
    values = np.random.default_rng(17).uniform(.1, .9, (5, 24, 32)).astype(np.float32)
    actual_rgb, actual_maps, actual_valid, field_valid = core.transform_geometry(image, values, angle=angle)
    # Independently project the raw field through the same full-frame transform.
    field = (~core.exterior_mask(np.asarray(image))).astype(np.float32)
    expanded = np.stack([np.asarray(Image.fromarray(m).resize(image.size, Image.Resampling.BILINEAR))
                         for m in values] + [field])
    expected_rgb, expected_maps, expected_valid = geometry.joint_geometry(image, expanded, train=True, angle=angle)
    legacy_rgb, legacy_maps, legacy_valid = geometry.joint_geometry(image, values, train=True, angle=angle)
    assert np.array_equal(np.asarray(actual_rgb), np.asarray(legacy_rgb))
    assert np.array_equal(np.asarray(actual_rgb), np.asarray(expected_rgb))
    assert torch.equal(actual_maps, legacy_maps)
    assert torch.equal(actual_valid, legacy_valid)
    assert torch.equal(field_valid, expected_valid & (expected_maps[-1] > .999))
    assert not (field_valid & ~actual_valid).any()
    assert 0 < field_valid.sum() < actual_valid.sum()


def relation_batch(batch_size=2, *, empty=False):
    generator = torch.Generator().manual_seed(81)
    target = torch.rand(batch_size, 1, 14, 14, generator=generator).expand(-1, 5, -1, -1).clone()
    valid = torch.ones(batch_size, 14, 14, dtype=torch.bool)
    field = valid.clone()
    field[:, :3] = False
    field[:, :, :3] = False
    if empty:
        valid.zero_()
        field.zero_()
    return {'image': torch.zeros(batch_size, 3, 14, 14), 'geometry': target, 'valid': valid, 'field_valid': field,
            'sample_seed': [123 + i for i in range(batch_size)],
            'rows': [{'image_name': f'{i}lt.png', 'image_sha256': f'{i + 1:064x}'}
                     for i in range(batch_size)], 'mask_metadata': [{} for _ in range(batch_size)]}


def test_pair_controls_preserve_positions_sign_counts_and_per_image_random_counts():
    batch = relation_batch()
    for b in range(2):
        selected = core.select_pairs(batch['geometry'][b], batch['valid'][b],
                                     batch['field_valid'][b], sample_seed=batch['sample_seed'][b])
        field, plain = selected['field_plain'], selected['legacy_plain']
        assert 0 < len(field['i']) < len(plain['i']) <= geometry.PAIR_COUNT
        legacy_positions = set(zip(plain['i'].tolist(), plain['j'].tolist()))
        assert set(zip(field['i'].tolist(), field['j'].tolist())) <= legacy_positions
        for seed in (42, 43, 44):
            shuffled = selected[f'field_shuffled_{seed}']
            random = selected[f'random_matched_{seed}']
            assert torch.equal(shuffled['i'], field['i']) and torch.equal(shuffled['j'], field['j'])
            assert torch.equal(shuffled['labels'].sort().values, field['labels'].sort().values)
            assert (shuffled['labels'] != field['labels']).any()
            assert len(random['i']) == len(field['i'])
            assert set(zip(random['i'].tolist(), random['j'].tolist())) <= legacy_positions
        again = core.select_pairs(batch['geometry'][b], batch['valid'][b],
                                  batch['field_valid'][b], sample_seed=batch['sample_seed'][b])
        for key in selected:
            assert torch.equal(selected[key]['labels'], again[key]['labels'])
            assert torch.equal(selected[key]['i'], again[key]['i'])


def test_plain_loss_is_exact_historical_mean_per_image_including_empty_images():
    batch = relation_batch()
    batch['valid'][1].zero_()
    batch['field_valid'][1].zero_()
    prediction = torch.linspace(-1, 1, 392).reshape(2, 1, 14, 14).requires_grad_()
    actual, stats = core.diagnostic_relation_loss(prediction, batch, 'legacy_plain')
    expected, expected_stats = geometry.relation_loss(prediction, batch['geometry'], batch['valid'])
    assert torch.allclose(actual, expected, rtol=0, atol=1e-7)
    assert stats['selected_pairs'] == expected_stats['selected_pairs'] > 0
    for variant in core.VARIANTS:
        loss, empty_stats = core.diagnostic_relation_loss(prediction, relation_batch(empty=True), variant)
        assert loss.requires_grad and loss.item() == 0 and empty_stats['selected_pairs'] == 0
        gradient, = torch.autograd.grad(loss, prediction)
        assert torch.equal(gradient, torch.zeros_like(prediction))


def test_real_cpu_student_preserves_classification_initialization_and_rng():
    torch.manual_seed(44)
    expected = geometry.build_resnet18_student(pretrained=False)
    expected_rng = torch.rand(7)
    torch.manual_seed(44)
    actual = core.build_student(pretrained=False)
    assert torch.equal(torch.rand(7), expected_rng)
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name


@pytest.mark.parametrize('raises', [False, True])
def test_student_wrapper_restores_all_cuda_rng_states_even_if_builder_raises(monkeypatch, raises):
    cuda_states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    before = [v.clone() for v in cuda_states]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [v.clone() for v in cuda_states])

    def restore(states):
        cuda_states[:] = [v.clone() for v in states]

    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', restore)

    def builder(**kwargs):
        cuda_states[0].add_(10)
        cuda_states[1].add_(20)
        if raises:
            raise RuntimeError('injected builder failure')
        return nn.Linear(1, 1)

    monkeypatch.setattr(core, 'build_resnet18_student', builder)
    if raises:
        with pytest.raises(RuntimeError, match='injected builder failure'):
            core.build_student(pretrained=False)
    else:
        core.build_student(pretrained=False)
    assert all(torch.equal(a, b) for a, b in zip(cuda_states, before))


@pytest.mark.parametrize('raises', [False, True])
def test_virtual_displacement_exact_control_norm_and_restoration_on_exception(raises):
    model = nn.Sequential(nn.Linear(2, 1, bias=False), nn.BatchNorm1d(1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2., -1.]]))
    model.train()
    model[0].eval()  # mixed modes must be restored individually
    original = model_state(model)
    parameter = model[0].weight
    ce = [torch.tensor([[3., 4.]])]
    aux = [torch.tensor([[0., -2.]])]
    seen = []

    def loss_fn():
        seen.append(parameter.detach().clone())
        model[1].running_mean.add_(11)
        if raises:
            raise RuntimeError('injected B loss failure')
        return parameter.square().sum()

    if raises:
        with pytest.raises(RuntimeError, match='injected B loss failure'):
            core.evaluate_virtual_loss(model, [parameter], ce, aux, .25, .025, loss_fn)
    else:
        result = core.evaluate_virtual_loss(model, [parameter], ce, aux, .25, .025, loss_fn)
        expected = original['tensors']['0.weight'] - .25 * ce[0] / 5 - .025 * aux[0] / 2
        assert torch.allclose(seen[-1], expected, atol=1e-7)
        assert result == pytest.approx(float(expected.square().sum()), abs=1e-7)
        control = core.evaluate_virtual_loss(model, [parameter], ce, None, .25, 0, loss_fn)
        assert torch.norm(original['tensors']['0.weight'] - seen[-1]).item() == pytest.approx(.25)
        assert control == pytest.approx(float(seen[-1].square().sum()))
    assert_state_equal(original, model)


@pytest.mark.parametrize('bad', [0., 1e-10, float('nan'), float('inf')])
def test_virtual_displacement_never_normalizes_empty_nearzero_or_nonfinite_gradient(bad):
    model = nn.Linear(1, 1, bias=False)
    original = model_state(model)
    with pytest.raises(ValueError):
        core.evaluate_virtual_loss(model, [model.weight], [torch.ones_like(model.weight)],
                                   [torch.full_like(model.weight, bad)], .1, .01,
                                   lambda: model.weight.square().sum())
    assert_state_equal(original, model)


@pytest.fixture(scope='module')
def synthetic_reference_template():
    """Use the already audited 45-run fixture writer; no historical data is read."""
    factory = reference_fixtures.GeometryReplicationTests
    factory.setUpClass()
    yield factory.template_root
    factory.tearDownClass()


@pytest.fixture
def diagnostic_fixture(tmp_path, monkeypatch, synthetic_reference_template):
    for name in ('images', 'cache', 'reference'):
        shutil.copytree(synthetic_reference_template / name, tmp_path / name)
    rows, metadata = audit_dataset(tmp_path / 'images', strict=False)
    for module in (legacy, replication, runner):
        if hasattr(module, 'audit_dataset'):
            monkeypatch.setattr(module, 'audit_dataset', lambda *a, **kw: (rows, metadata))
        if hasattr(module, 'EXPECTED_FINGERPRINT'):
            monkeypatch.setattr(module, 'EXPECTED_FINGERPRINT', metadata['dataset_fingerprint'])
    return SimpleNamespace(root=tmp_path, images=tmp_path / 'images', cache=tmp_path / 'cache',
                           reference=tmp_path / 'reference', output=tmp_path / 'diagnostic',
                           rows=rows, metadata=metadata, by_name={r['image_name']: r for r in rows})


def prepare_fixture(fixture, *, smoke=False):
    return runner.prepare(fixture.output, fixture.images, fixture.reference,
                          cache_root=fixture.cache, smoke=smoke)[1]


def tree_hashes(root):
    return {str(p.relative_to(root)): legacy.sha256_file(p)
            for p in root.rglob('*') if p.is_file()}


def test_plan_is_ten_frozen_original_supports_with_last_patient_only_in_B(diagnostic_fixture):
    f = diagnostic_fixture
    before = tree_hashes(f.reference)
    suite = prepare_fixture(f)
    jobs = suite['identity']['jobs']
    originals = legacy.read_json(f.reference / 'suite.json')['identity']['jobs']
    assert len(jobs) == 10
    assert {(j['fold'], j['budget']) for j in jobs} == {(i, b) for i in range(5) for b in ('5', '10')}
    assert suite['identity']['config']['training_seed'] == 44
    for job in jobs:
        old_job = next(j for j in originals if j['run_id'] == job['reference_run_id'])
        assert old_job['method'] == 'plain'
        assert job['support'] == old_job['support']
        assert job['B_patient'] == old_job['patient_order_prefix'][-1]
        assert len(job['B']) == 3
        assert len(job['A']) == (int(job['budget']) - 1) * 3
        assert set(job['A']) | set(job['B']) == set(job['support'])
        assert not set(job['A']) & set(job['B'])
        assert not set(job['support']) & set(old_job['eval'])
        assert {f.by_name[n]['patient_key'] for n in job['B']} == {str(job['B_patient'])}
        assert str(job['B_patient']) not in {f.by_name[n]['patient_key'] for n in job['A']}
    assert set(suite['identity']['source_sha256']) == set(replication.SOURCE_FILES) | {
        'src/geometry_diagnostic_core.py', 'src/geometry_diagnostic.py',
        'scripts/run_geometry_diagnostic.py', 'scripts/run_geometry_diagnostic_server.sh'}
    assert before == tree_hashes(f.reference)


def test_wrong_nonempty_overlapping_and_symlink_output_roots_rejected_without_write(diagnostic_fixture):
    f = diagnostic_fixture
    for protected in (f.images, f.cache, f.reference):
        before = tree_hashes(protected)
        for target in (protected, protected / 'nested'):
            with pytest.raises(ValueError):
                runner.prepare(target, f.images, f.reference, cache_root=f.cache)
        assert before == tree_hashes(protected)
    with pytest.raises(ValueError):
        runner.prepare(f.root, f.images, f.reference, cache_root=f.cache)
    f.output.mkdir()
    (f.output / 'unrelated.txt').write_text('preserve me')
    before = tree_hashes(f.output)
    with pytest.raises(ValueError):
        prepare_fixture(f)
    assert before == tree_hashes(f.output)
    alias = f.root / 'alias'
    alias.symlink_to(f.reference, target_is_directory=True)
    before = tree_hashes(f.reference)
    with pytest.raises(ValueError):
        runner.prepare(alias, f.images, f.reference, cache_root=f.cache)
    assert before == tree_hashes(f.reference)
    # The old suite must be rejected even if a distinct reference copy is passed.
    alternate = f.root / 'alternate_reference'
    shutil.copytree(f.reference, alternate)
    with pytest.raises(ValueError):
        runner.prepare(f.reference, f.images, alternate, cache_root=f.cache)
    assert before == tree_hashes(f.reference)


def test_resume_rejects_plan_source_and_reference_corruption(diagnostic_fixture, monkeypatch):
    f = diagnostic_fixture
    suite = prepare_fixture(f)
    assert prepare_fixture(f)['identity_sha256'] == suite['identity_sha256']
    plan = f.output / 'unit_plan.csv'
    saved = plan.read_bytes()
    plan.write_bytes(saved + b'corrupt\n')
    with pytest.raises(ValueError):
        prepare_fixture(f)
    plan.write_bytes(saved)
    actual_hash = runner.sha256_file
    with monkeypatch.context() as scoped:
        scoped.setattr(runner, 'sha256_file', lambda p: '0' * 64
                       if str(p).endswith('src/geometry_diagnostic_core.py') else actual_hash(p))
        with pytest.raises(ValueError):
            prepare_fixture(f)
    history = next((f.reference / 'runs').glob('*/history.csv'))
    history.write_bytes(history.read_bytes() + b'corrupt\n')
    with pytest.raises(ValueError):
        prepare_fixture(f)


def test_rehashed_changed_A_teacher_map_is_rejected_against_old_binding(diagnostic_fixture):
    f = diagnostic_fixture
    suite = prepare_fixture(f)
    manifest = geometry.load_cache_manifest(f.cache)
    name = suite['identity']['jobs'][0]['A'][0]
    entry = manifest['entries'][f.by_name[name]['image_sha256']]
    path = f.cache / entry['file']
    with np.load(path, allow_pickle=False) as saved:
        maps, size = saved['disparity'].copy(), saved['original_size'].copy()
    maps[:] = maps[..., ::-1]
    np.savez_compressed(path, disparity=maps, original_size=size)
    entry['sha256'] = geometry.file_hash(path)
    geometry.atomic_json(f.cache / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        prepare_fixture(f)


def test_make_batch_B_is_rgb_only_and_does_not_open_teacher_cache(diagnostic_fixture, monkeypatch):
    f = diagnostic_fixture
    _, jobs = runner.make_plan(f.rows)
    B = [f.by_name[n] for n in jobs[0]['B']]

    def forbidden(*args, **kwargs):
        raise AssertionError('B teacher targets must not be loaded')

    monkeypatch.setattr(geometry, '_read_entry', forbidden)
    monkeypatch.setattr(core, '_read_entry', forbidden)
    batch = core.make_batch(B, f.root / 'nonexistent-cache', need_geometry=False)
    assert batch['image'].shape == (3, 3, 224, 224)
    assert batch['label'].tolist() == [r['label_id'] for r in B]


def tiny_batches(fixture):
    _, jobs = runner.make_plan(fixture.rows)
    job = jobs[0]
    A = [fixture.by_name[n] for n in job['A']]
    B = [fixture.by_name[n] for n in job['B']]
    return job, A, B


def test_real_tiny_cpu_warmup_calibration_and_probes_use_only_A_and_restore_every_state(diagnostic_fixture, monkeypatch):
    f = diagnostic_fixture
    job, A, B = tiny_batches(f)
    torch.manual_seed(44)
    model = TinyStudent()
    initial = model_state(model)
    observed = []
    make_batch = core.make_batch

    def observe(rows, *args, **kwargs):
        rows = list(rows)
        observed.extend(r['image_name'] for r in rows)
        return make_batch(rows, *args, **kwargs)

    monkeypatch.setattr(core, 'make_batch', observe)
    config = core.fixed_config(smoke=True)
    result = core.prepare_model(model, A, f.cache, config, torch.device('cpu'))
    assert observed == job['A']
    assert not set(observed) & set(job['B'])
    assert [h['stage'] for h in result['history']] == ['head', 'head', 'calibration', 'calibration']
    assert all(h['image_occurrences'] == 16 and np.isfinite(h['loss']) for h in result['history'])
    evidence = result['evidence']
    assert evidence['actual_head_steps'] == evidence['actual_calibration_steps'] == 2
    assert evidence['preparation_image_names'] == job['A']
    for name, tensor in model.state_dict().items():
        if not name.startswith(('fc.', 'geometry_head.')):
            assert torch.equal(tensor, initial['tensors'][name]), name
    assert any(not torch.equal(v, initial['tensors'][k]) for k, v in model.state_dict().items() if k.startswith('fc.'))
    assert any(not torch.equal(v, initial['tensors'][k]) for k, v in model.state_dict().items() if k.startswith('geometry_head.'))
    assert evidence['backbone_unchanged'] and evidence['calibration_fc_unchanged'] and evidence['calibration_bn_unchanged']
    a_batch = make_batch(A, f.cache)
    b_batch = make_batch(B, None, need_geometry=False)
    model.train()
    model.layer2.eval()
    original = model_state(model)
    measured = core.probe(model, a_batch, b_batch, config)
    assert_state_equal(original, model)
    assert measured == core.probe(model, a_batch, b_batch, config), 'All branches must start at an identical deterministic basepoint'
    assert_state_equal(original, model)
    assert measured['evidence']['state_restored']
    expected_shared = sum(p.numel() for n, p in model.named_parameters()
                          if n.startswith(('conv1.', 'layer1.', 'layer2.', 'layer3.')))
    assert measured['evidence']['shared_parameter_count'] == expected_shared
    assert len(measured['records']) == len(core.VARIANTS) == 8
    assert all(row['valid'] for row in measured['records'])
    controls = {row['ce_control_B_loss'] for row in measured['records']}
    assert len(controls) == 1
    for row in measured['records']:
        assert row['relative_improvement'] == pytest.approx(row['ce_control_B_loss'] - row['variant_B_loss'])
        assert row['auxiliary_displacement_norm'] == pytest.approx(.1 * row['ce_displacement_norm'])
        assert row['weighted_auxiliary_gradient_norm'] == pytest.approx(.1 * row['auxiliary_gradient_norm'])
        assert -1.000001 <= row['cosine_aux_ce_A'] <= 1.000001
        assert -1.000001 <= row['cosine_aux_ce_B'] <= 1.000001
    # Failure inside a branch must restore the outer probe's modes and trainability.
    def fail(*args, **kwargs):
        raise RuntimeError('virtual failure')
    monkeypatch.setattr(core, 'evaluate_virtual_loss', fail)
    with pytest.raises(RuntimeError, match='virtual failure'):
        core.probe(model, a_batch, b_batch, config)
    assert_state_equal(original, model)


def test_probe_explicitly_skips_zero_auxiliary_gradient_and_empty_pairs(diagnostic_fixture):
    f = diagnostic_fixture
    _, A, B = tiny_batches(f)
    torch.manual_seed(44)
    model = TinyStudent()
    with torch.no_grad():
        model.geometry_head.weight.zero_()
    a_batch = core.make_batch(A, f.cache)
    b_batch = core.make_batch(B, None, need_geometry=False)
    original = model_state(model)
    measured = core.probe(model, a_batch, b_batch, core.fixed_config(smoke=True))
    assert_state_equal(original, model)
    for row in measured['records']:
        assert row['selected_pairs'] > 0
        assert not row['valid'] and row['skip_reason'] == 'invalid_auxiliary_gradient'
        assert row['auxiliary_gradient_norm'] == 0
        assert row['variant_B_loss'] is None and row['relative_improvement'] is None
        assert row['auxiliary_displacement_norm'] is None
    empty_batch = core.make_batch(A, f.cache)
    empty_batch['valid'].zero_()
    empty_batch['field_valid'].zero_()
    measured = core.probe(model, empty_batch, b_batch, core.fixed_config(smoke=True))
    assert all(not r['valid'] and r['skip_reason'] == 'zero_pairs' and r['relative_improvement'] is None
               for r in measured['records'])


def test_probe_rejects_A_B_patient_leakage_and_nonzero_probe_angle(diagnostic_fixture):
    f = diagnostic_fixture
    _, A, B = tiny_batches(f)
    a_batch = core.make_batch(A, f.cache)
    b_batch = core.make_batch(B, None, need_geometry=False)
    model = TinyStudent()
    original = model_state(model)
    overlap = dict(b_batch, rows=[dict(B[0], patient_key=A[0]['patient_key']), *B[1:]])
    with pytest.raises(ValueError, match='overlap'):
        core.probe(model, a_batch, overlap, core.fixed_config(smoke=True))
    with pytest.raises(ValueError, match='angle'):
        core.probe(model, dict(a_batch, angle=15), b_batch, core.fixed_config(smoke=True))
    assert_state_equal(original, model)


def save_runtime(fixture, suite):
    # This is a fabricated reference-compatible runtime for provenance tests.
    suite['runtime_identity'] = legacy.read_json(fixture.reference / 'suite.json')['runtime_identity']
    legacy._atomic_json(fixture.output / 'suite.json', suite)


def remark_completed(root, suite, job):
    directory = root / 'runs' / job['run_id']
    legacy._atomic_json(directory / 'complete.json', {
        'identity_sha256': suite['identity_sha256'], 'run_id': job['run_id'],
        'artifact_sha256': {name: legacy.sha256_file(directory / name) for name in runner.RUN_FILES[:-1]},
    })


def test_runner_real_tiny_smoke_archives_interruption_resumes_and_rejects_semantic_tamper(diagnostic_fixture):
    f = diagnostic_fixture
    suite = prepare_fixture(f, smoke=True)
    save_runtime(f, suite)
    job = suite['identity']['jobs'][0]
    directory = f.output / 'runs' / job['run_id']
    directory.mkdir(parents=True)
    (directory / 'interrupted.txt').write_text('partial attempt must be retained')
    built = []

    def factory(pretrained):
        assert pretrained is False, 'Synthetic smoke must never request pretrained weights'
        built.append(True)
        return TinyStudent()

    result = runner.train_one(f.output, suite, job, f.by_name, torch.device('cpu'),
                              f.cache, model_factory=factory)
    assert len(built) == 1
    assert result['history_rows'] == 4 and result['probe_rows'] == 8
    assert result['pair_audit_rows'] == 3 * 8 * len(job['A'])
    assert list((f.output / 'incomplete_attempts').glob('*/interrupted.txt'))
    assert runner.verify_complete_job(f.output, suite, job) == result
    assert runner.load_suite(f.output)['identity_sha256'] == suite['identity_sha256']
    before = tree_hashes(directory)
    assert runner.train_one(f.output, suite, job, f.by_name, torch.device('cpu'),
                             f.cache, model_factory=factory) == result
    assert len(built) == 1 and before == tree_hashes(directory)
    assert not list(f.output.rglob('*.pt')), 'The diagnostic does not export a trained classifier'
    with pytest.raises(ValueError, match='Smoke'):
        runner.pack_suite(f.output, runner.parser().parse_args(['pack', '--folds', '0', '--budgets', '5']))
    # A valid digest does not excuse changing the meaning of a completed result.
    path = directory / 'result.json'
    original = path.read_bytes()
    wrong = copy.deepcopy(result)
    wrong['evidence']['preparation']['preparation_image_names'][-1] = job['B'][0]
    legacy._atomic_json(path, wrong)
    remark_completed(f.output, suite, job)
    with pytest.raises(ValueError, match='A-only'):
        runner.verify_complete_job(f.output, suite, job)
    path.write_bytes(original)
    remark_completed(f.output, suite, job)
    probes = legacy.read_csv(directory / 'probes.csv')
    probes[0]['relative_improvement'] = str(float(probes[0]['relative_improvement']) + .01)
    runner._write_table(directory / 'probes.csv', probes)
    remark_completed(f.output, suite, job)
    with pytest.raises(ValueError, match='delta'):
        runner.verify_complete_job(f.output, suite, job)


def write_synthetic_formal_job(fixture, suite, job):
    """Fabricate formal evidence for archive tests, never run a formal fit.

    Numerical rows come from a real CPU tiny smoke, while the 140-step history
    and CUDA provenance are explicitly simulated fixture data. This exercises
    the offline validator without weights, a GPU, or patient images.
    """
    f = fixture
    torch.manual_seed(44)
    model = TinyStudent()
    A, B = [f.by_name[n] for n in job['A']], [f.by_name[n] for n in job['B']]
    preparation = core.prepare_model(model, A, f.cache, core.fixed_config(smoke=True), torch.device('cpu'))
    config = suite['identity']['config']
    stage_template = {row['stage']: row for row in preparation['history']}
    history = [dict(stage_template[stage], update=i + 1)
               for stage, count in [('head', config['head_steps']), ('calibration', config['calibration_steps'])]
               for i in range(count)]
    preparation['evidence'].update(actual_head_steps=config['head_steps'],
                                   actual_calibration_steps=config['calibration_steps'],
                                   rng={'cuda_rng_restored': True, 'cuda_device_count': 1,
                                        'cuda_before_sha256': ['5' * 64], 'cuda_after_sha256': ['5' * 64]})
    a_batch = core.make_batch(A, f.cache)
    b_batch = core.make_batch(B, None, need_geometry=False)
    probes = core.probe(model, a_batch, b_batch, config)
    pairs = []
    for angle in (-15, 0, 15):
        batch = a_batch if angle == 0 else core.make_batch(A, f.cache, angle=angle)
        pairs.extend(dict(row, angle=angle) for row in core.audit_pairs(batch))
    directory = f.output / 'runs' / job['run_id']
    directory.mkdir(parents=True)
    binding = suite['identity']['cache_bindings'][job['run_id']]
    legacy._atomic_json(directory / 'config.json', {
        'job': job, 'config': config, 'identity_sha256': suite['identity_sha256'],
        'cache_binding': binding, 'execution_runtime': suite['runtime_identity'],
    })
    runner._write_table(directory / 'history.csv', history)
    runner._write_table(directory / 'pair_audit.csv', pairs)
    runner._write_table(directory / 'probes.csv', probes['records'])
    result = {**{key: job[key] for key in ('run_id', 'fold', 'budget', 'repeat')},
              'cache_binding': binding, 'history_rows': len(history),
              'pair_audit_rows': len(pairs), 'probe_rows': len(probes['records']),
              'n_A_images': len(A), 'n_B_images': len(B),
              'evidence': {'preparation': preparation['evidence'], 'probe': probes['evidence']}}
    legacy._atomic_json(directory / 'result.json', result)
    remark_completed(f.output, suite, job)
    assert runner.verify_complete_job(f.output, suite, job) == result
    return directory


def test_selected_and_full_pack_are_self_contained_text_only_and_detect_corruption(diagnostic_fixture, monkeypatch, capsys):
    f = diagnostic_fixture
    suite = prepare_fixture(f)
    save_runtime(f, suite)
    selected_args = runner.parser().parse_args(['pack', '--folds', '0', '--budgets', '5'])
    with pytest.raises(ValueError, match='incomplete'):
        runner.pack_suite(f.output, selected_args)
    jobs = suite['identity']['jobs']
    selected = next(j for j in jobs if j['fold'] == 0 and j['budget'] == '5')
    write_synthetic_formal_job(f, suite, selected)
    runner.pack_suite(f.output, selected_args)
    scope = legacy.read_json(f.output / 'package_scope.json')
    assert scope['scope_complete'] and not scope['entire_suite_complete']
    assert scope['planned_runs'] == 10 and scope['selected_completed_runs'] == 1
    assert scope['selected_run_ids'] == [selected['run_id']]
    assert legacy.read_json(f.output / 'suite.json')['status'] != 'completed'
    for job in jobs:
        if job != selected:
            write_synthetic_formal_job(f, suite, job)
    for original_root in (f.images, f.cache, f.reference):
        shutil.rmtree(original_root)
    for name in ('private.png', 'teacher.npz', 'model.pt', 'third_party.py'):
        (f.output / name).write_bytes(b'these raw or executable files must never be packaged')

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline validation must not inspect live source/weights/cache')

    monkeypatch.setattr(runner, 'source_hashes', forbidden)
    monkeypatch.setattr(core, 'build_student', forbidden)
    monkeypatch.setattr(core, 'make_batch', forbidden)
    args = runner.parser().parse_args(['pack'])
    archive = runner.pack_suite(f.output, args)
    assert Path(str(archive) + '.sha256').read_text().split()[0] == legacy.sha256_file(archive)
    extracted = f.root / 'extracted'
    extracted.mkdir()
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        assert all(Path(name).suffix in ('.json', '.csv', '.md') for name in names)
        reference_names = [n for n in names if n.startswith(runner.REFERENCE_DIR + '/')]
        assert len(reference_names) == 243
        assert len([n for n in names if n.startswith('runs/') and n.endswith('/complete.json')]) == 10
        packed.extractall(extracted, filter='data')
    assert legacy.read_json(extracted / 'package_scope.json')['entire_suite_complete']
    assert legacy.read_json(extracted / 'suite.json')['completed_runs'] == 10
    runner.main(['status', '--output-root', str(extracted)])
    assert '10/10' in capsys.readouterr().out
    runner.pack_suite(extracted, args)
    reference_result = next((extracted / runner.REFERENCE_DIR / 'runs').glob('*/result.json'))
    reference_result.write_text('{}\n')
    with pytest.raises(ValueError):
        runner.main(['status', '--output-root', str(extracted)])


def test_pair_audit_reports_actual_counts_fractions_and_legacy_normalization():
    batch = relation_batch(batch_size=1)
    selected = core.select_pairs(batch['geometry'][0], batch['valid'][0], batch['field_valid'][0], 123)
    reports = core.audit_pairs(batch)
    assert len(reports) == 8
    candidates = torch.triu_indices(196, 196, offset=1)
    order = torch.randperm(candidates.shape[1], generator=torch.Generator().manual_seed(geometry.PAIR_SEED))[:1024]
    assert torch.equal(core.candidate_pairs(), candidates[:, order])
    low, high = torch.quantile(batch['geometry'][0, 0].flatten(), torch.tensor([.05, .95]))
    for report in reports:
        variant = report['variant']
        actual = selected[variant]
        count = len(actual['i'])
        assert report['scale'] == pytest.approx(float(high - low))
        assert report['selected_pairs'] == report['image_pair_denominator'] == count
        assert report['batch_image_denominator'] == 1
        assert report['near_pairs_before_eligibility'] + report['far_pairs_before_eligibility'] == 1024
        assert report['near_pairs_selected'] + report['far_pairs_selected'] == count
        assert report['positive_pairs'] + report['negative_pairs'] == count
        assert report['selected_to_legacy_fraction'] == pytest.approx(count / len(selected['legacy_plain']['i']))
        if variant.startswith('field_shuffled_'):
            changed = int((actual['labels'] != selected['field_plain']['labels']).sum())
            assert report['shuffled_changed_pairs'] == changed
            assert report['shuffled_changed_fraction'] == pytest.approx(changed / count)
        if variant.startswith('field_'):
            assert report['selected_touch_border_pairs'] == 0
    constant = torch.ones(5, 14, 14)
    none = core.select_pairs(constant, batch['valid'][0], batch['field_valid'][0])
    assert all(not value['audit']['valid'] and len(value['i']) == 0 for value in none.values())


def test_formal_resume_requires_same_new_runtime_before_starting_training(diagnostic_fixture, monkeypatch):
    f = diagnostic_fixture
    suite = prepare_fixture(f)
    save_runtime(f, suite)
    changed = dict(suite['runtime_identity'], gpu='another-synthetic-GPU')
    requested_cuda = []

    def runtime(require_cuda=False):
        requested_cuda.append(require_cuda)
        return changed

    monkeypatch.setattr(runner, 'environment', runtime)
    monkeypatch.setattr(runner, 'configure_torch', lambda: None)
    monkeypatch.setattr(legacy, 'check_dependencies', lambda: None)
    with patch.object(runner, 'train_one') as train:
        args = runner.parser().parse_args([
            'run', '--output-root', str(f.output), '--image-root', str(f.images),
            '--reference-root', str(f.reference), '--cache-root', str(f.cache),
            '--folds', '0', '--budgets', '5'])
        with pytest.raises(ValueError, match='Runtime changed'):
            runner.run_suite(args)
        train.assert_not_called()
    assert requested_cuda == [True]


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf')])
def test_nonfinite_auxiliary_loss_is_invalid_strict_json_without_state_leak(monkeypatch, nonfinite):
    import json
    a_batch = relation_batch()
    a_batch['image'] = torch.rand(2, 3, 14, 14, generator=torch.Generator().manual_seed(6))
    a_batch['label'] = torch.tensor([0, 1])
    a_batch['rows'] = [dict(row, patient_key='A') for row in a_batch['rows']]
    b_batch = dict(a_batch, rows=[dict(row, patient_key='B') for row in a_batch['rows']])
    model = TinyStudent()
    original = model_state(model)
    actual_loss = core.diagnostic_relation_loss

    def nonfinite_loss(*args, **kwargs):
        loss, stats = actual_loss(*args, **kwargs)
        return loss * nonfinite, stats

    monkeypatch.setattr(core, 'diagnostic_relation_loss', nonfinite_loss)
    result = core.probe(model, a_batch, b_batch, core.fixed_config(smoke=True))
    assert_state_equal(original, model)
    json.dumps(result, allow_nan=False)
    assert len(result['records']) == 8
    for row in result['records']:
        assert not row['valid'] and row['skip_reason'] == 'nonfinite_auxiliary_loss'
        assert row['geometry_A_loss'] is None and row['auxiliary_gradient_norm'] is None
        assert row['variant_B_loss'] is None and row['relative_improvement'] is None
