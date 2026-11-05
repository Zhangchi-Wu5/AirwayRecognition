"""Synthetic protocol/integrity tests and real tiny-network training; no patient data."""
import copy
import hashlib
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from src import geometry_experiment as g
from src.renew_data import audit_dataset


class GeometrySuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.images = self.base / 'images'
        self.images.mkdir()
        for patient in range(1, 32):
            for label, name in enumerate(('lt', 'yz', 'zz')):
                image = Image.new('RGB', (32, 18))
                image.putdata([((patient * 5 + x * 3) % 256, (label * 70 + y * 7) % 256, (x * y + 90) % 256) for y in range(18) for x in range(32)])
                image.save(self.images / f'{patient}{name}.png')
        self.rows, self.metadata = audit_dataset(self.images, strict=False)
        self.output = self.base / 'output'
        self.audit_patch = patch.object(g, 'audit_dataset', return_value=(self.rows, self.metadata))
        self.audit_patch.start()
        self.rows, self.suite = g.prepare(self.output, self.images)
        self.suite['runtime_identity'] = {'synthetic': True}
        g._atomic_json(self.output / 'suite.json', self.suite)

    def tearDown(self):
        self.audit_patch.stop()
        self.tmp.cleanup()

    def full(self):
        return [j for j in self.suite['identity']['jobs'] if j['budget'] == 'full']

    def fixture(self, job):
        import numpy as np
        d = self.output / 'runs' / job['run_id']
        d.mkdir(parents=True)
        by_name = {r['image_name']: r for r in self.rows}
        rows = [by_name[n] for n in job['eval']]
        probs = np.asarray([[.8 if i == r['label_id'] else .1 for i in range(3)] for r in rows], dtype=np.float32)
        binding = None if job['method'] == 'ce' else {'teacher_sha256': 'same-teacher', 'entries': {}}
        c = self.suite['identity']['config']
        g._atomic_json(d / 'config.json', {'identity_sha256': self.suite['identity_sha256'], 'job': job, 'config': c,
            'initial_state_sha256': 'initial', 'cache_binding': binding, 'execution_runtime': self.suite['runtime_identity']})
        hist = [{'stage': s, 'round': i + 1, 'steps': 28, 'image_occurrences': 448,
            'lr': .001 if s == 'head' else .0001 * (1 + math.cos(math.pi * i / 20)) / 2,
            'mean_train_loss': .5, 'classification_loss': .5, 'geometry_loss': 0., 'seconds': .001}
            for s, count in (('head', 5), ('finetune', 20)) for i in range(count)]
        g.write_csv(d / 'history.csv', hist)
        predictions = [{'patient_id': r['patient_id'], 'image_name': r['image_name'], 'label_id': r['label_id'],
            'predicted_id': int(p.argmax()), **{f'prob_{label}': float(p[i]) for label, i in g.LABEL_TO_ID.items()}}
            for r, p in zip(rows, probs)]
        g.write_csv(d / 'predictions.csv', predictions)
        (d / 'model.pt').write_bytes(b'synthetic checkpoint fixture')
        (d / 'student.pt').write_bytes(b'synthetic RGB-only fixture')
        result = {**{k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method')},
            'n_train_images': len(job['support']), 'n_train_patients': len(job['patient_order_prefix']),
            'initial_state_sha256': 'initial', 'cache_binding': binding, 'bn_buffers_unchanged': True,
            'head_stage_backbone_unchanged': True, 'finetune_backbone_changed': True,
            'bn_before_sha256': 'bn', 'bn_after_sha256': 'bn', 'initial_backbone_sha256': 'before', 'final_backbone_sha256': 'after',
            'final_state_sha256': 'final', 'reloaded_state_sha256': 'final', 'checkpoint_reload_verified': True,
            'student_export_sha256': g.sha256_file(d / 'student.pt'), 'student_export_verified': True,
            'actual_optimizer_steps': 700, 'actual_image_occurrences': 11200,
            **g.classification_metrics([r['label_id'] for r in rows], probs)}
        g._atomic_json(d / 'result.json', result)
        self.remark(d, job)
        return d

    def remark(self, d, job):
        g._atomic_json(d / 'complete.json', {'run_id': job['run_id'], 'identity_sha256': self.suite['identity_sha256'],
            'artifact_sha256': {n: g.sha256_file(d / n) for n in (*g.RUN_FILES[:-1], 'model.pt')}})

    def test_plan_matched_supports_nesting_and_no_leakage(self):
        jobs = self.suite['identity']['jobs']
        self.assertEqual(len(jobs), 45)
        self.assertEqual(len(self.full()), 5)
        for fold in range(5):
            few = [j for j in jobs if j['fold'] == fold and j['budget'] == '5']
            self.assertEqual(len(few), 4)
            self.assertTrue(all(j['support'] == few[0]['support'] for j in few))
            larger = next(j for j in jobs if j['fold'] == fold and j['budget'] == '10')
            self.assertTrue(set(few[0]['support']) < set(larger['support']))
            self.assertFalse(set(larger['patient_order_prefix']) & {r['patient_key'] for r in self.rows if r['image_name'] in larger['eval']})
        self.assertEqual(g.fixed_config()['geometry_phase'], 'finetune_only')

    def test_lock_initial_plan_cli_and_dependency_check(self):
        other = self.base / 'new'
        g.main(['check', '--image-root', str(self.images), '--output-root', str(other)])
        self.assertTrue((other / 'suite.json').is_file())
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            g.prepare(self.images / 'bad', self.images)

    def test_resume_and_plan_tamper_rejection(self):
        g.prepare(self.output, self.images)
        with patch.object(g, 'sha256_file', return_value='changed'), self.assertRaisesRegex(ValueError, 'Resume rejected'):
            g.prepare(self.output, self.images)
        (self.output / self.full()[0]['support_csv']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Plan file changed'):
            g.prepare(self.output, self.images)

    def test_completed_metrics_recomputed_even_if_marker_rehashed(self):
        job = self.full()[0]
        d = self.fixture(job)
        self.assertIsNotNone(g.validate_run(self.output, self.suite, job))
        r = g.read_json(d / 'result.json')
        r['accuracy'] = .123
        g._atomic_json(d / 'result.json', r)
        self.remark(d, job)
        with self.assertRaisesRegex(ValueError, 'Prediction metric mismatch'):
            g.validate_run(self.output, self.suite, job)

    def test_partial_oof_and_offline_pack_scope(self):
        self.fixture(self.full()[0])
        self.assertEqual(g.aggregate(self.output, self.suite)['oof'], [])
        args = g.parser().parse_args(['pack', '--phase', 'full'])
        with patch.object(g, 'EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint']), self.assertRaisesRegex(ValueError, 'incomplete'):
            g.pack_suite(self.output, args)
        for job in self.full()[1:]:
            self.fixture(job)
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['oof']), 1)
        self.assertEqual(len(data['predictions']), 93)
        shutil.rmtree(self.images)
        for p in self.output.glob('runs/*/*.pt'):
            p.unlink()
        with patch.object(g, 'EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint']):
            archive = g.pack_suite(self.output, args)
        with tarfile.open(archive) as f:
            names = f.getnames()
            self.assertFalse(any(n.endswith(('.pt', '.png', '.npz')) for n in names))
            self.assertIn('package_scope.json', names)
        scope = g.read_json(self.output / 'package_scope.json')
        self.assertFalse(scope['entire_suite_complete'])
        self.assertEqual(scope['selected_completed_runs'], 5)

    def test_all_jobs_aggregate_has_nine_oof_groups(self):
        for j in self.suite['identity']['jobs']:
            self.fixture(j)
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['oof']), 9)
        self.assertEqual(len(data['paired']), 6)
        self.assertEqual(len(data['results']), 45)

    def test_real_tiny_training_all_methods_bn_export_and_teacher_binding(self):
        import numpy as np
        import torch
        from src.geometry_distillation import cache_predictions
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Conv2d(3, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4)
                self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(4, 3))
                with torch.random.fork_rng():
                    torch.manual_seed(20261105)
                    self.geometry_head = torch.nn.Conv2d(4, 1, 1)
            def forward(self, x, return_geometry=False):
                features = self.bn(self.stem(torch.nn.functional.adaptive_avg_pool2d(x, (14, 14)))).relu()
                logits = self.fc(features.mean((2, 3)))
                return (logits, self.geometry_head(features)) if return_geometry else logits
        suite = copy.deepcopy(self.suite)
        suite['identity']['config'] = g.fixed_config(smoke=True)
        suite['identity_sha256'] = g._digest(suite['identity'])
        by_name = {r['image_name']: r for r in self.rows}
        jobs = [j for j in suite['identity']['jobs'] if j['fold'] == 0 and j['budget'] == '5']
        cache = self.base / 'cache'
        selected = [by_name[n] for n in jobs[0]['support']]
        cache_predictions(selected, cache, lambda im: np.tile(np.linspace(.1, .9, 16, dtype=np.float32), (8, 1)), {'fixture': True})
        results = []
        for job in jobs:
            result = g.train_one(self.output, suite, job, by_name, torch.device('cpu'), cache, model_factory=lambda _: Tiny())
            self.assertTrue(result['bn_buffers_unchanged'])
            self.assertTrue(result['student_export_verified'])
            hist = g.read_csv(self.output / 'runs' / job['run_id'] / 'history.csv')
            if job['method'] != 'ce':
                self.assertGreater(float(hist[-1]['aux_gradient_l1']), 0)
            results.append(result)
            g.train_one(self.output, suite, job, by_name, torch.device('cpu'), cache, model_factory=lambda _: Tiny())
            student = torch.load(self.output / 'runs' / job['run_id'] / 'student.pt', weights_only=True)
            self.assertFalse(any(k.startswith('geometry_head.') for k in student))
        self.assertEqual(len({r['initial_state_sha256'] for r in results}), 1)
        self.assertIsNone(results[0]['cache_binding'])
        # A cache modified after completion must not silently resume.
        (cache / 'manifest.json').write_text('{}')
        with self.assertRaises((ValueError, KeyError)):
            g.train_one(self.output, suite, jobs[1], by_name, torch.device('cpu'), cache, model_factory=lambda _: Tiny())


if __name__ == '__main__':
    unittest.main()
