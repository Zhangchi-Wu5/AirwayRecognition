"""Independent v2 protocol checks; all images/features/estimators are synthetic."""
from contextlib import ExitStack
import builtins
import copy
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src import frozen_screen as old_core
from src import frozen_screen_experiment as old
from src import frozen_linear_refinement as core
from src import frozen_linear_refinement_experiment as screen
from tests import test_frozen_screen_experiment as old_fixture


def tree_hashes(root):
    return {str(p.relative_to(root)): old.sha256_file(p)
            for p in root.rglob('*') if p.is_file()}


class LinearRefinementSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only fixture helpers, not the old TestCase's test methods.
        cls.fixture = old_fixture.FrozenScreenSuiteTests(methodName='runTest')
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.parent = cls.fixture.output
        cls.features = cls.fixture.prepare_synthetic_cache()
        runtime = screen.head_runtime()
        cls.fixture.suite['runtime_identity'] = {**runtime, 'synthetic_fixture': True}
        metadata = old.read_json(cls.parent / 'feature_cache/metadata.json')
        metadata['runtime_identity'] = cls.fixture.suite['runtime_identity']
        old._atomic_json(cls.parent / 'feature_cache/metadata.json', metadata)
        cls.fixture.remark_cache()
        original_fit = old.fit_classifier

        def old_reference_error(method, x, labels, patients):
            model, tuning = original_fit(method, x, labels, patients)
            if method == 'linear':
                # Deliberately fabricate two-class errors for paired accounting.
                model['coef'][0], model['coef'][1] = model['coef'][1], model['coef'][0]
            return model, tuning

        with patch.object(old_core, '_fit_linear', side_effect=cls.fixture.synthetic_linear):
            for job in cls.fixture.suite['identity']['jobs']:
                if job['phase'] == 'reference':
                    with patch.object(old, 'fit_classifier', side_effect=old_reference_error):
                        old.fit_one(cls.parent, cls.fixture.suite, job, cls.features)
                else:
                    old.fit_one(cls.parent, cls.fixture.suite, job, cls.features)
        old.write_aggregates(cls.parent, cls.fixture.suite)
        cls.parent_hashes = tree_hashes(cls.parent)
        cls.prepared = cls.fixture.base / 'v2_prepared'
        screen.prepare(cls.prepared, cls.parent)
        if tree_hashes(cls.parent) != cls.parent_hashes:
            raise AssertionError('Preparing v2 mutated the parent experiment')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.output = self.base / 'v2'
        shutil.copytree(self.prepared, self.output)
        self.suite = old.read_json(self.output / 'suite.json')

    def fake_fits(self):
        stack = ExitStack()
        stack.enter_context(patch.object(old_core, '_fit_linear', side_effect=self.fixture.synthetic_linear))
        if hasattr(core, '_fit_linear'):
            stack.enter_context(patch.object(core, '_fit_linear', side_effect=self.fixture.synthetic_linear))
        return stack

    def bind_runtime(self):
        self.suite['runtime_identity'] = screen.head_runtime()
        old._atomic_json(self.output / 'suite.json', self.suite)

    def remark_run(self, job):
        directory = self.output / 'runs' / job['run_id']
        marker = old.read_json(directory / 'complete.json')
        marker['artifact_sha256'] = {n: old.sha256_file(directory / n)
                                    for n in screen.RUN_FILES if n != 'complete.json'}
        old._atomic_json(directory / 'complete.json', marker)

    def test_plan_and_cache_exactly_match_parent_without_mutation(self):
        parent, features = screen.verify_artifacts(self.output, self.suite, require_arrays=True)
        expected = [j for j in parent['identity']['jobs'] if j['method'] == 'linear']
        self.assertEqual(self.suite['identity']['jobs'], expected)
        self.assertEqual(len(expected), 505)
        self.assertEqual(len({j['set_id'] for j in expected}), 101)
        np.testing.assert_array_equal(features, self.features)
        self.assertEqual(core.classifier_config()['linear']['C'], [.1, 1., 10., 100., 1000.])
        self.assertEqual(old_core.classifier_config()['linear']['C'], [.1, 1., 10.])
        self.assertEqual(tree_hashes(self.parent), self.parent_hashes)
        summary = screen.write_aggregates(self.output, self.suite)
        self.assertFalse(summary['results'])
        with self.assertRaises(ValueError):
            screen.pack_suite(self.output)

    def test_new_prepare_rejects_missing_or_corrupt_parent_cache(self):
        parent = self.base / 'parent'
        shutil.copytree(self.parent, parent)
        raw = parent / 'feature_cache/raw.npy'
        original = raw.read_bytes()
        raw.unlink()
        with self.assertRaises(ValueError):
            screen.prepare(self.base / 'missing_cache', parent)
        raw.write_bytes(original + b'corruption')
        with self.assertRaises(ValueError):
            screen.prepare(self.base / 'bad_cache', parent)

    def test_source_and_reference_snapshot_tampering_rejected(self):
        changed = copy.deepcopy(screen.source_hashes())
        first = next(iter(changed))
        changed[first] = '0' * 64
        with patch.object(screen, 'source_hashes', return_value=changed):
            with self.assertRaises(ValueError):
                screen.prepare(self.output, self.base / 'absent_parent')
        support = self.output / 'reference' / self.suite['identity']['jobs'][0]['support_csv']
        support.write_text('corrupted patient support\n')
        with self.assertRaises(ValueError):
            screen.verify_artifacts(self.output, self.suite, require_arrays=True)

    def test_support_only_fitting_resume_replay_and_shared_C_evidence(self):
        self.bind_runtime()
        job = self.suite['identity']['jobs'][0]
        rows = {r['image_name']: r for r in self.suite['identity']['data']}
        original_fit = screen.fit_classifier

        def spy(x, labels, patients):
            expected = [rows[n] for n in job['support']]
            np.testing.assert_array_equal(x, self.features[[r['source_index'] for r in expected]])
            self.assertEqual(labels, [r['label_id'] for r in expected])
            self.assertEqual(patients, [r['patient_key'] for r in expected])
            return original_fit(x, labels, patients)

        with self.fake_fits(), patch.object(screen, 'fit_classifier', side_effect=spy) as fit:
            result = screen.fit_one(self.output, self.suite, job, self.features)
        fit.assert_called_once()
        directory = self.output / 'runs' / job['run_id']
        self.assertEqual(len(old.read_csv(directory / 'tuning.csv')), 25)
        self.assertGreater(result['selected_C'], 10)
        with patch.object(screen, 'fit_classifier', side_effect=AssertionError('must resume')):
            self.assertEqual(result, screen.fit_one(self.output, self.suite, job, self.features))
        model = old.read_json(directory / 'model.json')
        model['coef'][0], model['coef'][1] = model['coef'][1], model['coef'][0]
        old._atomic_json(directory / 'model.json', model)
        self.remark_run(job)
        with self.assertRaisesRegex(ValueError, 'reproduce|replay'):
            screen.validate_run(self.output, self.suite, job, self.features)

    def test_shared_old_C_internal_losses_must_reproduce(self):
        self.bind_runtime()
        job = self.suite['identity']['jobs'][0]
        original = screen.fit_classifier

        def changed_loss(x, labels, patients):
            model, tuning = original(x, labels, patients)
            for t in tuning:
                if t['C'] == 1.:
                    t['val_log_loss'] += .01
                    t['mean_log_loss'] += .01
            for row in model['cv_scores']:
                if row['C'] == 1.:
                    row['mean_log_loss'] += .01
            return model, tuning

        with self.fake_fits(), patch.object(screen, 'fit_classifier', side_effect=changed_loss):
            with self.assertRaises(ValueError):
                screen.fit_one(self.output, self.suite, job, self.features)

    def test_cpu_only_run_and_runtime_change_rejection(self):
        args = SimpleNamespace(output_root=self.output, reference_root=self.base / 'absent_parent', phase='reference')
        original_import = builtins.__import__

        def cpu_import(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('CPU head runner must not import torch/CUDA')
            return original_import(name, *args, **kwargs)

        with self.fake_fits(), \
             patch.object(old, 'extract_features', side_effect=AssertionError('no extraction')), \
             patch.object(old_core, 'extract_features', side_effect=AssertionError('no extraction')), \
             patch.object(builtins, '__import__', side_effect=cpu_import):
            screen.run_suite(args)
        self.assertEqual(tree_hashes(self.parent), self.parent_hashes)
        changed = copy.deepcopy(screen.head_runtime())
        changed['imported_versions']['numpy'] = 'changed-version'
        with patch.object(screen, 'head_runtime', return_value=changed):
            with self.assertRaises(ValueError):
                screen.run_suite(args)

    def test_offline_runtime_provenance_is_checked_without_current_environment(self):
        self.bind_runtime()
        for path in self.output.rglob('*.npy'):
            path.unlink()
        with patch.object(screen, 'head_runtime', side_effect=AssertionError('offline runtime')):
            _, features = screen.verify_artifacts(self.output, self.suite, require_arrays=False)
            self.assertIsNone(features)
            bad_versions = copy.deepcopy(self.suite)
            bad_versions['runtime_identity']['imported_versions']['numpy'] = 'changed-version'
            bad_blas = copy.deepcopy(self.suite)
            bad_blas['runtime_identity']['numerical_libraries'][0]['version'] = 'changed-blas'
            bad_threads = copy.deepcopy(self.suite)
            bad_threads['runtime_identity']['thread_limit'] = 2
            for changed in (bad_versions, bad_blas, bad_threads):
                with self.assertRaises(ValueError):
                    screen.verify_artifacts(self.output, changed, require_arrays=False)

    def test_full_paired_summary_and_offline_package(self):
        self.bind_runtime()
        original_fit = screen.fit_classifier

        def v2_reference_error(x, labels, patients):
            model, tuning = original_fit(x, labels, patients)
            model['coef'][1], model['coef'][2] = model['coef'][2], model['coef'][1]
            return model, tuning

        with self.fake_fits():
            for job in self.suite['identity']['jobs']:
                if job['phase'] == 'reference':
                    with patch.object(screen, 'fit_classifier', side_effect=v2_reference_error):
                        screen.fit_one(self.output, self.suite, job, self.features)
                else:
                    screen.fit_one(self.output, self.suite, job, self.features)
        data = screen.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['results']), 505)
        self.assertEqual(len(data['oof']), 101)
        self.assertEqual(len(data['predictions']), 9393)
        self.assertTrue(all(r['n_draws'] == 100 for r in data['distributions']))
        self.assertTrue(all(r['mean'] == 1. for r in data['distributions'] if r['metric'] == 'accuracy'))
        # The fabricated reference has31 fixes AND31 newly wrong images.
        reference = next(r for r in data['paired'] if r['phase'] == 'reference')
        self.assertEqual((reference['v1_correct'], reference['v2_correct']), (31, 31))
        self.assertEqual((reference['fixed'], reference['new_errors'], reference['net_correct']), (31, 31, 0))
        self.assertEqual(tree_hashes(self.parent), self.parent_hashes)
        for p in self.output.rglob('*.npy'):
            p.unlink()
        (self.output / 'oof_metrics.csv').write_text('stale summary')
        unavailable = self.parent.with_name('parent_unavailable')
        self.parent.rename(unavailable)
        try:
            with patch.object(screen, 'head_runtime', side_effect=AssertionError('offline runtime')):
                archive = screen.pack_suite(self.output)
        finally:
            unavailable.rename(self.parent)
        self.assertTrue(Path(str(archive) + '.sha256').is_file())
        self.assertEqual(len(old.read_csv(self.output / 'oof_metrics.csv')), 101)
        with tarfile.open(archive) as packed:
            names = packed.getnames()
            self.assertEqual(len(names), 10141)
            self.assertEqual(sum(n.startswith('runs/') and n.endswith('/predictions.csv') for n in names), 505)
            self.assertEqual(sum(n.startswith('reference/runs/') and n.endswith('/predictions.csv') for n in names), 1010)
            self.assertFalse(any(n.endswith(('.npy', '.npz', '.png', '.jpg', '.pt', '.pth')) for n in names))
        self.assertEqual(tree_hashes(self.parent), self.parent_hashes)


if __name__ == '__main__':
    unittest.main()
