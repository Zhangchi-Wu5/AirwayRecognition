"""Independent synthetic checks of the frozen-feature screening protocol.

No private images, pretrained weight downloads, or real patient inference are
used. The fingerprint override exists only inside these test fixtures.
"""
import copy
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src import frozen_screen_experiment as screen
from src import frozen_screen as core
from src.renew_data import audit_dataset
from src.renew_experiment import fixed_config as old_config, make_plan as old_plan


class FrozenScreenSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.images = self.base / 'images'
        self.images.mkdir()
        for patient in range(1, 32):
            for label_id, label in enumerate(('lt', 'yz', 'zz')):
                Image.new('RGB', (19, 13), (patient * 7, label_id * 71, patient + label_id)).save(
                    self.images / f'{patient}{label}.png')
        self.rows, self.metadata = audit_dataset(self.images, strict=False)
        for name, value in [('audit_dataset', lambda *a, **k: (self.rows, self.metadata)),
                            ('EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint'])]:
            active = patch.object(screen, name, value)
            active.start()
            self.addCleanup(active.stop)
        self.output = self.base / 'output'
        _, self.suite = screen.prepare(self.output, self.images)

    def prepare_synthetic_cache(self):
        """Inject a labelled synthetic basis, not an ImageNet encoder result."""
        raw = np.zeros((93, 512), dtype=np.float32)
        for i, row in enumerate(self.rows):
            raw[i, row['label_id']] = 1.
            raw[i, 3 + int(row['patient_key'])] = .01
        digest = screen._digest('synthetic frozen state')
        evidence = {'architecture': 'resnet18', 'weights': 'ResNet18_Weights.IMAGENET1K_V1',
            'weights_url': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            'weights_file_sha256': 'f37072fd' + '0' * 56,
            'preprocessing': 'full_frame_eval', 'feature_layer': 'avgpool_flatten',
            'feature_dim': 512, 'batch_size': 16, 'num_workers': 0, 'images': 93,
            'encoder_eval': True, 'all_parameters_frozen': True, 'precision': 'float32',
            'device': 'cpu', 'bn_before_sha256': digest, 'bn_after_sha256': digest,
            'state_before_sha256': digest, 'state_after_sha256': digest,
            'row_order_sha256': screen._digest([r['image_name'] for r in self.rows]),
            'raw_feature_sha256': hashlib.sha256(raw.tobytes()).hexdigest(), 'elapsed_seconds': 0.01}
        self.suite.update(runtime_identity={'fixture': 'synthetic, no actual ImageNet extraction'}, device='cpu')
        with patch.object(screen, 'extract_features', return_value=(raw, evidence)) as extractor:
            features = screen.prepare_cache(self.output, self.suite, self.rows, device='cpu')
        extractor.assert_called_once()
        return features

    def remark_cache(self):
        path = self.output / 'feature_cache'
        marker = screen.read_json(path / 'complete.json')
        marker['artifact_sha256'] = {n: screen.sha256_file(path / n) for n in screen.CACHE_FILES}
        screen._atomic_json(path / 'complete.json', marker)
        self.suite['cache_complete_sha256'] = screen.sha256_file(path / 'complete.json')
        screen._atomic_json(self.output / 'suite.json', self.suite)

    def remark_run(self, job):
        directory = self.output / 'runs' / job['run_id']
        marker = screen.read_json(directory / 'complete.json')
        marker['artifact_sha256'] = {n: screen.sha256_file(directory / n) for n in screen.RUN_FILES[:-1]}
        screen._atomic_json(directory / 'complete.json', marker)

    @staticmethod
    def synthetic_linear(features, labels, c):
        """Fast fake estimator; the real solver is covered by core unit tests."""
        class Estimator:
            classes_ = np.array([0, 1, 2])
            n_iter_ = np.array([1])

            def __init__(self):
                self.coef_ = c * np.stack([features[labels == label].mean(0) for label in range(3)])
                self.intercept_ = np.zeros(3)

            def predict_proba(self, x):
                scores = x @ self.coef_.T
                exp = np.exp(scores - scores.max(1, keepdims=True))
                return exp / exp.sum(1, keepdims=True)

        return Estimator()

    def test_plan_preserves_reference_and_prespecifies_matched_disjoint_draws(self):
        folds, old_jobs = old_plan(self.rows, old_config())[::2]
        expected = {j['fold']: j for j in old_jobs
                    if j['method'] == 'full_frame' and j['budget'] == '5' and j['repeat'] == 0}
        identity = self.suite['identity']
        self.assertEqual(identity['folds'], folds)
        self.assertEqual(screen.make_plan(self.rows), (folds, identity['jobs']))
        jobs = identity['jobs']
        self.assertEqual(len(jobs), 1010)
        self.assertEqual(len({j['run_id'] for j in jobs}), 1010)
        self.assertEqual(len({j['support_csv'] for j in jobs}), 505)
        self.assertEqual({j['draw'] for j in jobs if j['phase'] == 'draws'}, set(range(100)))
        by_name = {r['image_name']: r for r in self.rows}
        groups = {}
        for job in jobs:
            groups.setdefault((job['set_id'], job['fold']), []).append(job)
            support = [by_name[n] for n in job['support']]
            held_out = [by_name[n] for n in job['eval']]
            self.assertEqual(len(support), 15)
            self.assertEqual(len({r['patient_key'] for r in support}), 5)
            self.assertEqual(sorted(r['label_id'] for r in support), [0] * 5 + [1] * 5 + [2] * 5)
            self.assertFalse({r['patient_key'] for r in support} & {r['patient_key'] for r in held_out})
            self.assertTrue(set(job['support']).issubset(set(folds[job['fold']]['train'])))
            if job['phase'] == 'reference':
                self.assertEqual(job['support'], expected[job['fold']]['support'])
                self.assertEqual(job['patient_order_prefix'], expected[job['fold']]['patient_order_prefix'])
        self.assertEqual(len(groups), 505)
        for pair in groups.values():
            self.assertEqual({j['method'] for j in pair}, {'prototype', 'linear'})
            self.assertEqual(pair[0]['support'], pair[1]['support'])
            self.assertEqual(pair[0]['eval'], pair[1]['eval'])
            self.assertEqual(pair[0]['sampling_seed'], pair[1]['sampling_seed'])

    def test_prepared_suite_is_pending_and_cannot_pack(self):
        screen.verify_artifacts(self.output, self.suite)
        summary = screen.write_aggregates(self.output, self.suite)
        self.assertEqual(summary['results'], [])
        self.assertEqual(summary['oof'], [])
        with self.assertRaises(ValueError):
            screen.pack_suite(self.output)

    def test_resume_rejects_changed_protocol_sources_and_support_bytes(self):
        screen.prepare(self.output, self.images)
        changed = copy.deepcopy(self.suite['identity']['source_sha256'])
        first = next(iter(changed))
        changed[first] = '0' * 64
        with patch.object(screen, 'source_hashes', return_value=changed):
            with self.assertRaises(ValueError):
                screen.prepare(self.output, self.images)
        support_path = self.output / self.suite['identity']['jobs'][0]['support_csv']
        support_path.write_text('changed support rows\n')
        with self.assertRaises(ValueError):
            screen.verify_artifacts(self.output, self.suite)

    def test_cache_reuses_extraction_and_detects_rehashed_normalization_or_row_changes(self):
        features = self.prepare_synthetic_cache()
        with patch.object(screen, 'extract_features', side_effect=AssertionError('must reuse cache')):
            np.testing.assert_array_equal(features, screen.prepare_cache(self.output, self.suite, self.rows, device='cpu'))
            # A process can stop after writing the cache marker but before
            # attaching its fingerprint to suite.json. Resume must bind it.
            self.suite['cache_complete_sha256'] = None
            screen._atomic_json(self.output / 'suite.json', self.suite)
            np.testing.assert_array_equal(features, screen.prepare_cache(self.output, self.suite, self.rows, device='cpu'))
            self.assertEqual(self.suite['cache_complete_sha256'],
                             screen.sha256_file(self.output / 'feature_cache' / 'complete.json'))
        directory = self.output / 'feature_cache'
        metadata = screen.read_json(directory / 'metadata.json')
        changed = copy.deepcopy(metadata)
        changed['image_names'][0], changed['image_names'][1] = changed['image_names'][1], changed['image_names'][0]
        screen._atomic_json(directory / 'metadata.json', changed)
        self.remark_cache()
        with self.assertRaisesRegex(ValueError, 'row identity'):
            screen.validate_cache(self.output, self.suite)
        screen._atomic_json(directory / 'metadata.json', metadata)
        np.save(directory / 'normalized.npy', features[:, ::-1])
        self.remark_cache()
        with self.assertRaisesRegex(ValueError, 'Normalized features'):
            screen.validate_cache(self.output, self.suite)

    def test_fit_is_support_only_resume_replays_and_corrupt_predictions_fail(self):
        features = self.prepare_synthetic_cache()
        job = self.suite['identity']['jobs'][0]
        by_name = {r['image_name']: r for r in self.rows}
        original_fit = screen.fit_classifier

        def spy(method, x, labels, patients):
            support = [by_name[n] for n in job['support']]
            np.testing.assert_array_equal(x, features[[r['source_index'] for r in support]])
            self.assertEqual(labels, [r['label_id'] for r in support])
            self.assertEqual(patients, [r['patient_key'] for r in support])
            return original_fit(method, x, labels, patients)

        with patch.object(screen, 'fit_classifier', side_effect=spy) as fitting:
            result = screen.fit_one(self.output, self.suite, job, features)
        fitting.assert_called_once()
        with patch.object(screen, 'fit_classifier', side_effect=AssertionError('must resume')):
            self.assertEqual(result, screen.fit_one(self.output, self.suite, job, features))
        directory = self.output / 'runs' / job['run_id']
        modified = screen.read_json(directory / 'result.json')
        modified['accuracy'] = .123
        screen._atomic_json(directory / 'result.json', modified)
        self.remark_run(job)
        with self.assertRaisesRegex(ValueError, 'Metric differs'):
            screen.validate_run(self.output, self.suite, job, features)
        screen._atomic_json(directory / 'result.json', result)
        model = screen.read_json(directory / 'model.json')
        model['prototypes'] = model['prototypes'][1:] + model['prototypes'][:1]
        screen._atomic_json(directory / 'model.json', model)
        self.remark_run(job)
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            screen.validate_run(self.output, self.suite, job, features)

    def test_offline_validation_rejects_malformed_model_and_convergence_evidence(self):
        features = self.prepare_synthetic_cache()
        job = next(j for j in self.suite['identity']['jobs'] if j['method'] == 'linear')
        with patch.object(core, '_fit_linear', side_effect=self.synthetic_linear):
            screen.fit_one(self.output, self.suite, job, features)
        directory = self.output / 'runs' / job['run_id']
        model = screen.read_json(directory / 'model.json')
        changed = copy.deepcopy(model)
        changed['classes'] = [2, 1, 0]
        screen._atomic_json(directory / 'model.json', changed)
        self.remark_run(job)
        with self.assertRaises(ValueError):
            screen.validate_run(self.output, self.suite, job, features=None)
        screen._atomic_json(directory / 'model.json', model)
        tuning = screen.read_csv(directory / 'tuning.csv')
        tuning[0]['n_iter'] = 10000
        screen.write_csv(directory / 'tuning.csv', tuning)
        self.remark_run(job)
        with self.assertRaises(ValueError):
            screen.validate_run(self.output, self.suite, job, features=None)

    def test_full_offline_pack_excludes_arrays_and_reference_from_draw_summary(self):
        features = self.prepare_synthetic_cache()
        jobs = self.suite['identity']['jobs']
        original_fit = screen.fit_classifier

        def fit_bad_reference(method, x, labels, patients):
            model, tuning = original_fit(method, x, labels, patients)
            if method == 'prototype':
                model['prototypes'] = model['prototypes'][1:] + model['prototypes'][:1]
            return model, tuning

        with patch.object(core, '_fit_linear', side_effect=self.synthetic_linear):
            for job in jobs:
                if job['phase'] == 'reference':
                    with patch.object(screen, 'fit_classifier', side_effect=fit_bad_reference):
                        screen.fit_one(self.output, self.suite, job, features)
                else:
                    screen.fit_one(self.output, self.suite, job, features)
        complete = screen.write_aggregates(self.output, self.suite)
        self.assertEqual(len(complete['results']), 1010)
        self.assertEqual(len(complete['oof']), 202)
        self.assertEqual(len(complete['predictions']), 18786)
        self.assertEqual(len(complete['paired']), 101)
        self.assertTrue(complete['feature_replay_verified'])
        reference = next(r for r in complete['oof'] if r['phase'] == 'reference' and r['method'] == 'prototype')
        self.assertEqual(reference['correct'], 0)
        for row in complete['distributions']:
            self.assertEqual(row['n_draws'], 100)
            if row['metric'] == 'accuracy':
                self.assertEqual(row['mean'], 1.)
        self.assertEqual(len(complete['errors']), 186)
        self.assertTrue(all(r['n_draws'] == 100 and r['errors'] == 0 for r in complete['errors']))
        self.assertEqual(sum(r['selected_count'] for r in complete['selection']), 2500)
        shutil.rmtree(self.images)
        for array in (self.output / 'feature_cache').glob('*.npy'):
            array.unlink()
        (self.output / 'oof_metrics.csv').write_text('stale summary; must regenerate')
        with patch.object(screen, 'extract_features', side_effect=AssertionError('offline pack')), \
             patch.object(screen, 'environment', side_effect=AssertionError('offline pack')):
            archive = screen.pack_suite(self.output)
        self.assertTrue(Path(str(archive) + '.sha256').is_file())
        self.assertEqual(len(screen.read_csv(self.output / 'oof_metrics.csv')), 202)
        scope = screen.read_json(self.output / 'package_scope.json')
        self.assertFalse(scope['feature_replay_verified_at_pack'])
        with tarfile.open(archive) as packed:
            names = packed.getnames()
            self.assertEqual(sum(n.endswith('/predictions.csv') for n in names), 1010)
            self.assertIn('feature_cache/metadata.json', names)
            self.assertIn('feature_cache/complete.json', names)
            self.assertFalse(any(n.endswith(('.npy', '.npz', '.png', '.jpg', '.pt', '.pth')) for n in names))


if __name__ == '__main__':
    unittest.main()
