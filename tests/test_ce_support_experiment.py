"""Independent support-sensitivity checks; generated images and tiny networks only."""
import copy
import math
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src import ce_support_experiment as g
from src import patient_relation_experiment as old
from src.renew_data import audit_dataset


class CESupportExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.images = self.base / 'images'
        self.images.mkdir()
        for patient in range(1, 32):
            for label_id, label in enumerate(('lt', 'yz', 'zz')):
                image = Image.new('RGB', (32, 18))
                image.putdata([((patient * 5 + x * 3) % 256,
                                (label_id * 70 + y * 7) % 256,
                                (x * y + patient + 90) % 256)
                               for y in range(18) for x in range(32)])
                image.save(self.images / f'{patient}{label}.png')
        self.rows, self.metadata = audit_dataset(self.images, strict=False)
        for mock in (patch.object(g, 'audit_dataset', return_value=(self.rows, self.metadata)),
                     patch.object(g, 'EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint'])):
            mock.start()
            self.addCleanup(mock.stop)
        self.output = self.base / 'output'
        self.rows, self.suite = g.prepare(self.output, self.images)
        self.suite['runtime_identity'] = {'synthetic': True}
        old._atomic_json(self.output / 'suite.json', self.suite)

    def fixture(self, job, wrong=0):
        """Fabricated bookkeeping for validation, not actual experimental results."""
        import numpy as np
        directory = self.output / 'runs' / job['run_id']
        directory.mkdir(parents=True)
        by_name = {r['image_name']: r for r in self.rows}
        evaluation = [by_name[n] for n in job['eval']]
        probabilities = np.asarray([
            [.8 if c == ((r['label_id'] + 1) % 3 if i < wrong else r['label_id']) else .1
             for c in range(3)] for i, r in enumerate(evaluation)], dtype=np.float32)
        config = old.run_config(self.suite['identity']['config'], job)
        initial = old._digest({'initial': 42})
        relation_hash = old._digest(job['relation_indices'])
        old._atomic_json(directory / 'config.json', {
            'identity_sha256': self.suite['identity_sha256'], 'config': config, 'job': job,
            'initial_state_sha256': initial, 'relation_indices_sha256': relation_hash,
            'execution_runtime': self.suite['runtime_identity']})
        history = []
        for stage, count in [('head', config['head_rounds']), ('finetune', config['finetune_rounds'])]:
            for i in range(count):
                history.append({
                    'stage': stage, 'round': i + 1, 'steps': config['steps_per_round'],
                    'image_occurrences': config['steps_per_round'] * 15,
                    'lr': config['head_lr'] if stage == 'head' else config['finetune_lr'] *
                          (1 + math.cos(math.pi * i / config['finetune_rounds'])) / 2,
                    'mean_train_loss': .5, 'classification_loss': .5,
                    'auxiliary_loss': 0., 'auxiliary_pairs': 0, 'seconds': .001})
        old.write_csv(directory / 'history.csv', history)
        predictions = [{
            'patient_id': r['patient_id'], 'image_name': r['image_name'], 'label_id': r['label_id'],
            'predicted_id': int(p.argmax()),
            **{f'prob_{label}': float(p[k]) for label, k in old.LABEL_TO_ID.items()}}
            for r, p in zip(evaluation, probabilities)]
        old.write_csv(directory / 'predictions.csv', predictions)
        (directory / 'model.pt').write_bytes(b'synthetic checkpoint, not trained weights')
        (directory / 'student.pt').write_bytes(b'synthetic exported checkpoint, not trained weights')
        result = {
            **{k: job[k] for k in old.JOB_KEYS}, 'n_train_images': 15, 'n_train_patients': 5,
            'initial_state_sha256': initial, 'relation_indices_sha256': relation_hash,
            'single_image_evaluation': True, 'student_export_verified': True,
            'student_parameter_count': 11178051,
            'student_export_sha256': old.sha256_file(directory / 'student.pt'),
            'bn_before_sha256': old._digest('bn'), 'bn_after_sha256': old._digest('bn'),
            'bn_buffers_unchanged': True,
            'initial_backbone_sha256': old._digest('before'),
            'final_backbone_sha256': old._digest('after'),
            'final_state_sha256': old._digest('final'),
            'reloaded_state_sha256': old._digest('final'),
            'checkpoint_reload_verified': True, 'head_stage_backbone_unchanged': True,
            'finetune_backbone_changed': True, 'effective_support_order': list(range(15)),
            'auxiliary_phase': 'finetune_only',
            'actual_optimizer_steps': len(history) * config['steps_per_round'],
            'actual_image_occurrences': len(history) * config['steps_per_round'] * 15,
            **{key: old._digest([42, key]) for key in
               ('training_rng_initial_sha256', 'training_rng_final_sha256')},
            **{key: old._digest([job['run_id'], key]) for key in
               ('augmentation_sha256', 'head_augmentation_sha256', 'finetune_augmentation_sha256')},
            **old.classification_metrics([r['label_id'] for r in evaluation], probabilities)}
        old._atomic_json(directory / 'result.json', result)
        self.remark(directory, job)
        return directory

    def remark(self, directory, job):
        old._atomic_json(directory / 'complete.json', {
            'identity_sha256': self.suite['identity_sha256'], 'run_id': job['run_id'],
            'artifact_sha256': {name: old.sha256_file(directory / name)
                               for name in (*old.RUN_FILES[:-1], 'model.pt')}})

    def test_fixed_plan_has_25_disjoint_triplet_supports_without_global_rng_effect(self):
        before = random.getstate()
        folds, jobs = g.make_plan(self.suite['identity']['data'])
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(jobs), 25)
        self.assertEqual(len({j['run_id'] for j in jobs}), 25)
        self.assertEqual(folds, old.make_plan(self.suite['identity']['data'])[0])
        self.assertEqual(jobs, self.suite['identity']['jobs'])
        by_name = {r['image_name']: r for r in self.rows}
        for job in jobs:
            fold = next(f for f in folds if f['fold'] == job['fold'])
            eligible = sorted(fold['train_patient_keys'])
            seed = 2026111201 + job['draw'] * 5 + job['fold']
            random.Random(seed).shuffle(eligible)
            self.assertEqual(job['sampling_seed'], seed)
            self.assertEqual(job['patient_order_prefix'], eligible[:5])
            self.assertEqual(job['training_seed'], 42)
            self.assertEqual(job['method'], 'ce')
            self.assertEqual(job['repeat'], job['draw'])
            support = [by_name[name] for name in job['support']]
            self.assertEqual(len(support), 15)
            self.assertEqual({r['patient_key'] for r in support}, set(eligible[:5]))
            self.assertFalse(set(eligible[:5]) & set(fold['eval_patient_keys']))
            for indices in job['relation_indices']:
                self.assertEqual([support[k]['label_id'] for k in indices], [0, 1, 2])
                self.assertEqual(len({support[k]['patient_key'] for k in indices}), 1)
            self.assertTrue((self.output / job['support_csv']).is_file())
        for draw in range(5):
            names = [n for job in jobs if job['draw'] == draw for n in job['eval']]
            self.assertEqual(sorted(names), sorted(by_name))

    def test_training_settings_match_old_ce(self):
        previous, current = old.fixed_config(), g.fixed_config()
        effective_keys = ('architecture', 'weights', 'dropout', 'batch_size', 'num_workers',
                          'head_rounds', 'finetune_rounds', 'steps_per_round', 'head_lr',
                          'finetune_lr', 'weight_decay', 'optimizer', 'finetune_scheduler',
                          'batch_norm', 'selection', 'precision', 'preprocessing',
                          'views_per_image', 'batching', 'aux_weight', 'auxiliary_phase')
        for key in effective_keys:
            self.assertEqual(current[key], previous[key], key)
        self.assertEqual((current['head_rounds'] + current['finetune_rounds']) * current['steps_per_round'], 700)
        self.assertEqual(current['training_seeds'], [42])

    def test_changed_source_plan_or_smoke_mode_cannot_resume(self):
        g.prepare(self.output, self.images)
        with patch.object(g, 'source_hashes', return_value={}):
            with self.assertRaises(ValueError):
                g.prepare(self.output, self.images)
        with self.assertRaises(ValueError):
            g.prepare(self.output, self.images, smoke=True)
        support = self.output / self.suite['identity']['jobs'][0]['support_csv']
        support.write_text('tampered support')
        with self.assertRaises(ValueError):
            g.prepare(self.output, self.images)

    def test_rehashed_identity_with_holdout_patient_is_rejected(self):
        bad = copy.deepcopy(self.suite)
        job = bad['identity']['jobs'][0]
        job['support'][0] = job['eval'][0]
        bad['identity_sha256'] = old._digest(bad['identity'])
        with self.assertRaises(ValueError):
            g.verify_artifacts(self.output, bad)

    def test_pending_and_partial_draw_do_not_report_complete_oof(self):
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(data['results'], [])
        self.assertEqual(data['oof'], [])
        self.assertEqual(data['summary']['status'], 'pending')
        self.fixture(self.suite['identity']['jobs'][0], wrong=3)
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['oof'], [])
        self.assertEqual(data['image_errors'], [])
        with self.assertRaises(ValueError):
            g.pack_suite(self.output)

    def test_complete_draws_and_pack_rebuild_aggregates_excluding_pixels_and_weights(self):
        jobs = self.suite['identity']['jobs']
        for job in jobs:
            self.fixture(job, wrong=job['draw'] if job['fold'] == 0 else 0)
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['results']), 25)
        self.assertEqual(len(data['oof']), 5)
        self.assertEqual(len(data['predictions']), 465)
        self.assertEqual(len(data['image_errors']), 93)
        self.assertEqual(data['summary']['status'], 'completed_descriptive')
        self.assertEqual({r['draw']: r['correct'] for r in data['oof']},
                         {draw: 93 - draw for draw in range(5)})
        distributions = {r['metric']: r for r in data['distribution']}
        self.assertAlmostEqual(distributions['correct']['mean'], 91.)
        self.assertAlmostEqual(distributions['correct']['sample_sd'], math.sqrt(2.5))
        self.assertAlmostEqual(distributions['accuracy']['mean'], 91. / 93.)
        self.assertAlmostEqual(distributions['accuracy']['sample_sd'], math.sqrt(2.5) / 93.)
        self.assertEqual(distributions['correct']['min'], 89.)
        self.assertEqual(distributions['correct']['max'], 93.)
        self.assertEqual(distributions['correct']['completed_draws'], 5)
        self.assertEqual(sum(r['error_count'] for r in data['image_errors']), 10)
        self.assertTrue(all(r['evaluations'] == 5 for r in data['image_errors']))
        shutil.rmtree(self.images)
        for checkpoint in self.output.glob('runs/*/*.pt'):
            checkpoint.unlink()
        (self.output / 'oof_metrics.csv').write_text('interrupted summary')
        archive = g.pack_suite(self.output)
        self.assertTrue(Path(str(archive) + '.sha256').is_file())
        with tarfile.open(archive) as packed:
            names = packed.getnames()
            self.assertIn('package_scope.json', names)
            self.assertEqual(sum(n.endswith('/predictions.csv') for n in names), 25)
            self.assertFalse(any(n.endswith(('.png', '.jpg', '.pt', '.npz')) for n in names))
        self.assertEqual(len(old.read_csv(self.output / 'oof_metrics.csv')), 5)

    def test_rehashed_metric_and_rng_drift_are_rejected_but_image_stream_may_differ(self):
        jobs = self.suite['identity']['jobs'][:2]
        directories = [self.fixture(j) for j in jobs]
        self.assertEqual(len(g.aggregate(self.output, self.suite)['results']), 2)
        result = old.read_json(directories[0] / 'result.json')
        original_accuracy = result['accuracy']
        result['accuracy'] = .123
        old._atomic_json(directories[0] / 'result.json', result)
        self.remark(directories[0], jobs[0])
        with self.assertRaises(ValueError):
            g.validate_run(self.output, self.suite, jobs[0])
        result['accuracy'] = original_accuracy
        old._atomic_json(directories[0] / 'result.json', result)
        self.remark(directories[0], jobs[0])
        other = old.read_json(directories[1] / 'result.json')
        other['training_rng_final_sha256'] = old._digest('different RNG stream')
        old._atomic_json(directories[1] / 'result.json', other)
        self.remark(directories[1], jobs[1])
        with self.assertRaises(ValueError):
            g.aggregate(self.output, self.suite)

    def test_removal_of_previously_recorded_complete_marker_fails_closed(self):
        job = self.suite['identity']['jobs'][0]
        directory = self.fixture(job)
        g.write_aggregates(self.output, self.suite)
        (directory / 'complete.json').unlink()
        with self.assertRaises(ValueError):
            g.aggregate(self.output, self.suite)

    def test_formal_run_requires_cuda_before_prepare(self):
        args = g.parser().parse_args(['run', '--output-root', str(self.base / 'never_prepared')])
        with patch.object(g, 'check_dependencies'), patch.object(g, 'configure_torch'), \
             patch.object(g, 'environment', side_effect=RuntimeError('no CUDA')) as env, \
             patch.object(g, 'prepare', side_effect=AssertionError('must not prepare')):
            with self.assertRaisesRegex(RuntimeError, 'no CUDA'):
                g.run_suite(args)
        env.assert_called_once_with(require_cuda=True)

    def test_filters_only_choose_prespecified_run_subset(self):
        args = g.parser().parse_args(['run', '--draws', '1', '3', '--folds', '0'])
        jobs = g.selected_jobs(self.suite, args)
        self.assertEqual([(j['draw'], j['fold']) for j in jobs], [(1, 0), (3, 0)])
        args.draws = []
        with self.assertRaises(ValueError):
            g.selected_jobs(self.suite, args)
        with self.assertRaises(ValueError):
            g.main(['pack', '--draws', '0', '--output-root', str(self.output)])

    def test_small_network_uses_old_training_path_exports_and_resumes(self):
        import torch
        torch.set_num_threads(1)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Conv2d(3, 6, 1)
                self.bn = torch.nn.BatchNorm2d(6)
                self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(6, 3))

            def forward(self, x, return_features=False):
                small = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
                features = self.bn(self.stem(small)).relu().mean((2, 3))
                logits = self.fc(features)
                return (logits, features) if return_features else logits

        root = self.base / 'smoke'
        rows, suite = g.prepare(root, self.images, smoke=True)
        suite['runtime_identity'] = {'synthetic': True}
        old._atomic_json(root / 'suite.json', suite)
        by_name = {r['image_name']: r for r in rows}
        jobs = [j for j in suite['identity']['jobs'] if j['fold'] == 0][:2]
        interrupted = root / 'runs' / jobs[0]['run_id']
        interrupted.mkdir(parents=True)
        (interrupted / 'partial.txt').write_text('preserve interrupted attempt')
        results = []
        for job in jobs:
            result = g.train_one(root, suite, job, by_name, torch.device('cpu'),
                                 model_factory=lambda _: Tiny())
            results.append(result)
            self.assertTrue(result['student_export_verified'])
            self.assertTrue(result['head_stage_backbone_unchanged'])
            self.assertTrue(result['finetune_backbone_changed'])
            self.assertTrue(result['bn_buffers_unchanged'])
            self.assertEqual(result['actual_optimizer_steps'], 2)
            self.assertEqual(result['actual_image_occurrences'], 30)
            with patch.object(old, 'train_updates', side_effect=AssertionError('completed fit must skip')):
                again = g.train_one(root, suite, job, by_name, torch.device('cpu'),
                                    model_factory=lambda _: Tiny())
            self.assertEqual(result, again)
        self.assertEqual(len({r['initial_state_sha256'] for r in results}), 1)
        self.assertEqual(len({r['training_rng_initial_sha256'] for r in results}), 1)
        self.assertEqual(len({r['training_rng_final_sha256'] for r in results}), 1)
        self.assertEqual(len({r['augmentation_sha256'] for r in results}), 2)
        self.assertEqual(len(list((root / 'incomplete_attempts').glob('*/partial.txt'))), 1)
        data = g.write_aggregates(root, suite)
        self.assertEqual(data['summary']['status'], 'smoke_not_formal')
        with self.assertRaises(ValueError):
            g.pack_suite(root)
        (root / 'runs' / jobs[0]['run_id'] / 'predictions.csv').write_text('corrupt')
        with self.assertRaises(ValueError):
            g.train_one(root, suite, jobs[0], by_name, torch.device('cpu'), model_factory=lambda _: Tiny())


if __name__ == '__main__':
    unittest.main()
