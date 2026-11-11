"""Independent protocol tests using generated images and tiny networks only."""
import copy
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src import patient_relation_experiment as suite_module
from src.renew_data import audit_dataset


class PatientRelationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
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
        self.output = self.base / 'output'
        self.audit = patch.object(suite_module, 'audit_dataset', return_value=(self.rows, self.metadata))
        self.fingerprint = patch.object(suite_module, 'EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint'])
        self.audit.start()
        self.fingerprint.start()
        self.addCleanup(self.audit.stop)
        self.addCleanup(self.fingerprint.stop)
        self.addCleanup(self.temp.cleanup)
        self.rows, self.suite = suite_module.prepare(self.output, self.images)
        self.suite['runtime_identity'] = {'synthetic': True}
        suite_module._atomic_json(self.output / 'suite.json', self.suite)

    def fixture(self, job, wrong=0):
        """Fabricated completed records, never represented as trained models."""
        import numpy as np
        g = suite_module
        directory = self.output / 'runs' / job['run_id']
        directory.mkdir(parents=True)
        rows_by_name = {r['image_name']: r for r in self.rows}
        evaluation = [rows_by_name[n] for n in job['eval']]
        probabilities = np.array([[.8 if c == ((r['label_id'] + 1) % 3 if i < wrong else r['label_id'])
                                   else .1 for c in range(3)] for i, r in enumerate(evaluation)], dtype=np.float32)
        c = g.run_config(self.suite['identity']['config'], job)
        initial = g._digest({'initial': job['training_seed']})
        relation_hash = g._digest(job['relation_indices'])
        g._atomic_json(directory / 'config.json', {'identity_sha256': self.suite['identity_sha256'],
            'config': c, 'job': job, 'initial_state_sha256': initial,
            'relation_indices_sha256': relation_hash, 'execution_runtime': self.suite['runtime_identity']})
        history = []
        for stage, count in [('head', c['head_rounds']), ('finetune', c['finetune_rounds'])]:
            for i in range(count):
                aux = 0. if stage == 'head' or job['method'] == 'ce' else .2
                pairs = 0 if stage == 'head' or job['method'] == 'ce' else 60 if job['method'] == 'supcon' else 30
                history.append({'stage': stage, 'round': i + 1, 'steps': c['steps_per_round'],
                    'image_occurrences': c['steps_per_round'] * 15,
                    'lr': c['head_lr'] if stage == 'head' else c['finetune_lr'] * (1 + math.cos(math.pi * i / c['finetune_rounds'])) / 2,
                    'mean_train_loss': .5 + c['aux_weight'] * aux, 'classification_loss': .5,
                    'auxiliary_loss': aux, 'auxiliary_pairs': pairs * c['steps_per_round'], 'seconds': .001})
        g.write_csv(directory / 'history.csv', history)
        predictions = [{'patient_id': r['patient_id'], 'image_name': r['image_name'], 'label_id': r['label_id'],
                        'predicted_id': int(p.argmax()), **{f'prob_{label}': float(p[k]) for label, k in g.LABEL_TO_ID.items()}}
                       for r, p in zip(evaluation, probabilities)]
        g.write_csv(directory / 'predictions.csv', predictions)
        (directory / 'model.pt').write_bytes(b'synthetic checkpoint fixture, not trained weights')
        (directory / 'student.pt').write_bytes(b'synthetic student fixture, not trained weights')
        result = {**{k: job[k] for k in g.JOB_KEYS}, 'n_train_images': 15, 'n_train_patients': 5,
            'initial_state_sha256': initial, 'relation_indices_sha256': relation_hash,
            'single_image_evaluation': True, 'student_export_verified': True, 'student_parameter_count': 11178051,
            'student_export_sha256': g.sha256_file(directory / 'student.pt'),
            'bn_before_sha256': g._digest('bn'), 'bn_after_sha256': g._digest('bn'), 'bn_buffers_unchanged': True,
            'initial_backbone_sha256': g._digest('before'), 'final_backbone_sha256': g._digest('after'),
            'final_state_sha256': g._digest('final'), 'reloaded_state_sha256': g._digest('final'),
            'checkpoint_reload_verified': True, 'head_stage_backbone_unchanged': True, 'finetune_backbone_changed': True,
            'effective_support_order': list(range(15)), 'auxiliary_phase': 'finetune_only',
            'actual_optimizer_steps': len(history) * c['steps_per_round'],
            'actual_image_occurrences': len(history) * c['steps_per_round'] * 15,
            **{key: g._digest([job['training_seed'], job['fold'], key])
               for key in ('augmentation_sha256', 'head_augmentation_sha256', 'finetune_augmentation_sha256',
                           'training_rng_initial_sha256', 'training_rng_final_sha256')},
            **g.classification_metrics([r['label_id'] for r in evaluation], probabilities)}
        g._atomic_json(directory / 'result.json', result)
        self.remark(directory, job)
        return directory

    def remark(self, directory, job):
        g = suite_module
        g._atomic_json(directory / 'complete.json', {'identity_sha256': self.suite['identity_sha256'],
            'run_id': job['run_id'], 'artifact_sha256': {name: g.sha256_file(directory / name)
            for name in (*g.RUN_FILES[:-1], 'model.pt')}})

    def test_plan_has_exact_matched_40_jobs_and_disjoint_patients(self):
        jobs = self.suite['identity']['jobs']
        by_name = {r['image_name']: r for r in self.rows}
        self.assertEqual(len(jobs), 40)
        self.assertEqual({j['budget'] for j in jobs}, {'5'})
        self.assertEqual({j['training_seed'] for j in jobs}, {42, 43})
        self.assertEqual({j['repeat'] for j in jobs}, {0})
        self.assertEqual(len({j['run_id'] for j in jobs}), 40)
        for fold in range(5):
            group = [j for j in jobs if j['fold'] == fold]
            self.assertEqual(len(group), 8)
            self.assertTrue(all(j['support'] == group[0]['support'] for j in group))
            self.assertTrue(all(j['eval'] == group[0]['eval'] for j in group))
            support = [by_name[n] for n in group[0]['support']]
            self.assertEqual(len(support), 15)
            self.assertEqual(len({r['patient_key'] for r in support}), 5)
            held_out = {by_name[n]['patient_key'] for n in group[0]['eval']}
            self.assertFalse(held_out & {r['patient_key'] for r in support})
            for job in group:
                indices = job['relation_indices']
                self.assertEqual(sorted(i for triple in indices for i in triple), list(range(15)))
                for triple in indices:
                    self.assertEqual([support[i]['label_id'] for i in triple], [0, 1, 2])
                    ids = {support[i]['patient_key'] for i in triple}
                    self.assertEqual(len(ids), 3 if job['method'] == 'regrouped' else 1)
        c = suite_module.fixed_config()
        self.assertEqual(c['batch_size'], 15)
        self.assertEqual((c['head_rounds'] + c['finetune_rounds']) * c['steps_per_round'], 700)

    def test_prepare_rejects_changed_sources_and_plan_bytes(self):
        suite_module.prepare(self.output, self.images)
        with patch.object(suite_module, 'sha256_file', return_value='changed'):
            with self.assertRaises(ValueError):
                suite_module.prepare(self.output, self.images)
        support = self.output / self.suite['identity']['jobs'][0]['support_csv']
        support.write_text('tampered support')
        with self.assertRaises(ValueError):
            suite_module.prepare(self.output, self.images)

    def test_prepared_suite_is_pending_and_cannot_pack(self):
        data = suite_module.write_aggregates(self.output, self.suite)
        self.assertEqual(data['results'], [])
        self.assertEqual(data['oof'], [])
        self.assertEqual(data['decision']['status'], 'pending')
        with self.assertRaises(ValueError):
            suite_module.pack_suite(self.output)

    def test_formal_run_requires_cuda_before_preparing_or_training(self):
        g = suite_module
        args = g.parser().parse_args(['run', '--output-root', str(self.base / 'never_prepared')])
        with patch.object(g, 'check_dependencies'), patch.object(g, 'configure_torch'), \
             patch.object(g, 'environment', side_effect=RuntimeError('no CUDA')) as env, \
             patch.object(g, 'prepare', side_effect=AssertionError('should not prepare')):
            with self.assertRaisesRegex(RuntimeError, 'no CUDA'):
                g.run_suite(args)
        env.assert_called_once_with(require_cuda=True)
        self.assertFalse(args.output_root.exists())

    def test_decision_needs_both_seeds_and_all_three_comparisons(self):
        g = suite_module
        rows = [{'training_seed': seed, 'method': method, 'correct': count}
                for seed in (42, 43) for method, count in [('ce', 80), ('supcon', 81), ('paired', 82), ('regrouped', 81)]]
        self.assertEqual(g.decision_from_oof(rows, False)['status'], 'pending')
        self.assertEqual(g.decision_from_oof(rows, True)['status'], 'continue_to_confirmation')
        self.assertEqual(g.decision_from_oof(rows, True, smoke=True)['status'], 'smoke_not_formal')
        for method, count in [('ce', 81), ('supcon', 82), ('regrouped', 82)]:
            bad = copy.deepcopy(rows)
            next(r for r in bad if r['training_seed'] == 43 and r['method'] == method)['correct'] = count
            self.assertEqual(g.decision_from_oof(bad, True)['status'], 'stop_current_candidate')
        with self.assertRaises(ValueError):
            g.decision_from_oof(rows[:4], True)

    def test_rehashed_metrics_and_shared_augmentation_mismatch_are_rejected(self):
        g = suite_module
        jobs = self.suite['identity']['jobs'][:2]
        directories = [self.fixture(j) for j in jobs]
        self.assertEqual(len(g.aggregate(self.output, self.suite)['results']), 2)
        d = directories[0]
        result = g.read_json(d / 'result.json')
        result['accuracy'] = .123
        g._atomic_json(d / 'result.json', result)
        self.remark(d, jobs[0])
        with self.assertRaisesRegex(ValueError, 'Prediction metric mismatch'):
            g.validate_run(self.output, self.suite, jobs[0])
        result['accuracy'] = 1.
        g._atomic_json(d / 'result.json', result)
        self.remark(d, jobs[0])
        other = g.read_json(directories[1] / 'result.json')
        for key in ('augmentation_sha256', 'training_rng_final_sha256'):
            original = other[key]
            other[key] = g._digest('different actual stochastic stream')
            g._atomic_json(directories[1] / 'result.json', other)
            self.remark(directories[1], jobs[1])
            with self.assertRaisesRegex(ValueError, 'stochastic stream differs'):
                g.aggregate(self.output, self.suite)
            other[key] = original
        g._atomic_json(directories[1] / 'result.json', other)
        history = g.read_csv(directories[1] / 'history.csv')
        history[0]['classification_loss'] = .6
        history[0]['mean_train_loss'] = .6
        g.write_csv(directories[1] / 'history.csv', history)
        self.remark(directories[1], jobs[1])
        with self.assertRaisesRegex(ValueError, 'head'):
            g.aggregate(self.output, self.suite)

    def test_full_offline_pack_rebuilds_summaries_and_excludes_images_and_weights(self):
        g = suite_module
        jobs = self.suite['identity']['jobs']
        self.fixture(jobs[0], wrong=3)
        partial = g.write_aggregates(self.output, self.suite)
        self.assertEqual(partial['oof'], [])
        self.assertEqual(partial['decision']['status'], 'pending')
        with self.assertRaises(ValueError):
            g.pack_suite(self.output)
        wrong = {'ce': 3, 'supcon': 2, 'paired': 0, 'regrouped': 2}
        for job in jobs[1:]:
            self.fixture(job, wrong=wrong[job['method']] if job['fold'] == 0 else 0)
        complete = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(complete['oof']), 8)
        self.assertEqual(len(complete['predictions']), 8 * 93)
        self.assertEqual(len(complete['image_changes']), 2 * 3 * 93)
        self.assertEqual(len(complete['patient_changes']), 2 * 3 * 31)
        self.assertEqual(complete['decision']['status'], 'continue_to_confirmation')
        shutil.rmtree(self.images)
        for checkpoint in self.output.glob('runs/*/*.pt'):
            checkpoint.unlink()
        (self.output / 'oof_metrics.csv').write_text('interrupted summary, must regenerate')
        archive = g.pack_suite(self.output)
        self.assertTrue(Path(str(archive) + '.sha256').is_file())
        with tarfile.open(archive) as packed:
            names = packed.getnames()
            self.assertIn('package_scope.json', names)
            self.assertIn('DECISION.json', names)
            self.assertEqual(sum(name.endswith('/predictions.csv') for name in names), 40)
            self.assertFalse(any(name.endswith(('.pt', '.png', '.jpg', '.npz')) for name in names))
        rebuilt = g.read_csv(self.output / 'oof_metrics.csv')
        self.assertEqual(len(rebuilt), 8)

    def test_small_network_all_arms_export_rng_and_resume(self):
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
        rows, suite = suite_module.prepare(root, self.images, smoke=True)
        suite['runtime_identity'] = {'synthetic': True}
        suite_module._atomic_json(root / 'suite.json', suite)
        by_name = {r['image_name']: r for r in rows}
        jobs = [j for j in suite['identity']['jobs'] if j['fold'] == 0 and j['training_seed'] == 42]
        interrupted = root / 'runs' / jobs[0]['run_id']
        interrupted.mkdir(parents=True)
        (interrupted / 'partial.txt').write_text('interrupted attempt should be preserved')
        results = []
        for job in jobs:
            result = suite_module.train_one(root, suite, job, by_name, torch.device('cpu'),
                                           model_factory=lambda _: Tiny())
            results.append(result)
            self.assertTrue(result['student_export_verified'])
            self.assertTrue(result['head_stage_backbone_unchanged'])
            self.assertTrue(result['finetune_backbone_changed'])
            self.assertTrue(result['bn_buffers_unchanged'])
            self.assertEqual(result['actual_image_occurrences'], 30)
            history = suite_module.read_csv(root / 'runs' / job['run_id'] / 'history.csv')
            self.assertEqual(float(history[0]['auxiliary_loss']), 0)
            self.assertEqual(float(history[0]['auxiliary_pairs']), 0)
            self.assertEqual(float(history[-1]['auxiliary_pairs']),
                             0 if job['method'] == 'ce' else 60 if job['method'] == 'supcon' else 30)
            with patch.object(suite_module, 'train_updates', side_effect=AssertionError('must skip completed fit')):
                repeated = suite_module.train_one(root, suite, job, by_name, torch.device('cpu'),
                                                 model_factory=lambda _: Tiny())
            self.assertEqual(result, repeated)
        for key in ('initial_state_sha256', 'augmentation_sha256', 'head_augmentation_sha256',
                    'finetune_augmentation_sha256', 'training_rng_initial_sha256', 'training_rng_final_sha256'):
            self.assertEqual(len({r[key] for r in results}), 1, key)
        archived = list((root / 'incomplete_attempts').glob('*/partial.txt'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), 'interrupted attempt should be preserved')
        with self.assertRaises(ValueError):
            suite_module.pack_suite(root)
        # A corrupt complete record must fail rather than silently being retrained.
        (root / 'runs' / jobs[0]['run_id'] / 'predictions.csv').write_text('corrupt')
        with self.assertRaises(ValueError):
            suite_module.train_one(root, suite, jobs[0], by_name, torch.device('cpu'),
                                   model_factory=lambda _: Tiny())

    def test_resnet_deployment_is_standard_single_image_model(self):
        import torch
        from torchvision.models import resnet18
        from src.patient_relation import build_model
        torch.set_num_threads(1)
        torch.manual_seed(21)
        training_model = build_model(pretrained=False).eval()
        deployment = resnet18(weights=None)
        deployment.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(512, 3))
        deployment.load_state_dict(training_model.state_dict(), strict=True)
        deployment.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            logits, features = training_model(images, return_features=True)
            ordinary = deployment(images)
            alone = deployment(images[:1])
        self.assertEqual(tuple(features.shape), (2, 512))
        self.assertTrue(torch.equal(logits, ordinary))
        self.assertTrue(torch.allclose(ordinary[:1], alone, atol=1e-5, rtol=1e-5))
        self.assertFalse(any('geometry' in key or 'relation' in key for key in deployment.state_dict()))


if __name__ == '__main__':
    unittest.main()
