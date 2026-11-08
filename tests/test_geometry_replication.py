"""Seed-43 replication: synthetic provenance fixtures and real CPU tiny fits."""
import copy
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src import geometry_experiment as old
from src import geometry_replication as g
from src.renew_data import audit_dataset


class GeometryReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from src.geometry_distillation import cache_predictions, validate_cache

        torch.set_num_threads(1)
        cls.template = tempfile.TemporaryDirectory()
        cls.template_root = Path(cls.template.name)
        images = cls.template_root / 'images'
        images.mkdir()
        for patient in range(1, 32):
            for label, name in enumerate(('lt', 'yz', 'zz')):
                image = Image.new('RGB', (32, 18))
                image.putdata([((patient * 5 + x * 3) % 256,
                                (label * 70 + y * 7) % 256, (x * y + 90) % 256)
                               for y in range(18) for x in range(32)])
                image.save(images / f'{patient}{name}.png')
        rows, cls.metadata = audit_dataset(images, strict=False)
        cache = cls.template_root / 'cache'
        cache_predictions(rows, cache,
                          lambda image: np.tile(np.linspace(.1, .9, 16, dtype=np.float32), (8, 1)),
                          {'synthetic_teacher': 'fixed for both seeds'})
        reference = cls.template_root / 'reference'
        with patch.object(old, 'audit_dataset', return_value=(rows, cls.metadata)):
            _, suite = old.prepare(reference, images)
        suite['runtime_identity'] = {'python': 'synthetic-python', 'packages': {'torch': 'synthetic'},
                                     'cuda': 'synthetic-cuda', 'cudnn': 1, 'gpu': 'synthetic-gpu'}
        old._atomic_json(reference / 'suite.json', suite)
        by_name = {r['image_name']: r for r in rows}
        for job in suite['identity']['jobs']:
            binding = None if job['method'] == 'ce' else validate_cache(
                cache, [by_name[n] for n in job['support']])
            cls.write_run(reference, suite, job, binding, initial='old-seed42-initial')
            if binding:
                suite['teacher_sha256'] = binding['teacher_sha256']
        old.write_aggregates(reference, suite)

    @classmethod
    def tearDownClass(cls):
        cls.template.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        for name in ('images', 'cache', 'reference'):
            shutil.copytree(self.template_root / name, self.base / name)
        self.images = self.base / 'images'
        self.cache = self.base / 'cache'
        self.reference = self.base / 'reference'
        self.output = self.base / 'seed43'
        self.rows, self.metadata = audit_dataset(self.images, strict=False)
        self.by_name = {r['image_name']: r for r in self.rows}
        self.old_suite = old.read_json(self.reference / 'suite.json')
        for module in (old, g):
            for name, value in [('audit_dataset', (self.rows, self.metadata))]:
                if hasattr(module, name):
                    p = patch.object(module, name, return_value=value)
                    p.start()
                    self.addCleanup(p.stop)
            if hasattr(module, 'EXPECTED_FINGERPRINT'):
                p = patch.object(module, 'EXPECTED_FINGERPRINT', self.metadata['dataset_fingerprint'])
                p.start()
                self.addCleanup(p.stop)

    def prepare(self, *, smoke=False):
        _, self.suite = g.prepare(self.output, self.images, self.reference,
                                 cache_root=self.cache, smoke=smoke)
        self.suite['runtime_identity'] = self.old_suite['runtime_identity']
        old._atomic_json(self.output / 'suite.json', self.suite)
        return self.suite

    @staticmethod
    def mark(root, suite, job):
        directory = root / 'runs' / job['run_id']
        old._atomic_json(directory / 'complete.json', {
            'run_id': job['run_id'], 'identity_sha256': suite['identity_sha256'],
            'artifact_sha256': {name: old.sha256_file(directory / name)
                                for name in (*old.RUN_FILES[:-1], 'model.pt')}})

    @classmethod
    def write_run(cls, root, suite, job, binding, *, initial='new-seed43-initial'):
        import numpy as np
        directory = root / 'runs' / job['run_id']
        directory.mkdir(parents=True)
        by_name = {r['image_name']: r for r in suite['identity']['data']}
        rows = [by_name[n] for n in job['eval']]
        probs = np.asarray([[.8 if i == r['label_id'] else .1 for i in range(3)]
                            for r in rows], dtype=np.float32)
        config = suite['identity']['config']
        old._atomic_json(directory / 'config.json', {
            'identity_sha256': suite['identity_sha256'], 'job': job, 'config': config,
            'initial_state_sha256': initial, 'cache_binding': binding,
            'execution_runtime': suite['runtime_identity']})
        hist = [{'stage': stage, 'round': i + 1,
                 'steps': config['steps_per_round'],
                 'image_occurrences': config['steps_per_round'] * config['batch_size'],
                 'lr': config['head_lr'] if stage == 'head' else config['finetune_lr'] *
                 (1 + math.cos(math.pi * i / config['finetune_rounds'])) / 2,
                 'mean_train_loss': .5, 'classification_loss': .5,
                 'geometry_loss': 0., 'seconds': .001}
                for stage, count in (('head', config['head_rounds']),
                                     ('finetune', config['finetune_rounds']))
                for i in range(count)]
        old.write_csv(directory / 'history.csv', hist)
        old.write_csv(directory / 'predictions.csv', [
            {'patient_id': r['patient_id'], 'image_name': r['image_name'],
             'label_id': r['label_id'], 'predicted_id': int(p.argmax()),
             **{f'prob_{label}': float(p[i]) for label, i in old.LABEL_TO_ID.items()}}
            for r, p in zip(rows, probs)])
        (directory / 'model.pt').write_bytes(b'synthetic checkpoint fixture')
        (directory / 'student.pt').write_bytes(b'synthetic RGB-only fixture')
        result = {**{k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method')},
                  'n_train_images': len(job['support']),
                  'n_train_patients': len(job['patient_order_prefix']),
                  'initial_state_sha256': initial, 'cache_binding': binding,
                  'bn_buffers_unchanged': True, 'head_stage_backbone_unchanged': True,
                  'finetune_backbone_changed': True,
                  'bn_before_sha256': 'bn', 'bn_after_sha256': 'bn',
                  'initial_backbone_sha256': 'before', 'final_backbone_sha256': 'after',
                  'final_state_sha256': 'final', 'reloaded_state_sha256': 'final',
                  'checkpoint_reload_verified': True,
                  'student_export_sha256': old.sha256_file(directory / 'student.pt'),
                  'student_export_verified': True,
                  'actual_optimizer_steps': len(hist) * config['steps_per_round'],
                  'actual_image_occurrences': len(hist) * config['steps_per_round'] * config['batch_size'],
                  **old.classification_metrics([r['label_id'] for r in rows], probs)}
        old._atomic_json(directory / 'result.json', result)
        cls.mark(root, suite, job)
        return directory

    def fixture(self, job, *, initial='new-seed43-initial'):
        original = next(j for j in self.old_suite['identity']['jobs']
                        if all(j[k] == job[k] for k in ('fold', 'budget', 'repeat', 'method')))
        binding = old.read_json(self.reference / 'runs' / original['run_id'] / 'result.json')['cache_binding']
        return self.write_run(self.output, self.suite, job, binding, initial=initial)

    @staticmethod
    def files(root):
        return {str(p.relative_to(root)): old.sha256_file(p)
                for p in root.rglob('*') if p.is_file()}

    def test_40_jobs_exactly_pair_old_support_and_eval_with_only_training_seed_changed(self):
        before = self.files(self.reference)
        suite = self.prepare()
        jobs = suite['identity']['jobs']
        self.assertEqual(len(jobs), 40)
        self.assertEqual({j['budget'] for j in jobs}, {'5', '10'})
        self.assertEqual(len({j['run_id'] for j in jobs}), 40)
        self.assertFalse({j['run_id'] for j in jobs} &
                         {j['run_id'] for j in self.old_suite['identity']['jobs']})
        for job in jobs:
            original = next(j for j in self.old_suite['identity']['jobs']
                            if all(j[k] == job[k] for k in ('fold', 'budget', 'repeat', 'method')))
            for key in ('support', 'eval', 'patient_order_prefix'):
                self.assertEqual(job[key], original[key])
            self.assertFalse(set(job['support']) & set(job['eval']))
            self.assertEqual(job['repeat'], 0)
        config = suite['identity']['config']
        self.assertEqual(config['training_seed'], 43)
        self.assertEqual(config['shuffled_target_seed'], 42)
        for key, value in old.fixed_config().items():
            if key != 'training_seed':
                self.assertEqual(config[key], value, key)
        self.assertEqual((config['head_rounds'] + config['finetune_rounds']) * config['steps_per_round'], 700)
        self.assertEqual(before, self.files(self.reference))

    def test_incomplete_and_semantically_corrupt_seed42_reference_are_rejected(self):
        job = self.old_suite['identity']['jobs'][0]
        directory = self.reference / 'runs' / job['run_id']
        marker = (directory / 'complete.json').read_bytes()
        (directory / 'complete.json').unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        (directory / 'complete.json').write_bytes(marker)
        result = old.read_json(directory / 'result.json')
        result['accuracy'] = .123
        old._atomic_json(directory / 'result.json', result)
        self.mark(self.reference, self.old_suite, job)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_changed_reference_protocol_and_current_source_are_rejected(self):
        suite_path = self.reference / 'suite.json'
        original = suite_path.read_bytes()
        wrong = copy.deepcopy(self.old_suite)
        wrong['identity']['config']['training_seed'] = 43
        wrong['identity_sha256'] = old._digest(wrong['identity'])
        old._atomic_json(suite_path, wrong)
        with self.assertRaises(ValueError):
            self.prepare()
        suite_path.write_bytes(original)
        wrong = copy.deepcopy(self.old_suite)
        wrong['identity']['source_sha256']['src/geometry_experiment.py'] = '0' * 64
        wrong['identity_sha256'] = old._digest(wrong['identity'])
        old._atomic_json(suite_path, wrong)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_output_overlap_rejected_without_mutating_inputs(self):
        for protected in (self.reference, self.images, self.cache):
            before = self.files(protected)
            for output in (protected, protected / 'nested-output'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    g.prepare(output, self.images, self.reference, cache_root=self.cache)
                with self.subTest(cli_output=output), self.assertRaises(ValueError):
                    g.main(['plan', '--output-root', str(output), '--image-root', str(self.images),
                            '--reference-root', str(self.reference), '--cache-root', str(self.cache)])
            self.assertEqual(before, self.files(protected))
        with self.assertRaises(ValueError):
            g.prepare(self.base, self.images, self.reference, cache_root=self.cache)
        before = self.files(self.reference)
        for mode in ('status', 'pack'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                g.main([mode, '--output-root', str(self.reference),
                        '--reference-root', str(self.reference)])
        self.assertEqual(before, self.files(self.reference))
        # The old output is still protected when another reference copy is passed.
        alternate_reference = self.base / 'alternate-reference'
        shutil.copytree(self.reference, alternate_reference)
        for mode in ('plan', 'check', 'run'):
            with self.subTest(mode=mode), patch.object(g, 'check_dependencies'), \
                    patch.object(g, 'configure_torch'), \
                    patch.object(g, 'environment', return_value=self.old_suite['runtime_identity']), \
                    patch.object(g, 'train_one') as train:
                with self.assertRaises(ValueError):
                    g.main([mode, '--output-root', str(self.reference), '--image-root', str(self.images),
                            '--reference-root', str(alternate_reference), '--cache-root', str(self.cache)])
                train.assert_not_called()
                self.assertEqual(before, self.files(self.reference))

    def test_resume_rejects_changed_plan_reference_and_source(self):
        self.prepare()
        g.prepare(self.output, self.images, self.reference, cache_root=self.cache)
        plan = self.output / 'support_plan.csv'
        contents = plan.read_bytes()
        plan.write_bytes(contents + b'corrupt\n')
        with self.assertRaises(ValueError):
            g.prepare(self.output, self.images, self.reference, cache_root=self.cache)
        plan.write_bytes(contents)
        actual_sha256 = g.sha256_file
        with patch.object(g, 'sha256_file', side_effect=lambda path: '0' * 64
                          if str(path).endswith('src/geometry_replication.py') else actual_sha256(path)):
            with self.assertRaises(ValueError):
                g.prepare(self.output, self.images, self.reference, cache_root=self.cache)
        job = self.old_suite['identity']['jobs'][0]
        path = self.reference / 'runs' / job['run_id'] / 'history.csv'
        path.write_bytes(path.read_bytes() + b'corrupt\n')
        with self.assertRaises(ValueError):
            g.prepare(self.output, self.images, self.reference, cache_root=self.cache)

    def test_current_cache_must_match_old_support_maps_not_only_teacher(self):
        import numpy as np
        from src.geometry_distillation import load_cache_manifest, atomic_json, file_hash
        self.prepare()
        job = next(j for j in self.suite['identity']['jobs'] if j['method'] == 'plain')
        g.preflight_cache(self.output, self.suite, [job], self.by_name, self.cache)
        manifest = load_cache_manifest(self.cache)
        key = self.by_name[job['support'][0]]['image_sha256']
        entry = manifest['entries'][key]
        path = self.cache / entry['file']
        with np.load(path, allow_pickle=False) as saved:
            maps, size = saved['disparity'].copy(), saved['original_size'].copy()
        maps[:] = maps[..., ::-1]
        np.savez_compressed(path, disparity=maps, original_size=size)
        entry['sha256'] = file_hash(path)
        atomic_json(self.cache / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            g.preflight_cache(self.output, self.suite, [job], self.by_name, self.cache)

    def test_rehashed_mixed_seed_result_and_wrong_metrics_cannot_resume(self):
        self.prepare()
        job = self.suite['identity']['jobs'][0]
        directory = self.fixture(job)
        self.assertIsNotNone(g.validate_run(self.output, self.suite, job))
        saved_path = directory / 'config.json'
        original = saved_path.read_bytes()
        saved = old.read_json(saved_path)
        saved['config']['training_seed'] = 42
        old._atomic_json(saved_path, saved)
        self.mark(self.output, self.suite, job)
        with self.assertRaises(ValueError):
            g.validate_run(self.output, self.suite, job)
        saved_path.write_bytes(original)
        result = old.read_json(directory / 'result.json')
        result['accuracy'] = .123
        old._atomic_json(directory / 'result.json', result)
        self.mark(self.output, self.suite, job)
        with self.assertRaises(ValueError):
            g.validate_run(self.output, self.suite, job)

    def test_partial_selected_pack_never_claims_entire_suite_completed(self):
        self.prepare()
        selected = [j for j in self.suite['identity']['jobs']
                    if j['method'] == 'ce' and j['budget'] == '5']
        self.fixture(selected[0])
        self.assertEqual(g.aggregate(self.output, self.suite)['oof'], [])
        args = g.parser().parse_args(['pack', '--methods', 'ce', '--budgets', '5'])
        with self.assertRaises(ValueError):
            g.pack_suite(self.output, args)
        for job in selected[1:]:
            self.fixture(job)
        g.pack_suite(self.output, args)
        state = old.read_json(self.output / 'suite.json')
        scope = old.read_json(self.output / 'package_scope.json')
        self.assertEqual(state['completed_runs'], 5)
        self.assertNotEqual(state['status'], 'completed')
        self.assertFalse(scope['entire_suite_complete'])
        self.assertEqual(scope['planned_runs'], 40)
        self.assertEqual(scope['selected_completed_runs'], 5)

    def test_full_pack_is_self_contained_offline_and_reference_corruption_is_detected(self):
        self.prepare()
        for job in self.suite['identity']['jobs']:
            self.fixture(job)
        data = g.write_aggregates(self.output, self.suite)
        self.assertEqual(len(data['results']), 40)
        self.assertEqual(len(data['oof']), 8)
        self.assertEqual(len(data['paired']), 6)
        for directory in (self.images, self.cache, self.reference):
            shutil.rmtree(directory)
        for path in self.output.rglob('*.pt'):
            path.unlink()
        args = g.parser().parse_args(['pack'])
        archive = g.pack_suite(self.output, args)
        extracted = self.base / 'extracted'
        extracted.mkdir()
        with tarfile.open(archive) as packed:
            names = packed.getnames()
            self.assertFalse(any(n.endswith(('.pt', '.png', '.npz')) for n in names))
            self.assertTrue(any(n.startswith('reference_seed42/') for n in names))
            packed.extractall(extracted, filter='data')
        g.main(['status', '--output-root', str(extracted)])
        g.pack_suite(extracted, args)
        state = old.read_json(extracted / 'suite.json')
        self.assertEqual(state['completed_runs'], 40)
        reference_result = next((extracted / 'reference_seed42' / 'runs').glob('*/result.json'))
        reference_result.write_text('{}\n')
        with self.assertRaises(ValueError):
            g.main(['status', '--output-root', str(extracted)])

    def test_aggregation_rejects_mixed_initializations(self):
        self.prepare()
        jobs = self.suite['identity']['jobs'][:2]
        self.fixture(jobs[0])
        self.fixture(jobs[1], initial='old-seed42-initial')
        with self.assertRaises(ValueError):
            g.aggregate(self.output, self.suite)

    def test_formal_runtime_mismatch_is_rejected_before_any_training(self):
        self.prepare()
        changed = {**self.old_suite['runtime_identity'], 'gpu': 'different-synthetic-gpu'}
        args = g.parser().parse_args(['run', '--output-root', str(self.output),
                                     '--reference-root', str(self.reference), '--image-root', str(self.images),
                                     '--cache-root', str(self.cache), '--methods', 'ce'])
        with patch.object(g, 'environment', return_value=changed), \
                patch.object(g, 'check_dependencies'), patch.object(g, 'configure_torch'), \
                patch.object(g, 'train_one') as train:
            with self.assertRaises(ValueError):
                g.run_suite(args)
            train.assert_not_called()

    def test_real_tiny_four_methods_pair_seed43_keep_target_seed42_and_export_rgb(self):
        import torch
        from src import geometry_distillation as geometry

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

        self.prepare(smoke=True)
        jobs = [j for j in self.suite['identity']['jobs'] if j['fold'] == 0 and j['budget'] == '5']
        samples = []
        dataset_class = geometry.GeometryImageDataset

        class ObserveDataset(dataset_class):
            def __getitem__(self, index):
                sample = super().__getitem__(index)
                samples.append((self.rows[index]['image_sha256'], sample['sample_seed']))
                return sample

        old.seed_training(42)
        seed42_fc = Tiny().fc[1].weight.detach().clone()
        old.seed_training(43)
        seed43_fc = Tiny().fc[1].weight.detach().clone()
        self.assertFalse(torch.equal(seed42_fc, seed43_fc))
        results = []
        with patch.object(geometry, 'GeometryImageDataset', ObserveDataset):
            for job in jobs:
                initialized = []

                def factory(pretrained):
                    self.assertFalse(pretrained, 'Synthetic smoke must not download weights')
                    model = Tiny()
                    initialized.append(model.fc[1].weight.detach().clone())
                    return model

                result = g.train_one(self.output, self.suite, job, self.by_name,
                                     torch.device('cpu'), self.cache, model_factory=factory)
                self.assertTrue(torch.equal(initialized[0], seed43_fc))
                self.assertTrue(result['bn_buffers_unchanged'])
                self.assertTrue(result['head_stage_backbone_unchanged'])
                self.assertTrue(result['finetune_backbone_changed'])
                self.assertTrue(result['student_export_verified'])
                self.assertEqual(result['actual_optimizer_steps'], 2)
                directory = self.output / 'runs' / job['run_id']
                student = torch.load(directory / 'student.pt', weights_only=True)
                self.assertFalse(any(k.startswith('geometry_head.') for k in student))
                history = old.read_csv(directory / 'history.csv')
                if job['method'] != 'ce':
                    self.assertGreater(float(history[-1]['aux_gradient_l1']), 0)
                results.append(result)
                before = self.files(directory)
                g.train_one(self.output, self.suite, job, self.by_name, torch.device('cpu'),
                            self.cache, model_factory=factory)
                self.assertEqual(before, self.files(directory))
        self.assertEqual(len({r['initial_state_sha256'] for r in results}), 1)
        self.assertTrue(samples)
        self.assertTrue(all(seed == (int(image_hash[:12], 16) ^ 42) for image_hash, seed in samples))
        # Cache loading must not change the RGB augmentation stream for paired methods.
        support = [self.by_name[n] for n in jobs[0]['support']]
        rgb = []
        for cache in (None, self.cache):
            old.seed_training(43)
            dataset = dataset_class(support, cache_root=cache, train=True, seed=42)
            rgb.append(dataset[0]['image'])
        self.assertTrue(torch.equal(*rgb))
        shutil.rmtree(self.cache)
        ce = next(j for j in jobs if j['method'] == 'ce')
        self.assertIsNotNone(g.train_one(self.output, self.suite, ce, self.by_name,
                                        torch.device('cpu'), None, model_factory=lambda _: Tiny()))
        geometry_job = next(j for j in jobs if j['method'] != 'ce')
        with self.assertRaises((ValueError, FileNotFoundError)):
            g.train_one(self.output, self.suite, geometry_job, self.by_name,
                        torch.device('cpu'), self.cache, model_factory=lambda _: Tiny())


if __name__ == '__main__':
    unittest.main()
