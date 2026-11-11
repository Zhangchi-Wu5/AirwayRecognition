"""Synthetic-only relation objectives, pairing and matched training streams."""
import math
import random
import unittest

import numpy as np
import torch
from torch import nn

from src import patient_relation as p
from src.fewshot import state_fingerprint
from src.fewshot_finetune import seed_training


def rows(count=5):
    # Deliberately interleaved patient ordering tests mapping by identity.
    return [{'patient_key': str(patient), 'label_id': label,
             'image_name': f'{patient}_{label}.png'}
            for label in range(3) for patient in reversed(range(count))]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 1)
        self.bn = nn.BatchNorm2d(6)
        self.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(6, 3))

    def forward(self, x, return_features=False):
        features = self.bn(self.stem(x)).tanh().mean((2, 3))
        logits = self.fc(features)
        return (logits, features) if return_features else logits


class SyntheticDataset:
    def __init__(self):
        self.rows = rows()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Uses all three global RNG families, so hashes expose cross-arm drift.
        shift = random.random() * .01 + float(np.random.random()) * .01
        image = torch.rand(3, 8, 8) * .1 + row['label_id'] * .2 + index * .01 + shift
        return image, row['label_id']


def config():
    return {'training_seed': 42, 'head_rounds': 1, 'finetune_rounds': 2,
            'steps_per_round': 2, 'head_lr': .001, 'finetune_lr': .001,
            'weight_decay': .0001, 'batch_size': 15, 'num_workers': 0,
            'aux_weight': .1, 'supcon_temperature': .1, 'difference_epsilon': 1e-6}


class PatientRelationCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_triplets_from_interleaved_rows_and_reject_bad_groups(self):
        data = rows()
        groups = p.triplet_indices(data)
        self.assertEqual(len(groups), 5)
        for group in groups:
            self.assertEqual([data[i]['label_id'] for i in group], [0, 1, 2])
            self.assertEqual(len({data[i]['patient_key'] for i in group}), 1)
        for invalid in (data[:-1], data + [dict(data[0])]):
            with self.assertRaises(ValueError):
                p.triplet_indices(invalid)

    def test_regroup_has_no_original_pair_and_preserves_columns_and_rng(self):
        for count in (3, 5, 10, 31):
            data = rows(count)
            groups = p.triplet_indices(data)
            seed_training(991)
            python_before, numpy_before = random.getstate(), np.random.get_state()
            torch_before = torch.get_rng_state().clone()
            shuffled = p.regroup_triplets(groups, 20261111)
            self.assertEqual(shuffled, p.regroup_triplets(groups, 20261111))
            self.assertEqual(random.getstate(), python_before)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_before[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
            self.assertEqual(sorted(i for g in shuffled for i in g), list(range(count * 3)))
            for group in shuffled:
                self.assertEqual([data[i]['label_id'] for i in group], [0, 1, 2])
                self.assertEqual(len({data[i]['patient_key'] for i in group}), 3)
        with self.assertRaises(ValueError):
            p.regroup_triplets([[0, 1, 2], [3, 4, 5]], 1)

    def test_direction_loss_matches_analytic_and_is_offset_invariant(self):
        features = torch.tensor([[0., 0.], [1., 0.], [0., 1.],
                                 [9., -7.], [10., -7.], [9., -6.]], dtype=torch.float64)
        indices = [[0, 1, 2], [3, 4, 5]]
        self.assertAlmostEqual(float(p.difference_loss(features, indices)), 0, places=12)
        opposite = features.clone()
        opposite[3:] = -features[:3]
        self.assertAlmostEqual(float(p.difference_loss(opposite, indices)), 2, places=12)
        perturbed = features.clone()
        perturbed[4] += torch.tensor([.4, .7])
        value = p.difference_loss(perturbed, indices)
        translated = perturbed + torch.tensor([[.5, -3.]] * 3 + [[-11., 9.]] * 3)
        self.assertAlmostEqual(float(p.difference_loss(translated, indices)), float(value), places=12)
        self.assertAlmostEqual(float(p.difference_loss(perturbed, indices[::-1])), float(value), places=12)

    def test_regroup_changes_relations_without_changing_images_or_labels(self):
        data = rows()
        groups = p.triplet_indices(data)
        templates = torch.tensor([[0., 0., 0.], [1., .2, .1], [.1, 1., .2]])
        features = torch.stack([templates[r['label_id']] +
                                torch.tensor([float(r['patient_key']) * 3,
                                              float(r['patient_key']) ** 2, 0.]) for r in data])
        self.assertLess(float(p.difference_loss(features, groups)), 1e-6)
        self.assertGreater(float(p.difference_loss(features, p.regroup_triplets(groups, 9))), .1)

    def test_difference_gradient_is_finite_nonzero_and_no_patient_offset_gradient(self):
        torch.manual_seed(11)
        features = torch.randn(15, 7, dtype=torch.float64, requires_grad=True)
        groups = p.triplet_indices(rows())
        p.difference_loss(features, groups).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        for group in groups:
            self.assertTrue(torch.allclose(features.grad[group].sum(0), torch.zeros(7, dtype=torch.float64), atol=1e-12))
        zero = torch.zeros(15, 7, requires_grad=True)
        zero_loss = p.difference_loss(zero, groups)
        zero_loss.backward()
        self.assertEqual(float(zero_loss), 1.)
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_supcon_known_constant_case_and_nonzero_finite_gradients(self):
        labels = torch.tensor([row['label_id'] for row in rows()])
        constant = torch.ones(15, 4)
        self.assertAlmostEqual(float(p.supervised_contrastive_loss(constant, labels)), math.log(14), places=5)
        torch.manual_seed(5)
        features = torch.randn(15, 4, requires_grad=True)
        p.supervised_contrastive_loss(features, labels).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            p.supervised_contrastive_loss(features[:3], [0, 1, 2])
        with self.assertRaises(ValueError):
            p.supervised_contrastive_loss(features, labels, temperature=0)

    def test_invalid_relation_indices_and_labels_fail(self):
        features = torch.randn(6, 4)
        for indices in ([[0, 1, 2], [0, 4, 5]], [[0, 1], [2, 3]], [[0., 1, 2], [3, 4, 5]]):
            with self.assertRaises(ValueError):
                p.difference_loss(features, indices)
        dataset = SyntheticDataset()
        swapped = p.triplet_indices(dataset.rows)
        swapped[0][0], swapped[0][1] = swapped[0][1], swapped[0][0]
        with self.assertRaisesRegex(ValueError, 'class order'):
            p.train_updates(Tiny(), dataset, config(), 'cpu', 'paired', swapped)

    def test_all_arms_matched_streams_stages_bn_and_repeatability(self):
        dataset = SyntheticDataset()
        matched = p.triplet_indices(dataset.rows)
        rematched = p.regroup_triplets(matched, 20261111)
        results, initial, trained, histories = {}, {}, {}, {}
        for method in p.METHODS:
            seed_training(42)
            model = Tiny()
            initial[method] = state_fingerprint(model)
            history, evidence = p.train_updates(model, dataset, config(), 'cpu', method,
                                                rematched if method == 'regrouped' else matched)
            results[method], trained[method], histories[method] = evidence, state_fingerprint(model), history
            self.assertEqual(evidence['actual_optimizer_steps'], 6)
            self.assertEqual(evidence['actual_image_occurrences'], 90)
            self.assertEqual(evidence['effective_support_order'], list(range(15)))
            self.assertEqual(evidence['bn_before_sha256'], evidence['bn_after_sha256'])
            self.assertNotEqual(evidence['initial_backbone_sha256'], evidence['final_backbone_sha256'])
            self.assertTrue(evidence['head_stage_backbone_unchanged'])
            self.assertEqual(history[0]['auxiliary_loss'], 0)
            self.assertEqual(history[0]['auxiliary_pairs'], 0)
            self.assertEqual([r['lr'] for r in history], [.001, .001, .0005])
            expected_pairs = {'ce': 0, 'supcon': 120, 'paired': 60, 'regrouped': 60}[method]
            self.assertTrue(all(h['auxiliary_pairs'] == expected_pairs for h in history[1:]))
            if method == 'ce':
                self.assertTrue(all(h['auxiliary_loss'] == 0 for h in history))
            else:
                self.assertGreater(history[1]['auxiliary_loss'], 0)
        self.assertEqual(len(set(initial.values())), 1)
        head_rows = [{k: v for k, v in history[0].items() if k != 'seconds'}
                     for history in histories.values()]
        self.assertTrue(all(row == head_rows[0] for row in head_rows))
        for key in ('augmentation_sha256', 'head_augmentation_sha256', 'finetune_augmentation_sha256',
                    'training_rng_initial_sha256', 'training_rng_final_sha256'):
            self.assertEqual(len({r[key] for r in results.values()}), 1, key)
        self.assertEqual(len(set(trained.values())), 4)
        seed_training(42)
        again = Tiny()
        history, evidence = p.train_updates(again, dataset, config(), 'cpu', 'paired', matched)
        self.assertEqual(evidence, results['paired'])
        self.assertEqual(state_fingerprint(again), trained['paired'])
        for before, after in zip(histories['paired'], history):
            self.assertEqual({k: v for k, v in before.items() if k != 'seconds'},
                             {k: v for k, v in after.items() if k != 'seconds'})

    def test_wrong_batch_budget_rejected_before_training(self):
        settings = config()
        settings['batch_size'] = 16
        dataset = SyntheticDataset()
        with self.assertRaisesRegex(ValueError, 'entire support'):
            p.train_updates(Tiny(), dataset, settings, 'cpu', 'ce', p.triplet_indices(dataset.rows))


if __name__ == '__main__':
    unittest.main()
