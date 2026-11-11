"""Training-only class relations between matched examinations.

All arms have the same single-image ResNet-18 and full-support image stream.
Differences constrain class relations, not causal removal of patient features.
No held-out data, depth teacher, auxiliary head or inference-time pairing is used.
"""
from __future__ import annotations

import hashlib
import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet

from src.baseline_reproduction import require
from src.fewshot import LABEL_TO_ID
from src.fewshot_finetune import (
    _backbone_fingerprint, bn_fingerprint, seed_training, set_training_stage,
)

METHODS = ('ce', 'supcon', 'paired', 'regrouped')


class PatientRelationResNet18(ResNet):
    """Standard ResNet state keys; extra return value is training-only."""

    def __init__(self, pretrained=True, dropout=.3):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 3))

    def forward(self, images, return_features=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        features = torch.flatten(self.avgpool(x), 1)
        logits = self.fc(features)
        return (logits, features) if return_features else logits


def build_model(pretrained=True, dropout=.3):
    return PatientRelationResNet18(pretrained=pretrained, dropout=dropout)


def triplet_indices(rows):
    """Map support-list positions to patient rows, with lt/yz/zz columns."""
    require(tuple(LABEL_TO_ID.values()) == (0, 1, 2), 'Unexpected class ordering')
    groups = {}
    names = set()
    for index, row in enumerate(rows):
        key = str(row['patient_key'])
        label = int(row['label_id'])
        require(bool(key) and label in (0, 1, 2), 'Invalid patient/class identity')
        if 'image_name' in row:
            require(row['image_name'] not in names, 'Repeated support image')
            names.add(row['image_name'])
        group = groups.setdefault(key, {})
        require(label not in group, 'Repeated patient/class support image')
        group[label] = index
    require(len(groups) >= 2, 'Relations require at least two complete patient groups')
    require(all(set(group) == {0, 1, 2} for group in groups.values()),
            'Each patient must contribute all three classes')
    return [[groups[key][label] for label in (0, 1, 2)] for key in sorted(groups)]


def _validate_indices(indices, n_images):
    require(len(indices) >= 2 and all(len(row) == 3 for row in indices),
            'Relation indices must contain complete three-class groups')
    flattened = [index for row in indices for index in row]
    require(all(isinstance(index, int) and not isinstance(index, bool) for index in flattened),
            'Relation indices must be integers')
    require(sorted(flattened) == list(range(n_images)),
            'Relation groups must use each support image exactly once')


def regroup_triplets(indices, seed):
    """Preserve class columns and image counts, destroying every patient pair.

    A locally shuffled patient order and three distinct circular offsets make
    each pseudo-group contain three distinct source patients. Global Python,
    NumPy and Torch RNG streams are untouched.
    """
    _validate_indices(indices, len(indices) * 3)
    count = len(indices)
    require(count >= 3, 'Regrouping requires at least three patients')
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    offsets = rng.sample(range(count), 3)
    regrouped = [[indices[order[(p + offsets[c]) % count]][c] for c in range(3)]
                 for p in range(count)]
    _validate_indices(regrouped, count * 3)
    owners = {index: p for p, group in enumerate(indices) for index in group}
    require(all(len({owners[index] for index in row}) == 3 for row in regrouped),
            'Regrouping retained a matched patient pair')
    return regrouped


def supervised_contrastive_loss(features, labels, temperature=.1):
    """One-view supervised contrastive objective; all same-class peers positive.

    This is the SupCon loss used in joint fine-tuning, not a replication of the
    original paper's multi-stage training recipe. There is no projection head.
    """
    require(features.ndim == 2 and features.shape[0] >= 2, 'Expected feature matrix')
    require(math.isfinite(temperature) and temperature > 0, 'Temperature must be positive')
    labels = torch.as_tensor(labels, device=features.device)
    require(labels.ndim == 1 and len(labels) == len(features), 'Label/feature mismatch')
    normalized = F.normalize(features, dim=1, eps=1e-6)
    similarities = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    require(bool((counts > 0).all()), 'Every anchor must have another same-class image')
    log_denominator = torch.logsumexp(similarities.masked_fill(diagonal, -torch.inf), dim=1)
    # Avoid 0 * -inf on the diagonal by summing finite positive similarities.
    positive_mean = similarities.masked_fill(~positives, 0).sum(1) / counts
    return (log_denominator - positive_mean).mean()


def difference_loss(features, relation_indices, epsilon=1e-6):
    """Mean direction disagreement for the same ordered class pair across IDs.

    The three directions are lt->yz, lt->zz, yz->zz. Each unordered patient
    pair contributes once for each direction. A zero difference has normalized
    direction zero and disagreement one, without silently discarding pairs.
    """
    require(features.ndim == 2, 'Expected feature matrix')
    require(math.isfinite(epsilon) and epsilon > 0, 'Epsilon must be positive')
    _validate_indices(relation_indices, len(features))
    index = torch.as_tensor(relation_indices, dtype=torch.long, device=features.device)
    groups = features[index]
    differences = groups[:, (1, 2, 2)] - groups[:, (0, 0, 1)]
    directions = F.normalize(differences, dim=-1, eps=epsilon)
    cosine = torch.einsum('pcd,qcd->cpq', directions, directions).clamp(-1, 1)
    left, right = torch.triu_indices(len(groups), len(groups), offset=1, device=features.device)
    return (1 - cosine[:, left, right]).mean()


def _rng_fingerprint(device):
    import numpy as np
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    state = np.random.get_state()
    digest.update(state[0].encode())
    digest.update(state[1].tobytes())
    digest.update(repr(state[2:]).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.device(device).type == 'cuda':
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


def _update_batch_digest(digest, images, labels):
    digest.update(str(tuple(images.shape)).encode())
    digest.update(str(images.dtype).encode())
    digest.update(images.detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())


def train_updates(model, dataset, config, device, method, relation_indices):
    """Fit on the provided support only, returning measured stream evidence.

    Each step reads every support image once in stable index order and uses
    one augmented view. Every arm runs the same forward path and RNG stream.
    Dataset access never includes evaluation data or a worker RNG stream.
    """
    require(method in METHODS, 'Unknown patient relation method')
    _validate_indices(relation_indices, len(dataset))
    for key in ('head_rounds', 'finetune_rounds', 'steps_per_round'):
        require(isinstance(config[key], int) and config[key] > 0, 'Invalid training schedule: ' + key)
    weight = config.get('aux_weight', .1)
    require(math.isfinite(weight) and weight > 0, 'Auxiliary weight must be positive')
    require(config.get('num_workers', 0) == 0, 'Full-support training uses no loader workers')
    if 'batch_size' in config:
        require(config['batch_size'] == len(dataset), 'Batch must contain the entire support exactly once')
    # Reset AFTER model initialization, identically for all arms. There is no
    # auxiliary-head initialization or CPU-only fork that changes CUDA RNG.
    seed_training(config['training_seed'])
    rng_before = _rng_fingerprint(device)
    bn_before = bn_fingerprint(model)
    backbone_before = _backbone_fingerprint(model)
    stream = hashlib.sha256()
    stage_streams = {stage: hashlib.sha256() for stage in ('head', 'finetune')}
    history = []
    verified_labels = None
    for stage, rounds, learning_rate in (
        ('head', config['head_rounds'], config['head_lr']),
        ('finetune', config['finetune_rounds'], config['finetune_lr']),
    ):
        set_training_stage(model, stage)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=learning_rate, weight_decay=config['weight_decay'])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds)
                     if stage == 'finetune' else None)
        for iteration in range(rounds):
            started = time.perf_counter()
            sums = dict(total=0., ce=0., auxiliary=0., pairs=0)
            current_lr = optimizer.param_groups[0]['lr']
            for _ in range(config['steps_per_round']):
                batch = [dataset[index] for index in range(len(dataset))]
                images_cpu = torch.stack([item[0] for item in batch])
                labels_cpu = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
                require(bool(torch.isfinite(images_cpu).all()), 'Nonfinite training image')
                if verified_labels is None:
                    require(all([int(labels_cpu[index]) for index in group] == [0, 1, 2]
                                for group in relation_indices), 'Relation columns do not match class order')
                    verified_labels = labels_cpu.clone()
                else:
                    require(torch.equal(labels_cpu, verified_labels), 'Dataset label order changed')
                _update_batch_digest(stream, images_cpu, labels_cpu)
                _update_batch_digest(stage_streams[stage], images_cpu, labels_cpu)
                images, labels = images_cpu.to(device), labels_cpu.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model(images, return_features=True)
                ce = F.cross_entropy(logits, labels)
                auxiliary = features.new_zeros(())
                pairs = 0
                if stage == 'finetune' and method == 'supcon':
                    auxiliary = supervised_contrastive_loss(
                        features, labels, temperature=config.get('supcon_temperature', .1))
                    pairs = int(sum(int((labels == label).sum()) - 1 for label in labels))
                elif stage == 'finetune' and method in ('paired', 'regrouped'):
                    auxiliary = difference_loss(features, relation_indices,
                                                epsilon=config.get('difference_epsilon', 1e-6))
                    count = len(relation_indices)
                    pairs = 3 * count * (count - 1) // 2
                loss = ce + weight * auxiliary
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                            if p.grad is not None), 'Nonfinite gradient')
                optimizer.step()
                sums['total'] += float(loss.detach())
                sums['ce'] += float(ce.detach())
                sums['auxiliary'] += float(auxiliary.detach())
                sums['pairs'] += pairs
            steps = config['steps_per_round']
            history.append({
                'stage': stage, 'round': iteration + 1, 'steps': steps,
                'image_occurrences': steps * len(dataset), 'lr': current_lr,
                'mean_train_loss': sums['total'] / steps,
                'classification_loss': sums['ce'] / steps,
                'auxiliary_loss': sums['auxiliary'] / steps,
                'auxiliary_pairs': sums['pairs'], 'seconds': time.perf_counter() - started,
            })
            print(f"  {stage} {iteration + 1}/{rounds}: ce={sums['ce']/steps:.5f}, "
                  f"aux={sums['auxiliary']/steps:.5f}", flush=True)
            if scheduler is not None:
                scheduler.step()
        if stage == 'head':
            require(_backbone_fingerprint(model) == backbone_before, 'Head stage changed backbone')
    bn_after = bn_fingerprint(model)
    backbone_after = _backbone_fingerprint(model)
    require(bn_before == bn_after, 'BatchNorm running statistics changed')
    require(backbone_before != backbone_after, 'Whole-network fine-tuning did not change backbone')
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
            'Nonfinite final model state')
    evidence = {
        'bn_before_sha256': bn_before, 'bn_after_sha256': bn_after, 'bn_buffers_unchanged': True,
        'initial_backbone_sha256': backbone_before, 'final_backbone_sha256': backbone_after,
        'head_stage_backbone_unchanged': True, 'finetune_backbone_changed': True,
        'actual_optimizer_steps': sum(row['steps'] for row in history),
        'actual_image_occurrences': sum(row['image_occurrences'] for row in history),
        'augmentation_sha256': stream.hexdigest(),
        'head_augmentation_sha256': stage_streams['head'].hexdigest(),
        'finetune_augmentation_sha256': stage_streams['finetune'].hexdigest(),
        'training_rng_initial_sha256': rng_before,
        'training_rng_final_sha256': _rng_fingerprint(device),
        'effective_support_order': list(range(len(dataset))), 'auxiliary_phase': 'finetune_only',
    }
    return history, evidence
