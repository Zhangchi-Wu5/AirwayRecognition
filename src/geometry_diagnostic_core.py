"""Numerical primitives for the isolated geometry mechanism diagnostic.

This is not an accuracy experiment. The field mask is a boundary-connected,
near-constant dark-frame heuristic; internal dark holes and blue marks remain.
All variants share the historical candidate pool, legacy scale and gap filter.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.geometry_distillation import (
    GRID_SIZE, PAIR_COUNT, PAIR_SEED, MIN_RELATIVE_GAP,
    build_resnet18_student, joint_geometry, load_cache_manifest, _read_entry,
)

CONTROL_SEEDS = (42, 43, 44)
VARIANTS = ('legacy_plain', 'field_plain') + tuple(
    f'{kind}_{seed}' for kind in ('random_matched', 'field_shuffled') for seed in CONTROL_SEEDS)


def fixed_config(smoke=False):
    """Frozen constants; no held-out values determine these diagnostic choices."""
    return dict(mode='smoke' if smoke else 'formal', training_seed=44,
                weights='none' if smoke else 'ResNet18_Weights.IMAGENET1K_V1',
                architecture='resnet18', preprocessing='full_frame',
                head_steps=2 if smoke else 140, calibration_steps=2 if smoke else 140,
                batch_size=16, head_lr=1e-3, geometry_lr=1e-3, weight_decay=1e-4,
                head_dropout=.3, geometry_weight=.1, ce_relative_displacement=1e-4,
                auxiliary_relative_displacement=.1, minimum_gradient_norm=1e-8,
                grid_size=GRID_SIZE, pair_count=PAIR_COUNT, pair_seed=PAIR_SEED,
                minimum_relative_gap=MIN_RELATIVE_GAP, control_seeds=list(CONTROL_SEEDS),
                variants=list(VARIANTS), probe_angle=0, audit_angles=[-15, 0, 15],
                field_rgb_center=15, field_rgb_tolerance=2, field_rgb_max=25,
                field_pool_threshold=.999, near_pair_distance_cells=3.0,
                field_definition='4-connected boundary near-(15,15,15) exterior; blue marks retained',
                pair_normalization='legacy valid-grid per-image 5%-95% scale shared by all variants',
                loss_reduction='mean of per-image means; zero-pair images contribute zero',
                calibration='legacy_plain geometry head only on A; may favor legacy',
                warmup='fc only; fixed angle0 RGB without color jitter; cached frozen features',
                virtual_update='stateless SGD-style directions; not an AdamW trajectory',
                diagnostic_only=True)


def _hash_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model, include=None):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if include is not None and not include(name):
            continue
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_student(pretrained=True, dropout=.3):
    """Keep the old CPU initialization sequence, restoring every CUDA RNG state.

    The old auxiliary constructor calls torch.manual_seed in a CPU-only fork;
    that also seeds CUDA. Restoring CUDA explicitly fixes only that side effect.
    """
    cpu_before = _hash_tensor(torch.get_rng_state())
    cuda_states = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    before = [_hash_tensor(s) for s in cuda_states]
    try:
        model = build_resnet18_student(pretrained=pretrained, dropout=dropout)
    finally:
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    after = [_hash_tensor(s) for s in torch.cuda.get_rng_state_all()] if cuda_states else []
    if before != after:
        raise RuntimeError('Student construction changed a CUDA RNG state')
    model._diagnostic_rng_evidence = dict(cuda_device_count=len(cuda_states),
        cuda_before_sha256=before, cuda_after_sha256=after, cuda_rng_restored=before == after,
        cpu_before_sha256=cpu_before, cpu_after_sha256=_hash_tensor(torch.get_rng_state()),
        cpu_sequence='original builder sequence retained')
    return model


def exterior_mask(rgb):
    """True only for near-15 pixels 4-connected to an original-image boundary."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError('Expected nonempty H x W x 3 RGB image')
    values = rgb.astype(np.float32)
    candidate = (np.abs(values - 15) <= 2).all(2) & (values.max(2) <= 25)
    height, width = candidate.shape
    exterior = np.zeros_like(candidate)
    todo = deque()
    for y, x in [(0, x) for x in range(width)] + [(height - 1, x) for x in range(width)] + [(y, 0) for y in range(height)] + [(y, width - 1) for y in range(height)]:
        if candidate[y, x] and not exterior[y, x]:
            exterior[y, x] = True
            todo.append((y, x))
    while todo:
        y, x = todo.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and candidate[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                todo.append((ny, nx))
    return exterior


def transform_geometry(image, maps, angle=0):
    """Project the original field through exactly the old RGB/map operations."""
    field = (~exterior_mask(np.asarray(image.convert('RGB')))).astype(np.float32)
    # joint_geometry first resizes each map to the resized RGB canvas. Giving it
    # the raw-resolution field separately preserves that exact projection while
    # the teacher maps retain their original cached (resized) resolution.
    rgb, geometry, valid = joint_geometry(image, maps, train=True, angle=angle)
    _, projected, valid_check = joint_geometry(image, field[None], train=True, angle=angle)
    if not torch.equal(valid, valid_check):
        raise RuntimeError('Mask and target transforms are not co-registered')
    return rgb, geometry, valid, valid & (projected[0] > .999)


def make_batch(rows, cache_root, angle=0, need_geometry=True):
    """Read only supplied images, and only their targets when explicitly needed."""
    rows = list(rows)
    if not rows:
        raise ValueError('A batch must contain at least one image')
    manifest = load_cache_manifest(cache_root) if need_geometry else None
    images, targets, valid, field, metadata = [], [], [], [], []
    for row in rows:
        with Image.open(row['path']) as opened:
            image = opened.convert('RGB')
        outside = exterior_mask(np.asarray(image))
        if need_geometry:
            maps, original_size, _ = _read_entry(cache_root, row, manifest)
            if tuple(original_size) != image.size:
                raise ValueError('Image and teacher source dimensions differ')
        else:
            maps = np.zeros((5, 16, 16), dtype=np.float32)
        rgb, maps, mask, field_mask = transform_geometry(image, maps, angle)
        images.append(TF.normalize(TF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD))
        targets.append(maps)
        valid.append(mask)
        field.append(field_mask)
        metadata.append(dict(raw_exterior_pixels=int(outside.sum()), raw_pixels=int(outside.size),
                             raw_exterior_fraction=float(outside.mean())))
    return dict(image=torch.stack(images), label=torch.tensor([int(r['label_id']) for r in rows]),
                geometry=torch.stack(targets), valid=torch.stack(valid), field_valid=torch.stack(field),
                sample_seed=[int(r['image_sha256'][:12], 16) ^ 44 for r in rows],
                rows=rows, mask_metadata=metadata, angle=float(angle), has_geometry=bool(need_geometry))


def candidate_pairs():
    generator = torch.Generator().manual_seed(PAIR_SEED)
    candidates = torch.triu_indices(GRID_SIZE ** 2, GRID_SIZE ** 2, offset=1)
    return candidates[:, torch.randperm(candidates.shape[1], generator=generator)[:PAIR_COUNT]]


def select_pairs(geometry, valid, field_valid, sample_seed=0):
    """Produce eight controls with one common legacy scale and eligibility set."""
    target = geometry.detach().cpu().float().reshape(geometry.shape[0], -1)
    legacy = valid.detach().cpu().bool().flatten() & torch.isfinite(target).all(0)
    field = field_valid.detach().cpu().bool().flatten() & legacy
    if target.shape[1] != GRID_SIZE ** 2 or legacy.numel() != GRID_SIZE ** 2:
        raise ValueError('Expected the historical 14 x 14 geometry grid')
    i, j = candidate_pairs()
    near = ((i // GRID_SIZE - j // GRID_SIZE).float().square() +
            (i % GRID_SIZE - j % GRID_SIZE).float().square()) <= 3.0 ** 2
    valid_pairs = legacy[i] & legacy[j]
    field_positions = field[i] & field[j]
    touch = valid_pairs & ~field_positions
    scale = low = high = float('nan')
    labels = torch.zeros(PAIR_COUNT)
    eligible = torch.zeros(PAIR_COUNT, dtype=torch.bool)
    if int(legacy.sum()) >= 2:
        low_t, high_t = torch.quantile(target[0, legacy], torch.tensor([.05, .95]))
        scale_t = (high_t - low_t).clamp_min(1e-8)
        low, high, scale = float(low_t), float(high_t), float(scale_t)
        delta = (target[0, i] - target[0, j]) / scale_t
        labels = delta.sign()
        eligible = valid_pairs & (delta.abs() >= MIN_RELATIVE_GAP)
    legacy_selected = torch.nonzero(eligible).flatten()
    field_selected = torch.nonzero(eligible & field_positions).flatten()
    result = {}
    for variant in VARIANTS:
        selected = legacy_selected if variant == 'legacy_plain' else field_selected
        seed = int(variant.rsplit('_', 1)[1]) if variant.startswith(('random_matched_', 'field_shuffled_')) else None
        if variant.startswith('random_matched_'):
            generator = torch.Generator().manual_seed(int(sample_seed) ^ seed)
            selected = legacy_selected[torch.randperm(len(legacy_selected), generator=generator)[:len(field_selected)]]
        y = labels[selected].clone()
        changed = 0
        if variant.startswith('field_shuffled_') and len(selected):
            generator = torch.Generator().manual_seed(int(sample_seed) ^ seed)
            shuffled = y[torch.randperm(len(y), generator=generator)]
            changed = int((y != shuffled).sum())
            y = shuffled
        # Null instead of NaN keeps evidence valid strict JSON when no grid exists.
        actual_scale = scale if math.isfinite(scale) else None
        audit = dict(variant=variant, control_seed=seed, candidate_pairs=PAIR_COUNT,
            legacy_grid_cells=int(legacy.sum()), field_grid_cells=int(field.sum()),
            field_grid_fraction=float(field.sum()) / int(legacy.sum()) if int(legacy.sum()) else 0.,
            legacy_pairs_before_gap=int(valid_pairs.sum()), field_pairs_before_gap=int((valid_pairs & field_positions).sum()),
            legacy_pairs=int(eligible.sum()), field_pairs=len(field_selected), selected_pairs=len(selected),
            selected_to_legacy_fraction=len(selected) / len(legacy_selected) if len(legacy_selected) else 0.,
            touch_border_pairs_before_gap=int(touch.sum()), touch_border_pairs_after_gap=int((touch & eligible).sum()),
            selected_touch_border_pairs=int((~field_positions[selected]).sum()),
            near_pairs_before_eligibility=int(near.sum()), far_pairs_before_eligibility=int((~near).sum()),
            near_pairs_after_legacy=int((near & eligible).sum()), far_pairs_after_legacy=int((~near & eligible).sum()),
            near_pairs_selected=int(near[selected].sum()), far_pairs_selected=int((~near[selected]).sum()),
            positive_pairs=int((y > 0).sum()), negative_pairs=int((y < 0).sum()),
            shuffled_changed_pairs=changed, shuffled_changed_fraction=changed / len(selected) if len(selected) else 0.0,
            scale=actual_scale, quantile05=low if math.isfinite(low) else None,
            quantile95=high if math.isfinite(high) else None,
            image_pair_denominator=len(selected), loss_normalization='per-image selected-pair mean; batch image mean',
            valid=bool(len(selected)))
        result[variant] = dict(i=i[selected], j=j[selected], labels=y, scale=actual_scale, audit=audit)
    return result


def _batch_pairs(batch):
    # Pair preparation is immutable CPU metadata; do not retain any model graph.
    batch_size = len(batch['geometry'])
    if '_diagnostic_pairs' not in batch:
        batch['_diagnostic_pairs'] = [select_pairs(batch['geometry'][n], batch['valid'][n],
            batch['field_valid'][n], batch.get('sample_seed', [0] * batch_size)[n])
            for n in range(batch_size)]
    return batch['_diagnostic_pairs']


def audit_pairs(batch):
    if not batch.get('has_geometry', True):
        raise ValueError('Pair audit requires explicitly requested geometry targets')
    records = []
    batch_size = len(batch['geometry'])
    for n, variants in enumerate(_batch_pairs(batch)):
        row = batch.get('rows', [{}] * batch_size)[n]
        metadata = batch.get('mask_metadata', [{}] * batch_size)[n]
        for variant in VARIANTS:
            records.append(dict(image_name=row.get('image_name', str(n)), patient_key=row.get('patient_key', ''),
                angle=batch.get('angle', 0), **metadata, **variants[variant]['audit'],
                batch_image_denominator=batch_size))
    return records


def diagnostic_relation_loss(predicted, batch, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown diagnostic variant: ' + variant)
    prediction = predicted.flatten(1)
    if prediction.shape != (len(batch['geometry']), GRID_SIZE ** 2):
        raise ValueError('Prediction must be B x 1 x 14 x 14')
    losses, count, empty = [], 0, 0
    for n, variants in enumerate(_batch_pairs(batch)):
        pair = variants[variant]
        count += len(pair['i'])
        if not len(pair['i']):
            losses.append(prediction[n].sum() * 0)
            empty += 1
        else:
            i, j, labels = [pair[key].to(prediction.device) for key in ('i', 'j', 'labels')]
            losses.append(torch.nn.functional.softplus(-labels * (prediction[n, i] - prediction[n, j])).mean())
    return torch.stack(losses).mean(), dict(selected_pairs=count, zero_pair_images=empty,
        batch_image_denominator=len(losses), valid=count > 0)


def _subset_batch(batch, indices):
    ids = indices.tolist()
    n = len(batch['image'])
    selected = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and len(value) == n:
            selected[key] = value[indices]
        elif isinstance(value, list) and len(value) == n:
            selected[key] = [value[i] for i in ids]
        else:
            selected[key] = value
    return selected


def _bn_digest(model):
    return state_digest(model, lambda n: n.endswith(('running_mean', 'running_var', 'num_batches_tracked')))


def prepare_model(model, Arows, cache_root, config, device):
    """Fit fc then geometry_head on A only; the shared backbone never changes."""
    rows = list(Arows)
    if not rows:
        raise ValueError('Preparation requires A rows')
    torch.manual_seed(config['training_seed'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batch = make_batch(rows, cache_root, angle=0, need_geometry=True)
    pairs = _batch_pairs(batch)
    pair_count = sum(p['legacy_plain']['audit']['selected_pairs'] for p in pairs)
    if pair_count == 0:
        raise ValueError('A contains no eligible legacy pairs; calibration is invalid')
    backbone = lambda n: not n.startswith(('fc.', 'geometry_head.'))
    initial_backbone = state_digest(model, backbone)
    initial_bn = _bn_digest(model)
    features = {}
    handles = [model.fc.register_forward_pre_hook(lambda m, args: features.__setitem__('fc', args[0].detach())),
               model.geometry_head.register_forward_pre_hook(lambda m, args: features.__setitem__('geometry', args[0].detach()))]
    try:
        with torch.no_grad():
            model(batch['image'].to(device), return_geometry=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(features) != {'fc', 'geometry'}:
        raise ValueError('Model must call fc and geometry_head with cached feature inputs')
    history = []
    generator = torch.Generator().manual_seed(config['training_seed'])
    labels = batch['label'].to(device)
    for stage, head, steps, lr in [('head', model.fc, config['head_steps'], config['head_lr']),
                                  ('calibration', model.geometry_head, config['calibration_steps'], config['geometry_lr'])]:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)
        model.eval()
        head.train(stage == 'head')
        if stage == 'calibration':
            fc_before = state_digest(model, lambda n: n.startswith('fc.'))
            bn_before = _bn_digest(model)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=config['weight_decay'])
        for step in range(steps):
            indices = torch.randint(len(rows), (config['batch_size'],), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            if stage == 'head':
                loss = torch.nn.functional.cross_entropy(head(features['fc'][indices.to(device)]), labels[indices.to(device)])
            else:
                predictions = head(features['geometry'][indices.to(device)])
                loss, _ = diagnostic_relation_loss(predictions, _subset_batch(batch, indices), 'legacy_plain')
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite preparation loss')
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.grad is not None):
                raise ValueError('Nonfinite preparation gradient')
            optimizer.step()
            scalar = float(loss.detach())
            history.append(dict(stage=stage, update=step + 1, loss=scalar,
                ce_loss=scalar if stage == 'head' else 0., geometry_loss=scalar if stage == 'calibration' else 0.,
                lr=lr, image_occurrences=len(indices)))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    final_backbone = state_digest(model, backbone)
    fc_after, bn_after = state_digest(model, lambda n: n.startswith('fc.')), _bn_digest(model)
    evidence = dict(A_image_count=len(rows), A_patient_count=len({r['patient_key'] for r in rows}),
        actual_head_steps=config['head_steps'], actual_calibration_steps=config['calibration_steps'],
        initial_backbone_sha256=initial_backbone, final_backbone_sha256=final_backbone,
        backbone_unchanged=initial_backbone == final_backbone,
        calibration_fc_before_sha256=fc_before, calibration_fc_after_sha256=fc_after,
        calibration_fc_unchanged=fc_before == fc_after,
        calibration_bn_before_sha256=bn_before, calibration_bn_after_sha256=bn_after,
        calibration_bn_unchanged=bn_before == bn_after == initial_bn,
        calibration_pair_count=pair_count, calibration_bias='legacy_plain calibration may favor legacy',
        rng=getattr(model, '_diagnostic_rng_evidence', {'injected_test_model': True}),
        preparation_image_names=[r['image_name'] for r in rows])
    if not all(evidence[k] for k in ('backbone_unchanged', 'calibration_fc_unchanged', 'calibration_bn_unchanged')):
        raise RuntimeError('Preparation changed frozen backbone/BN/classifier state')
    return dict(history=history, evidence=evidence)


def gradient_norm(values):
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))


def gradient_cosine(left, right):
    a, b = gradient_norm(left), gradient_norm(right)
    if not math.isfinite(a * b) or a <= 1e-8 or b <= 1e-8:
        return None
    return sum(float((x.detach().double() * y.detach().double()).sum()) for x, y in zip(left, right)) / (a * b)


def _gradient(loss, parameters):
    if not loss.requires_grad:
        return tuple(torch.zeros_like(p) for p in parameters)
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads))


def evaluate_virtual_loss(model, parameters, ce_grad, aux_grad, ce_norm, aux_norm, loss_fn):
    """Evaluate an exact norm-controlled displacement and restore state on error."""
    parameters = list(parameters)
    if len(parameters) != len(ce_grad) or (aux_grad is not None and len(parameters) != len(aux_grad)):
        raise ValueError('Gradient and parameter counts differ')
    ce_length = gradient_norm(ce_grad)
    aux_length = gradient_norm(aux_grad) if aux_grad is not None else None
    if (not math.isfinite(ce_length) or ce_length <= 1e-8 or ce_norm < 0 or not math.isfinite(ce_norm)
            or not math.isfinite(aux_norm) or aux_norm < 0
            or (aux_grad is not None and (not math.isfinite(aux_length) or aux_length <= 1e-8))):
        raise ValueError('Invalid/near-zero gradient or displacement norm')
    snapshot = {n: v.detach().clone() for n, v in model.state_dict().items()}
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.eval()
        with torch.no_grad():
            for index, p in enumerate(parameters):
                p.add_(ce_grad[index], alpha=-ce_norm / ce_length)
                if aux_grad is not None:
                    p.add_(aux_grad[index], alpha=-aux_norm / aux_length)
            value = float(loss_fn())
        if not math.isfinite(value):
            raise ValueError('Nonfinite virtual B loss')
        return value
    finally:
        model.load_state_dict(snapshot)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training


def probe(model, A_batch, B_batch, config):
    """Measure gradients and fixed virtual displacements from one shared basepoint."""
    a_patients = {r['patient_key'] for r in A_batch.get('rows', [])}
    b_patients = {r['patient_key'] for r in B_batch.get('rows', [])}
    if a_patients & b_patients:
        raise ValueError('A and B patient sets overlap')
    if A_batch.get('angle', 0) != 0 or B_batch.get('angle', 0) != 0:
        raise ValueError('Virtual probes require deterministic angle zero')
    device = next(model.parameters()).device
    original_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    original_modes = [(m, m.training) for m in model.modules()]
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    before = state_digest(model)
    parameters = [p for n, p in model.named_parameters() if n.startswith(('conv1.', 'layer1.', 'layer2.', 'layer3.'))]
    if not parameters:
        raise ValueError('Model has no shared conv1/layer1-3 parameters')
    records = []
    try:
        model.eval()
        shared_ids = {id(p) for p in parameters}
        for p in model.parameters():
            p.requires_grad_(id(p) in shared_ids)
        a_images, a_labels = A_batch['image'].to(device), A_batch['label'].to(device)
        b_images, b_labels = B_batch['image'].to(device), B_batch['label'].to(device)
        def loss_b():
            return torch.nn.functional.cross_entropy(model(b_images), b_labels)
        loss_a = torch.nn.functional.cross_entropy(model(a_images), a_labels)
        if not bool(torch.isfinite(loss_a)):
            raise ValueError('Nonfinite basepoint A classification loss')
        g_a = _gradient(loss_a, parameters)
        b_loss = loss_b()
        if not bool(torch.isfinite(b_loss)):
            raise ValueError('Nonfinite basepoint B classification loss')
        g_b = _gradient(b_loss, parameters)
        norm_a, norm_b = gradient_norm(g_a), gradient_norm(g_b)
        parameter_norm = gradient_norm(parameters)
        if not math.isfinite(parameter_norm):
            raise ValueError('Nonfinite shared basepoint parameter norm')
        ce_disp = config['ce_relative_displacement'] * parameter_norm
        aux_disp = config['auxiliary_relative_displacement'] * ce_disp
        ce_valid = math.isfinite(norm_a) and norm_a > config['minimum_gradient_norm'] and math.isfinite(parameter_norm) and parameter_norm > 0
        ce_first_order = (-ce_disp * sum(float((x.double() * y.double()).sum()) for x, y in zip(g_b, g_a)) / norm_a
                          if ce_valid and math.isfinite(norm_b) else None)
        control = evaluate_virtual_loss(model, parameters, g_a, None, ce_disp, 0., loss_b) if ce_valid else None
        for variant in VARIANTS:
            _, prediction = model(a_images, return_geometry=True)
            loss_geo, stats = diagnostic_relation_loss(prediction, A_batch, variant)
            geometry_loss_finite = bool(torch.isfinite(loss_geo))
            g_geo = _gradient(loss_geo, parameters) if geometry_loss_finite else tuple(torch.zeros_like(p) for p in parameters)
            norm_geo = gradient_norm(g_geo) if geometry_loss_finite else float('nan')
            # B's gradient is not normalized, so a zero gradient is permissible;
            # a nonfinite B gradient invalidates the mechanistic comparison.
            valid = ce_valid and math.isfinite(norm_b) and geometry_loss_finite and stats['valid'] and math.isfinite(norm_geo) and norm_geo > config['minimum_gradient_norm']
            reason = '' if valid else ('invalid_ce_gradient' if not ce_valid else
                'invalid_ce_B_gradient' if not math.isfinite(norm_b) else
                'nonfinite_auxiliary_loss' if not geometry_loss_finite else
                'zero_pairs' if not stats['valid'] else 'invalid_auxiliary_gradient')
            value = evaluate_virtual_loss(model, parameters, g_a, g_geo, ce_disp, aux_disp, loss_b) if valid else None
            dot_b_geo = sum(float((x.double() * y.double()).sum()) for x, y in zip(g_b, g_geo)) if math.isfinite(norm_geo) and math.isfinite(norm_b) else None
            changed = sum(p[variant]['audit']['shuffled_changed_pairs'] for p in _batch_pairs(A_batch))
            record = dict(variant=variant, valid=bool(valid), skip_reason=reason,
                ce_A_loss=float(loss_a.detach()), ce_B_base_loss=float(b_loss.detach()), geometry_A_loss=float(loss_geo.detach()) if geometry_loss_finite else None,
                ce_A_gradient_norm=norm_a if math.isfinite(norm_a) else None,
                ce_B_gradient_norm=norm_b if math.isfinite(norm_b) else None,
                auxiliary_gradient_norm=norm_geo if math.isfinite(norm_geo) else None,
                weighted_auxiliary_gradient_norm=config['geometry_weight'] * norm_geo if math.isfinite(norm_geo) else None,
                weighted_auxiliary_to_ce_ratio=config['geometry_weight'] * norm_geo / norm_a if ce_valid and math.isfinite(norm_geo) else None,
                cosine_aux_ce_A=gradient_cosine(g_geo, g_a), cosine_aux_ce_B=gradient_cosine(g_geo, g_b),
                first_order_raw_weighted_ce_B_change=-config['geometry_weight'] * dot_b_geo if dot_b_geo is not None else None,
                first_order_normalized_ce_B_change=-aux_disp * dot_b_geo / norm_geo if valid and dot_b_geo is not None else None,
                first_order_ce_control_B_change=ce_first_order,
                first_order_combined_B_change=ce_first_order - aux_disp * dot_b_geo / norm_geo if valid and dot_b_geo is not None and ce_first_order is not None else None,
                ce_displacement_norm=ce_disp, auxiliary_displacement_norm=aux_disp if valid else None,
                ce_control_B_loss=control, variant_B_loss=value,
                relative_improvement=control - value if valid else None,
                selected_pairs=stats['selected_pairs'], zero_pair_images=stats['zero_pair_images'],
                shuffled_changed_pairs=changed, shuffled_changed_fraction=changed / stats['selected_pairs'] if stats['selected_pairs'] else 0.)
            records.append(record)
    finally:
        model.load_state_dict(original_state)
        for p, flag in original_flags:
            p.requires_grad_(flag)
        for module, training in original_modes:
            module.training = training
    after = state_digest(model)
    if before != after:
        raise RuntimeError('Virtual probes failed to restore the model state')
    evidence = dict(state_before_sha256=before, state_after_sha256=after, state_restored=before == after,
        shared_parameter_count=sum(p.numel() for p in parameters), shared_parameter_norm=parameter_norm,
        ce_displacement_norm=ce_disp, auxiliary_displacement_norm=aux_disp,
        A_image_count=len(A_batch['image']), B_image_count=len(B_batch['image']), patient_disjoint=True,
        virtual_optimizer='stateless norm-controlled SGD-style; not AdamW',
        all_branches_share_basepoint=True, geometry_head_frozen=True)
    return dict(records=records, evidence=evidence)
