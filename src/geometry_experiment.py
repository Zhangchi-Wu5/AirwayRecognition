"""Isolated ResNet-18 geometry pilot; immutable 45-job patient-grouped plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import re
from pathlib import Path
import shutil
import tarfile
import time

from src.baseline_reproduction import output_lock, read_json, read_csv, require, verify_prediction_metrics
from src.fewshot import PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics, sha256_file, state_fingerprint, write_csv
from src.fewshot_finetune import (_atomic_json, _digest, RepeatedBatchSampler, ImageRows,
    _worker_seed, seed_training, set_training_stage, bn_fingerprint,
    save_reload_final, evaluate, validate_completed)
from src.renew_data import audit_dataset, EXPECTED_FINGERPRINT, get_transforms
from src.renew_experiment import make_plan as renew_plan, fixed_config as renew_config, portable, configure_torch, environment, runtime_key

VERSION = 'renew_r18_geometry_v1'
METHODS = ('ce', 'plain', 'stable', 'shuffled')
SOURCE_FILES = ('src/geometry_experiment.py', 'src/geometry_distillation.py',
    'src/renew_data.py', 'src/renew_experiment.py', 'src/fewshot_finetune.py',
    'src/fewshot.py', 'src/baseline_reproduction.py', 'src/data.py',
    'scripts/run_geometry_experiment.py', 'scripts/run_geometry_server.sh')
RUN_FILES = ('config.json', 'history.csv', 'predictions.csv', 'result.json', 'complete.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_config(smoke=False):
    c = renew_config()
    c.update(mode='smoke' if smoke else 'formal', weights='none' if smoke else 'ResNet18_Weights.IMAGENET1K_V1',
             architecture='resnet18', preprocessing='full_frame', num_workers=0,
             geometry_weight=0.1, geometry_phase='finetune_only',
             geometry_head='layer3 1x1 conv; frozen during head stage; discarded at inference',
             support_repeats=[0], selection='fixed final checkpoint; no holdout selection',
             geometry='fixed disparity ordering; photo-stability proxy, not ground-truth depth')
    if smoke:
        c.update(head_rounds=1, finetune_rounds=1, steps_per_round=1, batch_size=2)
    return c


def make_plan(rows):
    folds, _, original = renew_plan(rows, renew_config())
    jobs = []
    for j in original:
        if j['method'] != 'full_frame' or j['repeat'] != 0:
            continue
        for method in (('ce',) if j['budget'] == 'full' else METHODS):
            job = dict(j)
            stem = j['run_id'].removesuffix('_full_frame')
            job.update(run_id=stem + '_r18_' + method, method=method)
            jobs.append(job)
    require(len(jobs) == 45, 'Expected exactly 45 pilot jobs')
    return folds, jobs


def artifacts(identity):
    data = identity['data']
    by_name = {r['image_name']: r for r in data}
    result = {'dataset.csv': data}
    for job in identity['jobs']:
        result[job['support_csv']] = [by_name[n] for n in job['support']]
    result['support_plan.csv'] = [{k: j[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method', 'support_csv')} for j in identity['jobs']]
    return result


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong geometry suite version')
    require(identity.get('config') == fixed_config(identity.get('config', {}).get('mode') == 'smoke'), 'Changed fixed protocol')
    require(set(identity.get('source_sha256', {})) == set(SOURCE_FILES), 'Source manifest incomplete')
    rows = identity['data']
    require(len(rows) == 93 and len({r['patient_key'] for r in rows}) == 31, 'Wrong cohort')
    require([r['image_name'] for r in rows] == sorted(r['image_name'] for r in rows), 'Data order changed')
    for index, row in enumerate(rows):
        match = re.fullmatch(r'([0-9]+)\s*(lt|yz|zz)\.png', row['image_name'], flags=re.IGNORECASE)
        require(match is not None and row['source_index'] == index, 'Invalid source image identity')
        require(row['label'] == match.group(2).lower() and row['label_id'] == LABEL_TO_ID[row['label']], 'Image label identity changed')
        require(row['patient_id'] == match.group(1) and row['patient_key'] == str(int(match.group(1))), 'Patient identity changed')
    payload = ''.join(r['image_name'] + '\t' + r['image_sha256'] + '\n' for r in rows)
    import hashlib
    require(hashlib.sha256(payload.encode()).hexdigest() == EXPECTED_FINGERPRINT, 'Wrong source fingerprint')
    folds, jobs = make_plan(rows)
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Changed support/fold plan')


def verify_artifacts(root, suite):
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Suite identity corrupted')
    expected = artifacts(suite['identity'])
    require(set(expected) == set(suite['plan_artifact_sha256']), 'Plan manifest incomplete')
    for name, rows in expected.items():
        path = root / name
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == suite['plan_artifact_sha256'][name], f'Plan file changed: {name}')
        require(read_csv(path) == [{k: str(v) for k, v in r.items()} for r in rows], f'Plan values differ: {name}')


def prepare(root, image_root, *, smoke=False):
    root = Path(root).resolve()
    image_root = Path(image_root).expanduser().resolve()
    require(not root.is_relative_to(image_root) and not image_root.is_relative_to(root), 'Image and output roots must be disjoint')
    try:
        rows, metadata = audit_dataset(image_root)
    except ValueError as error:
        if 'No first-level PNG' in str(error):
            raise ValueError(str(error) + '; set --image-root to the directory directly containing the audited 93 PNGs (for example dataset/renew/气管镜)') from error
        raise
    folds, jobs = make_plan(portable(rows))
    identity = {'version': VERSION, 'config': fixed_config(smoke), 'data': portable(rows),
                'folds': folds, 'jobs': jobs,
                'source_sha256': {n: sha256_file(PROJECT_ROOT / n) for n in SOURCE_FILES}}
    if (root / 'suite.json').exists():
        suite = read_json(root / 'suite.json')
        require(suite['identity'] == identity, 'Resume rejected: source/config/data changed; use a new output root')
        verify_artifacts(root, suite)
        return rows, suite
    require(not root.exists() or not any(p.name != '.reproduction.lock' for p in root.iterdir()), 'Refusing nonempty output without suite.json')
    root.mkdir(parents=True, exist_ok=True)
    for name, values in artifacts(identity).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        write_csv(root / name, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
             'runtime_identity': None, 'status': 'prepared_not_run', 'completed_runs': 0,
             'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
             'plan_artifact_sha256': {n: sha256_file(root / n) for n in artifacts(identity)}}
    _atomic_json(root / 'suite.json', suite)
    return rows, suite


def selected_jobs(suite, args):
    selected = []
    for j in suite['identity']['jobs']:
        if args.phase != 'all' and (j['budget'] == 'full') != (args.phase == 'full'):
            continue
        if args.methods and j['method'] not in args.methods:
            continue
        if args.budgets and j['budget'] not in args.budgets:
            continue
        if args.folds is not None and j['fold'] not in args.folds:
            continue
        selected.append(j)
    require(bool(selected), 'Run filters select no jobs')
    return selected


def cache_binding(cache_root, rows):
    from src.geometry_distillation import validate_cache
    require(cache_root is not None, 'Geometry runs require --cache-root; CE runs do not')
    # This summary must depend only on teacher/config and the requested support.
    return validate_cache(cache_root, rows)


def validate_run(root, suite, job, require_model=True):
    directory = root / 'runs' / job['run_id']
    result = validate_completed(directory, suite['identity_sha256'], job['run_id'], verify_model=require_model)
    if result is None:
        return None
    saved = read_json(directory / 'config.json')
    c = suite['identity']['config']
    require(saved.get('config') == c and saved.get('job') == job and saved.get('identity_sha256') == suite['identity_sha256'], 'Run config changed')
    require(saved.get('execution_runtime') == suite['runtime_identity'], 'Run runtime mismatch')
    require(saved.get('cache_binding') == result.get('cache_binding'), 'Teacher provenance differs')
    require((saved.get('cache_binding') is None) == (job['method'] == 'ce'), 'Missing/unexpected cache provenance')
    for k in ('run_id', 'fold', 'budget', 'repeat', 'method'):
        require(result.get(k) == job[k], 'Result job mismatch: ' + k)
    require(result.get('n_train_images') == len(job['support']) and result.get('n_train_patients') == len(job['patient_order_prefix']), 'Support counts differ')
    require(saved.get('initial_state_sha256') == result.get('initial_state_sha256'), 'Initialization differs')
    require(result.get('student_export_verified') is True and isinstance(result.get('student_export_sha256'), str), 'Missing deployment export evidence')
    if require_model:
        require((directory / 'student.pt').is_file() and sha256_file(directory / 'student.pt') == result['student_export_sha256'], 'Student export corrupted')
    require(result.get('head_stage_backbone_unchanged') and result.get('finetune_backbone_changed'), 'Missing stage invariance evidence')
    for x, y in (('bn_before_sha256', 'bn_after_sha256'), ('final_state_sha256', 'reloaded_state_sha256')):
        require(isinstance(result.get(x), str) and result[x] == result.get(y), 'State evidence inconsistent')
    require(result['initial_backbone_sha256'] != result['final_backbone_sha256'], 'No backbone change')
    hist = read_csv(directory / 'history.csv')
    expected = [(stage, i + 1) for stage, count in (('head', c['head_rounds']), ('finetune', c['finetune_rounds'])) for i in range(count)]
    require([(r['stage'], int(r['round'])) for r in hist] == expected, 'Schedule changed')
    for r in hist:
        lr = c['head_lr'] if r['stage'] == 'head' else c['finetune_lr'] * (1 + math.cos(math.pi * (int(r['round']) - 1) / c['finetune_rounds'])) / 2
        require(int(r['steps']) == c['steps_per_round'] and int(r['image_occurrences']) == c['steps_per_round'] * c['batch_size'], 'Update budget changed')
        require(math.isclose(float(r['lr']), lr, rel_tol=1e-9) and all(math.isfinite(float(r[k])) for k in ('mean_train_loss', 'classification_loss', 'geometry_loss')), 'Invalid history')
        if r['stage'] == 'head' or job['method'] == 'ce':
            require(float(r['geometry_loss']) == 0, 'Geometry loss used in forbidden stage')
    steps = len(expected) * c['steps_per_round']
    require(result.get('actual_optimizer_steps') == steps and result.get('actual_image_occurrences') == steps * c['batch_size'], 'Budget evidence differs')
    by_name = {r['image_name']: r for r in suite['identity']['data']}
    verify_prediction_metrics(directory / 'predictions.csv', result, [by_name[n] for n in job['eval']])
    return result


def backbone_fingerprint(model):
    return _digest({n: v.detach().cpu().contiguous().numpy().tobytes().hex()
                    for n, v in model.state_dict().items()
                    if not n.startswith(('fc.', 'geometry_head.'))})


def train_updates(model, dataset, config, device, method):
    import torch
    from torch.utils.data import DataLoader
    from src.geometry_distillation import relation_loss
    before_bn, before_backbone = bn_fingerprint(model), backbone_fingerprint(model)
    history, round_number = [], 0
    for stage, rounds, lr in (('head', config['head_rounds'], config['head_lr']), ('finetune', config['finetune_rounds'], config['finetune_lr'])):
        set_training_stage(model, stage)
        # The CE control has the same head/init but never updates this auxiliary head.
        for p in model.geometry_head.parameters():
            p.requires_grad = stage == 'finetune' and method != 'ce'
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds) if stage == 'finetune' else None
        for iteration in range(rounds):
            started = time.perf_counter()
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
            generator = torch.Generator().manual_seed(config['training_seed'] + round_number)
            sampler = RepeatedBatchSampler(len(dataset), config['batch_size'], config['steps_per_round'], config['training_seed'] + round_number)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config['num_workers'], worker_init_fn=_worker_seed, generator=generator)
            loss_sum = ce_sum = geo_sum = valid_sum = selected_sum = changed_sum = aux_grad_sum = 0.0
            examples = steps = 0
            current_lr = optimizer.param_groups[0]['lr']
            for batch in loader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                use_geometry = stage == 'finetune' and method != 'ce'
                if use_geometry:
                    logits, pred = model(images, return_geometry=True)
                    geo, stats = relation_loss(pred, batch['geometry'].to(device), batch['valid'].to(device), variant=method, sample_seeds=batch['sample_seed'])
                    valid_sum += stats['valid_pairs']
                    selected_sum += stats['selected_pairs']
                    changed_sum += stats['shuffled_changed_pairs']
                else:
                    logits, geo = model(images), images.new_zeros(())
                ce = torch.nn.functional.cross_entropy(logits, labels)
                loss = ce + config['geometry_weight'] * geo
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None), 'Nonfinite gradient')
                aux_grad_sum += sum(float(p.grad.detach().abs().sum()) for p in model.geometry_head.parameters() if p.grad is not None)
                optimizer.step()
                examples += len(labels)
                steps += 1
                loss_sum += float(loss.detach()) * len(labels)
                ce_sum += float(ce.detach()) * len(labels)
                geo_sum += float(geo.detach()) * len(labels)
            history.append({'stage': stage, 'round': iteration + 1, 'steps': steps, 'image_occurrences': examples,
                            'lr': current_lr, 'mean_train_loss': loss_sum / examples, 'classification_loss': ce_sum / examples,
                            'geometry_loss': geo_sum / examples, 'valid_pairs': valid_sum, 'selected_pairs': selected_sum,
                            'selection_rate': selected_sum / valid_sum if valid_sum else 0.0,
                            'shuffled_changed_pairs': changed_sum,
                            'shuffled_changed_fraction': changed_sum / selected_sum if selected_sum else 0.0,
                            'aux_gradient_l1': aux_grad_sum, 'seconds': time.perf_counter() - started})
            print(f"  {stage} {iteration+1}/{rounds}: loss={loss_sum/examples:.5f}, geometry={geo_sum/examples:.5f}", flush=True)
            if scheduler:
                scheduler.step()
            round_number += 1
        if stage == 'head':
            require(backbone_fingerprint(model) == before_backbone, 'Head stage changed backbone')
    after_bn, after_backbone = bn_fingerprint(model), backbone_fingerprint(model)
    require(before_bn == after_bn and before_backbone != after_backbone, 'BN/backbone training invariance failed')
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), 'Nonfinite final tensor')
    return history, {'bn_before_sha256': before_bn, 'bn_after_sha256': after_bn, 'bn_buffers_unchanged': True,
        'initial_backbone_sha256': before_backbone, 'final_backbone_sha256': after_backbone,
        'head_stage_backbone_unchanged': True, 'finetune_backbone_changed': True,
        'actual_optimizer_steps': sum(r['steps'] for r in history), 'actual_image_occurrences': sum(r['image_occurrences'] for r in history)}


def export_student(model, path, factory, device):
    """Store and verify an RGB-only state dict, excluding training-only head."""
    import torch
    state = {n: t.detach().cpu() for n, t in model.state_dict().items() if not n.startswith('geometry_head.')}
    torch.save(state, path)
    student = factory(False).to(device)
    del student.geometry_head
    student.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.eval()
    student.eval()
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        require(torch.equal(model(probe), student(probe)), 'RGB-only export prediction mismatch')
    return {'student_export_sha256': sha256_file(path), 'student_export_verified': True,
            'student_parameter_count': sum(p.numel() for p in student.parameters())}


def train_one(root, suite, job, by_name, device, cache_root=None, model_factory=None):
    from src.geometry_distillation import GeometryImageDataset, build_resnet18_student
    c = suite['identity']['config']
    support = [by_name[n] for n in job['support']]
    binding = cache_binding(cache_root, support) if job['method'] != 'ce' else None
    if binding is not None:
        require(suite.get('teacher_sha256') in (None, binding['teacher_sha256']), 'Frozen teacher changed across jobs')
        suite['teacher_sha256'] = binding['teacher_sha256']
        _atomic_json(root / 'suite.json', suite)
    completed = validate_run(root, suite, job)
    if completed:
        require(completed['cache_binding'] == binding, 'Resume teacher/cache changed')
        return completed
    directory = root / 'runs' / job['run_id']
    if directory.exists():
        archived = root / 'incomplete_attempts'
        archived.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archived / (job['run_id'] + '_' + str(time.time_ns()))))
    directory.mkdir(parents=True)
    factory = model_factory or (lambda pretrained: build_resnet18_student(pretrained=pretrained, dropout=c['dropout']))
    seed_training(c['training_seed'])
    model = factory(c['mode'] == 'formal').to(device)
    initial = state_fingerprint(model)
    _atomic_json(directory / 'config.json', {'job': job, 'config': c, 'identity_sha256': suite['identity_sha256'],
        'initial_state_sha256': initial, 'cache_binding': binding, 'execution_runtime': suite['runtime_identity']})
    dataset = GeometryImageDataset(support, cache_root=cache_root if binding is not None else None, train=True, seed=c['training_seed'])
    hist, evidence = train_updates(model, dataset, c, device, job['method'])
    write_csv(directory / 'history.csv', hist)
    reloaded, checkpoint = save_reload_final(model, directory / 'model.pt', lambda: factory(False), device)
    exported = export_student(reloaded, directory / 'student.pt', factory, device)
    del model
    eval_rows = [by_name[n] for n in job['eval']]
    probs = evaluate(reloaded, ImageRows(eval_rows, get_transforms('full_frame')), c, device)
    predictions = [{'patient_id': r['patient_id'], 'image_name': r['image_name'], 'label_id': r['label_id'],
                    'predicted_id': int(p.argmax()), **{f'prob_{label}': float(p[i]) for label, i in LABEL_TO_ID.items()}}
                   for r, p in zip(eval_rows, probs)]
    write_csv(directory / 'predictions.csv', predictions)
    result = {**{k: job[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method')},
              'n_train_images': len(support), 'n_train_patients': len(job['patient_order_prefix']),
              'initial_state_sha256': initial, 'cache_binding': binding,
              **classification_metrics([r['label_id'] for r in eval_rows], probs), **evidence, **checkpoint, **exported}
    _atomic_json(directory / 'result.json', result)
    _atomic_json(directory / 'complete.json', {'identity_sha256': suite['identity_sha256'], 'run_id': job['run_id'],
        'artifact_sha256': {n: sha256_file(directory / n) for n in (*RUN_FILES[:-1], 'model.pt')}})
    validate_run(root, suite, job)
    return result


def aggregate(root, suite):
    import numpy as np
    results, predictions = [], []
    for job in suite['identity']['jobs']:
        result = validate_run(root, suite, job, require_model=False)
        if result:
            results.append(result)
            predictions.extend([{**{k: job[k] for k in ('run_id', 'fold', 'budget', 'method', 'repeat')}, **p}
                                for p in read_csv(root / 'runs' / job['run_id'] / 'predictions.csv')])
    require(len({r['initial_state_sha256'] for r in results}) <= 1, 'Paired model initialization differs')
    require(len({r['cache_binding']['teacher_sha256'] for r in results if r['cache_binding']}) <= 1, 'Different frozen teachers mixed')
    oof, paired = [], []
    expected = sorted(r['image_name'] for r in suite['identity']['data'])
    for method in METHODS:
        for budget in ('full', '5', '10'):
            group = [p for p in predictions if p['method'] == method and p['budget'] == budget]
            if {p['fold'] for p in group} != set(range(5)):
                continue
            require(sorted(p['image_name'] for p in group) == expected, 'OOF coverage differs')
            probs = np.asarray([[float(p[f'prob_{label}']) for label in LABEL_TO_ID] for p in group], dtype=np.float32)
            oof.append({'method': method, 'budget': budget, 'repeat': 0, **classification_metrics([int(p['label_id']) for p in group], probs)})
    for row in oof:
        if row['method'] == 'ce':
            continue
        baseline = next((r for r in oof if r['method'] == 'ce' and r['budget'] == row['budget']), None)
        if baseline:
            paired.append({'method': row['method'], 'budget': row['budget'], **{k + '_delta_vs_ce': row[k] - baseline[k] for k in METRIC_NAMES}})
    return {'results': results, 'predictions': predictions, 'oof': oof, 'paired': paired}


def write_aggregates(root, suite):
    data = aggregate(root, suite)
    metrics = [{k: r[k] for k in ('run_id', 'fold', 'budget', 'repeat', 'method', 'n_train_images', 'n_train_patients', *METRIC_NAMES)} for r in data['results']]
    outputs = {'per_run_metrics.csv': metrics, 'oof_predictions.csv': data['predictions'], 'oof_metrics.csv': data['oof'], 'oof_comparison.csv': data['paired']}
    empty_fields = {'per_run_metrics.csv': ['run_id', 'fold', 'budget', 'repeat', 'method', *METRIC_NAMES],
                    'oof_predictions.csv': ['run_id', 'fold', 'budget', 'method', 'repeat', 'patient_id', 'image_name', 'label_id', 'predicted_id', *['prob_' + l for l in LABEL_TO_ID]],
                    'oof_metrics.csv': ['method', 'budget', 'repeat', *METRIC_NAMES],
                    'oof_comparison.csv': ['method', 'budget', *[k + '_delta_vs_ce' for k in METRIC_NAMES]]}
    for name, rows in outputs.items():
        write_csv(root / name, rows, fieldnames=None if rows else empty_fields[name])
    lines = ['# ResNet-18 选择性几何开发筛查', '', f"已完成 {len(data['results'])}/45 次；只有完整五折才产生OOF指标。", '',
             '仅支持集repeat 0、训练seed 42，不能宣称减少支持抽样波动；不是外部验证。',
             '几何教师来自额外外部预训练。稳定视差不等于真实深度准确。单图推理不用教师。', '',
             '| 方法 | 每类预算 | OOF准确率 | 宏F1 |', '|---|---|---:|---:|']
    for job in suite['identity']['jobs']:
        if job['method'] == 'shuffled' and any(r['run_id'] == job['run_id'] for r in data['results']):
            hist = read_csv(root / 'runs' / job['run_id'] / 'history.csv')
            if sum(float(h.get('shuffled_changed_pairs', 0)) for h in hist) == 0:
                lines.append('警告：' + job['run_id'] + ' 的打乱目标未改变任何关系，此负对照无辨别力。')
    for r in data['oof']:
        lines.append(f"| {r['method']} | {r['budget']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    suite.update(completed_runs=len(data['results']), status='completed' if len(data['results']) == 45 else 'partial',
                 aggregate_sha256={n: sha256_file(root / n) for n in (*outputs, 'REPORT.md')})
    _atomic_json(root / 'suite.json', suite)
    return data


def check_dependencies():
    try:
        import numpy
        import torch
        import torchvision
        from sklearn.metrics import accuracy_score
        accuracy_score([0, 1], [0, 1])
    except Exception as error:
        raise RuntimeError('Dependency check failed. Use the existing airway environment with mutually compatible numpy/scipy/scikit-learn and torch/torchvision; NumPy 2 with scikit-learn 1.3 is incompatible. Original error: ' + str(error)) from error


def run_suite(args):
    check_dependencies()
    configure_torch()
    import torch
    runtime = environment(require_cuda=not args.smoke)
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(args.output_root):
        rows, suite = prepare(args.output_root, args.image_root, smoke=args.smoke)
        key = runtime_key(runtime)
        require(suite['runtime_identity'] in (None, key), 'Runtime changed; use a new output root')
        suite['runtime_identity'] = key
        suite.setdefault('attempts', []).append({'at_utc': now(), 'phase': args.phase, 'methods': args.methods,
                                                'budgets': args.budgets, 'folds': args.folds})
        _atomic_json(args.output_root / 'suite.json', suite)
        jobs, by_name = selected_jobs(suite, args), {r['image_name']: r for r in rows}
        # Preflight the entire selected scope, before any expensive fit.
        for j in jobs:
            if j['method'] != 'ce':
                cache_binding(args.cache_root, [by_name[n] for n in j['support']])
        try:
            for i, job in enumerate(jobs, 1):
                print(f"[{i}/{len(jobs)}] {job['run_id']}", flush=True)
                train_one(args.output_root, suite, job, by_name, device, args.cache_root)
                write_aggregates(args.output_root, suite)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as error:
            suite['last_failure'] = {'at_utc': now(), 'message': str(error)}
            _atomic_json(args.output_root / 'suite.json', suite)
            raise


def pack_suite(root, args):
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(root / 'suite.json')
        verify_identity(suite['identity'])
        verify_artifacts(root, suite)
        require(suite['identity']['config']['mode'] == 'formal', 'Smoke results cannot be packaged as formal results')
        jobs = selected_jobs(suite, args)
        require(all(validate_run(root, suite, j, require_model=False) for j in jobs), 'Selected packing scope is incomplete')
        # Recompute all summaries from validated predictions, never trust aggregate values.
        write_aggregates(root, suite)
        manifest = {'version': VERSION, 'created_utc': now(), 'selected_run_ids': [j['run_id'] for j in jobs],
                    'planned_runs': 45, 'selected_completed_runs': len(jobs), 'scope_complete': True,
                    'entire_suite_complete': suite['completed_runs'] == 45,
                    'aggregate_scope': 'all completed jobs in suite; selected runs listed explicitly',
                    'excluded': ['images', 'weights', 'teacher maps', 'teacher code']}
        _atomic_json(root / 'package_scope.json', manifest)
        archive = root.parent / f'{root.name}_{args.phase}_{time.time_ns()}_results.tar.gz'
        names = ['suite.json', 'package_scope.json', *suite['plan_artifact_sha256'], *suite['aggregate_sha256']]
        # Include every completed run because aggregates summarize all completed runs.
        for j in suite['identity']['jobs']:
            if validate_run(root, suite, j, require_model=False):
                names.extend(f"runs/{j['run_id']}/{n}" for n in RUN_FILES)
        with tarfile.open(archive, 'x:gz') as out:
            for name in sorted(set(names)):
                path = root / name
                require(path.is_file() and not path.is_symlink(), 'Unsafe package member: ' + name)
                out.add(path, arcname=name, recursive=False)
        Path(str(archive) + '.sha256').write_text(sha256_file(archive) + '  ' + archive.name + '\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='Independent 45-fit ResNet18 geometry pilot; existing renew results untouched.')
    p.add_argument('mode', choices=['check', 'plan', 'run', 'status', 'pack'])
    p.add_argument('--image-root', type=Path, default=Path('dataset/renew'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/renew_geometry_fewshot'))
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--phase', choices=['full', 'fewshot', 'all'], default='all')
    p.add_argument('--methods', nargs='+', choices=METHODS)
    p.add_argument('--budgets', nargs='+', choices=['full', '5', '10'])
    p.add_argument('--folds', nargs='+', type=int, choices=range(5))
    p.add_argument('--smoke', action='store_true', help='CPU, random weights, 2 updates; separate output only; never formal pack')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    if args.mode == 'run':
        run_suite(args)
    elif args.mode == 'pack':
        print(pack_suite(args.output_root, args))
    elif args.mode == 'status':
        with output_lock(args.output_root):
            suite = read_json(args.output_root / 'suite.json')
            verify_identity(suite['identity'])
            verify_artifacts(args.output_root, suite)
            data = write_aggregates(args.output_root, suite)
            print(f"Completed {len(data['results'])}/45; full={sum(r['budget']=='full' for r in data['results'])}/5")
    else:
        if args.mode == 'check':
            check_dependencies()
        with output_lock(args.output_root):
            rows, suite = prepare(args.output_root, args.image_root, smoke=args.smoke)
            print(f'PASS: {len(rows)} images, 31 patient IDs, 45 planned fits, mode={suite["identity"]["config"]["mode"]}')
            if args.cache_root:
                jobs = [j for j in selected_jobs(suite, args) if j['method'] != 'ce']
                by_name = {r['image_name']: r for r in rows}
                for job in jobs:
                    cache_binding(args.cache_root, [by_name[n] for n in job['support']])
                print(f'PASS: teacher cache checked for {len(jobs)} selected geometry jobs')
            print('Next: run --phase full; or run --phase fewshot --methods ce. Geometry methods require prepared cache.')
