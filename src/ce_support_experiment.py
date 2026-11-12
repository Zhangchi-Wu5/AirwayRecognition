"""Prespecified five-draw support sensitivity check of the existing CE baseline.

Training delegates to the original patient-relation CE implementation. Only the
five training patients change; outer folds and training seed remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import re
import statistics
import tarfile
import time

from src import patient_relation_experiment as baseline
from src.baseline_reproduction import output_lock, read_csv, read_json, require
from src.fewshot import PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest
from src.patient_relation import triplet_indices
from src.renew_data import EXPECTED_FINGERPRINT, audit_dataset
from src.renew_experiment import make_plan as renew_plan, fixed_config as renew_config, portable, configure_torch, environment, runtime_key

VERSION = 'renew_ce_support_sensitivity_v1'
DRAW_COUNT = 5
EXPECTED_RUNS = 25
SAMPLING_SEED = 2026111201
TRAINING_SEED = 42
SOURCE_FILES = (*baseline.SOURCE_FILES, 'src/ce_support_experiment.py',
                'scripts/run_ce_support.py', 'scripts/run_ce_support_server.sh')
RUN_FILES = baseline.RUN_FILES
JOB_KEYS = (*baseline.JOB_KEYS, 'draw')
HASH = baseline.HASH
now = baseline.now
safe_file = baseline.safe_file
run_config = baseline.run_config
check_dependencies = baseline.check_dependencies


def fixed_config(smoke=False):
    config = baseline.fixed_config(smoke)
    config.update(training_seeds=[TRAINING_SEED], support_repeats=list(range(DRAW_COUNT)),
        sampling_seed=SAMPLING_SEED, draws=DRAW_COUNT,
        sampling='seed=2026111201+draw*5+fold; shuffle sorted outer-train patient keys; first five',
        pairing='fixed initialization and training RNG; only support patients vary',
        interpretation='five prespecified support draws on the same 31 development patients; conditional on training seed 42; not independent validation',
        continue_rule='descriptive support sensitivity only; no automatic method or best-support selection')
    return config


def make_plan(rows):
    folds, _, _ = renew_plan(rows, renew_config())
    by_name = {r['image_name']: r for r in rows}
    jobs = []
    for draw in range(DRAW_COUNT):
        for fold in folds:
            index = fold['fold']
            seed = SAMPLING_SEED + draw * 5 + index
            patients = sorted(fold['train_patient_keys'])
            random.Random(seed).shuffle(patients)
            patients = patients[:5]
            selected = set(patients)
            support = [n for n in fold['train'] if by_name[n]['patient_key'] in selected]
            jobs.append({'run_id': f'draw_{draw:02d}_fold_{index:02d}_seed_42_ce',
                'draw': draw, 'fold': index, 'method': 'ce', 'budget': '5', 'repeat': draw,
                'training_seed': TRAINING_SEED, 'sampling_seed': seed, 'pairing_seed': seed,
                'patient_order_prefix': patients,
                'support_csv': f'supports/draw_{draw:02d}_fold_{index:02d}.csv',
                'support': support, 'eval': fold['eval'],
                'relation_indices': triplet_indices([by_name[n] for n in support])})
    require(len(jobs) == len({j['run_id'] for j in jobs}) == EXPECTED_RUNS, 'Expected 25 unique CE jobs')
    return folds, jobs


def artifacts(identity):
    values = baseline.artifacts(identity)
    values['support_plan.csv'] = [{k: j[k] for k in (*JOB_KEYS, 'sampling_seed', 'support_csv')}
                                  for j in identity['jobs']]
    return values


def source_hashes():
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong CE support suite version')
    config = identity.get('config', {})
    require(config == fixed_config(config.get('mode') == 'smoke'), 'Changed fixed CE support protocol')
    hashes = identity.get('source_sha256', {})
    require(set(hashes) == set(SOURCE_FILES) and all(isinstance(v, str) and HASH.fullmatch(v)
            for v in hashes.values()), 'Source manifest incomplete')
    rows = identity['data']
    require(len(rows) == 93 and len({r['patient_key'] for r in rows}) == 31, 'Expected 93 images and 31 patients')
    names = [r['image_name'] for r in rows]
    require(names == sorted(names) and len(set(names)) == 93, 'Invalid source order/uniqueness')
    for index, row in enumerate(rows):
        match = re.fullmatch(r'([0-9]+)\s*(lt|yz|zz)\.png', row['image_name'], re.IGNORECASE)
        require(match is not None and row['source_index'] == index, 'Invalid source identity')
        require(row['patient_id'] == match.group(1) and row['patient_key'] == str(int(match.group(1))), 'Invalid patient identity')
        require(row['label'] == match.group(2).lower() and row['label_id'] == LABEL_TO_ID[row['label']], 'Invalid class identity')
        require(isinstance(row['image_sha256'], str) and HASH.fullmatch(row['image_sha256']), 'Invalid image hash')
    triplet_indices(rows)
    fingerprint = hashlib.sha256(''.join(r['image_name']+'\t'+r['image_sha256']+'\n' for r in rows).encode()).hexdigest()
    require(fingerprint == identity.get('dataset_fingerprint') == EXPECTED_FINGERPRINT, 'Dataset fingerprint changed')
    folds, jobs = make_plan(rows)
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Fold/support plan changed')
    by_name = {r['image_name']: r for r in rows}
    for job in jobs:
        require(len(job['support']) == len(set(job['support'])) == 15, 'Support must contain fifteen unique images')
        selected = {by_name[n]['patient_key'] for n in job['support']}
        require(len(selected) == 5 and not selected & {by_name[n]['patient_key'] for n in job['eval']}, 'Patient leakage/support count')


def verify_artifacts(root, suite):
    root = Path(root)
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Suite identity corrupted')
    verify_identity(suite['identity'])
    expected = artifacts(suite['identity'])
    require(set(expected) == set(suite.get('plan_artifact_sha256', {})), 'Plan manifest incomplete')
    for name, rows in expected.items():
        path = safe_file(root, name)
        require(sha256_file(path) == suite['plan_artifact_sha256'][name], 'Plan artifact changed: '+name)
        require(read_csv(path) == [{k: str(v) for k, v in row.items()} for row in rows], 'Plan content changed: '+name)
    recorded = suite.get('completed_run_sha256', {})
    require(isinstance(recorded, dict) and set(recorded) <= {j['run_id'] for j in suite['identity']['jobs']}, 'Invalid completed-run ledger')
    for run_id, digest in recorded.items():
        require(sha256_file(safe_file(root, f'runs/{run_id}/complete.json')) == digest, 'Completed run marker changed: '+run_id)


def prepare(root, image_root, *, smoke=False):
    root, image_root = Path(root).resolve(), Path(image_root).expanduser().resolve()
    require(not root.is_relative_to(image_root) and not image_root.is_relative_to(root), 'Image and output roots must be disjoint')
    try:
        rows, metadata = audit_dataset(image_root)
    except ValueError as error:
        if 'No first-level PNG' in str(error):
            raise ValueError(str(error)+'; --image-root must directly contain the 93 PNGs, e.g. dataset/renew/气管镜') from error
        raise
    data = portable(rows)
    folds, jobs = make_plan(data)
    identity = {'version': VERSION, 'config': fixed_config(smoke), 'data': data,
        'dataset_fingerprint': metadata['dataset_fingerprint'], 'folds': folds, 'jobs': jobs,
        'source_sha256': source_hashes()}
    verify_identity(identity)
    if (root/'suite.json').exists():
        suite = read_json(safe_file(root, 'suite.json'))
        require(suite['identity'] == identity, 'Resume rejected: source/config/data changed; use a new output root')
        verify_artifacts(root, suite)
        return rows, suite
    require(not root.exists() or not any(p.name != '.reproduction.lock' for p in root.iterdir()), 'Nonempty output without suite.json')
    root.mkdir(parents=True, exist_ok=True)
    for name, values in artifacts(identity).items():
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        write_csv(root/name, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
        'runtime_identity': None, 'completed_runs': 0, 'status': 'prepared_not_run',
        'completed_run_sha256': {},
        'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
        'plan_artifact_sha256': {n: sha256_file(root/n) for n in artifacts(identity)}}
    _atomic_json(root/'suite.json', suite)
    return rows, suite


def selected_jobs(suite, args):
    jobs = [j for j in suite['identity']['jobs']
            if (args.draws is None or j['draw'] in args.draws)
            and (args.folds is None or j['fold'] in args.folds)]
    require(bool(jobs), 'Filters select no jobs')
    return jobs


def validate_run(root, suite, job, require_model=True):
    require(job in suite['identity']['jobs'], 'Job is outside the fixed plan')
    recorded = suite.get('completed_run_sha256', {}).get(job['run_id'])
    if recorded is not None:
        marker = safe_file(root, f"runs/{job['run_id']}/complete.json")
        require(sha256_file(marker) == recorded, 'Completed run marker changed: '+job['run_id'])
    result = baseline.validate_run(root, suite, job, require_model=require_model)
    return None if result is None else {**result, 'draw': job['draw']}


def train_one(root, suite, job, by_name, device, model_factory=None):
    completed = validate_run(root, suite, job)
    if completed is not None:
        return completed
    # All optimization, augmentation, BN and checkpoint logic is unchanged.
    result = baseline.train_one(root, suite, job, by_name, device, model_factory=model_factory)
    return {**result, 'draw': job['draw']}


def aggregate(root, suite):
    import numpy as np
    root = Path(root)
    verify_artifacts(root, suite)
    results, predictions = [], []
    for job in suite['identity']['jobs']:
        result = validate_run(root, suite, job, require_model=False)
        if result is None:
            continue
        results.append(result)
        predictions.extend([{**{k: job[k] for k in JOB_KEYS}, **p}
                            for p in read_csv(root/'runs'/job['run_id']/'predictions.csv')])
    for key in ('initial_state_sha256', 'training_rng_initial_sha256', 'training_rng_final_sha256', 'student_parameter_count'):
        require(len({r[key] for r in results}) <= 1, 'Fixed-seed CE evidence differs: '+key)
    # Actual augmented tensors and losses differ when the input patients change.
    expected_names = sorted(r['image_name'] for r in suite['identity']['data'])
    oof = []
    for draw in range(DRAW_COUNT):
        group = [p for p in predictions if p['draw'] == draw]
        if {p['fold'] for p in group} != set(range(5)):
            continue
        require(sorted(p['image_name'] for p in group) == expected_names, 'OOF coverage differs')
        probabilities = np.asarray([[float(p['prob_'+label]) for label in LABEL_TO_ID] for p in group], dtype=np.float32)
        oof.append({'draw': draw, 'training_seed': TRAINING_SEED, 'method': 'ce', 'budget': '5',
            'n_images': 93, 'correct': sum(int(p['label_id']) == int(p['predicted_id']) for p in group),
            **classification_metrics([int(p['label_id']) for p in group], probabilities)})
    distribution = []
    for metric in ('correct', *METRIC_NAMES):
        values = [float(row[metric]) for row in oof]
        if values:
            distribution.append({'metric': metric, 'completed_draws': len(values),
                'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values) > 1 else '',
                'min': min(values), 'max': max(values), 'range': max(values)-min(values)})
    fold_summary = []
    for fold in range(5):
        group = [r for r in results if r['fold'] == fold]
        if group:
            values = [r['accuracy'] for r in group]
            fold_summary.append({'fold': fold, 'completed_draws': len(group), 'accuracy_mean': statistics.mean(values),
                'accuracy_sample_sd': statistics.stdev(values) if len(values) > 1 else '',
                'accuracy_min': min(values), 'accuracy_max': max(values)})
    complete_draws = {r['draw'] for r in oof}
    image_errors = []
    for name in expected_names:
        group = [p for p in predictions if p['image_name'] == name and p['draw'] in complete_draws]
        if group:
            wrong = sum(int(p['label_id']) != int(p['predicted_id']) for p in group)
            image_errors.append({'image_name': name, 'patient_id': group[0]['patient_id'],
                'label_id': int(group[0]['label_id']), 'evaluations': len(group), 'error_count': wrong,
                'error_fraction': wrong/len(group),
                **{f'predicted_{label}': sum(int(p['predicted_id']) == label_id for p in group)
                   for label, label_id in LABEL_TO_ID.items()}})
    formal = suite['identity']['config']['mode'] == 'formal'
    complete = len(results) == EXPECTED_RUNS
    require(not complete or len(oof) == DRAW_COUNT, 'Completed suite lacks five OOF groups')
    summary = {'status': ('completed_descriptive' if complete else 'pending') if formal else 'smoke_not_formal',
        'formal': formal, 'completed_runs': len(results), 'planned_runs': EXPECTED_RUNS,
        'complete_25_runs': complete, 'completed_draws': len(oof), 'training_seed': TRAINING_SEED,
        'sampling_seed': SAMPLING_SEED, 'unique_patients': 31, 'unique_images': 93,
        'independent_external_validation': False, 'algorithm_efficacy_demonstrated': False,
        'interpretation': 'descriptive variation over five prespecified supports conditional on training seed 42; repeated cohort, no best-support selection',
        'distribution': distribution}
    return {'results': results, 'predictions': predictions, 'oof': oof, 'distribution': distribution,
            'fold_summary': fold_summary, 'image_errors': image_errors, 'summary': summary}


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root, suite)
    outputs = {
        'per_run_metrics.csv': ([{k: r[k] for k in (*JOB_KEYS, 'n_train_images', 'n_train_patients', *METRIC_NAMES)}
                                  for r in data['results']], [*JOB_KEYS, 'n_train_images', 'n_train_patients', *METRIC_NAMES]),
        'oof_predictions.csv': (data['predictions'], [*JOB_KEYS, 'patient_id', 'image_name', 'label_id', 'predicted_id', *['prob_'+l for l in LABEL_TO_ID]]),
        'oof_metrics.csv': (data['oof'], ['draw', 'training_seed', 'method', 'budget', 'n_images', 'correct', *METRIC_NAMES]),
        'draw_distribution.csv': (data['distribution'], ['metric', 'completed_draws', 'mean', 'sample_sd', 'min', 'max', 'range']),
        'fold_summary.csv': (data['fold_summary'], ['fold', 'completed_draws', 'accuracy_mean', 'accuracy_sample_sd', 'accuracy_min', 'accuracy_max']),
        'image_error_frequency.csv': (data['image_errors'], ['image_name', 'patient_id', 'label_id', 'evaluations', 'error_count', 'error_fraction', *['predicted_'+l for l in LABEL_TO_ID]])}
    for name, (rows, fields) in outputs.items():
        baseline.atomic_csv(root/name, rows, fields)
    _atomic_json(root/'SUMMARY.json', data['summary'])
    lines = ['# 整体微调CE：5套训练患者敏感性检查', '',
        f"已完成 {len(data['results'])}/25 次训练，{len(data['oof'])}/5 组完整五折；模式：{suite['identity']['config']['mode']}。", '',
        '每折5位训练患者、每类5张；ResNet-18整体微调，训练种子固定42。',
        '只更换训练患者；不根据留出成绩挑选支持，不扩参数网格，不混入旧支持结果。',
        '五套仍来自同一31位患者，均值/标准差仅描述支持变化，不是外部人群置信区间。',
        '固定种子无法估计全部训练随机性；本检查不证明选样算法有效或专利新颖性。', '',
        '| draw | 正确/93 | 准确率 | macro-F1 | AUC | lt召回 | yz召回 | zz召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['oof']:
        lines.append(f"| {row['draw']} | {row['correct']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['auc_ovr_macro']:.6f} | {row['recall_lt']:.6f} | {row['recall_yz']:.6f} | {row['recall_zz']:.6f} |")
    tmp = root/'REPORT.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tmp.replace(root/'REPORT.md')
    suite.update(completed_runs=len(data['results']), status='completed' if len(data['results']) == EXPECTED_RUNS else 'partial',
        completed_run_sha256={r['run_id']: sha256_file(root/'runs'/r['run_id']/'complete.json') for r in data['results']},
        aggregate_sha256={name: sha256_file(root/name) for name in (*outputs, 'SUMMARY.json', 'REPORT.md')})
    _atomic_json(root/'suite.json', suite)
    return data


def run_suite(args):
    check_dependencies(); configure_torch()
    import torch
    runtime = environment(require_cuda=not args.smoke)
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(args.output_root):
        rows, suite = prepare(args.output_root, args.image_root, smoke=args.smoke)
        key = runtime_key(runtime)
        require(suite['runtime_identity'] in (None, key), 'Runtime changed; use a new output root')
        suite['runtime_identity'] = key
        suite.setdefault('attempts', []).append({'at_utc': now(), 'draws': args.draws, 'folds': args.folds})
        _atomic_json(args.output_root/'suite.json', suite)
        jobs = selected_jobs(suite, args)
        by_name = {r['image_name']: r for r in rows}
        try:
            for index, job in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] {job['run_id']}", flush=True)
                train_one(args.output_root, suite, job, by_name, device)
                write_aggregates(args.output_root, suite)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as error:
            suite['last_failure'] = {'at_utc': now(), 'message': str(error)}
            _atomic_json(args.output_root/'suite.json', suite)
            raise
        print(f"Completed {suite['completed_runs']}/25. Pack after all 25 complete.", flush=True)


def pack_suite(root, args=None):
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(safe_file(root, 'suite.json'))
        verify_artifacts(root, suite)
        require(suite['identity']['config']['mode'] == 'formal', 'Smoke results cannot be packaged as formal')
        require(all(validate_run(root, suite, j, require_model=False) is not None for j in suite['identity']['jobs']), 'Full 25-run suite is incomplete; partial packing is not formal completion')
        data = write_aggregates(root, suite)
        require(len(data['oof']) == DRAW_COUNT, 'Pack needs five complete OOF groups')
        _atomic_json(root/'package_scope.json', {'version': VERSION, 'created_utc': now(),
            'planned_runs': EXPECTED_RUNS, 'completed_runs': EXPECTED_RUNS, 'entire_suite_complete': True,
            'scope': 'five prespecified support draws x five folds, CE only, training seed 42',
            'excluded': ['images', 'weights'], 'claim_status': 'developmental support sensitivity, not independent validation or algorithm efficacy'})
        archive = root.parent/(root.name+'_'+str(time.time_ns())+'_results.tar.gz')
        names = ['suite.json', 'package_scope.json', *suite['plan_artifact_sha256'], *suite['aggregate_sha256']]
        names.extend(f"runs/{j['run_id']}/{name}" for j in suite['identity']['jobs'] for name in RUN_FILES)
        with tarfile.open(archive, 'x:gz') as packed:
            for name in sorted(set(names)):
                packed.add(safe_file(root, name), arcname=name, recursive=False)
        Path(str(archive)+'.sha256').write_text(sha256_file(archive)+'  '+archive.name+'\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='25-fit CE support sensitivity check; fixed training seed and five prespecified supports.')
    p.add_argument('mode', choices=['check', 'plan', 'run', 'status', 'pack'])
    p.add_argument('--image-root', type=Path, default=Path('dataset/renew'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/renew_ce_support'))
    p.add_argument('--draws', nargs='+', type=int, choices=range(DRAW_COUNT))
    p.add_argument('--folds', nargs='+', type=int, choices=range(5))
    p.add_argument('--smoke', action='store_true', help='CPU/random weights/two updates; separate output; never formal results')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    require(args.mode == 'run' or (args.draws is None and args.folds is None), 'Draw/fold filters are valid only for run')
    if args.mode == 'run':
        run_suite(args)
    elif args.mode == 'pack':
        require(not args.smoke, 'Formal pack cannot use --smoke')
        print(pack_suite(args.output_root))
    elif args.mode == 'status':
        with output_lock(args.output_root):
            suite = read_json(safe_file(args.output_root, 'suite.json'))
            verify_artifacts(args.output_root, suite)
            data = write_aggregates(args.output_root, suite)
            print(f"Completed {len(data['results'])}/25; complete OOF draws {len(data['oof'])}/5; {data['summary']['status']}")
    else:
        if args.mode == 'check':
            check_dependencies()
        with output_lock(args.output_root):
            rows, suite = prepare(args.output_root, args.image_root, smoke=args.smoke)
            print(f"PASS: {len(rows)} images, 31 patient IDs, 25 planned CE fits, mode={suite['identity']['config']['mode']}")
            print('Five prespecified supports x five folds; seed 42; existing full-network CE; no teacher/cache required.')


if __name__ == '__main__':
    main()
