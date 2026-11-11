"""Immutable 40-fit, 5-shot patient-relation development experiment.

Training-only matched relations; independent single-image deployment. Existing
renew/geometry runs and source modules are never modified by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import shutil
import tarfile
import time

from src.baseline_reproduction import output_lock, read_csv, read_json, require, verify_prediction_metrics
from src.fewshot import PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics, sha256_file, state_fingerprint, write_csv
from src.fewshot_finetune import ImageRows, _atomic_json, _digest, evaluate, save_reload_final, seed_training, validate_completed
from src.renew_data import EXPECTED_FINGERPRINT, audit_dataset, get_transforms
from src.renew_experiment import make_plan as renew_plan, fixed_config as renew_config, portable, configure_torch, environment, runtime_key
from src.patient_relation import METHODS, build_model, triplet_indices, regroup_triplets, train_updates

VERSION = 'renew_patient_relation_5shot_v1'
SEEDS = (42, 43)
SOURCE_FILES = ('src/patient_relation_experiment.py', 'src/patient_relation.py',
    'src/renew_experiment.py', 'src/renew_data.py', 'src/fewshot_finetune.py',
    'src/fewshot.py', 'src/baseline_reproduction.py', 'src/data.py', 'src/models.py',
    'scripts/run_patient_relation.py', 'scripts/run_patient_relation_server.sh')
RUN_FILES = ('config.json', 'history.csv', 'predictions.csv', 'result.json', 'complete.json')
HASH = re.compile(r'[0-9a-f]{64}')
JOB_KEYS = ('run_id', 'fold', 'budget', 'repeat', 'training_seed', 'method')


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_config(smoke=False):
    c = renew_config()
    c.update(mode='smoke' if smoke else 'formal', architecture='resnet18',
        weights='none' if smoke else 'ResNet18_Weights.IMAGENET1K_V1',
        training_seeds=list(SEEDS), batch_size=15, num_workers=0,
        preprocessing='full_frame', views_per_image=1,
        batching='all 15 support images, fixed dataset order, once per update',
        aux_weight=.1, supcon_temperature=.1, difference_epsilon=1e-6,
        auxiliary_phase='finetune_only', projection_head='none; pooled pre-dropout features',
        support_repeats=[0], budgets=['5'],
        pairing='identical support, initialization, actual augmentation tensor stream; local pairing RNG only',
        identity_confirmation={'source': 'user report', 'date': '2026-11-11',
            'same_patient_and_exam': True, 'independent_clinical_audit': False},
        inference='one RGB image, no patient identifier or other patient images',
        interpretation='exploratory development CV; repeated cohort, not independent external validation',
        continue_rule='both seeds: paired correct minus CE >= 2/93; paired strictly exceeds supcon and regrouped')
    if smoke:
        c.update(head_rounds=1, finetune_rounds=1, steps_per_round=1,
                 update_budget='engineering smoke only: one head + one finetune update')
    return c


def run_config(config, job):
    return {**config, 'training_seed': job['training_seed']}


def make_plan(rows):
    folds, _, old = renew_plan(rows, renew_config())
    by_name = {r['image_name']: r for r in rows}
    jobs = []
    for original in old:
        if original['method'] != 'full_frame' or original['budget'] != '5' or original['repeat'] != 0:
            continue
        support = [by_name[n] for n in original['support']]
        matched = triplet_indices(support)
        pairing_seed = 20261111 + original['fold']
        regrouped = regroup_triplets(matched, pairing_seed)
        for seed in SEEDS:
            for method in METHODS:
                j = dict(original)
                j.update(run_id=f"fold_{original['fold']:02d}_shot_005_repeat_00_seed_{seed}_{method}",
                    method=method, training_seed=seed, pairing_seed=pairing_seed,
                    relation_indices=regrouped if method == 'regrouped' else matched)
                jobs.append(j)
    require(len(jobs) == 40 and len({j['run_id'] for j in jobs}) == 40, 'Expected exactly 40 unique 5-shot fits')
    return folds, jobs


def artifacts(identity):
    by_name = {r['image_name']: r for r in identity['data']}
    values = {'dataset.csv': identity['data']}
    for fold in identity['folds']:
        for split in ('train', 'eval'):
            values[f"folds/fold_{fold['fold']:02d}_{split}.csv"] = [by_name[n] for n in fold[split]]
    for job in identity['jobs']:
        values[job['support_csv']] = [by_name[n] for n in job['support']]
    values['support_plan.csv'] = [{k: j[k] for k in (*JOB_KEYS, 'support_csv', 'pairing_seed')} for j in identity['jobs']]
    values['relation_plan.csv'] = [{'run_id': j['run_id'], 'group': g, 'label_id': label,
        'support_index': idx, 'image_name': j['support'][idx],
        'true_patient_key': by_name[j['support'][idx]]['patient_key']}
        for j in identity['jobs'] for g, group in enumerate(j['relation_indices']) for label, idx in enumerate(group)]
    return values


def source_hashes():
    return {n: sha256_file(PROJECT_ROOT / n) for n in SOURCE_FILES}


def verify_identity(identity):
    require(identity.get('version') == VERSION, 'Wrong patient-relation suite version')
    c = identity.get('config', {})
    require(c == fixed_config(c.get('mode') == 'smoke'), 'Changed fixed protocol')
    hashes = identity.get('source_sha256', {})
    require(set(hashes) == set(SOURCE_FILES) and all(HASH.fullmatch(v) for v in hashes.values()), 'Source manifest incomplete')
    rows = identity['data']
    require(len(rows) == 93 and len({r['patient_key'] for r in rows}) == 31, 'Expected 93 images and 31 patients')
    names = [r['image_name'] for r in rows]
    require(names == sorted(names) and len(set(names)) == 93, 'Source order/uniqueness changed')
    for i, r in enumerate(rows):
        m = re.fullmatch(r'([0-9]+)\s*(lt|yz|zz)\.png', r['image_name'], re.IGNORECASE)
        require(m is not None and r['source_index'] == i, 'Invalid source image identity')
        require(r['patient_id'] == m.group(1) and r['patient_key'] == str(int(m.group(1))), 'Invalid patient identity')
        require(r['label'] == m.group(2).lower() and r['label_id'] == LABEL_TO_ID[r['label']], 'Invalid class identity')
        require(HASH.fullmatch(r['image_sha256']) is not None, 'Invalid image SHA256')
    triplet_indices(rows)
    fingerprint = hashlib.sha256(''.join(r['image_name']+'\t'+r['image_sha256']+'\n' for r in rows).encode()).hexdigest()
    require(fingerprint == identity.get('dataset_fingerprint') == EXPECTED_FINGERPRINT, 'Dataset fingerprint changed')
    folds, jobs = make_plan(rows)
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Fold/support/relation plan changed')
    by_name = {r['image_name']: r for r in rows}
    for j in jobs:
        require(len(j['support']) == 15 and len(set(j['support'])) == 15, 'Wrong support count')
        require(not {by_name[n]['patient_key'] for n in j['support']} & {by_name[n]['patient_key'] for n in j['eval']}, 'Patient leakage')
        if j['method'] == 'regrouped':
            require(all(len({by_name[j['support'][idx]]['patient_key'] for idx in group}) == 3
                        for group in j['relation_indices']), 'Regrouped control retains original patient matching')


def safe_file(root, relative):
    root = Path(root)
    path = root / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe relative path')
    components = [root]
    for part in Path(relative).parts:
        components.append(components[-1] / part)
    require(path.is_file() and not any(p.is_symlink() for p in components), 'Missing/unsafe artifact: '+str(relative))
    return path


def verify_artifacts(root, suite):
    root = Path(root)
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Suite identity corrupted')
    verify_identity(suite['identity'])
    expected = artifacts(suite['identity'])
    require(set(expected) == set(suite.get('plan_artifact_sha256', {})), 'Plan manifest incomplete')
    for n, rows in expected.items():
        path = safe_file(root, n)
        require(sha256_file(path) == suite['plan_artifact_sha256'][n], 'Plan artifact changed: '+n)
        require(read_csv(path) == [{k: str(v) for k, v in r.items()} for r in rows], 'Plan content changed: '+n)


def prepare(root, image_root, *, smoke=False):
    root, image_root = Path(root).resolve(), Path(image_root).expanduser().resolve()
    require(not root.is_relative_to(image_root) and not image_root.is_relative_to(root), 'Image and output roots must be disjoint')
    try:
        rows, metadata = audit_dataset(image_root)
    except ValueError as e:
        if 'No first-level PNG' in str(e):
            raise ValueError(str(e)+'; --image-root must directly contain the 93 PNGs, e.g. dataset/renew/气管镜') from e
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
    for n, values in artifacts(identity).items():
        (root/n).parent.mkdir(parents=True, exist_ok=True)
        write_csv(root/n, values)
    suite = {'identity': identity, 'identity_sha256': _digest(identity), 'created_utc': now(),
        'runtime_identity': None, 'completed_runs': 0, 'status': 'prepared_not_run',
        'data_metadata': {k: v for k, v in metadata.items() if k != 'image_root'},
        'plan_artifact_sha256': {n: sha256_file(root/n) for n in artifacts(identity)}}
    _atomic_json(root/'suite.json', suite)
    return rows, suite


def selected_jobs(suite, args):
    jobs = [j for j in suite['identity']['jobs']
        if (not args.methods or j['method'] in args.methods)
        and (args.folds is None or j['fold'] in args.folds)
        and (not args.seeds or j['training_seed'] in args.seeds)]
    require(bool(jobs), 'Filters select no jobs')
    return jobs


def validate_run(root, suite, job, require_model=True):
    root = Path(root)
    d = root/'runs'/job['run_id']
    if not (d/'complete.json').exists():
        return None
    for n in RUN_FILES:
        safe_file(root, f"runs/{job['run_id']}/{n}")
    if require_model:
        safe_file(root, f"runs/{job['run_id']}/model.pt")
        safe_file(root, f"runs/{job['run_id']}/student.pt")
    result = validate_completed(d, suite['identity_sha256'], job['run_id'], verify_model=require_model)
    saved = read_json(d/'config.json')
    c = run_config(suite['identity']['config'], job)
    require(saved.get('config') == c and saved.get('job') == job and saved.get('identity_sha256') == suite['identity_sha256'], 'Run configuration changed')
    require(suite.get('runtime_identity') is not None and saved.get('execution_runtime') == suite['runtime_identity'], 'Runtime identity mismatch')
    require(saved.get('initial_state_sha256') == result.get('initial_state_sha256'), 'Initialization evidence mismatch')
    require(saved.get('relation_indices_sha256') == result.get('relation_indices_sha256') == _digest(job['relation_indices']), 'Relation mapping mismatch')
    for k in JOB_KEYS:
        require(result.get(k) == job[k], 'Result job mismatch: '+k)
    require(result.get('n_train_images') == 15 and result.get('n_train_patients') == 5, 'Support counts differ')
    for key in ('initial_state_sha256', 'student_export_sha256', 'augmentation_sha256', 'head_augmentation_sha256',
                'finetune_augmentation_sha256', 'training_rng_initial_sha256', 'training_rng_final_sha256',
                'initial_backbone_sha256', 'final_backbone_sha256', 'final_state_sha256', 'reloaded_state_sha256'):
        require(isinstance(result.get(key), str) and HASH.fullmatch(result[key]), 'Missing state/augmentation evidence: '+key)
    require(result.get('student_export_verified') is True and result.get('single_image_evaluation') is True, 'Missing single-image deployment evidence')
    require(type(result.get('student_parameter_count')) is int and result['student_parameter_count'] > 0,
            'Invalid deployment parameter count')
    if require_model:
        require(sha256_file(d/'student.pt') == result['student_export_sha256'], 'Student export corrupted')
    require(result.get('head_stage_backbone_unchanged') and result.get('finetune_backbone_changed'), 'Training stage invariance missing')
    require(result['initial_backbone_sha256'] != result['final_backbone_sha256'], 'Backbone not updated')
    for a, b in (('bn_before_sha256','bn_after_sha256'), ('final_state_sha256','reloaded_state_sha256')):
        require(isinstance(result.get(a), str) and result[a] == result.get(b), 'State evidence inconsistent: '+a)
    require(result.get('effective_support_order') == list(range(15)) and result.get('auxiliary_phase') == 'finetune_only', 'Training order/phase changed')
    history = read_csv(d/'history.csv')
    schedule = [(s, i+1) for s,n in (('head',c['head_rounds']),('finetune',c['finetune_rounds'])) for i in range(n)]
    require([(h['stage'],int(h['round'])) for h in history] == schedule, 'Training schedule changed')
    for h in history:
        lr = c['head_lr'] if h['stage']=='head' else c['finetune_lr']*(1+math.cos(math.pi*(int(h['round'])-1)/c['finetune_rounds']))/2
        require(int(h['steps']) == c['steps_per_round'] and int(h['image_occurrences']) == c['steps_per_round']*15, 'Training budget changed')
        require(math.isclose(float(h['lr']),lr,rel_tol=1e-9), 'Learning rate schedule changed')
        for k in ('mean_train_loss','classification_loss','auxiliary_loss','seconds'):
            require(math.isfinite(float(h[k])) and float(h[k]) >= -1e-6, 'Invalid training history: '+k)
        require(math.isclose(float(h['mean_train_loss']),float(h['classification_loss'])+c['aux_weight']*float(h['auxiliary_loss']),rel_tol=1e-5,abs_tol=1e-6), 'Training loss components inconsistent')
        pairs = 0 if h['stage']=='head' or job['method']=='ce' else (60 if job['method']=='supcon' else 30)*c['steps_per_round']
        require(int(h['auxiliary_pairs']) == pairs, 'Auxiliary pair count changed')
        if pairs == 0:
            require(float(h['auxiliary_loss']) == 0., 'Forbidden auxiliary loss')
    steps = len(schedule)*c['steps_per_round']
    require(result.get('actual_optimizer_steps') == steps and result.get('actual_image_occurrences') == steps*15, 'Run update budget differs')
    by_name = {r['image_name']:r for r in suite['identity']['data']}
    verify_prediction_metrics(d/'predictions.csv',result,[by_name[n] for n in job['eval']])
    return result


def export_student(model, path, factory, device):
    import torch
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, path)
    student = factory(False).to(device)
    student.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
    model.eval(); student.eval()
    with torch.inference_mode():
        for value in (0., .5):
            probe = torch.full((1,3,224,224),value,device=device)
            require(torch.equal(model(probe),student(probe)), 'Single-image export output differs')
    return student, {'student_export_sha256':sha256_file(path),'student_export_verified':True,
                     'student_parameter_count':sum(p.numel() for p in student.parameters())}


def train_one(root, suite, job, by_name, device, model_factory=None):
    root = Path(root)
    completed = validate_run(root,suite,job)
    if completed is not None:
        return completed
    d = root/'runs'/job['run_id']
    if d.exists():
        require(not d.is_symlink(), 'Unsafe incomplete run directory')
        archived = root/'incomplete_attempts'
        archived.mkdir(exist_ok=True)
        shutil.move(str(d),str(archived/(job['run_id']+'_'+str(time.time_ns()))))
    d.mkdir(parents=True)
    c = run_config(suite['identity']['config'],job)
    factory = model_factory or (lambda pretrained:build_model(pretrained=pretrained,dropout=c['dropout']))
    seed_training(job['training_seed'])
    model = factory(c['mode']=='formal').to(device)
    initial = state_fingerprint(model)
    relation_sha = _digest(job['relation_indices'])
    _atomic_json(d/'config.json',{'job':job,'config':c,'identity_sha256':suite['identity_sha256'],
        'initial_state_sha256':initial,'relation_indices_sha256':relation_sha,'execution_runtime':suite['runtime_identity']})
    dataset = ImageRows([by_name[n] for n in job['support']],get_transforms('full_frame',train=True))
    history, evidence = train_updates(model,dataset,c,device,job['method'],job['relation_indices'])
    write_csv(d/'history.csv',history)
    reloaded, checkpoint = save_reload_final(model,d/'model.pt',lambda:factory(False),device)
    student, exported = export_student(reloaded,d/'student.pt',factory,device)
    del model, reloaded
    eval_rows = [by_name[n] for n in job['eval']]
    probabilities = evaluate(student,ImageRows(eval_rows,get_transforms('full_frame')),{**c,'batch_size':1},device)
    predictions = [{'patient_id':r['patient_id'],'image_name':r['image_name'],'label_id':r['label_id'],
        'predicted_id':int(p.argmax()),**{f'prob_{label}':float(p[i]) for label,i in LABEL_TO_ID.items()}}
        for r,p in zip(eval_rows,probabilities)]
    write_csv(d/'predictions.csv',predictions)
    result = {**{k:job[k] for k in JOB_KEYS},'n_train_images':15,'n_train_patients':5,
        'initial_state_sha256':initial,'relation_indices_sha256':relation_sha,'single_image_evaluation':True,
        **classification_metrics([r['label_id'] for r in eval_rows],probabilities),**evidence,**checkpoint,**exported}
    _atomic_json(d/'result.json',result)
    _atomic_json(d/'complete.json',{'identity_sha256':suite['identity_sha256'],'run_id':job['run_id'],
        'artifact_sha256':{n:sha256_file(d/n) for n in (*RUN_FILES[:-1],'model.pt')}})
    validate_run(root,suite,job)
    return result


def decision_from_oof(oof, complete, smoke=False):
    decision = {'status':'smoke_not_formal' if smoke else 'pending','formal':not smoke,
        'complete_40_runs':bool(complete),'criterion':'each seed: paired minus ce >= 2 correct of 93; paired > supcon and regrouped',
        'independent_external_validation':False,'per_seed':[]}
    for seed in SEEDS:
        selected = [r for r in oof if r['training_seed']==seed]
        rows = {r['method']:r for r in selected}
        require(len(rows)==len(selected), 'Duplicate OOF method/seed')
        if set(rows) != set(METHODS):
            continue
        counts = {m:int(rows[m]['correct']) for m in METHODS}
        require(all(counts[m] == rows[m]['correct'] and 0 <= counts[m] <= 93 for m in METHODS), 'Invalid OOF correct count')
        decision['per_seed'].append({'training_seed':seed,'correct':counts,
            'net_correct_vs_ce':counts['paired']-counts['ce'],
            'pass':counts['paired']-counts['ce']>=2 and counts['paired']>counts['supcon'] and counts['paired']>counts['regrouped']})
    if not smoke and complete:
        require(len(decision['per_seed'])==2,'Completed suite lacks both seeds OOF')
        decision['status'] = 'continue_to_confirmation' if all(r['pass'] for r in decision['per_seed']) else 'stop_current_candidate'
    return decision


def aggregate(root, suite):
    import numpy as np
    root = Path(root)
    results, predictions = [], []
    for job in suite['identity']['jobs']:
        result = validate_run(root,suite,job,require_model=False)
        if result is not None:
            results.append(result)
            predictions.extend([{**{k:job[k] for k in JOB_KEYS},**p} for p in read_csv(root/'runs'/job['run_id']/'predictions.csv')])
    for seed in SEEDS:
        same_seed = [r for r in results if r['training_seed']==seed]
        require(len({r['initial_state_sha256'] for r in same_seed})<=1,'Paired initialization differs')
        for fold in range(5):
            group = [r for r in same_seed if r['fold']==fold]
            for key in ('augmentation_sha256','head_augmentation_sha256','finetune_augmentation_sha256',
                        'training_rng_initial_sha256','training_rng_final_sha256'):
                require(len({r[key] for r in group})<=1,'Paired stochastic stream differs: '+key)
            head_histories = []
            for r in group:
                h = read_csv(root/'runs'/r['run_id']/'history.csv')
                head_histories.append(_digest([{k:v for k,v in row.items() if k != 'seconds'}
                                               for row in h if row['stage']=='head']))
            require(len(set(head_histories))<=1, 'Paired head-stage training differs')
    require(len({r['student_parameter_count'] for r in results})<=1, 'Deployment parameter counts differ')
    expected = sorted(r['image_name'] for r in suite['identity']['data'])
    oof, paired, image_changes, patient_changes = [], [], [], []
    for seed in SEEDS:
        for method in METHODS:
            group = [p for p in predictions if p['training_seed']==seed and p['method']==method]
            if {p['fold'] for p in group} != set(range(5)):
                continue
            require(sorted(p['image_name'] for p in group)==expected,'OOF coverage differs')
            probs = np.asarray([[float(p['prob_'+l]) for l in LABEL_TO_ID] for p in group],dtype=np.float32)
            correct = sum(int(p['label_id'])==int(p['predicted_id']) for p in group)
            oof.append({'training_seed':seed,'method':method,'budget':'5','repeat':0,'n_images':93,'correct':correct,
                        **classification_metrics([int(p['label_id']) for p in group],probs)})
        for method in METHODS:
            if method=='ce':
                continue
            base = next((r for r in oof if r['training_seed']==seed and r['method']=='ce'),None)
            other = next((r for r in oof if r['training_seed']==seed and r['method']==method),None)
            if base is None or other is None:
                continue
            paired.append({'training_seed':seed,'method':method,'net_correct_vs_ce':other['correct']-base['correct'],
                           **{k+'_delta_vs_ce':other[k]-base[k] for k in METRIC_NAMES}})
            base_pred = {p['image_name']:p for p in predictions if p['training_seed']==seed and p['method']=='ce'}
            method_pred = {p['image_name']:p for p in predictions if p['training_seed']==seed and p['method']==method}
            for name in expected:
                a,b = base_pred[name],method_pred[name]
                ac,bc = int(a['predicted_id'])==int(a['label_id']),int(b['predicted_id'])==int(b['label_id'])
                image_changes.append({'training_seed':seed,'method':method,'patient_id':a['patient_id'],
                    'image_name':name,'label_id':a['label_id'],'ce_predicted_id':a['predicted_id'],
                    'method_predicted_id':b['predicted_id'],'ce_correct':int(ac),'method_correct':int(bc),
                    'change':'corrected' if bc and not ac else 'new_error' if ac and not bc else 'unchanged'})
            changes = [r for r in image_changes if r['training_seed']==seed and r['method']==method]
            for patient in sorted({r['patient_id'] for r in changes}):
                g = [r for r in changes if r['patient_id']==patient]
                require(len(g)==3,'Patient comparison lacks three images')
                patient_changes.append({'training_seed':seed,'method':method,'patient_id':patient,
                    'ce_correct':sum(r['ce_correct'] for r in g),'method_correct':sum(r['method_correct'] for r in g),
                    'corrected':sum(r['change']=='corrected' for r in g),'new_errors':sum(r['change']=='new_error' for r in g)})
    decision = decision_from_oof(oof,len(results)==40,suite['identity']['config']['mode']=='smoke')
    return {'results':results,'predictions':predictions,'oof':oof,'paired':paired,
            'image_changes':image_changes,'patient_changes':patient_changes,'decision':decision}


def atomic_csv(path, rows, fields):
    tmp = path.with_name(path.name+'.tmp')
    write_csv(tmp,rows,fieldnames=None if rows else fields)
    tmp.replace(path)


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root,suite)
    metrics = [{k:r[k] for k in (*JOB_KEYS,'n_train_images','n_train_patients',*METRIC_NAMES)} for r in data['results']]
    outputs = {'per_run_metrics.csv':(metrics,[*JOB_KEYS,*METRIC_NAMES]),
        'oof_predictions.csv':(data['predictions'],[*JOB_KEYS,'patient_id','image_name','label_id','predicted_id',*['prob_'+l for l in LABEL_TO_ID]]),
        'oof_metrics.csv':(data['oof'],['training_seed','method','budget','repeat','n_images','correct',*METRIC_NAMES]),
        'oof_comparison.csv':(data['paired'],['training_seed','method','net_correct_vs_ce',*[k+'_delta_vs_ce' for k in METRIC_NAMES]]),
        'image_changes.csv':(data['image_changes'],['training_seed','method','patient_id','image_name','label_id','ce_predicted_id','method_predicted_id','ce_correct','method_correct','change']),
        'patient_changes.csv':(data['patient_changes'],['training_seed','method','patient_id','ce_correct','method_correct','corrected','new_errors'])}
    for n,(rows,fields) in outputs.items():
        atomic_csv(root/n,rows,fields)
    _atomic_json(root/'DECISION.json',data['decision'])
    lines = ['# 同患者类别差约束：5-shot开发筛选','',f"已完成 {len(data['results'])}/40 次；模式：{suite['identity']['config']['mode']}。",
        f"工程决策：{data['decision']['status']}。",'','相同支持集合的两个训练种子，不是独立患者重复。当前93张已参与多轮研究，非独立外部验证。',
        '同患者同次检查关系为用户于2026-11-11确认，未实施独立临床复核。差分约束不证明去偏或新颖性。',
        '所有评价为单张RGB输入，完整五折后才产生OOF；阈值为工程资源决策，不是统计显著性标准。','',
        '| seed | 方法 | 正确/93 | 准确率 | macro-F1 | lt召回 | yz召回 | zz召回 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in data['oof']:
        lines.append(f"| {r['training_seed']} | {r['method']} | {r['correct']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['recall_lt']:.6f} | {r['recall_yz']:.6f} | {r['recall_zz']:.6f} |")
    tmp = root/'REPORT.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); tmp.replace(root/'REPORT.md')
    suite.update(completed_runs=len(data['results']),status='completed' if len(data['results'])==40 else 'partial',
        aggregate_sha256={n:sha256_file(root/n) for n in (*outputs,'DECISION.json','REPORT.md')})
    _atomic_json(root/'suite.json',suite)
    return data


def check_dependencies():
    try:
        import numpy
        import torch
        import torchvision
        from sklearn.metrics import accuracy_score
        accuracy_score([0,1],[0,1])
    except Exception as e:
        raise RuntimeError('Dependency check failed; use compatible numpy/scipy/sklearn and torch/torchvision in airway environment. '+str(e)) from e


def run_suite(args):
    check_dependencies(); configure_torch()
    import torch
    runtime = environment(require_cuda=not args.smoke)
    device = torch.device('cpu' if args.smoke else 'cuda')
    with output_lock(args.output_root):
        rows,suite = prepare(args.output_root,args.image_root,smoke=args.smoke)
        key = runtime_key(runtime)
        require(suite['runtime_identity'] in (None,key),'Runtime changed; use a new output root')
        suite['runtime_identity'] = key
        suite.setdefault('attempts',[]).append({'at_utc':now(),'methods':args.methods,'folds':args.folds,'seeds':args.seeds})
        _atomic_json(args.output_root/'suite.json',suite)
        jobs = selected_jobs(suite,args)
        by_name = {r['image_name']:r for r in rows}
        try:
            for i,j in enumerate(jobs,1):
                print(f"[{i}/{len(jobs)}] {j['run_id']}",flush=True)
                train_one(args.output_root,suite,j,by_name,device)
                write_aggregates(args.output_root,suite)
                if device.type=='cuda':
                    torch.cuda.empty_cache()
        except Exception as e:
            suite['last_failure'] = {'at_utc':now(),'message':str(e)}
            _atomic_json(args.output_root/'suite.json',suite)
            raise
        print(f"Completed {suite['completed_runs']}/40. Pack after all 40 complete.",flush=True)


def pack_suite(root, args=None):
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(safe_file(root,'suite.json'))
        verify_artifacts(root,suite)
        require(suite['identity']['config']['mode']=='formal','Smoke results cannot be packaged as formal')
        require(all(validate_run(root,suite,j,require_model=False) is not None for j in suite['identity']['jobs']), 'Full 40-run suite is incomplete; partial packing is not formal completion')
        write_aggregates(root,suite)
        _atomic_json(root/'package_scope.json',{'version':VERSION,'created_utc':now(),'planned_runs':40,'completed_runs':40,
            'entire_suite_complete':True,'scope':'all 40 runs, 5-shot only, both seeds, all four methods',
            'excluded':['images','weights'],'claim_status':'exploratory mechanism screen, not independent validation'})
        archive = root.parent/(root.name+'_'+str(time.time_ns())+'_results.tar.gz')
        names = ['suite.json','package_scope.json',*suite['plan_artifact_sha256'],*suite['aggregate_sha256']]
        names.extend(f"runs/{j['run_id']}/{n}" for j in suite['identity']['jobs'] for n in RUN_FILES)
        with tarfile.open(archive,'x:gz') as out:
            for n in sorted(set(names)):
                out.add(safe_file(root,n),arcname=n,recursive=False)
        Path(str(archive)+'.sha256').write_text(sha256_file(archive)+'  '+archive.name+'\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='40-fit 5-shot patient-relation screen; preserves existing experiments.')
    p.add_argument('mode',choices=['check','plan','run','status','pack'])
    p.add_argument('--image-root',type=Path,default=Path('dataset/renew'))
    p.add_argument('--output-root',type=Path,default=Path('outputs/renew_patient_relation_5shot'))
    p.add_argument('--methods',nargs='+',choices=METHODS)
    p.add_argument('--folds',nargs='+',type=int,choices=range(5))
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS)
    p.add_argument('--smoke',action='store_true',help='Random weights, CPU, two updates; separate output; never formal results')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    if args.mode=='run':
        run_suite(args)
    elif args.mode=='pack':
        require(not args.methods and args.folds is None and not args.seeds and not args.smoke,'Pack accepts only full formal suite; omit filters/--smoke')
        print(pack_suite(args.output_root))
    elif args.mode=='status':
        with output_lock(args.output_root):
            suite = read_json(safe_file(args.output_root,'suite.json'))
            verify_artifacts(args.output_root,suite)
            data = write_aggregates(args.output_root,suite)
            print(f"Completed {len(data['results'])}/40; decision={data['decision']['status']}")
    else:
        if args.mode=='check':
            check_dependencies()
        with output_lock(args.output_root):
            rows,suite = prepare(args.output_root,args.image_root,smoke=args.smoke)
            print(f"PASS: {len(rows)} images, 31 patient IDs, 40 planned fits, mode={suite['identity']['config']['mode']}")
            print('Methods: ce/supcon/paired/regrouped; seeds 42/43; 5-shot only; no teacher/cache needed.')


if __name__=='__main__':
    main()
