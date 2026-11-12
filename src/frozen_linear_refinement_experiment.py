"""One bounded, support-only C-grid amendment over a frozen v1 feature cache.

The original experiment is read-only. This CPU-only runner neither extracts
image features nor uses outer evaluation scores to choose a C or support set.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time

from src import frozen_screen_experiment as v1
from src.baseline_reproduction import output_lock, read_csv, read_json, require
from src.fewshot import PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest
from src.frozen_linear_refinement import classifier_config, fit_classifier, predict_classifier

VERSION = 'renew_frozen_linear_refinement_suite_v2'
EXPECTED_RUNS = 505
JOB_KEYS = v1.JOB_KEYS
RUN_FILES = v1.RUN_FILES
TUNING_FIELDS = v1.TUNING_FIELDS
SOURCE_FILES = (*v1.SOURCE_FILES, 'src/frozen_linear_refinement.py',
    'src/frozen_linear_refinement_experiment.py', 'scripts/run_frozen_linear_refinement.py',
    'scripts/run_frozen_linear_refinement_server.sh')
ARRAY_NAMES = ('reference/feature_cache/raw.npy', 'reference/feature_cache/normalized.npy')


def fixed_config():
    return {'mode':'formal', 'classifier':classifier_config(), 'head_threads':1,
        'jobs':505, 'inner_fits_per_job':25, 'final_fits_per_job':1,
        'features':'exact frozen v1 cache; no extraction or fitted preprocessing',
        'supports':'exact v1 reference repeat0 and 100 prespecified draws',
        'selection':'support-only leave-one-patient-out mean log loss; smaller C within 1e-12',
        'shared_C_reproduction':{'candidates':[0.1,1.0,10.0], 'atol':1e-8, 'rtol':1e-8},
        'amendment':'after observing v1 internal C=10 boundary; developmental, not preregistered confirmation',
        'stopping':'one bounded grid; no automatic further expansion or new algorithm',
        'interpretation':'same 31 development patients reused; not independent clinical replication',
        'claim_status':'standard linear baseline completion, not algorithm or patent novelty'}


def source_hashes():
    return {n:sha256_file(PROJECT_ROOT/n) for n in SOURCE_FILES}


def make_plan(parent):
    jobs = [dict(j) for j in parent['identity']['jobs'] if j['method']=='linear']
    require(len(jobs)==EXPECTED_RUNS and len({j['run_id'] for j in jobs})==EXPECTED_RUNS,
            'Expected exactly 505 inherited linear jobs')
    return jobs


def reference_names(parent):
    names = ['suite.json', *parent['plan_artifact_sha256'],
             'feature_cache/metadata.json', 'feature_cache/complete.json']
    names.extend(f"runs/{j['run_id']}/{n}" for j in parent['identity']['jobs'] for n in RUN_FILES)
    return sorted(set(names))


def head_runtime():
    # Import the solver first so BLAS discovery is stable before the first fit.
    import numpy as np
    import scipy
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_info
    libraries = [{k:lib.get(k) for k in ('user_api','internal_api','prefix','version','threading_layer','architecture')}
                 for lib in threadpool_info() if lib['user_api']=='blas']
    return {'python':sys.version, 'platform':platform.platform(),
        'imported_versions':{'numpy':np.__version__, 'scipy':scipy.__version__, 'scikit-learn':sklearn.__version__},
        'thread_limit':1, 'numerical_libraries':libraries}


def verify_numerical_runtime(runtime, parent):
    require(isinstance(runtime,dict) and set(runtime)=={'python','platform','imported_versions','thread_limit','numerical_libraries'}
            and isinstance(runtime['python'],str) and isinstance(runtime['platform'],str)
            and runtime['thread_limit']==1 and isinstance(runtime['imported_versions'],dict)
            and isinstance(runtime['numerical_libraries'],list), 'Invalid CPU head runtime identity')
    old = parent.get('runtime_identity') or {}
    require(runtime['imported_versions'] == old.get('imported_versions'),
            'Numerical package versions differ from v1; use the original airway environment')
    old_blas = [lib for lib in old.get('numerical_libraries',[]) if lib.get('user_api')=='blas']
    require(runtime['numerical_libraries']==old_blas, 'BLAS implementation differs from v1')


def verify_artifacts(root, suite, require_arrays=False):
    root = Path(root)
    identity = suite['identity']
    require(suite.get('identity_sha256') == _digest(identity), 'V2 identity corrupted')
    require(identity.get('version')==VERSION and identity.get('config')==fixed_config(), 'Changed fixed v2 protocol')
    hashes = identity.get('source_sha256',{})
    require(set(hashes)==set(SOURCE_FILES) and all(v1.HASH.fullmatch(h) for h in hashes.values()), 'Source manifest incomplete')
    parent_root = root/'reference'
    parent = read_json(v1.safe_file(parent_root,'suite.json'))
    if suite.get('runtime_identity') is not None:
        # Validate stored evidence only: offline audits need no current GPU or
        # matching installed numerical stack when feature arrays are absent.
        verify_numerical_runtime(suite['runtime_identity'],parent)
    require(identity.get('parent_identity_sha256')==parent['identity_sha256'], 'Parent identity changed')
    require(identity.get('parent_cache_complete_sha256')==parent.get('cache_complete_sha256')
            ==suite.get('cache_complete_sha256') and suite['cache_complete_sha256'] is not None, 'Parent cache identity changed')
    expected = reference_names(parent)
    manifest = identity.get('reference_file_sha256',{})
    require(set(manifest)==set(expected), 'Parent snapshot manifest incomplete')
    for n in expected:
        require(sha256_file(v1.safe_file(parent_root,n))==manifest[n], 'Parent snapshot changed: '+n)
    v1.verify_artifacts(parent_root,parent)
    require(all(hashes[n]==h for n,h in parent['identity']['source_sha256'].items()), 'V2 does not preserve parent source identity')
    require(identity['data']==parent['identity']['data'] and identity['folds']==parent['identity']['folds']
            and identity['jobs']==make_plan(parent), 'V2 data/folds/supports differ from v1')
    expected_plan = v1.artifacts(identity)
    require(set(suite.get('plan_artifact_sha256',{}))==set(expected_plan), 'V2 plan manifest incomplete')
    for n,rows in expected_plan.items():
        path = v1.safe_file(root,n)
        require(sha256_file(path)==suite['plan_artifact_sha256'][n], 'V2 plan changed: '+n)
        require(read_csv(path)==[{k:str(v) for k,v in r.items()} for r in rows], 'V2 plan content changed: '+n)
    cache = v1.validate_cache(parent_root,parent,require_arrays=require_arrays)
    if require_arrays:
        require(cache is not None, 'The original raw/normalized feature cache is required; no extraction fallback')
    return parent, None if cache is None else cache[1]


def verify_parent_complete(reference_root, parent, features):
    count = 0
    for job in parent['identity']['jobs']:
        require(v1.validate_run(reference_root,parent,job,features) is not None, 'Parent v1 is incomplete: '+job['run_id'])
        count += 1
    require(count==1010, 'Parent must contain the entire 1010-job v1 experiment')


def _copy_file(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.is_symlink() and not any(p.is_symlink() for p in target.parents), 'Unsafe snapshot destination')
    tmp = target.with_name(target.name+'.copying')
    require(not tmp.is_symlink(), 'Unsafe snapshot temporary file')
    shutil.copyfile(source,tmp)
    tmp.replace(target)


def prepare(root, reference_root):
    root, reference_root = Path(root).expanduser().resolve(), Path(reference_root).expanduser().resolve()
    require(not root.is_relative_to(reference_root) and not reference_root.is_relative_to(root),
            'V1 and v2 output directories must be disjoint, not nested')
    if (root/'suite.json').exists():
        suite = read_json(v1.safe_file(root,'suite.json'))
        require(suite['identity']['source_sha256']==source_hashes(), 'Resume rejected: source changed; use a new output root')
        verify_artifacts(root,suite,require_arrays=True)
        return suite
    # This read-only validation never locks, summarizes or updates the v1 root.
    parent = read_json(v1.safe_file(reference_root,'suite.json'))
    v1.verify_artifacts(reference_root,parent)
    require(parent['identity']['source_sha256']==v1.source_hashes(), 'Parent source code differs from the trusted v1 implementation')
    cache = v1.validate_cache(reference_root,parent,require_arrays=True)
    require(cache is not None and parent.get('cache_complete_sha256') is not None, 'Parent feature cache is incomplete')
    verify_parent_complete(reference_root,parent,cache[1])
    names = reference_names(parent)
    manifest = {n:sha256_file(v1.safe_file(reference_root,n)) for n in names}
    identity = {'version':VERSION, 'config':fixed_config(), 'data':parent['identity']['data'],
        'folds':parent['identity']['folds'], 'jobs':make_plan(parent),
        'parent_identity_sha256':parent['identity_sha256'],
        'parent_cache_complete_sha256':parent['cache_complete_sha256'],
        'reference_file_sha256':manifest, 'source_sha256':source_hashes()}
    intent = root/'preparing.json'
    if intent.exists():
        require(read_json(v1.safe_file(root,'preparing.json'))=={'identity_sha256':_digest(identity)},
                'Interrupted preparation belongs to a different source or parent')
    else:
        require(not root.exists() or not any(p.name!='.reproduction.lock' for p in root.iterdir()),
                'Nonempty output without a recognized v2 preparation')
        root.mkdir(parents=True,exist_ok=True)
        _atomic_json(intent,{'identity_sha256':_digest(identity)})
    for n in names+['feature_cache/raw.npy','feature_cache/normalized.npy']:
        _copy_file(v1.safe_file(reference_root,n),root/'reference'/n)
    for n,rows in v1.artifacts(identity).items():
        (root/n).parent.mkdir(parents=True,exist_ok=True)
        v1.atomic_csv(root/n,rows,list(rows[0]) if rows else [])
    suite = {'identity':identity, 'identity_sha256':_digest(identity), 'created_utc':v1.now(),
        'cache_complete_sha256':parent['cache_complete_sha256'], 'runtime_identity':None,
        'completed_runs':0, 'status':'prepared_not_run',
        'plan_artifact_sha256':{n:sha256_file(root/n) for n in v1.artifacts(identity)}}
    # Validate copied snapshots before installing the usable suite marker.
    verify_artifacts(root,suite,require_arrays=True)
    _atomic_json(root/'suite.json',suite)
    intent.unlink(missing_ok=True)
    return suite


def shared_cv_delta(root, job, tuning):
    """The unchanged candidates must reproduce v1 support-only validation."""
    old = read_csv(Path(root)/'reference'/'runs'/job['run_id']/'tuning.csv')
    old_by_key = {(float(t['C']),t['heldout_patient']):t for t in old}
    current = {(float(t['C']),str(t['heldout_patient'])):t for t in tuning
               if float(t['C']) in (.1,1.,10.)}
    require(set(old_by_key)==set(current) and len(current)==15, 'Shared-C validation coverage changed')
    differences = []
    for key,a in old_by_key.items():
        b = current[key]
        require(json.loads(a['train_patients'])==json.loads(b['train_patients'])
                and int(a['n_train'])==int(b['n_train']) and int(a['n_val'])==int(b['n_val']),
                'Shared-C patient membership changed')
        for name in ('val_log_loss','mean_log_loss'):
            av,bv = float(a[name]),float(b[name])
            require(math.isclose(av,bv,rel_tol=1e-8,abs_tol=1e-8),
                    'Shared-C validation does not reproduce v1; this is not a C-grid-only comparison')
            differences.append(abs(av-bv))
    return max(differences)


def validate_run(root, suite, job, features=None):
    import numpy as np
    root = Path(root)
    directory = root/'runs'/job['run_id']
    if not (directory/'complete.json').exists():
        return None
    marker = read_json(v1.safe_file(root,f"runs/{job['run_id']}/complete.json"))
    require(marker.get('identity_sha256')==suite['identity_sha256'] and marker.get('run_id')==job['run_id'], 'Wrong completion identity')
    hashes = marker.get('artifact_sha256',{})
    require(set(hashes)==set(RUN_FILES[:-1]), 'Run manifest incomplete')
    for n in RUN_FILES[:-1]:
        require(sha256_file(v1.safe_file(root,f"runs/{job['run_id']}/{n}"))==hashes[n], 'Run artifact corrupted: '+n)
    saved = read_json(directory/'config.json')
    require(saved=={'identity_sha256':suite['identity_sha256'],'job':job,
        'runtime_identity':suite['runtime_identity'],'cache_complete_sha256':suite['cache_complete_sha256']}, 'Run configuration changed')
    require(suite['runtime_identity'] is not None, 'Missing head runtime')
    model, result = read_json(directory/'model.json'), read_json(directory/'result.json')
    probe = np.zeros((1,512)); probe[0,0] = 1
    predict_classifier(model,probe)
    require(all(result.get(k)==job[k] for k in JOB_KEYS), 'Result identity changed')
    require(result.get('n_train_images')==15 and result.get('n_train_patients')==5, 'Training budget changed')
    require(result.get('cache_complete_sha256')==suite['cache_complete_sha256'], 'Result cache changed')
    by_name = {r['image_name']:r for r in suite['identity']['data']}
    patients = sorted({by_name[n]['patient_key'] for n in job['support']})
    tuning = read_csv(directory/'tuning.csv')
    grid = classifier_config()['linear']['C']
    require(len(tuning)==25 and sorted((float(t['C']),t['heldout_patient']) for t in tuning)
            ==sorted((c,p) for c in grid for p in patients), 'Inner validation coverage changed')
    max_iter = classifier_config()['linear']['max_iter']
    require(type(model.get('n_iter')) is int and 0<=model['n_iter']<max_iter, 'Invalid final convergence evidence')
    for t in tuning:
        require(int(t['n_train'])==12 and int(t['n_val'])==3, 'Inner budget changed')
        require(sorted(json.loads(t['train_patients']))==sorted(set(patients)-{t['heldout_patient']}), 'Inner patient leakage')
        require(math.isfinite(float(t['val_log_loss'])) and float(t['val_log_loss'])>=0, 'Invalid inner loss')
        require(0<=int(t['n_iter'])<max_iter, 'Invalid inner convergence evidence')
    scores = {c:sum(float(t['val_log_loss']) for t in tuning if float(t['C'])==c)/5 for c in grid}
    require([s['C'] for s in model['cv_scores']]==grid and all(math.isclose(s['mean_log_loss'],scores[s['C']],rel_tol=1e-12,abs_tol=1e-12)
            for s in model['cv_scores']), 'Saved CV scores changed')
    best = min(c for c in grid if scores[c]<=min(scores.values())+1e-12)
    require(model['selected_C']==result.get('selected_C')==best, 'C was not chosen by support-only validation')
    for t in tuning:
        require(math.isclose(float(t['mean_log_loss']),scores[float(t['C'])],rel_tol=1e-12,abs_tol=1e-12), 'Inner mean changed')
        require(t['selected'] in ('True','False') and (t['selected']=='True')==(float(t['C'])==best), 'Selected C evidence changed')
    delta = shared_cv_delta(root,job,tuning)
    require(math.isclose(result.get('shared_C_max_abs_delta',float('inf')),delta,rel_tol=0,abs_tol=1e-15),
            'Shared-C reproduction evidence changed')
    rows = [by_name[n] for n in job['eval']]
    # Existing float64 metric audit does not inspect the model with features=None.
    v1._verify_predictions(directory/'predictions.csv',result,rows,model,None)
    if features is not None:
        predictions = read_csv(directory/'predictions.csv')
        probs = np.asarray([[float(p['prob_'+l]) for l in LABEL_TO_ID] for p in predictions])
        replay = predict_classifier(model,features[[r['source_index'] for r in rows]])
        require(np.allclose(probs,replay,rtol=1e-10,atol=1e-12), 'Saved v2 head does not reproduce predictions')
    return result


def fit_one(root, suite, job, features):
    root = Path(root)
    completed = validate_run(root,suite,job,features)
    if completed is not None:
        return completed
    directory = root/'runs'/job['run_id']
    v1._archive_incomplete(root,directory)
    directory.mkdir(parents=True)
    by_name = {r['image_name']:r for r in suite['identity']['data']}
    support, heldout = ([by_name[n] for n in job[k]] for k in ('support','eval'))
    _atomic_json(directory/'config.json', {'identity_sha256':suite['identity_sha256'],'job':job,
        'runtime_identity':suite['runtime_identity'],'cache_complete_sha256':suite['cache_complete_sha256']})
    start = time.monotonic()
    # Outer rows are not passed to fitting or C selection.
    model,tuning = fit_classifier(features[[r['source_index'] for r in support]],
        [r['label_id'] for r in support],[r['patient_key'] for r in support])
    shared_delta = shared_cv_delta(root,job,tuning)
    _atomic_json(directory/'model.json',model)
    probs = predict_classifier(read_json(directory/'model.json'),features[[r['source_index'] for r in heldout]])
    write_csv(directory/'tuning.csv',tuning,fieldnames=TUNING_FIELDS)
    predictions = [{'patient_id':r['patient_id'],'image_name':r['image_name'],'label_id':r['label_id'],
        'predicted_id':int(p.argmax()),**{'prob_'+l:float(p[i]) for l,i in LABEL_TO_ID.items()}}
        for r,p in zip(heldout,probs)]
    write_csv(directory/'predictions.csv',predictions)
    result = {**{k:job[k] for k in JOB_KEYS},'n_train_images':15,'n_train_patients':5,
        'cache_complete_sha256':suite['cache_complete_sha256'],'seconds':time.monotonic()-start,
        'selected_C':model['selected_C'], 'shared_C_max_abs_delta':shared_delta,
        **classification_metrics([r['label_id'] for r in heldout],probs)}
    _atomic_json(directory/'result.json',result)
    _atomic_json(directory/'complete.json',{'identity_sha256':suite['identity_sha256'],'run_id':job['run_id'],
        'artifact_sha256':{n:sha256_file(directory/n) for n in RUN_FILES[:-1]}})
    return validate_run(root,suite,job,features)


def _metrics(predictions):
    import numpy as np
    probs = np.asarray([[float(p['prob_'+l]) for l in LABEL_TO_ID] for p in predictions])
    return {'correct':sum(int(p['label_id'])==int(p['predicted_id']) for p in predictions),
        **classification_metrics([int(p['label_id']) for p in predictions],probs)}


def aggregate(root, suite):
    import numpy as np
    root = Path(root)
    parent,features = verify_artifacts(root,suite,require_arrays=False)
    verify_parent_complete(root/'reference',parent,features)
    results, predictions, comparisons, changes, curves = [], [], [], [], []
    for job in suite['identity']['jobs']:
        result = validate_run(root,suite,job,features)
        if result is None:
            continue
        results.append(result)
        tags = {k:job[k] for k in JOB_KEYS}
        new = read_csv(root/'runs'/job['run_id']/'predictions.csv')
        old = read_csv(root/'reference'/'runs'/job['run_id']/'predictions.csv')
        require([p['image_name'] for p in old]==[p['image_name'] for p in new], 'Unpaired v1/v2 predictions')
        fixed = introduced = 0
        for a,b in zip(old,new):
            ac,bc = int(a['predicted_id'])==int(a['label_id']),int(b['predicted_id'])==int(b['label_id'])
            fixed += (not ac and bc); introduced += (ac and not bc)
            changes.append({**tags,'patient_id':b['patient_id'],'image_name':b['image_name'],
                'label_id':b['label_id'],'v1_predicted_id':a['predicted_id'],'v2_predicted_id':b['predicted_id'],
                'v1_correct':ac,'v2_correct':bc,'fixed':not ac and bc,'new_error':ac and not bc})
        old_metrics,new_metrics = _metrics(old),_metrics(new)
        comparisons.append({**tags,'v1_correct':old_metrics['correct'],'v2_correct':new_metrics['correct'],
            'fixed':fixed,'new_errors':introduced,'net_correct':fixed-introduced,
            **{k+'_v2_minus_v1':new_metrics[k]-old_metrics[k] for k in METRIC_NAMES}})
        predictions.extend({**tags,**p} for p in new)
        model = read_json(root/'runs'/job['run_id']/'model.json')
        curves.extend({**tags,**s,'selected':s['C']==model['selected_C']} for s in model['cv_scores'])
    expected = sorted(r['image_name'] for r in suite['identity']['data'])
    oof, complete_predictions, paired = [], [], []
    for draw in range(-1,100):
        group = [p for p in predictions if p['draw']==draw]
        if {p['fold'] for p in group}!=set(range(5)):
            continue
        require(sorted(p['image_name'] for p in group)==expected, 'Incomplete/duplicate OOF coverage')
        tags = {k:group[0][k] for k in ('set_id','phase','draw','method')}
        oof.append({**tags,'n_images':93,**_metrics(group)})
        complete_predictions.extend(group)
        run_pairs = [r for r in comparisons if r['draw']==draw]
        old_predictions = []
        for job in suite['identity']['jobs']:
            if job['draw']==draw:
                old_predictions.extend(read_csv(root/'reference'/'runs'/job['run_id']/'predictions.csv'))
        old_metrics,new_metrics = _metrics(old_predictions),_metrics(group)
        paired.append({**tags,'v1_correct':old_metrics['correct'],'v2_correct':new_metrics['correct'],
            'fixed':sum(r['fixed'] for r in run_pairs),'new_errors':sum(r['new_errors'] for r in run_pairs),
            'net_correct':new_metrics['correct']-old_metrics['correct'],
            **{k+'_v2_minus_v1':new_metrics[k]-old_metrics[k] for k in METRIC_NAMES}})
    distributions,errors,c_counts,cv_summary = [],[],[],[]
    draws = [r for r in oof if r['phase']=='draws']
    for metric in ('correct',*METRIC_NAMES):
        values = np.asarray([r[metric] for r in draws],dtype=np.float64)
        if len(values):
            distributions.append({'metric':metric,'n_draws':len(values),'mean':float(values.mean()),
                'sd':float(values.std(ddof=1)) if len(values)>1 else 0.,
                **{n:float(np.quantile(values,q)) for n,q in [('q05',.05),('q25',.25),('median',.5),('q75',.75),('q95',.95)]}})
    for row in suite['identity']['data']:
        group = [p for p in complete_predictions if p['phase']=='draws' and p['image_name']==row['image_name']]
        if group:
            count = sum(int(p['predicted_id'])!=int(p['label_id']) for p in group)
            errors.append({'patient_id':row['patient_id'],'image_name':row['image_name'],'label_id':row['label_id'],
                'n_draws':len(group),'errors':count,'error_fraction':count/len(group)})
    for phase in ('reference','draws','all'):
        selected = [r for r in results if phase=='all' or r['phase']==phase]
        for c in classifier_config()['linear']['C']:
            c_counts.append({'phase':phase,'C':c,'completed_heads':len(selected),
                'selected_count':sum(r['selected_C']==c for r in selected),'upper_boundary':c==1000})
            values = [r['mean_log_loss'] for r in curves if r['C']==c and (phase=='all' or r['phase']==phase)]
            if values:
                cv_summary.append({'phase':phase,'C':c,'n_heads':len(values),'mean_inner_log_loss':float(np.mean(values))})
    return {'results':results,'oof':oof,'predictions':complete_predictions,'paired':paired,
        'per_run_comparison':comparisons,'image_changes':changes,'distributions':distributions,
        'errors':errors,'c_counts':c_counts,'cv_curves':curves,'cv_summary':cv_summary,
        'feature_replay_verified':features is not None}


def write_aggregates(root, suite):
    root = Path(root)
    data = aggregate(root,suite)
    tables = {'per_run_metrics.csv':('results',list(JOB_KEYS)+list(METRIC_NAMES)),
        'oof_metrics.csv':('oof',['set_id','phase','draw','method','n_images','correct',*METRIC_NAMES]),
        'oof_predictions.csv':('predictions',[*JOB_KEYS,'patient_id','image_name','label_id','predicted_id']),
        'paired_v1_v2_per_run.csv':('per_run_comparison',[*JOB_KEYS,'v1_correct','v2_correct','net_correct']),
        'paired_v1_v2_oof.csv':('paired',['set_id','phase','draw','method','v1_correct','v2_correct','net_correct']),
        'paired_v1_v2_image_changes.csv':('image_changes',[*JOB_KEYS,'image_name','v1_correct','v2_correct']),
        'draw_distribution.csv':('distributions',['metric','n_draws','mean','sd']),
        'image_error_frequency.csv':('errors',['patient_id','image_name','label_id','n_draws','errors','error_fraction']),
        'selected_C_counts.csv':('c_counts',['phase','C','completed_heads','selected_count','upper_boundary']),
        'inner_cv_curves.csv':('cv_curves',[*JOB_KEYS,'C','mean_log_loss','selected']),
        'inner_cv_summary.csv':('cv_summary',['phase','C','n_heads','mean_inner_log_loss'])}
    for name,(key,fields) in tables.items():
        v1.atomic_csv(root/name,data[key],fields)
    boundary = sum(r['selected_C']==1000 for r in data['results'])
    lines = ['# 冻结线性头正则范围补全 v2','',
        f"已完成 {len(data['results'])}/505 个最终线性头；完整五折 OOF {len(data['oof'])}/101。",
        'C固定为0.1、1、10、100、1000；只在5位支持患者内部留一验证选参。',
        '复用v1特征和支持；没有新特征提取或主干微调。每C不单独评价外层或据其选参。',
        '这是看到v1内部上界后的一次开发性方案修订，不是原预注册确认性试验或专利创新。',
        f'已完成任务中选择新上界C=1000的数量：{boundary}。不自动扩展下一轮网格。',
        f"本次审计有真实缓存可重放预测：{data['feature_replay_verified']}；无数组时只复核文本记录和指标。",'',
        '| 原repeat0 | v1正确/93 | v2正确/93 | 修复 | 新增错误 |', '|---|---:|---:|---:|---:|']
    for r in data['paired']:
        if r['phase']=='reference':
            lines.append(f"| linear | {r['v1_correct']} | {r['v2_correct']} | {r['fixed']} | {r['new_errors']} |")
    lines += ['', '100套分布仅包含完整draws，不含reference；未完成结果不能作为最终分布。',
        '同一31位患者重复使用；SD/分位数是支持抽样描述，不是独立临床重复或外部泛化置信区间。',
        '对数损失改善不保证正确数增加；更高C不保证达到CE或98%。不按外层分数挑选最好患者集合。']
    tmp = root/'REPORT.md.tmp'; tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); tmp.replace(root/'REPORT.md')
    suite.update(completed_runs=len(data['results']),status='completed' if len(data['results'])==EXPECTED_RUNS else 'partial',
        aggregate_sha256={n:sha256_file(root/n) for n in (*tables,'REPORT.md')})
    _atomic_json(root/'suite.json',suite)
    return data


def run_suite(args):
    from threadpoolctl import threadpool_limits
    runtime = head_runtime()
    with output_lock(args.output_root), threadpool_limits(limits=1):
        suite = prepare(args.output_root,args.reference_root)
        parent,features = verify_artifacts(args.output_root,suite,require_arrays=True)
        verify_numerical_runtime(runtime,parent)
        require(suite['runtime_identity'] in (None,runtime), 'Head runtime changed; use the same environment')
        suite['runtime_identity'] = runtime
        _atomic_json(args.output_root/'suite.json',suite)
        jobs = [j for j in suite['identity']['jobs'] if args.phase=='all' or j['phase']==args.phase]
        try:
            for i,job in enumerate(jobs,1):
                print(f"[{i}/{len(jobs)}] {job['run_id']}",flush=True)
                fit_one(args.output_root,suite,job,features)
            write_aggregates(args.output_root,suite)
        except Exception as e:
            suite['last_failure'] = {'at_utc':v1.now(),'message':str(e)}
            _atomic_json(args.output_root/'suite.json',suite)
            raise
        print(f"Completed {suite['completed_runs']}/505. Pack after all 505 complete.",flush=True)


def pack_suite(root):
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(v1.safe_file(root,'suite.json'))
        data = write_aggregates(root,suite)
        require(len(data['results'])==505 and len(data['oof'])==101 and len(data['predictions'])==9393,
                'The entire 505-head suite is required; partial results cannot be packed as complete')
        _atomic_json(root/'package_scope.json',{'version':VERSION,'created_utc':v1.now(),'planned_runs':505,
            'completed_runs':505,'entire_suite_complete':True,'scope':'v2 linear C grid; original supports plus100draws',
            'parent_identity_sha256':suite['identity']['parent_identity_sha256'],
            'excluded':['images','encoder weights','raw and normalized feature arrays'],
            'feature_replay_verified_at_pack':data['feature_replay_verified'],
            'independent_feature_extraction_replay_possible_from_package':False,
            'claim_status':'bounded developmental baseline amendment; not a new algorithm or independent validation'})
        names = ['suite.json','package_scope.json',*suite['plan_artifact_sha256'],*suite['aggregate_sha256']]
        names += ['reference/'+n for n in suite['identity']['reference_file_sha256']]
        names += [f"runs/{j['run_id']}/{n}" for j in suite['identity']['jobs'] for n in RUN_FILES]
        archive = root.parent/(root.name+'_'+str(time.time_ns())+'_results.tar.gz')
        with tarfile.open(archive,'x:gz') as out:
            for n in sorted(set(names)):
                out.add(v1.safe_file(root,n),arcname=n,recursive=False)
        Path(str(archive)+'.sha256').write_text(sha256_file(archive)+'  '+archive.name+'\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='CPU-only bounded linear C-grid completion using the existing frozen screen.')
    p.add_argument('mode',choices=('check','run','status','pack'))
    p.add_argument('--reference-root',type=Path,default=Path('outputs/renew_frozen_screen'))
    p.add_argument('--output-root',type=Path,default=Path('outputs/renew_frozen_linear_v2'))
    p.add_argument('--phase',choices=('reference','draws','all'),default='all')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    if args.mode=='run':
        run_suite(args)
    elif args.mode=='pack':
        require(args.phase=='all','Pack requires the entire suite')
        print(pack_suite(args.output_root))
    elif args.mode=='status':
        with output_lock(args.output_root):
            suite = read_json(v1.safe_file(args.output_root,'suite.json'))
            data = write_aggregates(args.output_root,suite)
            print(f"Completed {len(data['results'])}/505; complete OOF groups {len(data['oof'])}/101")
    else:
        from threadpoolctl import threadpool_limits
        runtime = head_runtime()
        with output_lock(args.output_root), threadpool_limits(limits=1):
            suite = prepare(args.output_root,args.reference_root)
            parent,_ = verify_artifacts(args.output_root,suite,require_arrays=True)
            verify_numerical_runtime(runtime,parent)
            print('PASS: same 93 images,31patients,505supports;505 CPU linear heads; C=0.1,1,10,100,1000')
            print('Existing feature cache verified and copied; no new ResNet extraction or backbone training.')


if __name__=='__main__':
    main()
