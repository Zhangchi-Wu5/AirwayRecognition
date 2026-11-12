"""Fixed-feature, patient-support sensitivity screen; no backbone training.

All supports are committed to the suite before extracting features. Reference
repeat 0 is separate from the 100 random draws; no held-out subset selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import re
import shutil
import tarfile
import time

from src.baseline_reproduction import output_lock, read_csv, read_json, require
from src.fewshot import PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest
from src.renew_data import EXPECTED_FINGERPRINT, audit_dataset
from src.renew_experiment import make_plan as renew_plan, fixed_config as renew_config, portable, environment, runtime_key, configure_torch
from src.frozen_screen import CLASSIFIERS, classifier_config, extract_features, l2_normalize, fit_classifier, predict_classifier

VERSION = 'renew_frozen_support_screen_v1'
DRAW_COUNT = 100
EXPECTED_RUNS = 1010
SOURCE_FILES = ('src/frozen_screen_experiment.py', 'src/frozen_screen.py',
    'src/renew_experiment.py', 'src/renew_data.py', 'src/fewshot_finetune.py',
    'src/fewshot.py', 'src/baseline_reproduction.py', 'src/data.py', 'src/models.py',
    'scripts/run_frozen_screen.py', 'scripts/run_frozen_screen_server.sh')
JOB_KEYS = ('run_id', 'set_id', 'phase', 'draw', 'fold', 'method')
RUN_FILES = ('config.json', 'model.json', 'tuning.csv', 'predictions.csv', 'result.json', 'complete.json')
CACHE_FILES = ('raw.npy', 'normalized.npy', 'metadata.json')
TUNING_FIELDS = ('C', 'heldout_patient', 'train_patients', 'n_train', 'n_val',
                 'val_log_loss', 'n_iter', 'mean_log_loss', 'selected')
HASH = re.compile(r'[0-9a-f]{64}')


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_config():
    return {'mode': 'formal', 'architecture': 'resnet18',
        'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'feature_dim': 512,
        'feature_layer': 'avgpool_flatten', 'preprocessing': 'full_frame_eval',
        'feature_normalization': 'independent per-image L2; float64 classifiers',
        'encoder': 'eval; all parameters frozen; unchanged BN statistics',
        'extraction_batch_size': 16, 'extraction_num_workers': 0,
        'head_threads': 1, 'classifier': classifier_config(),
        'fold_seed': 42, 'sampling_seed': 20261112, 'draws': DRAW_COUNT,
        'sampling': 'seed=20261112+draw*5+fold; shuffle sorted outer-train patient keys; first five',
        'reference': 'existing renew full_frame 5-shot repeat 0; sampling seed 20261104+fold*100',
        'budget': 5, 'support_images': 15,
        'selection': 'inner leave-one-support-patient-out log loss only; no outer-score selection',
        'interpretation': 'descriptive development screen; 100 draws reuse 31 patients; not independent validation',
        'claim_status': 'standard baselines, not a new algorithm or a patent novelty claim'}


def make_plan(rows):
    folds, _, old = renew_plan(rows, renew_config())
    references = {j['fold']: j for j in old if j['method'] == 'full_frame'
                  and j['budget'] == '5' and j['repeat'] == 0}
    jobs = []
    for draw in range(-1, DRAW_COUNT):
        phase = 'reference' if draw == -1 else 'draws'
        set_id = 'reference_repeat00' if draw == -1 else f'draw_{draw:03d}'
        for fold in folds:
            index = fold['fold']
            if draw == -1:
                original = references[index]
                patients, seed = original['patient_order_prefix'], original['sampling_seed']
                support = original['support']
            else:
                seed = 20261112 + draw * 5 + index
                patients = list(fold['train_patient_keys'])
                random.Random(seed).shuffle(patients)
                patients = patients[:5]
                chosen = set(patients)
                support = [r['image_name'] for r in rows if r['patient_key'] in chosen]
            for method in CLASSIFIERS:
                jobs.append({'run_id': f'{set_id}_fold_{index:02d}_{method}',
                    'set_id': set_id, 'phase': phase, 'draw': draw, 'fold': index,
                    'method': method, 'sampling_seed': seed,
                    'patient_order_prefix': patients,
                    'support_csv': f'supports/{set_id}_fold_{index:02d}.csv',
                    'support': support, 'eval': fold['eval']})
    require(len(jobs) == EXPECTED_RUNS, 'Expected 1010 final classifier jobs')
    return folds, jobs


def source_hashes():
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def artifacts(identity):
    by_name = {r['image_name']: r for r in identity['data']}
    values = {'dataset.csv': identity['data']}
    for fold in identity['folds']:
        for split in ('train', 'eval'):
            values[f"folds/fold_{fold['fold']:02d}_{split}.csv"] = [by_name[n] for n in fold[split]]
    for job in identity['jobs']:
        values[job['support_csv']] = [by_name[n] for n in job['support']]
    values['support_plan.csv'] = [{k: j[k] for k in (*JOB_KEYS, 'sampling_seed', 'support_csv')}
                                  for j in identity['jobs']]
    return values


def safe_file(root, relative):
    root, rel = Path(root), Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe artifact path')
    components = [root]
    for part in rel.parts:
        components.append(components[-1] / part)
    path = root / rel
    require(path.is_file() and not any(p.is_symlink() for p in components), 'Missing/unsafe artifact: ' + str(relative))
    return path


def verify_identity(identity):
    require(identity.get('version') == VERSION and identity.get('config') == fixed_config(), 'Changed fixed screen protocol')
    hashes = identity.get('source_sha256', {})
    require(set(hashes) == set(SOURCE_FILES) and all(HASH.fullmatch(v) for v in hashes.values()), 'Source manifest incomplete')
    rows = identity['data']
    require(len(rows) == 93 and len({r['patient_key'] for r in rows}) == 31, 'Expected 93 images and 31 patients')
    names = [r['image_name'] for r in rows]
    require(names == sorted(names) and len(set(names)) == 93, 'Invalid source order')
    for i, r in enumerate(rows):
        m = re.fullmatch(r'([0-9]+)\s*(lt|yz|zz)\.png', r['image_name'], re.IGNORECASE)
        require(m is not None and r['source_index'] == i, 'Invalid source identity')
        require(r['patient_id'] == m.group(1) and r['patient_key'] == str(int(m.group(1))), 'Invalid patient identity')
        require(r['label'] == m.group(2).lower() and r['label_id'] == LABEL_TO_ID[r['label']], 'Invalid class identity')
        require(HASH.fullmatch(r['image_sha256']) is not None, 'Invalid image SHA256')
    fingerprint = hashlib.sha256(''.join(r['image_name']+'\t'+r['image_sha256']+'\n' for r in rows).encode()).hexdigest()
    require(fingerprint == identity.get('dataset_fingerprint') == EXPECTED_FINGERPRINT, 'Dataset fingerprint changed')
    folds, jobs = make_plan(rows)
    require(identity['folds'] == folds and identity['jobs'] == jobs, 'Fold/support plan changed')
    by_name = {r['image_name']: r for r in rows}
    for j in jobs:
        require(len(j['support']) == len(set(j['support'])) == 15, 'Support must have fifteen unique images')
        require(not {by_name[n]['patient_key'] for n in j['support']} &
                {by_name[n]['patient_key'] for n in j['eval']}, 'Patient leakage')


def verify_artifacts(root, suite):
    require(suite.get('identity_sha256') == _digest(suite['identity']), 'Suite identity corrupted')
    verify_identity(suite['identity'])
    expected = artifacts(suite['identity'])
    require(set(expected) == set(suite.get('plan_artifact_sha256', {})), 'Plan manifest incomplete')
    for name, rows in expected.items():
        path = safe_file(root, name)
        require(sha256_file(path) == suite['plan_artifact_sha256'][name], 'Plan artifact changed: '+name)
        require(read_csv(path) == [{k: str(v) for k, v in row.items()} for row in rows], 'Plan content changed: '+name)


def prepare(root, image_root):
    root, image_root = Path(root).resolve(), Path(image_root).expanduser().resolve()
    require(not root.is_relative_to(image_root) and not image_root.is_relative_to(root), 'Image and output roots must be disjoint')
    try:
        rows, metadata = audit_dataset(image_root)
    except ValueError as e:
        if 'No first-level PNG' in str(e):
            raise ValueError(str(e)+'; --image-root must directly contain 93 PNGs, e.g. dataset/renew/气管镜') from e
        raise
    data = portable(rows)
    folds, jobs = make_plan(data)
    identity = {'version': VERSION, 'config': fixed_config(), 'data': data,
        'dataset_fingerprint': metadata['dataset_fingerprint'], 'folds': folds,
        'jobs': jobs, 'source_sha256': source_hashes()}
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
        'runtime_identity': None, 'device': None, 'cache_complete_sha256': None,
        'completed_runs': 0, 'status': 'prepared_not_run',
        'plan_artifact_sha256': {n: sha256_file(root/n) for n in artifacts(identity)}}
    _atomic_json(root/'suite.json', suite)
    return rows, suite


def _archive_incomplete(root, directory):
    if directory.exists():
        require(not directory.is_symlink(), 'Unsafe incomplete artifact directory')
        archived = root/'incomplete_attempts'
        archived.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archived/(directory.name+'_'+str(time.time_ns()))))


def validate_cache(root, suite, require_arrays=True):
    import numpy as np
    root = Path(root)
    directory = root/'feature_cache'
    if not (directory/'complete.json').exists():
        require(suite.get('cache_complete_sha256') is None, 'Completed cache marker missing')
        return None
    marker_path = safe_file(root, 'feature_cache/complete.json')
    complete = read_json(marker_path)
    require(complete.get('identity_sha256') == suite['identity_sha256'], 'Cache belongs to another suite')
    require(suite.get('cache_complete_sha256') in (None, sha256_file(marker_path)), 'Cache completion fingerprint changed')
    hashes = complete.get('artifact_sha256', {})
    require(set(hashes) == set(CACHE_FILES) and all(HASH.fullmatch(v) for v in hashes.values()), 'Cache manifest incomplete')
    metadata_path = safe_file(root, 'feature_cache/metadata.json')
    require(sha256_file(metadata_path) == hashes['metadata.json'], 'Cache metadata changed')
    metadata = read_json(metadata_path)
    names = [r['image_name'] for r in suite['identity']['data']]
    require(metadata.get('identity_sha256') == suite['identity_sha256'] and metadata.get('image_names') == names, 'Cache row identity changed')
    require(metadata.get('runtime_identity') == suite.get('runtime_identity') and metadata.get('runtime_identity') is not None, 'Cache runtime identity changed')
    evidence = metadata['evidence']
    for key, value in {'architecture':'resnet18', 'weights':'ResNet18_Weights.IMAGENET1K_V1',
            'preprocessing':'full_frame_eval', 'feature_layer':'avgpool_flatten', 'feature_dim':512,
            'batch_size':16, 'num_workers':0, 'images':93, 'encoder_eval':True,
            'all_parameters_frozen':True, 'precision':'float32'}.items():
        require(evidence.get(key) == value, 'Invalid encoder evidence: '+key)
    require(evidence.get('device') == suite.get('device'), 'Extraction device changed')
    for before, after in (('bn_before_sha256','bn_after_sha256'), ('state_before_sha256','state_after_sha256')):
        require(isinstance(evidence.get(before), str) and HASH.fullmatch(evidence[before])
                and evidence[before] == evidence.get(after), 'Encoder/BN changed during extraction')
    require(HASH.fullmatch(evidence.get('weights_file_sha256', '')) is not None
            and evidence['weights_file_sha256'].startswith('f37072fd')
            and evidence.get('weights_url') == 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            'Missing/incorrect pretrained weights provenance')
    require(evidence.get('row_order_sha256') == _digest(names), 'Encoder row order changed')
    present = [(directory/name).exists() for name in CACHE_FILES[:2]]
    if not require_arrays and not any(present):
        return None
    require(all(present), 'Feature arrays are missing or incomplete')
    arrays = []
    for name in CACHE_FILES[:2]:
        path = safe_file(root, 'feature_cache/'+name)
        require(sha256_file(path) == hashes[name], 'Feature cache corrupted: '+name)
        arrays.append(np.load(path, allow_pickle=False))
    raw, normalized = arrays
    require(raw.shape == normalized.shape == (93,512), 'Feature dimensions changed')
    require(raw.dtype == np.float32 and normalized.dtype == np.float64, 'Feature precision changed')
    require(np.array_equal(l2_normalize(raw), normalized), 'Normalized features differ from raw cache')
    require(hashlib.sha256(raw.tobytes(order='C')).hexdigest() == evidence.get('raw_feature_sha256'), 'Raw feature evidence changed')
    return raw, normalized


def prepare_cache(root, suite, rows, device='cuda'):
    import numpy as np
    root = Path(root)
    previous = validate_cache(root, suite)
    if previous is not None:
        # Recover a process exit after the cache marker but before suite.json.
        if suite.get('cache_complete_sha256') is None:
            suite['cache_complete_sha256'] = sha256_file(root/'feature_cache/complete.json')
            _atomic_json(root/'suite.json',suite)
        return previous[1]
    require(suite.get('runtime_identity') is not None and suite.get('device') == str(device), 'Set runtime/device before extraction')
    directory = root/'feature_cache'
    _archive_incomplete(root, directory)
    directory.mkdir()
    raw, evidence = extract_features(rows, device=device)
    normalized = l2_normalize(raw)
    for name, array in (('raw.npy',raw), ('normalized.npy',normalized)):
        tmp = directory/(name+'.tmp')
        with tmp.open('wb') as out:
            np.save(out, array, allow_pickle=False)
        tmp.replace(directory/name)
    _atomic_json(directory/'metadata.json', {'identity_sha256':suite['identity_sha256'],
        'image_names':[r['image_name'] for r in rows], 'runtime_identity':suite['runtime_identity'],
        'evidence':evidence, 'created_utc':now()})
    _atomic_json(directory/'complete.json', {'identity_sha256':suite['identity_sha256'],
        'artifact_sha256':{n:sha256_file(directory/n) for n in CACHE_FILES}})
    suite['cache_complete_sha256'] = sha256_file(directory/'complete.json')
    _atomic_json(root/'suite.json', suite)
    return validate_cache(root,suite)[1]


def _verify_predictions(path, result, rows, model, features):
    import numpy as np
    predictions = read_csv(path)
    require([(p['patient_id'],p['image_name'],int(p['label_id'])) for p in predictions] ==
            [(r['patient_id'],r['image_name'],r['label_id']) for r in rows], 'Prediction order/identity changed')
    probs = np.asarray([[float(p['prob_'+label]) for label in LABEL_TO_ID] for p in predictions], dtype=np.float64)
    require(np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
            and np.allclose(probs.sum(1),1,rtol=0,atol=1e-12), 'Invalid probabilities')
    require(probs.argmax(1).tolist() == [int(p['predicted_id']) for p in predictions], 'Prediction labels changed')
    for key,value in classification_metrics([r['label_id'] for r in rows],probs).items():
        require(math.isclose(float(result[key]),value,rel_tol=1e-9,abs_tol=1e-9), 'Metric differs from predictions: '+key)
    if features is not None:
        replay = predict_classifier(model,features[[r['source_index'] for r in rows]])
        require(np.allclose(replay,probs,rtol=1e-10,atol=1e-12), 'Saved head does not reproduce predictions')


def validate_run(root, suite, job, features=None):
    import numpy as np
    root = Path(root)
    directory = root/'runs'/job['run_id']
    if not (directory/'complete.json').exists():
        return None
    complete = read_json(safe_file(root,f"runs/{job['run_id']}/complete.json"))
    require(complete.get('identity_sha256') == suite['identity_sha256'] and complete.get('run_id') == job['run_id'], 'Wrong run completion identity')
    hashes = complete.get('artifact_sha256', {})
    require(set(hashes) == set(RUN_FILES[:-1]), 'Run manifest incomplete')
    for name in RUN_FILES[:-1]:
        path = safe_file(root,f"runs/{job['run_id']}/{name}")
        require(sha256_file(path) == hashes[name], 'Run artifact corrupted: '+str(path))
    saved = read_json(directory/'config.json')
    require(saved == {'identity_sha256':suite['identity_sha256'], 'job':job,
        'runtime_identity':suite['runtime_identity'], 'cache_complete_sha256':suite['cache_complete_sha256']}, 'Run configuration/provenance changed')
    require(saved['runtime_identity'] is not None and saved['cache_complete_sha256'] is not None, 'Missing run provenance')
    model, result = read_json(directory/'model.json'), read_json(directory/'result.json')
    require(model.get('method') == job['method'] and model.get('config') == classifier_config(), 'Wrong saved classifier')
    probe = np.zeros((1,512),dtype=np.float64)
    probe[0,0] = 1.
    predict_classifier(model,probe)  # Validate model structure even in text-only archives.
    require(all(result.get(k) == job[k] for k in JOB_KEYS), 'Result identity changed')
    require(result.get('n_train_images') == 15 and result.get('n_train_patients') == 5, 'Support count changed')
    require(result.get('cache_complete_sha256') == suite['cache_complete_sha256'], 'Result feature provenance changed')
    tuning = read_csv(directory/'tuning.csv')
    by_name = {r['image_name']:r for r in suite['identity']['data']}
    patients = sorted({by_name[n]['patient_key'] for n in job['support']})
    if job['method'] == 'prototype':
        require(not tuning, 'Prototype unexpectedly fitted hyperparameters')
    else:
        require(len(tuning) == 15, 'Expected 15 inner validation fits')
        pairs = [(float(t['C']),t['heldout_patient']) for t in tuning]
        require(sorted(pairs) == sorted((c,p) for c in (.1,1.,10.) for p in patients), 'Inner-validation coverage changed')
        import json
        max_iter = classifier_config()['linear']['max_iter']
        require(type(model.get('n_iter')) is int and 0 <= model['n_iter'] < max_iter, 'Invalid final convergence evidence')
        for t in tuning:
            require(int(t['n_train']) == 12 and int(t['n_val']) == 3, 'Inner validation budget changed')
            require(sorted(json.loads(t['train_patients'])) == sorted(set(patients)-{t['heldout_patient']}), 'Inner patient leakage')
            require(math.isfinite(float(t['val_log_loss'])) and float(t['val_log_loss']) >= 0, 'Invalid inner loss')
            require(0 <= int(t['n_iter']) < max_iter, 'Invalid inner convergence evidence')
        scores = {c:sum(float(t['val_log_loss']) for t in tuning if float(t['C'])==c)/5 for c in (.1,1.,10.)}
        require(isinstance(model.get('cv_scores'),list) and len(model['cv_scores']) == 3, 'Missing model CV evidence')
        require([r['C'] for r in model['cv_scores']] == [.1,1.,10.], 'Model CV candidates changed')
        require(all(math.isclose(r['mean_log_loss'],scores[r['C']],rel_tol=1e-12,abs_tol=1e-12)
                    for r in model['cv_scores']), 'Model CV evidence differs from inner validation')
        best = min(c for c in scores if scores[c] <= min(scores.values())+1e-12)
        require(model.get('selected_C') == result.get('selected_C') == best, 'C was not selected by support-only validation')
        for t in tuning:
            require(math.isclose(float(t['mean_log_loss']),scores[float(t['C'])],rel_tol=1e-12,abs_tol=1e-12), 'Inner mean loss changed')
            require(t['selected'] in ('True','False') and (t['selected']=='True') == (float(t['C'])==best), 'Selected C evidence changed')
    _verify_predictions(directory/'predictions.csv',result,[by_name[n] for n in job['eval']],model,features)
    return result


def fit_one(root, suite, job, features):
    root = Path(root)
    previous = validate_run(root,suite,job,features)
    if previous is not None:
        return previous
    directory = root/'runs'/job['run_id']
    _archive_incomplete(root,directory)
    directory.mkdir(parents=True)
    by_name = {r['image_name']:r for r in suite['identity']['data']}
    support = [by_name[n] for n in job['support']]
    heldout = [by_name[n] for n in job['eval']]
    _atomic_json(directory/'config.json', {'identity_sha256':suite['identity_sha256'],
        'job':job, 'runtime_identity':suite['runtime_identity'], 'cache_complete_sha256':suite['cache_complete_sha256']})
    start = time.monotonic()
    # Only these 15 selected rows reach fitting or hyperparameter selection.
    model,tuning = fit_classifier(job['method'],features[[r['source_index'] for r in support]],
                                  [r['label_id'] for r in support],[r['patient_key'] for r in support])
    _atomic_json(directory/'model.json',model)
    probabilities = predict_classifier(read_json(directory/'model.json'),features[[r['source_index'] for r in heldout]])
    write_csv(directory/'tuning.csv',tuning,fieldnames=TUNING_FIELDS)
    predictions = [{'patient_id':r['patient_id'],'image_name':r['image_name'],'label_id':r['label_id'],
        'predicted_id':int(p.argmax()),**{'prob_'+label:float(p[i]) for label,i in LABEL_TO_ID.items()}}
        for r,p in zip(heldout,probabilities)]
    write_csv(directory/'predictions.csv',predictions)
    result = {**{k:job[k] for k in JOB_KEYS}, 'n_train_images':15, 'n_train_patients':5,
        'cache_complete_sha256':suite['cache_complete_sha256'], 'seconds':time.monotonic()-start,
        'selected_C':model.get('selected_C'),
        **classification_metrics([r['label_id'] for r in heldout],probabilities)}
    _atomic_json(directory/'result.json',result)
    _atomic_json(directory/'complete.json', {'identity_sha256':suite['identity_sha256'],
        'run_id':job['run_id'],'artifact_sha256':{n:sha256_file(directory/n) for n in RUN_FILES[:-1]}})
    return validate_run(root,suite,job,features)


def aggregate(root, suite):
    import numpy as np
    root = Path(root)
    cache = validate_cache(root,suite,require_arrays=False)
    features = None if cache is None else cache[1]
    results, predictions = [], []
    for job in suite['identity']['jobs']:
        result = validate_run(root,suite,job,features)
        if result is not None:
            results.append(result)
            predictions.extend({**{k:job[k] for k in JOB_KEYS},**p}
                               for p in read_csv(root/'runs'/job['run_id']/'predictions.csv'))
    expected = sorted(r['image_name'] for r in suite['identity']['data'])
    oof, complete_predictions, paired = [], [], []
    for draw in range(-1,DRAW_COUNT):
        for method in CLASSIFIERS:
            group = [p for p in predictions if p['draw']==draw and p['method']==method]
            if {p['fold'] for p in group} != set(range(5)):
                continue
            require(sorted(p['image_name'] for p in group) == expected, 'OOF coverage changed')
            probs = np.asarray([[float(p['prob_'+l]) for l in LABEL_TO_ID] for p in group],dtype=np.float64)
            oof.append({k:group[0][k] for k in ('set_id','phase','draw','method')} |
                       {'n_images':93,'correct':sum(int(p['label_id'])==int(p['predicted_id']) for p in group),
                        **classification_metrics([int(p['label_id']) for p in group],probs)})
            complete_predictions.extend(group)
        matched = {r['method']:r for r in oof if r['draw']==draw}
        if set(matched) == set(CLASSIFIERS):
            a,b = matched['prototype'],matched['linear']
            paired.append({'set_id':a['set_id'],'phase':a['phase'],'draw':draw,
                'net_correct_linear_minus_prototype':b['correct']-a['correct'],
                **{k+'_linear_minus_prototype':b[k]-a[k] for k in METRIC_NAMES}})
    distributions, errors, selection = [], [], []
    for method in CLASSIFIERS:
        group = [r for r in oof if r['phase']=='draws' and r['method']==method]
        for metric in ('correct',*METRIC_NAMES):
            values = np.asarray([r[metric] for r in group],dtype=np.float64)
            if len(values):
                distributions.append({'method':method,'metric':metric,'n_draws':len(values),
                    'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if len(values)>1 else 0.,
                    **{name:float(np.quantile(values,q)) for name,q in [('q05',.05),('q25',.25),('median',.5),('q75',.75),('q95',.95)]}})
        for row in suite['identity']['data']:
            selected = [p for p in complete_predictions if p['phase']=='draws' and p['method']==method and p['image_name']==row['image_name']]
            if selected:
                count = sum(int(p['label_id']) != int(p['predicted_id']) for p in selected)
                errors.append({'method':method,'patient_id':row['patient_id'],'image_name':row['image_name'],
                    'label_id':row['label_id'],'n_draws':len(selected),'errors':count,'error_fraction':count/len(selected)})
    for fold in suite['identity']['folds']:
        jobs = [j for j in suite['identity']['jobs'] if j['phase']=='draws' and j['fold']==fold['fold'] and j['method']=='prototype']
        for patient in fold['train_patient_keys']:
            selection.append({'fold':fold['fold'],'patient_key':patient,'planned_draws':DRAW_COUNT,
                'selected_count':sum(patient in j['patient_order_prefix'] for j in jobs)})
    return {'results':results,'predictions':complete_predictions,'oof':oof,'paired':paired,
            'distributions':distributions,'errors':errors,'selection':selection,
            'feature_replay_verified':features is not None}


def atomic_csv(path, rows, fields):
    tmp = path.with_name(path.name+'.tmp')
    write_csv(tmp,rows,fieldnames=None if rows else fields)
    tmp.replace(path)


def write_aggregates(root,suite):
    root = Path(root)
    data = aggregate(root,suite)
    outputs = {
        'per_run_metrics.csv':(data['results'],[*JOB_KEYS,*METRIC_NAMES]),
        'oof_predictions.csv':(data['predictions'],[*JOB_KEYS,'patient_id','image_name','label_id','predicted_id',*['prob_'+l for l in LABEL_TO_ID]]),
        'oof_metrics.csv':(data['oof'],['set_id','phase','draw','method','n_images','correct',*METRIC_NAMES]),
        'oof_comparison.csv':(data['paired'],['set_id','phase','draw','net_correct_linear_minus_prototype']),
        'draw_distribution.csv':(data['distributions'],['method','metric','n_draws','mean','sd','q05','q25','median','q75','q95']),
        'image_error_frequency.csv':(data['errors'],['method','patient_id','image_name','label_id','n_draws','errors','error_fraction']),
        'support_selection_frequency.csv':(data['selection'],['fold','patient_key','planned_draws','selected_count'])}
    for name,(rows,fields) in outputs.items():
        atomic_csv(root/name,rows,fields)
    lines = ['# 冻结特征与训练患者抽样筛查','',
        f"已完成 {len(data['results'])}/{EXPECTED_RUNS} 个最终分类器任务；完整 OOF 组 {len(data['oof'])}/202。",
        '每张图仅提取一次固定 ImageNet R18 特征；没有微调主干。',
        'reference 为原 repeat 0 支持；100 套 draws 使用预先固定的新支持。不得按外层分数挑选最好患者组。',
        '100 套抽样重复使用同一31位患者，属于开发性描述，不是100次独立临床验证。',
        '冻结分类器的支持敏感性不能代表完整微调的支持敏感性；此筛查不是新算法或专利效果证据。',
        '已有 CE 只可与同支持 reference 比较，不能与100套新支持均值作配对效果比较。',
        '原型概率使用固定温度1的余弦 softmax；与线性概率的损失/置信度差异不单独解释为分类能力差异。',
        f"本次审计可重放缓存特征上的分类器预测：{data['feature_replay_verified']}。缺少数组时只核对文本哈希、配置与预测指标。",'',
        '| reference 方法 | 正确/93 | 准确率 | macro-F1 |', '|---|---:|---:|---:|']
    for row in data['oof']:
        if row['phase']=='reference':
            lines.append(f"| {row['method']} | {row['correct']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} |")
    lines.extend(['','100套完整抽样分布见 draw_distribution.csv；只汇总完成五折的组，未完成不能视为最终分布。',
        'image_error_frequency.csv 仅统计 draws；support_selection_frequency.csv 是计划中的频次，不是患者推荐名单。'])
    tmp = root/'REPORT.md.tmp'
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); tmp.replace(root/'REPORT.md')
    suite.update(completed_runs=len(data['results']),status='completed' if len(data['results'])==EXPECTED_RUNS else 'partial',
        aggregate_sha256={n:sha256_file(root/n) for n in (*outputs,'REPORT.md')})
    _atomic_json(root/'suite.json',suite)
    return data


def check_dependencies():
    try:
        import numpy
        import torch
        import torchvision
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        accuracy_score([0,1],[0,1])
    except Exception as e:
        raise RuntimeError('Dependency check failed; use compatible numpy/scipy/sklearn and torch/torchvision in airway environment. '+str(e)) from e


def run_suite(args):
    check_dependencies(); configure_torch()
    from threadpoolctl import threadpool_limits, threadpool_info
    import numpy as np
    import scipy
    import sklearn
    import platform
    runtime = runtime_key(environment(require_cuda=args.device=='cuda'))
    runtime.update(platform=platform.platform(), imported_versions={
        'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__},
        thread_limit=1, numerical_libraries=[{k:library.get(k) for k in
            ('user_api','internal_api','prefix','version','threading_layer','architecture')}
            for library in threadpool_info()])
    with output_lock(args.output_root):
        rows,suite = prepare(args.output_root,args.image_root)
        require(suite['runtime_identity'] in (None,runtime) and suite['device'] in (None,args.device), 'Runtime/device changed; use a new output root')
        suite.update(runtime_identity=runtime,device=args.device)
        _atomic_json(args.output_root/'suite.json',suite)
        try:
            features = prepare_cache(args.output_root,suite,rows,device=args.device)
            jobs = [j for j in suite['identity']['jobs'] if args.phase=='all' or j['phase']==args.phase]
            with threadpool_limits(limits=1):
                for i,job in enumerate(jobs,1):
                    print(f"[{i}/{len(jobs)}] {job['run_id']}",flush=True)
                    fit_one(args.output_root,suite,job,features)
                write_aggregates(args.output_root,suite)
        except Exception as e:
            suite['last_failure'] = {'at_utc':now(),'message':str(e)}
            _atomic_json(args.output_root/'suite.json',suite)
            raise
        print(f"Completed {suite['completed_runs']}/{EXPECTED_RUNS}. Pack after the whole suite completes.",flush=True)


def pack_suite(root):
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(safe_file(root,'suite.json'))
        verify_artifacts(root,suite)
        data = write_aggregates(root,suite)
        require(len(data['results']) == EXPECTED_RUNS and len(data['oof']) == 202 and len(data['predictions']) == 18786, 'Full 1010-job suite is incomplete; partial packing is not formal completion')
        _atomic_json(root/'package_scope.json', {'version':VERSION,'created_utc':now(),
            'planned_runs':EXPECTED_RUNS,'completed_runs':EXPECTED_RUNS,'entire_suite_complete':True,
            'scope':'original repeat0 plus 100 prespecified draws; 5 folds; prototype and linear; 5-shot only',
            'excluded':['patient images','encoder weights','raw and normalized feature matrices'],
            'includes':['frozen feature hashes and encoder provenance','small classifier JSON','support manifests','inner tuning','raw predictions','metrics'],
            'feature_replay_verified_at_pack':data['feature_replay_verified'],
            'independent_feature_extraction_replay_possible_from_package':False,
            'claim_status':'standard baseline development screen; not independent validation or patent novelty evidence'})
        archive = root.parent/(root.name+'_'+str(time.time_ns())+'_results.tar.gz')
        names = ['suite.json','package_scope.json',*suite['plan_artifact_sha256'],*suite['aggregate_sha256'],
                 'feature_cache/metadata.json','feature_cache/complete.json']
        names.extend(f"runs/{j['run_id']}/{n}" for j in suite['identity']['jobs'] for n in RUN_FILES)
        with tarfile.open(archive,'x:gz') as out:
            for name in sorted(set(names)):
                out.add(safe_file(root,name),arcname=name,recursive=False)
        Path(str(archive)+'.sha256').write_text(sha256_file(archive)+'  '+archive.name+'\n')
        return archive


def parser():
    p = argparse.ArgumentParser(description='Frozen R18 feature/support screen: original repeat0 plus 100 prespecified 5-shot draws.')
    p.add_argument('mode',choices=('check','run','status','pack'))
    p.add_argument('--image-root',type=Path,default=Path('dataset/renew'))
    p.add_argument('--output-root',type=Path,default=Path('outputs/renew_frozen_screen'))
    p.add_argument('--phase',choices=('reference','draws','all'),default='all')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda',help='Encoder extraction device; CPU must be chosen explicitly. Heads always use CPU.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    if args.mode=='run':
        run_suite(args)
    elif args.mode=='pack':
        require(args.phase=='all','Pack requires the entire suite; omit --phase filters')
        print(pack_suite(args.output_root))
    elif args.mode=='status':
        with output_lock(args.output_root):
            suite = read_json(safe_file(args.output_root,'suite.json'))
            verify_artifacts(args.output_root,suite)
            data = write_aggregates(args.output_root,suite)
            print(f"Completed {len(data['results'])}/{EXPECTED_RUNS}; complete OOF groups {len(data['oof'])}/202")
    else:
        check_dependencies()
        with output_lock(args.output_root):
            rows,suite = prepare(args.output_root,args.image_root)
            print(f"PASS: {len(rows)} images, 31 patient IDs, 505 fixed supports, {EXPECTED_RUNS} final classifier jobs")
            print('One frozen R18 extraction; original repeat0 + 100 draws; two standard heads; no backbone training.')


if __name__=='__main__':
    main()
