"""Validate recorded application identities and complete numerical outputs."""
from hashlib import sha256
import json
from pathlib import Path
import math
import re
import subprocess

def verify(root):
    data=json.loads((root/'docs/inference-cpu-evidence.json').read_text())
    files=data['source_files']
    revision=data['source_revision']
    if re.fullmatch('[0-9a-f]{40}',revision) is None:
        raise ValueError('invalid recorded source revision')
    for record in files:
        path=root/record['path']
        if path.resolve().parent != root.resolve() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('source path escapes repository')
        if sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            # Verify the older execution's immutable source, not later fixes.
            try:
                historical=subprocess.check_output(['git','show',revision+':'+record['path']],cwd=root)
            except (OSError,subprocess.CalledProcessError) as error:
                raise ValueError('recorded source snapshot is required: '+record['path']) from error
            if sha256(historical).hexdigest()!=record['sha256']:
                raise ValueError('recorded source identity changed: '+record['path'])
    fingerprint=sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if fingerprint!=data['source_fingerprint']:raise ValueError('source inventory identity changed')
    app=data['application'];reference=data['reference']
    if app['model_sha256']!=data['model_sha256'] or not reference['full_output_pass']:
        raise ValueError('model/reference identity mismatch')
    if app['batch_shape']!=[8,1,28,28] or len(app['logits'])!=8 or any(len(x)!=10 for x in app['logits']):
        raise ValueError('incomplete application output')
    for logits,probabilities,prediction in zip(app['logits'],app['probabilities'],app['predictions'],strict=True):
        if not all(math.isfinite(x) for x in logits+probabilities):raise ValueError('nonfinite output')
        if max(range(10),key=lambda n:logits[n])!=prediction:raise ValueError('prediction mismatch')
        if not math.isclose(sum(probabilities),1,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('invalid probabilities')
    if not app['usable'] or app['status']!='completed':raise ValueError('missing completed outcome')
    if app['placement']['provider_events']!={'CPUExecutionProvider':64}:raise ValueError('recorded CPU placement changed')
    if any(x['exit_code'] for x in data['checks']):raise ValueError('a recorded check failed')
    if data['contract_tests_passed']!=26:raise ValueError('unexpected executed contract count')
    comparison=json.loads((root/'docs/inference-cpu-reuse.json').read_text())
    if comparison['model_sha256']!=data['model_sha256'] or (comparison['samples'],comparison['warmups'])!=(20,5):
        raise ValueError('comparison contract changed')
    if comparison['installed_consumer_source_fingerprint']!=fingerprint or len(comparison['rows'])!=240:
        raise ValueError('comparison source or row coverage changed')
    for count in (1,2,8):
        for mode in ('cpp-fresh','cpp-reused','python-cpu-reused'):
            rows=[r for r in comparison['rows'] if r['batch']==count and r['mode']==mode and r['phase']=='sample']
            if len(rows)!=20 or {r['sample'] for r in rows}!=set(range(20)):
                raise ValueError('incomplete or duplicate comparison samples')
    for row in comparison['rows']:
        if not math.isfinite(row['seconds']) or row['seconds']<0:
            raise ValueError('invalid completed-call duration')
    memory=json.loads((root/'docs/inference-cpu-memory.json').read_text())
    if memory['status']!='completed' or (memory['completed_requests'],memory['full_output_checks'])!=(64,64):
        raise ValueError('incomplete memory probe')
    if memory['installed_consumer_source_fingerprint']!=fingerprint or memory['model_sha256']!=data['model_sha256']:
        raise ValueError('memory probe source or model changed')
    if sha256((root/'examples/image_classifier/memory_check.py').read_bytes()).hexdigest()!=memory['source_sha256']:
        raise ValueError('memory probe identity changed')
    if [r['completed_requests'] for r in memory['snapshots']]!=[0,0,8,16,32,64,64,64]:
        raise ValueError('incomplete memory checkpoints')
    for row in memory['snapshots']:
        if row['outstanding_requests']!=0 or min(row['working_set_bytes'],row['private_usage_bytes'])<0:
            raise ValueError('invalid quiescent memory sample')
        if row['peak_working_set_bytes']<row['working_set_bytes']:
            raise ValueError('invalid process peak')
    return data['source_fingerprint']

if __name__=='__main__':
    print('Recorded inference evidence verified:',verify(Path(__file__).resolve().parents[1]))
