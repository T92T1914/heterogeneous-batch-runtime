"""Validate recorded application identities and complete numerical outputs."""
from hashlib import sha256
import json
from pathlib import Path
import math

def verify(root):
    data=json.loads((root/'docs/inference-cpu-evidence.json').read_text())
    files=data['source_files']
    for record in files:
        path=root/record['path']
        if path.resolve().parent != root.resolve() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('source path escapes repository')
        if sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('source identity changed: '+record['path'])
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
    return data['source_fingerprint']

if __name__=='__main__':
    print('Recorded inference evidence verified:',verify(Path(__file__).resolve().parents[1]))
