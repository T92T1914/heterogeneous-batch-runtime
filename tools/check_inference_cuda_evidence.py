"""Check retained CUDA execution identities and complete-output coverage."""
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


def tables(data):
    timing = ['| Images | Path | Samples | Median ms | Minimum ms | Maximum ms |',
              '| --- | --- | --- | --- | --- | --- |']
    for count in (1, 2, 8):
        for mode in ('cpp-fresh', 'cpp-reused', 'python-cpu-reused'):
            values = [r['seconds'] * 1000 for r in data['comparison']['rows']
                      if r['batch'] == count and r['mode'] == mode and r['phase'] == 'sample']
            timing.append(f'| {count} | {mode} | {len(values)} | {statistics.median(values):.5f} | '
                          f'{min(values):.5f} | {max(values):.5f} |')
    memory = ['| Phase | Completed requests | Working set MiB | Private commit MiB | Shared device free MiB |',
              '| --- | --- | --- | --- | --- |']
    for row in data['corrected_memory']['snapshots']:
        memory.append(f"| {row['phase']} | {row['completed_requests']} | "
                      f"{row['working_set_bytes']/2**20:.3f} | {row['private_usage_bytes']/2**20:.3f} | "
                      f"{row['device_free_bytes']/2**20:.3f} |")
    return {'timing': '\n'.join(timing), 'memory': '\n'.join(memory)}


def source(root, revision, records):
    if re.fullmatch('[0-9a-f]{40}',revision) is None:raise ValueError('invalid source revision')
    for record in records:
        path=record['path']
        if not (root/path).resolve().is_relative_to(root.resolve()):raise ValueError('source path escapes repository')
        blob=subprocess.check_output(['git','show',revision+':'+path],cwd=root)
        if sha256(blob).hexdigest()!=record['sha256']:raise ValueError('historical source identity differs: '+path)


def verify(root, write_tables=False):
    data=json.loads((root/'docs/inference-cuda-evidence.json').read_text())
    for revision,records in [('application_source_revision','application_source_files'),
        ('comparison_source_revision','comparison_source_files'),('corrected_memory_source_revision','memory_source_files'),
        ('gpu_contract_source_revision','contract_test_files')]:source(root,data[revision],data[records])
    if sha256((root/'docs/inference-cuda-protocol.md').read_bytes()).hexdigest()!=data['protocol_sha256']:
        raise ValueError('CUDA protocol identity differs')
    source(root, data['protocol_commit'], [{'path': 'docs/inference-cuda-protocol.md',
                                          'sha256': data['protocol_sha256']}])
    app,ref=data['application'],data['reference']
    model_sha=json.loads((root/'examples/image_classifier/artifacts.json').read_text())['model']['sha256']
    if any(record['model_sha256']!=model_sha for record in
           (app,data['comparison'],data['corrected_memory'])):
        raise ValueError('reviewed model identity differs')
    if (ref['rtol'],ref['atol'])!=(1e-5,2e-5):raise ValueError('reference tolerance differs')
    if app['status']!='completed' or not app['usable'] or app['batch_shape']!=[8,1,28,28]:raise ValueError('application incomplete')
    if app['provider_requested']!='cuda-required' or app['placement']['provider_events']!={'CUDAExecutionProvider':64}:
        raise ValueError('required CUDA placement not established')
    if len(app['placement']['node_events'])!=64 or any(x['provider']!='CUDAExecutionProvider' for x in app['placement']['node_events']):
        raise ValueError('incomplete operation placement')
    if len(app['logits'])!=8 or len(ref['expected_logits'])!=8:raise ValueError('incomplete full outputs')
    maximum=0
    for actual,expected in zip(app['logits'],ref['expected_logits'],strict=True):
        if len(actual)!=10 or len(expected)!=10:raise ValueError('incomplete logit row')
        for a,b in zip(actual,expected,strict=True):
            if not math.isfinite(a+b) or abs(a-b)>2e-5+1e-5*abs(b):raise ValueError('full-output tolerance failed')
            maximum=max(maximum,abs(a-b))
    if not ref['full_output_pass'] or maximum!=ref['max_absolute_logit_difference']:raise ValueError('reference summary differs')
    comparison=data['comparison']
    if len(comparison['rows'])!=240 or (comparison['samples'],comparison['warmups'])!=(20,5):raise ValueError('comparison coverage incomplete')
    if any(r['phase'] not in ('first_invocation','warmup','sample','setup','retirement')
           or r['batch'] not in (1,2,8) or r['mode'] not in ('cpp-fresh','cpp-reused','python-cpu-reused')
           for r in comparison['rows']):raise ValueError('unrecognized comparison row')
    for count in (1,2,8):
        batch_rows=[r for r in comparison['rows'] if r['batch']==count]
        identities={r['input_sha256'] for r in batch_rows}
        if len(identities)!=1 or re.fullmatch('[0-9a-f]{64}',next(iter(identities))) is None:
            raise ValueError('comparison input identity differs')
        if count==8 and identities!={app['input_sha256']}:raise ValueError('application and comparison inputs differ')
        for phase in ('setup','retirement'):
            boundaries=[r for r in batch_rows if r['phase']==phase]
            if len(boundaries)!=1 or boundaries[0]['mode']!='cpp-reused':
                raise ValueError('reused session boundary coverage differs')
        for mode in ('cpp-fresh','cpp-reused','python-cpu-reused'):
            rows=[r for r in comparison['rows'] if r['batch']==count and r['mode']==mode
                  and r['phase'] in ('first_invocation','warmup','sample')]
            if len(rows)!=26 or {r['sample'] for r in rows if r['phase']=='sample'}!=set(range(20)):
                raise ValueError('sample/warmup coverage differs')
            if [r['phase'] for r in rows].count('first_invocation')!=1 or [r['phase'] for r in rows].count('warmup')!=5:
                raise ValueError('first invocation or warmup coverage differs')
            expected_provider='cpu' if mode=='python-cpu-reused' else 'cuda-required'
            if any(r['provider_requested']!=expected_provider for r in rows):raise ValueError('backend identity differs')
            if any(not math.isfinite(r['max_absolute_logit_difference']) or
                   r['max_absolute_logit_difference']>2e-5 for r in rows):
                raise ValueError('complete comparison validation differs')
    for row in comparison['rows']:
        if not math.isfinite(row['seconds']) or row['seconds']<0:raise ValueError('invalid completed-call interval')
    memory=data['corrected_memory']
    if memory['status']!='completed' or (memory['completed_requests'],memory['full_output_checks'])!=(64,64) or memory.get('cleanup_errors'):
        raise ValueError('corrected memory observation incomplete')
    if memory['provider_requested']!='cuda-required' or memory['input_shape']!=[8,1,28,28] or memory['input_sha256']!=app['input_sha256']:
        raise ValueError('memory execution contract differs')
    if (memory['rtol'],memory['atol'])!=(1e-5,2e-5) or not 0<=memory['maximum_absolute_logit_difference']<=2e-5:
        raise ValueError('memory full-output summary differs')
    if [r['completed_requests'] for r in memory['snapshots']]!=[0,0,8,16,32,64,64,64]:raise ValueError('memory checkpoint coverage differs')
    if memory['source_sha256']!=data['memory_source_files'][0]['sha256']:raise ValueError('memory probe execution identity differs')
    for row in memory['snapshots']:
        if row['outstanding_requests']!=0 or not 0<=row['device_free_bytes']<=row['device_total_bytes']:
            raise ValueError('invalid quiescent device sample')
        if row['working_set_bytes']>row['peak_working_set_bytes']:raise ValueError('invalid process peak')
    if data['checks']['actual_required_cuda_contracts']['passed']!=9 or data['checks']['memory_failure_fixtures']['passed']!=3:
        raise ValueError('execution/cleanup coverage differs')
    for dependency in data['dependencies']:
        if not dependency['url'].startswith('https://files.pythonhosted.org/') or re.fullmatch('[0-9a-f]{64}',dependency['sha256']) is None:
            raise ValueError('dependency identity missing')
    report_path=root/'docs/inference-cuda-results.md'
    report=report_path.read_text(encoding='utf8')
    for name, table in tables(data).items():
        expression=f'(<!-- {name}-begin -->\\n)(.*?)(<!-- {name}-end -->)'
        match=re.search(expression, report, re.DOTALL)
        if match is None:raise ValueError('generated table markers missing: '+name)
        if write_tables:
            report=report[:match.start(2)]+table+'\n'+report[match.end(2):]
        elif match.group(2)!=table+'\n':raise ValueError('result table differs from retained rows: '+name)
    if write_tables:report_path.write_text(report,encoding='utf8',newline='\n')
    return sha256((root/'docs/inference-cuda-evidence.json').read_bytes()).hexdigest()


if __name__=='__main__':
    if sys.argv[1:] not in ([], ['--write-tables']):raise SystemExit('Use --write-tables or no arguments')
    print('Recorded CUDA consumer evidence verified:',
          verify(Path(__file__).resolve().parents[1],bool(sys.argv[1:])))
