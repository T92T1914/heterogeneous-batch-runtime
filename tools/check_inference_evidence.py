"""Validate recorded application identities and complete numerical outputs."""
from hashlib import sha256
import json
from pathlib import Path
import math
import re
import subprocess
import statistics
import random

def percentile(values, percent):
    ordered=sorted(values)
    position=(len(ordered)-1)*percent/100
    lower=math.floor(position);upper=math.ceil(position)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)

def same_number(actual,expected):
    if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-15):
        raise ValueError('batching analysis arithmetic changed')

def verify_batching(root):
    """Replay the new complete evidence without executing a model or benchmark."""
    data=json.loads((root/'docs/batching-evidence.json').read_text())
    revision=data['source_revision']
    if re.fullmatch('[0-9a-f]{40}',revision) is None:
        raise ValueError('invalid batching source revision')
    for record in data['source_files']:
        path=root/record['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('batching source escapes repository')
        if sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            historical=subprocess.check_output(['git','show',revision+':'+record['path']],cwd=root)
            if sha256(historical).hexdigest()!=record['sha256']:
                raise ValueError('batching source identity changed: '+record['path'])
    fingerprint=sha256(json.dumps(data['source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if fingerprint!=data['source_fingerprint']:
        raise ValueError('batching source inventory changed')
    source_hashes={row['path']:row['sha256'] for row in data['source_files']}
    if len(data['public_reports'])!=2 or {row['provider'] for row in data['public_reports']}!={'cpu','cuda'}:
        raise ValueError('both batching provider reports required')
    for record in data['public_reports']:
        path=root/record['public_file']
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest()!=record['public_sha256']:
            raise ValueError('batching report identity changed')
        report=json.loads(path.read_text())
        requested='cpu' if record['provider']=='cpu' else 'cuda-required'
        if report['status']!='completed' or report['provider_requested']!=requested or report['completed_requests']!=468:
            raise ValueError('incomplete batching collection')
        if report['source_model_sha256']!=data['source_model_sha256'] or report['source_identities']!=report['source_identities_after']:
            raise ValueError('batching model or installed source changed')
        identity_paths={'collector':'examples/image_classifier/compare_batching.py',
            'memory_check':'examples/image_classifier/memory_check.py'}
        for module in ('model','batching','session','executor'):
            identity_paths['hbr_image_classifier.'+module]='examples/image_classifier/python/hbr_image_classifier/'+module+'.py'
        for name,path in identity_paths.items():
            if report['source_identities'][name]['sha256']!=source_hashes[path]:
                raise ValueError('executed installed source differs from recorded source: '+name)
        if 'onnxruntime_library' not in report['source_identities'] or (requested=='cuda-required' and 'onnxruntime_cuda_provider' not in report['source_identities']):
            raise ValueError('actual execution library identity missing')
        config=report['configuration']
        if (config['batches'],config['blocks'],config['samples_per_mode_block'],config['warmups_per_mode_block'])!=([1,2,8],3,20,5):
            raise ValueError('declared batching coverage changed')
        if (config['rtol'],config['atol'])!=(1e-5,2e-5):
            raise ValueError('batching tolerance changed')
        rows=report['rows']
        if any(row['record'] in ('request_failure','condition_failure','close_failure') for row in rows):
            raise ValueError('batching failure cannot be completed')
        requests=[row for row in rows if row['record']=='request']
        if len(requests)!=468 or len({row['request_id'] for row in requests})!=468:
            raise ValueError('batching request coverage incomplete or duplicated')
        reference,=[row for row in rows if row['record']=='reference_setup']
        for row in requests:
            result=row['result'];values=result['value'];expected=reference['expected_logits'][str(row['batch'])]
            graph=data['source_model_sha256'] if row['mode']=='serial' else data['execution_model_sha256']
            if row['execution_model_sha256']!=graph or not row['validation_passed'] or result['status']!='completed' or result['stale'] or result['cancellation_requested']:
                raise ValueError('batching output provenance or adoption changed')
            if len(values)!=row['batch'] or any(len(value)!=10 for value in values):
                raise ValueError('incomplete batching logits')
            for actual,target in zip(values,expected,strict=True):
                for value,wanted in zip(actual,target,strict=True):
                    if not math.isfinite(value) or not math.isfinite(wanted) or abs(value-wanted)>config['atol']+config['rtol']*abs(wanted):
                        raise ValueError('batching full-output equivalence failed')
            if not math.isfinite(row['primary_seconds']) or row['primary_seconds']<=0:
                raise ValueError('invalid batching duration')
        for count in (1,2,8):
            block_medians={}
            orders=[row for row in rows if row['record']=='run_order' and row['batch']==count]
            if len(orders)!=3 or {row['mode_order'][0] for row in orders}!={'serial','batched'}:
                raise ValueError('batching mode-order coverage missing')
            for mode in ('serial','batched'):
                summary,=[row for row in report['analysis']['conditions'] if row['batch']==count and row['mode']==mode]
                samples=[row['primary_seconds'] for row in requests if row['batch']==count and row['mode']==mode and row['phase']=='sample']
                if len(samples)!=60 or summary['median_seconds']!=statistics.median(samples):
                    raise ValueError('batching sample summary changed')
                if not math.isclose(summary['images_per_second_over_samples'],count*60/sum(samples),rel_tol=1e-12):
                    raise ValueError('batching throughput changed')
                same_number(summary['minimum_seconds'],min(samples))
                same_number(summary['maximum_seconds'],max(samples))
                same_number(summary['p95_seconds'],percentile(samples,95))
                block_medians[mode]=[]
                for block in range(3):
                    group=[row for row in requests if (row['batch'],row['mode'],row['block'])==(count,mode,block)]
                    if len(group)!=26 or [row['phase'] for row in group].count('first_request')!=1 or [row['phase'] for row in group].count('warmup')!=5:
                        raise ValueError('batching reused-session coverage changed')
                    if {row['sample'] for row in group if row['phase']=='sample'}!=set(range(20)):
                        raise ValueError('batching deciding samples duplicated or absent')
                    block_values=[row['primary_seconds'] for row in group if row['phase']=='sample']
                    block_medians[mode].append(statistics.median(block_values))
                    block_summary,=[row for row in summary['block_summaries'] if row['block']==block]
                    for name,wanted in (('median_seconds',statistics.median(block_values)),('minimum_seconds',min(block_values)),
                                        ('maximum_seconds',max(block_values)),('p95_seconds',percentile(block_values,95))):
                        same_number(block_summary[name],wanted)
                    setup,=[row for row in rows if row['record']=='setup' and (row['batch'],row['mode'],row['block'])==(count,mode,block)]
                    close,=[row for row in rows if row['record']=='close' and (row['batch'],row['mode'],row['block'])==(count,mode,block)]
                    if not close['owner_retired'] or close['outstanding_requests']!=0 or setup['execution_provenance']['execution_mode']!=mode:
                        raise ValueError('batching owner retirement or mode failed')
            differences=[serial-batched for serial,batched in zip(block_medians['serial'],block_medians['batched'],strict=True)]
            for block in range(3):
                pair,=[row for row in report['analysis']['paired'] if row['batch']==count and row.get('block')==block]
                same_number(pair['serial_minus_batched_block_median_seconds'],differences[block])
                indexed={mode:{row['sample']:row['primary_seconds'] for row in requests if (row['batch'],row['mode'],row['block'],row['phase'])==(count,mode,block,'sample')} for mode in ('serial','batched')}
                if len(pair['same_index_request_differences_seconds'])!=20:
                    raise ValueError('batching request pairing incomplete')
                for index,value in enumerate(pair['same_index_request_differences_seconds']):
                    same_number(value,indexed['serial'][index]-indexed['batched'][index])
            decision,=[row for row in report['analysis']['paired'] if row['batch']==count and 'blocks' in row]
            if (decision['blocks'],decision['bootstrap_seed'],decision['bootstrap_replicates'])!=(3,10414+count,2000):
                raise ValueError('batching block analysis design changed')
            rng=random.Random(10414+count)
            bootstrap=[statistics.median(rng.choices(differences,k=3)) for _ in range(2000)]
            low,high=percentile(bootstrap,2.5),percentile(bootstrap,97.5)
            for actual,wanted in zip(decision['block_bootstrap_interval_seconds'],(low,high),strict=True):
                same_number(actual,wanted)
            same_number(decision['median_serial_minus_batched_block_median_seconds'],statistics.median(differences))
            expected_decision='inconclusive_interval_spans_zero' if low<=0<=high else ('batched_faster_in_declared_blocks' if low>0 else 'serial_faster_in_declared_blocks')
            if decision['decision']!=expected_decision:
                raise ValueError('batching declared decision changed')
        qualification=data['placement_qualification'][record['provider']]
        if qualification['actual_profile_model_runs']!=1 or qualification['execution_model_sha256']!=data['execution_model_sha256']:
            raise ValueError('batching actual model invocation evidence missing')
        expected_provider='CPUExecutionProvider' if requested=='cpu' else 'CUDAExecutionProvider'
        if qualification['placement']['provider_events']!={expected_provider:8}:
            raise ValueError('batching operation placement changed')
        if len(qualification['complete_logits'])!=8 or len(qualification['original_reference_logits'])!=8:
            raise ValueError('installed CLI output incomplete')
        for actual,target in zip(qualification['complete_logits'],qualification['original_reference_logits'],strict=True):
            if len(actual)!=10 or len(target)!=10:
                raise ValueError('installed CLI logits incomplete')
            for value,wanted in zip(actual,target,strict=True):
                if not math.isfinite(value) or not math.isfinite(wanted) or abs(value-wanted)>2e-5+1e-5*abs(wanted):
                    raise ValueError('installed CLI complete-output equivalence failed')
    for operation in data['owned_operations']:
        if operation['exit_code']!=0 or not all(operation[key] for key in ('retired','job_empty','job_assigned_before_input')):
            raise ValueError('bounded batching process did not retire')
    return fingerprint

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
    print('Recorded batching evidence verified:',verify_batching(Path(__file__).resolve().parents[1]))
