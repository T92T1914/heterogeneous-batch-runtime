"""Validate recorded application identities and complete numerical outputs."""
from hashlib import sha256
from collections import Counter
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

def verify_request_cost(root):
    """Check retained identities, coverage and arithmetic without running inference.

    The reference outputs and process receipts are retained evidence. These
    checks do not authenticate their external origin or repeat the experiment.
    """
    root=Path(root).resolve()
    data=json.loads((root/'docs/request-cost-evidence.json').read_text(encoding='utf8'))
    revision='6ec762f4a81d172c797f03557aaeffa05f69f6ad'
    original='2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf'
    derived='cfba8152caa873db9563fef0daa1bebe97d2f3dfc3ec23cf72cc5db31a4ee539'
    providers=('cpu','cuda')
    instruments=('plain','worker','ort')
    phases=('preparation_seconds','admission_seconds','wait_reconciliation_seconds','caller_gap_seconds')

    def require(condition,message):
        if not condition:
            raise ValueError('request-cost evidence: '+message)

    def digest(value):
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid SHA256')

    def number(value,minimum=0):
        require(type(value) in (int,float) and math.isfinite(value) and value>=minimum,'invalid finite quantity')
        return value

    def equal_number(actual,expected):
        number(actual,-math.inf)
        require(math.isfinite(expected) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-15),'summary arithmetic changed')

    def index(rows,keys):
        result={tuple(row[key] for key in keys):row for row in rows}
        require(len(result)==len(rows),'duplicate '+','.join(keys))
        return result

    def summary(actual,values):
        require(bool(values),'empty sample summary')
        expected={'count':len(values),'mean':sum(values)/len(values),'median':statistics.median(values),
                  'minimum':min(values),'maximum':max(values),'p95':percentile(values,95)}
        require(set(actual)==set(expected),'summary fields changed')
        require(type(actual['count']) is int and actual['count']==len(values),'summary count changed')
        for key in expected.keys()-{'count'}:
            equal_number(actual[key],expected[key])

    def logits(actual,expected,count):
        require(isinstance(actual,list) and isinstance(expected,list) and len(actual)==len(expected)==count,'incomplete output rows')
        maximum=0.0
        for values,targets in zip(actual,expected,strict=True):
            require(len(values)==len(targets)==10,'incomplete output columns')
            for value,wanted in zip(values,targets,strict=True):
                number(value,-math.inf);number(wanted,-math.inf)
                difference=abs(value-wanted)
                require(difference<=2e-5+1e-5*abs(wanted),'complete-logit reference comparison failed')
                maximum=max(maximum,difference)
        return maximum

    require(data['schema_version']==1 and data['source_revision']==revision,'source revision or schema changed')
    expected_paths={
        'examples/image_classifier/diagnose_request_cost.py','docs/request-cost-protocol.md',
        'examples/image_classifier/compare_batching.py','examples/image_classifier/memory_check.py',
        'examples/image_classifier/session.cpp',
        *('examples/image_classifier/python/hbr_image_classifier/'+name+'.py' for name in ('model','session','executor','batching'))}
    source_files=data['source_files']
    require(len(source_files)==9 and {row['path'] for row in source_files}==expected_paths,'source inventory coverage changed')
    fingerprint=sha256(json.dumps(source_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(fingerprint==data['source_fingerprint'],'source inventory fingerprint changed')
    source_hashes={}
    for record in source_files:
        path=root/record['path']
        require(path.resolve().is_relative_to(root),'source path escapes repository')
        digest(record['sha256'])
        try:
            historical=subprocess.check_output(['git','show',revision+':'+record['path']],cwd=root,stderr=subprocess.PIPE)
        except (OSError,subprocess.CalledProcessError) as error:
            raise ValueError('request-cost recorded Git source is required: '+record['path']) from error
        require(sha256(historical).hexdigest()==record['sha256'],'recorded Git source identity changed: '+record['path'])
        source_hashes[record['path']]=record['sha256']

    identity_paths={'collector':'examples/image_classifier/compare_batching.py',
                    'memory_check':'examples/image_classifier/memory_check.py'}
    identity_paths.update({'hbr_image_classifier.'+name:'examples/image_classifier/python/hbr_image_classifier/'+name+'.py'
                           for name in ('model','session','executor','batching')})
    inherited={provider:json.loads((root/('docs/batching-'+provider+'-evidence.json')).read_text(encoding='utf8'))
               for provider in providers}
    keys={(provider,instrument,replicate) for provider in providers for instrument in instruments for replicate in range(3)}
    reports=index(data['reports'],('provider','instrumentation','replicate'))
    require(set(reports)==keys,'eighteen complete diagnostic reports required')
    conditions=index(data['analysis']['process_conditions'],('provider','instrumentation','replicate'))
    require(set(conditions)==keys,'process summary coverage changed')
    partitions=index(data['analysis']['worker_partitions'],('provider','replicate'))
    trace_index=index(data['traces'],('provider','replicate'))
    provider_replicates={(provider,replicate) for provider in providers for replicate in range(3)}
    require(set(partitions)==set(trace_index)==provider_replicates,'worker or trace coverage changed')
    reference_bank=None
    frame_hashes=None
    samples_by_key={}
    native_identity=None
    for key,record in reports.items():
        provider,instrument,replicate=key
        requested='cpu' if provider=='cpu' else 'cuda-required'
        report=record['data']
        digest(record['private_original_sha256']);digest(record['private_journal_sha256'])
        require((report['schema_version'],report['status'],report['provider'],report['instrumentation'],report['replicate'])
                ==(1,'completed',requested,instrument,replicate),'report identity or completion changed')
        require('failure' not in report,'failed report labeled completed')
        require(report['driver_sha256']==source_hashes['examples/image_classifier/diagnose_request_cost.py'],'diagnostic driver changed')
        require(report['source_model_sha256']==original,'original model identity changed')
        require(report['source_identities']==report['source_identities_after'],'installed source changed during collection')
        identities=report['source_identities']
        for name,path in identity_paths.items():
            require(identities[name]['sha256']==source_hashes[path],'installed Python/source join changed: '+name)
        require(set(identities)==set(inherited[provider]['source_identities']),'installed identity coverage changed')
        for name,identity in identities.items():
            digest(identity['sha256'])
            if 'bytes' in identity:
                require(type(identity['bytes']) is int and identity['bytes']>0,'invalid installed file size')
            require(identity==inherited[provider]['source_identities'][name],'retained installed consumer identity changed: '+name)
        if native_identity is None:
            native_identity=identities['hbr_image_classifier._session']
        require(identities['hbr_image_classifier._session']==native_identity,'native consumer differs between environments')
        config=report['configuration']
        require(config=={'batch':8,'mode':'batched','first_requests':1,'warmups':5,'samples':20,'capacity':1,
                         'max_seconds':8.0,'rtol':1e-5,'atol':2e-5},'diagnostic configuration changed')
        number(report['elapsed_seconds'])
        require(report['elapsed_seconds']<=8,'internal collection bound exceeded')
        for clock in report['clocks'].values():
            require(clock['monotonic'] is True and clock['adjustable'] is False,'clock qualification changed')
            require(number(clock['resolution'])>0,'invalid clock resolution')
        require(set(report['clocks'])=={'perf_counter','monotonic'} and
                report['clocks']['perf_counter']['implementation']=='QueryPerformanceCounter()','caller/worker QPC clock missing')
        rows=report['rows']
        counts=Counter(row['record'] for row in rows)
        expected_counts={'reference':1,'memory':2,'setup':1,'request':26,'close':1}
        if instrument=='ort':expected_counts['ort_trace']=1
        require(counts==expected_counts,'report row coverage or failure state changed')
        reference,=[row for row in rows if row['record']=='reference']
        bank=reference['original_reference_logits']
        require(len(bank)==len(reference['frames_sha256'])==4,'four original reference frames required')
        for values in bank:logits(values,values,8)
        for value in reference['frames_sha256']:digest(value)
        require(len(set(reference['frames_sha256']))==4,'input frame identities duplicated')
        if reference_bank is None:
            reference_bank=bank;frame_hashes=reference['frames_sha256']
        require(bank==reference_bank and reference['frames_sha256']==frame_hashes,'reference frames differ between reports')
        require(bank[0]==inherited[provider]['rows'][0]['expected_logits']['8'],'original reference changed from accepted frame')
        setup,=[row for row in rows if row['record']=='setup']
        require(setup['provenance']=={'model_sha256':original,'source_model_sha256':original,'execution_model_sha256':derived,
                'execution_mode':'batched','execution_recipe':'mnist-opset8-batch-preserving-reshape-v1'},'execution graph provenance changed')
        number(setup['seconds'])
        requests=[row for row in rows if row['record']=='request']
        for request,row in enumerate(requests):
            ident=f'r{replicate}-{instrument}-{request}'
            phase='first' if request==0 else ('warmup' if request<=5 else 'sample')
            require((row['request_id'],row['frame'],row['phase'],row['sample'])==
                    (ident,request%4,phase,request-6 if request>5 else None),'request identity/frame/sample sequence changed')
            stamps=[row[name] for name in ('started_ns','prepared_ns','admission_started_ns','admitted_ns','finished_ns')]
            require(all(type(value) is int and value>=0 for value in stamps) and stamps==sorted(stamps),'invalid caller QPC timestamps')
            start,prepared,admission_start,admitted,end=stamps
            differences={'primary_seconds':end-start,'preparation_seconds':prepared-start,
                'admission_seconds':admitted-admission_start,'wait_reconciliation_seconds':end-admitted,
                'caller_gap_seconds':admission_start-prepared}
            require(differences['primary_seconds']>0,'empty completed request interval')
            for name,nanoseconds in differences.items():
                require(type(row[name]) in (int,float) and row[name]==nanoseconds/1e9,'exact QPC phase arithmetic changed: '+name)
            require(sum(differences[name] for name in phases)==end-start,'caller integer partition changed')
            require(row['validation_passed'] is True,'unvalidated request labeled completed')
            number(row['validation_seconds'])
            result=row['result']
            require(result['dispatch']=={'id':ident,'resource':'inference','actual_start':0,'finish':1,'status':'dispatched'} and
                    result['generation']==1 and result['status']=='completed' and result['stale'] is False and
                    result['cancellation_requested'] is False and result['error'] is None,'completed result identity or adoption changed')
            maximum=logits(result['value'],bank[row['frame']],8)
            require(row['max_absolute_difference']==maximum,'reported output difference changed')
            times=[result[name] for name in ('submitted_at','started_at','completed_at','reconciled_at')]
            for value in times:number(value)
            require(times==sorted(times),'executor observation ordering changed')
            if instrument=='worker':
                worker=row['worker']
                entered,returned=worker['started_ns'],worker['finished_ns']
                require(type(entered) is int and type(returned) is int and start<=entered<=returned<=end and
                        worker['error_type'] is None,'worker QPC containment or failure changed')
                for name,nanoseconds in (('prefix_seconds',entered-start),('session_run_host_seconds',returned-entered),('suffix_seconds',end-returned)):
                    require(worker[name]==nanoseconds/1e9,'exact worker QPC arithmetic changed')
                require(worker['starts_before_admission_return'] is (entered<admitted),'worker/admission overlap flag changed')
            else:
                require('worker' not in row,'worker instrumentation present in another control')
        samples=[row for row in requests if row['phase']=='sample']
        samples_by_key[key]=samples
        condition=conditions[key]
        summary(condition['primary_seconds'],[row['primary_seconds'] for row in samples])
        require(set(condition['caller_phases'])==set(phases),'caller summary fields changed')
        for name in phases:summary(condition['caller_phases'][name],[row[name] for row in samples])
        if instrument=='worker':
            partition=partitions[(provider,replicate)]
            total=sum(row['primary_seconds'] for row in samples)
            equal_number(partition['primary_sum_seconds'],total)
            parts=('prefix_seconds','session_run_host_seconds','suffix_seconds')
            require(set(partition['parts'])==set(partition['fractions'])==set(parts),'worker partition fields changed')
            for name in parts:
                values=[row['worker'][name] for row in samples]
                summary(partition['parts'][name],values)
                equal_number(partition['fractions'][name],sum(values)/total)
            equal_number(sum(partition['fractions'].values()),1)
            require(partition['starts_before_admission_return']==sum(row['worker']['starts_before_admission_return'] for row in samples),
                    'worker overlap summary changed')
        close,=[row for row in rows if row['record']=='close']
        require(close['owner_retired'] is True and close['outstanding']==0 and close['retired_results']==[],'owner retirement incomplete')
        number(close['seconds'])
        observations=close['observations']
        require(len(observations)==104,'complete executor observations required')
        for request,row in enumerate(requests):
            group=observations[request*4:request*4+4]
            require([item['kind'] for item in group]==['submitted','started','completed','reconciled'],'executor transition coverage changed')
            for item,field in zip(group,('submitted_at','started_at','completed_at','reconciled_at'),strict=True):
                require(item['id']==row['request_id'] and item['generation']==1 and item['resource']=='inference' and
                        item['at']==row['result'][field],'result/observation identity or clock changed')
                require(item['stale'] is (False if item['kind']=='reconciled' else None),'observation stale state changed')
        memories=[row for row in rows if row['record']=='memory']
        require([row['phase'] for row in memories]==['before_setup','after_requests'],'memory phase coverage changed')
        for memory in memories:
            for name in ('working_set_bytes','peak_working_set_bytes','private_usage_bytes','peak_commit_bytes'):number(memory[name])
            require(memory['peak_working_set_bytes']>=memory['working_set_bytes'] and memory['peak_commit_bytes']>=memory['private_usage_bytes'],
                    'invalid cumulative process peak')
        for name in ('peak_working_set_bytes','peak_commit_bytes'):
            require(memories[1][name]>=memories[0][name],'cumulative process peak decreased')
        if instrument=='ort':
            trace=trace_index[(provider,replicate)]
            trace_summary,=[row for row in rows if row['record']=='ort_trace']
            digest(trace['sha256'])
            require(trace_summary['sha256']==trace['sha256'],'trace/report identity changed')
            events=trace['events']
            categories=dict(Counter(event.get('cat','unknown') for event in events))
            model_runs=sum(event.get('name')=='model_run' for event in events)
            wanted='CPUExecutionProvider' if provider=='cpu' else 'CUDAExecutionProvider'
            node_providers=Counter(event.get('args',{}).get('provider') for event in events if event.get('cat')=='Node' and 'provider' in event.get('args',{}))
            require(categories==trace_summary['categories']=={'Session':54,'Node':208} and
                    model_runs==trace_summary['model_runs']==26,'trace categories or actual model-run coverage changed')
            require(node_providers=={wanted:208} and trace_summary['node_providers']==[wanted],'trace provider-event coverage changed')
            for event in events:
                require(event.get('ph')=='X','unexpected trace event type')
                number(event['ts']);number(event['dur'])

    observer_rows=index(data['analysis']['extra_observer_differences'],('provider','instrumentation'))
    require(set(observer_rows)=={(provider,instrument) for provider in providers for instrument in ('worker','ort')},'observer comparison coverage changed')
    for (provider,instrument),row in observer_rows.items():
        pairs=index(row['pairs'],('replicate',))
        require(set(pairs)=={(replicate,) for replicate in range(3)},'observer process pairing changed')
        differences=[]
        for replicate in range(3):
            plain=statistics.median(item['primary_seconds'] for item in samples_by_key[(provider,'plain',replicate)])
            measured=statistics.median(item['primary_seconds'] for item in samples_by_key[(provider,instrument,replicate)])
            pair=pairs[(replicate,)];difference=measured-plain;differences.append(difference)
            for name,expected in (('plain_median_seconds',plain),('instrumented_median_seconds',measured),
                                  ('instrumented_minus_plain_seconds',difference),('instrumented_over_plain',measured/plain)):
                equal_number(pair[name],expected)
        equal_number(row['median_process_difference_seconds'],statistics.median(differences))
        require(len(row['range_seconds'])==2,'observer range incomplete')
        for actual,expected in zip(row['range_seconds'],(min(differences),max(differences)),strict=True):equal_number(actual,expected)

    retained=index(data['retained_host_phases'],('provider','mode','batch'))
    require(set(retained)=={(provider,mode,count) for provider in providers for mode in ('serial','batched') for count in (1,2,8)},
            'historical host-phase coverage changed')
    for (provider,mode,count),row in retained.items():
        filename='docs/batching-'+provider+'-evidence.json'
        require(row['input_file']==filename and row['input_sha256']==sha256((root/filename).read_bytes()).hexdigest(),
                'historical host-phase input identity changed')
        samples=[item for item in inherited[provider]['rows'] if item['record']=='request' and
                 item['mode']==mode and item['batch']==count and item['phase']=='sample']
        require(row['samples']==len(samples)==60,'historical host-phase samples incomplete')
        names=phases[:3]+('primary_seconds',)
        require(set(row['means'])==set(row['fractions_of_total'])==set(names),'historical host-phase fields changed')
        total=sum(item['primary_seconds'] for item in samples)
        for name in names:
            equal_number(row['means'][name],sum(item[name] for item in samples)/len(samples))
            equal_number(row['fractions_of_total'][name],sum(item[name] for item in samples)/total)
        gaps=[item['primary_seconds']-sum(item[name] for name in phases[:3]) for item in samples]
        summary(row['gap_seconds'],gaps)

    require(set(data['qualification'])==set(providers),'both correctness qualifications required')
    for provider in providers:
        qualification=data['qualification'][provider]
        requested='cpu' if provider=='cpu' else 'cuda-required'
        require((qualification['status'],qualification['provider'],qualification['all_sizes'],qualification['modes'],qualification['complete_output_cases'])
                ==('completed',requested,list(range(9)),['serial','batched'],54),'all-size qualification changed')
        cases=index(qualification['rows'],('mode','batch','frame'))
        require(set(cases)=={(mode,count,frame) for mode in ('serial','batched') for count in range(9) for frame in (0,2,3)},
                'qualification cases duplicated or missing')
        for (mode,count,frame),row in cases.items():
            require(row['actual_invocations']==(count if mode=='serial' else int(count>0)),'qualified native invocation count changed')
            require(row['original_reference']==reference_bank[frame][:count],'qualified original reference frame changed')
            logits(row['logits'],row['original_reference'],count)

    receipt=data['resource_receipt']
    require(receipt['total_wall_seconds']==300 and 0<number(receipt['aggregate_elapsed_seconds'])<=300,'aggregate owned-operation bound exceeded')
    operations=receipt['operations']
    expected_order=['goal-c-qualify-cpu-all-sizes','goal-c-qualify-cuda-all-sizes']
    for provider in providers:
        for replicate in range(3):
            order=instruments[replicate:]+instruments[:replicate]
            expected_order.extend(f'goal-c-{provider}-{instrument}-r{replicate}' for instrument in order)
    require([row['id'] for row in operations]==expected_order,'twenty admitted operations or declared run order changed')
    for operation in operations:
        require(operation['exit_code']==0 and all(operation[name] is True for name in ('retired','job_empty','job_assigned_before_input')),
                'owned operation failed or did not retire')
        number(operation['seconds'])
        bound=20 if '-qualify-' in operation['id'] else 10
        require(0<operation['seconds']<=bound,'owned operation exceeded its process bound')
        provider='cuda' if '-cuda-' in operation['id'] else 'cpu'
        require(operation['job_commit_bytes']==(2147483648 if provider=='cuda' else 536870912),'owned aggregate memory bound changed')
    require(sum(row['seconds'] for row in operations)<=receipt['aggregate_elapsed_seconds']+1e-9,'aggregate elapsed time excludes owned work')
    operation_index={row['id']:row for row in operations}
    for (provider,instrument,replicate),record in reports.items():
        require(record['data']['elapsed_seconds']<=operation_index[f'goal-c-{provider}-{instrument}-r{replicate}']['seconds'],
                'internal collection exceeds its enclosing operation')
    require(data['disposition']=='no_consumer_optimization_justified','diagnostics relabeled as an optimization')
    return fingerprint

if __name__=='__main__':
    print('Recorded inference evidence verified:',verify(Path(__file__).resolve().parents[1]))
    print('Recorded batching evidence verified:',verify_batching(Path(__file__).resolve().parents[1]))
    print('Recorded request-cost consistency verified:',verify_request_cost(Path(__file__).resolve().parents[1]))
