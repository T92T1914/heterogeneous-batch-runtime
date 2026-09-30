"""Prepared-input, completed-result comparison. Run only in a free performance slot."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from adaptive_timing.runtime import Dispatch
from hbr_image_classifier.executor import InferenceExecutor
from hbr_image_classifier.model import load_model, prepare_arrays, example_images

def run_owned(execution, batch, ident):
    execution.submit_batch(Dispatch(ident,'inference',0,1,'dispatched'),1,batch)
    result,=execution.wait(1,timeout=30)
    if not result.usable:raise RuntimeError('completed usable inference required')
    execution.observations()
    return result.value

def collect(model, *, provider='cpu', samples=20, warmups=5, deadline_seconds=120):
    if (samples,warmups)!=(20,5):raise ValueError('the committed protocol requires 20 samples and five warmups')
    import onnxruntime as ort
    if ort.__version__!='1.30.0':raise ValueError('protocol requires ONNX Runtime 1.30.0')
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    rows=[];deadline=time.perf_counter()+deadline_seconds
    reference=ReferenceEvaluator(onnx.load_model_from_string(model.data))
    for count in (8,1,2):
        batch=prepare_arrays(example_images()[:count]);input_hash=sha256(batch.tobytes()).hexdigest()
        expected=np.concatenate([reference.run(None,{'Input3':x[None]})[0] for x in batch])
        setup=time.perf_counter();reused=InferenceExecutor(model,provider=provider);setup_seconds=time.perf_counter()-setup
        rows.append({'batch':count,'mode':'cpp-reused','phase':'setup','seconds':setup_seconds,'input_sha256':input_hash})
        python=ort.InferenceSession(model.data,opts,providers=['CPUExecutionProvider'])
        try:
            for iteration in range(-warmups,samples):
                # Alternating order is fixed in advance, not chosen after a fast sample.
                modes=('cpp-fresh','cpp-reused','python-cpu-reused') if iteration%2==0 else ('python-cpu-reused','cpp-reused','cpp-fresh')
                for mode in modes:
                    if time.perf_counter()>deadline:raise TimeoutError('bounded comparison stopped; search/measurement incomplete')
                    start=time.perf_counter()
                    if mode=='cpp-fresh':
                        with InferenceExecutor(model,provider=provider) as fresh:
                            values=run_owned(fresh,batch,'fresh')
                    elif mode=='cpp-reused':values=run_owned(reused,batch,str(iteration))
                    else:
                        owned=batch.copy()
                        values=np.concatenate([python.run(None,{'Input3':x[None]})[0] for x in owned])
                    elapsed=time.perf_counter()-start
                    # Reference validation is outside all three timing boundaries,
                    # remains mandatory for every result and has a separate duration.
                    check=time.perf_counter();np.testing.assert_allclose(values,expected,rtol=1e-5,atol=2e-5)
                    validation_seconds=time.perf_counter()-check
                    rows.append({'batch':count,'mode':mode,'phase':'warmup' if iteration<0 else 'sample',
                        'sample':iteration,'seconds':elapsed,'validation_seconds':validation_seconds,'input_sha256':input_hash,
                        'max_absolute_logit_difference':float(np.abs(values-expected).max()),'provider_requested':provider if mode.startswith('cpp') else 'cpu'})
        finally:
            start=time.perf_counter();reused.close();retirement=time.perf_counter()-start
            rows.append({'batch':count,'mode':'cpp-reused','phase':'retirement','seconds':retirement,'input_sha256':input_hash})
    return {'schema_version':1,'contract':'prepared tensor, owned admission/input copy, completed host logits; PNG decode and normalization excluded equally',
        'model_sha256':model.sha256,'protocol':'docs/inference-protocol.md','ort_version':ort.__version__,
        'platform':platform.system(),'python':platform.python_version(),'samples':samples,'warmups':warmups,'rows':rows,
        'memory':'not instrumented by this timing runner; separate process/device memory acceptance required',
        'device_intervals':'not supplied; host completed-call seconds are not kernel intervals',
        'provider_placement':'requires a separately retained actual operator profile for the executed provider'}

def table(data):
    result=['| Batch | Path | Samples | Median seconds | Minimum | Maximum |','| --- | --- | --- | --- | --- | --- |']
    for count in (1,2,8):
        for mode in ('cpp-fresh','cpp-reused','python-cpu-reused'):
            values=[r['seconds'] for r in data['rows'] if r['batch']==count and r['mode']==mode and r['phase']=='sample']
            if len(values)!=data['samples']:raise ValueError('incomplete sample group')
            result.append(f'| {count} | {mode} | {len(values)} | {statistics.median(values):.8f} | {min(values):.8f} | {max(values):.8f} |')
    return '\n'.join(result)+'\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--provider',choices=['cpu','cuda-required'],default='cpu');args=parser.parse_args()
    data=collect(load_model(args.model),provider=args.provider)
    Path(args.output).write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print(table(data))

if __name__=='__main__':main()
