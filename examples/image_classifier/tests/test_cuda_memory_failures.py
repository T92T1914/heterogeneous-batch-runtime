"""Host fixtures for deadline and retirement evidence, without GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
from test_inference import model


def fake_probe(monkeypatch, model, *, primary=None, fail_device_call=None):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    import cuda_memory_check as probe
    import hbr_image_classifier.executor as executor
    from hbr_image_classifier.model import example_images, prepare_arrays
    expected = np.concatenate([ReferenceEvaluator(onnx.load_model_from_string(model.data)).run(None, {"Input3": x[None]})[0]
        for x in prepare_arrays(example_images()[:8])])
    state={"closed":False,"reads":0,"wait_limits":[]}
    class Device:
        library_sha256='fixture'
        def read(self):
            state['reads']+=1
            if state['reads']==fail_device_call:raise OSError('injected device snapshot failure')
            return {'device_free_bytes':10,'device_total_bytes':20}
    class Execution:
        outstanding=0
        def submit_batch(self,*args):pass
        def observations(self):return ()
        def wait(self,*args,timeout):
            state['wait_limits'].append(timeout)
            if primary=='empty':return ()
            if primary=='failure':raise ValueError('original inference failure')
            return (SimpleNamespace(usable=True,value=expected.copy()),)
        def close(self):state['closed']=True
    monkeypatch.setattr(probe,'CurrentProcessMemory',lambda:SimpleNamespace(read=lambda:{}))
    monkeypatch.setattr(probe,'DeviceMemory',Device)
    monkeypatch.setattr(executor,'InferenceExecutor',lambda *args,**kwargs:Execution())
    return probe,state


def test_failed_close_snapshot_preserves_original_failure_and_last_observation(model, monkeypatch):
    probe,state=fake_probe(monkeypatch,model,primary='failure',fail_device_call=3)
    report={'snapshots':[]}
    with pytest.raises(ValueError,match='original inference failure'):probe.probe(__import__('os').environ['HBR_MNIST_MODEL'],report)
    assert state['closed'] and state['reads']==4
    assert report['cleanup_errors']==[{'phase':'after_explicit_close','type':'OSError'}]
    assert report['snapshots'][-1]['phase']=='after_close_executor_release_and_gc'


def test_wait_expiry_is_explicit_timeout_and_retires_owner(model, monkeypatch):
    probe,state=fake_probe(monkeypatch,model,primary='empty')
    with pytest.raises(TimeoutError,match='physical retirement'):probe.probe(__import__('os').environ['HBR_MNIST_MODEL'],{'snapshots':[]})
    assert state['closed'] and len(state['wait_limits'])==1 and 0<=state['wait_limits'][0]<30


def test_snapshot_failure_cannot_publish_completed_status(model, monkeypatch):
    probe,state=fake_probe(monkeypatch,model,fail_device_call=7)
    report={'status':'incomplete','snapshots':[]}
    with pytest.raises(RuntimeError,match='cleanup incomplete'):probe.probe(__import__('os').environ['HBR_MNIST_MODEL'],report)
    assert state['closed'] and report['completed_requests']==64
    assert report['status']=='incomplete' and report['snapshots'][-1]['phase']=='after_close_executor_release_and_gc'
