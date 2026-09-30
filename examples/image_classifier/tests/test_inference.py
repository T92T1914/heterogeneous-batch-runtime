from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import threading
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
from PIL import Image
from adaptive_timing.runtime import Dispatch
from hbr_image_classifier.model import (MODEL_SHA256, example_images, load_model,
    prepare_arrays, probabilities, read_images)
from hbr_image_classifier.session import Session
from hbr_image_classifier.executor import InferenceExecutor

@pytest.fixture(scope="module")
def model():
    path = os.environ.get("HBR_MNIST_MODEL")
    if not path:
        pytest.fail("HBR_MNIST_MODEL must name the separately acquired reviewed model")
    return load_model(path)

def dispatch(ident):
    return Dispatch(ident, "inference", 0, 1, "dispatched")

@pytest.mark.parametrize("count", [0,1,2,8])
def test_entire_logits_against_independent_onnx_engine(model, count):
    batch = prepare_arrays(example_images()[:count])
    session = Session(model)
    try:
        actual = session.run(batch)
        assert actual.shape == (count,10) and actual.dtype == np.float32
        reference = ReferenceEvaluator(onnx.load_model_from_string(model.data))
        expected = np.concatenate([reference.run(None, {"Input3": x[None]})[0] for x in batch]) if count else np.empty((0,10))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
        first = actual.copy()
        session.run(1-batch)
        np.testing.assert_array_equal(first, actual)
        assert not np.shares_memory(actual, batch)
    finally:
        session.close()

def test_preprocessing_original_png_and_alpha(tmp_path):
    pixels = example_images()[3]
    path = tmp_path / "digit.png"
    Image.fromarray(pixels).save(path)
    prepared = read_images([path])
    np.testing.assert_array_equal(prepared[0,0], pixels.astype(np.float32)/np.float32(255))
    rgba = np.full((28,28,4),255,np.uint8);rgba[:,:,3]=0
    Image.fromarray(rgba).save(path)
    assert not read_images([path]).any()

def test_input_file_bounds_and_animation(tmp_path):
    path = tmp_path / "wrong.png"
    Image.new("L",(257,1)).save(path)
    with pytest.raises(ValueError,match="dimensions"): read_images([path])
    Image.new("L",(28,28)).save(path,format="GIF")
    with pytest.raises(ValueError,match="PNG"): read_images([path])
    path.write_bytes(b"x"*1048577)
    with pytest.raises(ValueError,match="byte limit"): read_images([path])
    with pytest.raises(ValueError,match="eight"): read_images([path]*9)

def test_model_corruption_rejected_before_runtime(tmp_path):
    path = tmp_path / "model.onnx";path.write_bytes(b"not the reviewed graph")
    with pytest.raises(ValueError,match="reviewed"): load_model(path)

@pytest.mark.parametrize("bad", [np.zeros((1,1,28,28),np.float64), np.zeros((9,1,28,28),np.float32),
    np.full((1,1,28,28),np.nan,np.float32),np.full((1,1,28,28),np.inf,np.float32),np.full((1,1,28,28),1.1,np.float32),
    np.zeros((1,1,28,28),np.float32)[:,:,:,::-1]])
def test_exact_input_contract(model, bad):
    session=Session(model)
    try:
        with pytest.raises((TypeError,ValueError)):session.run(bad)
    finally:session.close()

def test_native_owner_and_closed_session(model):
    session=Session(model)
    batch=prepare_arrays(example_images()[:1])
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError,match="owning thread"):pool.submit(session.run,batch).result()
    session.close();session.close()
    with pytest.raises(RuntimeError,match="closed"):session.run(batch)

def test_queued_cancel_running_cancel_stale_input_and_output_ownership(model):
    entered, release=threading.Event(),threading.Event()
    execution=InferenceExecutor(model,capacity=2)
    batch=prepare_arrays(example_images()[:2]);before=batch.copy()
    def gate(cancel):
        entered.set()
        assert release.wait(5)
    try:
        execution.submit_batch(dispatch("running"),1,batch,gate=gate)
        assert entered.wait(2)
        batch[:]=0
        execution.submit_batch(dispatch("queued"),1,batch)
        with pytest.raises(RuntimeError,match="capacity"):execution.submit_batch(dispatch("full"),1,batch)
        assert execution.cancel("queued")
        assert not execution.cancel("running")
        queued,=execution.reconcile(2)
        assert queued.status=="canceled" and execution.outstanding==1
        release.set()
        result,=execution.wait(2,timeout=5)
        assert result.status=="completed" and result.stale and result.cancellation_requested and not result.usable
        reference=Session(model)
        try:np.testing.assert_array_equal(result.value,reference.run(before))
        finally:reference.close()
        execution.submit_batch(dispatch("later"),2,batch)
        later,=execution.wait(2,timeout=5)
        assert later.usable and not np.shares_memory(result.value,later.value)
        kinds=[r.kind for r in execution.observations() if r.id=="running"]
        assert kinds.index("cancel_requested")<kinds.index("completed")<kinds.index("reconciled")
    finally:release.set();execution.close()

def test_shutdown_with_work_and_failure_retirement(model):
    execution=InferenceExecutor(model)
    batch=prepare_arrays(example_images()[:1])
    def fail(cancel):raise ValueError("injected host failure before Run")
    execution.submit_batch(dispatch("error"),1,batch,gate=fail)
    error,=execution.wait(1,timeout=5)
    assert error.status=="failed" and 'injected host failure' in error.error
    execution.close()
    assert execution.outstanding==0
    with pytest.raises(RuntimeError,match="closed"):execution.submit_batch(dispatch("new"),1,batch)
    with pytest.raises(ValueError,match="capacity"):InferenceExecutor(model,capacity=5)

def test_close_waits_for_running_physical_completion(model):
    entered,release=threading.Event(),threading.Event()
    execution=InferenceExecutor(model)
    def gate(cancel):
        entered.set()
        assert release.wait(5)
    execution.submit_batch(dispatch('closing'),1,prepare_arrays(example_images()[:1]),gate=gate)
    assert entered.wait(2)
    timer=threading.Timer(0.02,release.set);timer.start()
    try:
        execution.close()
        result,=execution.reconcile(1)
        assert result.status=='completed' and result.cancellation_requested and not result.usable
        assert release.is_set() and execution.outstanding==0
        execution.close()
    finally:release.set();timer.join();execution.close()

def test_duplicate_and_finite_session_history(model):
    execution=InferenceExecutor(model)
    empty=np.empty((0,1,28,28),np.float32)
    try:
        execution.submit_batch(dispatch('first'),1,empty)
        execution.wait(1,timeout=5)
        with pytest.raises(ValueError,match='twice'):execution.submit_batch(dispatch('first'),1,empty)
        for n in range(1,256):
            execution.submit_batch(dispatch(str(n)),1,empty)
            result,=execution.wait(1,timeout=5)
            assert result.usable
        with pytest.raises(RuntimeError,match='request limit'):execution.submit_batch(dispatch('limit'),1,empty)
        assert execution.outstanding==0
    finally:execution.close()

def test_profile_records_actual_cpu_placement(model,tmp_path):
    from hbr_image_classifier.cli import summarize_profile
    execution=InferenceExecutor(model,profile_prefix=tmp_path/'ort')
    execution.submit_batch(dispatch("profile"),1,prepare_arrays(example_images()[:2]))
    result,=execution.wait(1,timeout=5)
    assert result.usable
    execution.close()
    summary=summarize_profile(execution.profile_path)
    assert summary['provider_events'] and set(summary['provider_events'])=={'CPUExecutionProvider'}
    assert summary['raw_sha256']

def test_cpu_package_cannot_silently_satisfy_required_cuda(model):
    import onnxruntime
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        pytest.skip('this test exercises a CPU-only provider package, not GPU acceptance')
    with pytest.raises(RuntimeError):InferenceExecutor(model,provider='cuda-required')

@pytest.mark.parametrize('provider', ['cpu', 'cuda-required'])
def test_automatic_runtime_preloads_only_for_required_cuda(model, monkeypatch, provider):
    import sys
    from types import SimpleNamespace
    from hbr_image_classifier import _session
    from hbr_image_classifier import session as wrapper
    calls=[]
    runtime=SimpleNamespace(get_available_providers=lambda:['CUDAExecutionProvider'],
        preload_dlls=lambda **kwargs:calls.append(('preload',kwargs)))
    monkeypatch.setitem(sys.modules,'onnxruntime',runtime)
    monkeypatch.setattr(wrapper,'runtime_library',lambda:'reviewed-runtime')
    def construct(library, data, requested, prefix):
        calls.append(('construct',library,requested))
        return SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(_session,'Session',construct)
    wrapped=Session(model,provider=provider)
    wrapped.close()
    if provider=='cuda-required':
        assert calls==[('preload',{'cuda':True,'cudnn':True,'msvc':True,'directory':''}),
                       ('construct','reviewed-runtime',provider)]
    else:
        assert calls==[('construct','reviewed-runtime',provider)]

def test_automatic_required_cuda_rejects_unavailable_provider_before_construction(model, monkeypatch):
    import sys
    from types import SimpleNamespace
    from hbr_image_classifier import _session
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(get_available_providers=lambda:['CPUExecutionProvider']))
    monkeypatch.setattr(_session,'Session',lambda *args:pytest.fail('unavailable CUDA cannot construct a session'))
    with pytest.raises(RuntimeError,match='CUDAExecutionProvider'):
        Session(model,provider='cuda-required')

def test_explicit_library_preserves_caller_dependency_loading_contract(model, monkeypatch):
    from types import SimpleNamespace
    from hbr_image_classifier import _session
    from hbr_image_classifier import session as wrapper
    monkeypatch.setattr(wrapper,'runtime_library',lambda:pytest.fail('explicit library must not be replaced'))
    calls=[]
    monkeypatch.setattr(_session,'Session',lambda library,*args:(calls.append(library) or SimpleNamespace(close=lambda:None)))
    wrapped=Session(model,provider='cuda-required',library='explicit-reviewed-library')
    wrapped.close()
    assert calls==['explicit-reviewed-library']

def test_probability_math():
    logits=np.array([[1000,1001]+[0]*8],np.float32)
    values=probabilities(logits)
    assert np.isfinite(values).all() and values[0].sum()==pytest.approx(1)
    assert values.argmax()==1
