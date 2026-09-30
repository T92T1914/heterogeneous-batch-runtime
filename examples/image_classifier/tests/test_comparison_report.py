import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('hbr_compare',Path(__file__).parents[1]/'compare.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def rows():
    return [{'batch':n,'mode':mode,'phase':'sample','seconds':float(i+1)}
        for n in (1,2,8) for mode in ('cpp-fresh','cpp-reused','python-cpu-reused') for i in range(20)]

def test_table_uses_entire_retained_sample_group():
    text=module.table({'samples':20,'rows':rows()})
    assert text.count('10.50000000')==9
    assert '| 8 | cpp-reused | 20 |' in text

def test_incomplete_comparison_is_not_reported_as_a_complete_table():
    with pytest.raises(ValueError,match='incomplete'):module.table({'samples':20,'rows':rows()[:-1]})

@pytest.mark.parametrize('fail_at',('baseline','setup_record'))
def test_post_creation_failure_retires_the_created_owner(monkeypatch,fail_at):
    from types import SimpleNamespace
    import numpy as np
    import onnxruntime as ort
    owners=[];retained=[]
    class Owner:
        def __init__(self,*args,**kwargs):
            self.closed=False;owners.append(self)
        def close(self):self.closed=True
    class Reference:
        def run(self,*args,**kwargs):return [np.zeros((1,10),dtype=np.float32)]
    def fail(*args,**kwargs):raise RuntimeError('baseline initialization failed')
    def record(row):
        retained.append(row)
        if fail_at=='setup_record' and row['phase']=='setup':
            raise RuntimeError('journal write failed')
    monkeypatch.setattr(module,'InferenceExecutor',Owner)
    monkeypatch.setattr(module,'ReferenceEvaluator',lambda *args:Reference())
    monkeypatch.setattr(module.onnx,'load_model_from_string',lambda *args:object())
    monkeypatch.setattr(ort,'InferenceSession',fail)
    with pytest.raises(RuntimeError,match='baseline initialization failed|journal write failed'):
        module.collect(SimpleNamespace(data=b''),on_record=record)
    assert len(owners)==1 and owners[0].closed
    assert [row['phase'] for row in retained]==['setup','retirement']
