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
