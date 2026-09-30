"""Reject relabelled retained evidence without starting a GPU operation."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'cuda_evidence_check', ROOT / 'tools/check_inference_cuda_evidence.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


@pytest.mark.parametrize('change', ['model', 'memory_provider', 'session_boundaries',
                                  'input', 'tolerance', 'memory_outputs'])
def test_relabelled_cuda_record_is_rejected(monkeypatch, change):
    path = ROOT / 'docs/inference-cuda-evidence.json'
    data = json.loads(path.read_text(encoding='utf8'))
    if change == 'model':
        data['comparison']['model_sha256'] = '0' * 64
    elif change == 'memory_provider':
        data['corrected_memory']['provider_requested'] = 'cpu'
    elif change == 'session_boundaries':
        for row in data['comparison']['rows']:
            if row['phase'] in ('setup', 'retirement'):
                row['phase'] = 'unknown'
    elif change == 'input':
        data['comparison']['rows'][0]['input_sha256'] = '0' * 64
    elif change == 'tolerance':
        data['reference']['atol'] = 1
    else:
        data['corrected_memory']['maximum_absolute_logit_difference'] = float('nan')
    original_read = Path.read_text

    def read(file, *args, **kwargs):
        return json.dumps(data) if file == path else original_read(file, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(ValueError):
        CHECKER.verify(ROOT)
