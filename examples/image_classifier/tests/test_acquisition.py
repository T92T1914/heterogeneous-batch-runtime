from hashlib import sha256
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('hbr_acquire',Path(__file__).parents[1]/'acquire.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_atomic_publication_retry_and_changed_file(tmp_path):
    target=tmp_path/'model.onnx';data=b'reviewed public fixture';digest=sha256(data).hexdigest()
    module.publish(target,data,digest)
    module.publish(target,data,digest)
    assert target.read_bytes()==data
    target.write_bytes(b'existing different content')
    with pytest.raises(ValueError,match='differs'):module.publish(target,data,digest)
    assert target.read_bytes()==b'existing different content'
    assert not list(tmp_path.glob('.acquire-*'))

def test_failure_before_publication_leaves_no_partial_final_name(tmp_path,monkeypatch):
    target=tmp_path/'model.onnx';data=b'fixture';digest=sha256(data).hexdigest()
    original=module.os.fsync
    def fail(fd):raise OSError('injected flush failure')
    monkeypatch.setattr(module.os,'fsync',fail)
    with pytest.raises(OSError,match='flush failure'):module.publish(target,data,digest)
    assert not target.exists() and not list(tmp_path.glob('.acquire-*'))
    monkeypatch.setattr(module.os,'fsync',original)
    module.publish(target,data,digest)
    assert target.read_bytes()==data

def test_wrong_identity_never_published(tmp_path):
    target=tmp_path/'model.onnx'
    with pytest.raises(ValueError,match='unverified'):module.publish(target,b'wrong','0'*64)
    assert not target.exists()
