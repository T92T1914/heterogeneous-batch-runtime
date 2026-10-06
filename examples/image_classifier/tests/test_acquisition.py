from hashlib import sha256
import importlib.util
from pathlib import Path
import pytest
import json

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

@pytest.mark.parametrize('received',[b'short',b'fixture model',b'fixture model too long'])
def test_acquire_rejects_response_with_deciding_identity_without_leaking_body(
        tmp_path,monkeypatch,capsys,received):
    expected=b'fixture bytes'
    record={'name':'fixture.onnx','url':'https://example.invalid/model?private=signed-query',
            'bytes':len(expected),'sha256':sha256(expected).hexdigest()}
    (tmp_path/'artifacts.json').write_text(json.dumps({'model':record,'headers':[]}))
    monkeypatch.setattr(module,'__file__',str(tmp_path/'acquire.py'))
    reads=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,limit):
            reads.append(limit)
            return received[:limit]
    calls=[]
    def open_response(request,timeout):
        calls.append((request,timeout))
        return Response()
    monkeypatch.setattr(module.urllib.request,'urlopen',open_response)
    with pytest.raises(ValueError,match='failed size or identity validation') as failure:
        module.acquire(tmp_path/'artifacts')
    message=str(failure.value)
    bounded=received[:len(expected)+1]
    assert 'fixture.onnx' in message
    assert f"expected {len(expected)} bytes and sha256 {record['sha256']}" in message
    assert f"received {len(bounded)} bytes and sha256 {sha256(bounded).hexdigest()}" in message
    assert 'no artifact published' in message
    assert 'signed-query' not in message and bounded.decode() not in message
    assert reads==[len(expected)+1] and len(calls)==1 and calls[0][1]==30
    assert not list((tmp_path/'artifacts').iterdir())
    logged=capsys.readouterr().out
    assert 'fixture.onnx' in logged and 'expected bytes' in logged
    assert 'https://' not in logged and 'signed-query' not in logged

def test_acquire_verified_response_and_existing_retry_do_not_redownload(
        tmp_path,monkeypatch,capsys):
    expected=b'fixture bytes'
    record={'name':'fixture.onnx','url':'https://example.invalid/model',
            'bytes':len(expected),'sha256':sha256(expected).hexdigest()}
    (tmp_path/'artifacts.json').write_text(json.dumps({'model':record,'headers':[]}))
    monkeypatch.setattr(module,'__file__',str(tmp_path/'acquire.py'))
    calls=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,limit):
            assert limit==len(expected)+1
            return expected
    def open_response(request,timeout):
        calls.append((request,timeout))
        return Response()
    monkeypatch.setattr(module.urllib.request,'urlopen',open_response)
    destination=tmp_path/'artifacts'
    module.acquire(destination)
    module.acquire(destination)
    assert (destination/'fixture.onnx').read_bytes()==expected
    assert len(calls)==1
    assert not list(destination.glob('.acquire-*'))
    assert 'fixture.onnx' in capsys.readouterr().out
