"""Host scheduling and instrumentation use independent logical-client state."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys
from threading import Event, Lock
import time
import types

import pytest

from test_effusion_runtime import pack, load


def module():
    return importlib.import_module('effusion_host')


def test_expected_round_contract_binds_run_recipe_and_client_count(tmp_path):
    path, _, _, _ = pack(tmp_path)
    data = load(path)
    cfg = module().expected_config(data) | {'server_round': 2}
    assert module().validate_config(data, cfg) == 2
    for key in module().expected_config(data):
        changed = cfg | {key: 'changed'}
        with pytest.raises(ValueError):
            module().validate_config(data, changed)
    for value in (0, 6, '1.5', True):
        with pytest.raises(ValueError):
            module().validate_config(data, cfg | {'server_round': value})


def test_fit_limiter_enforces_two_active_fits_and_records_peak():
    limiter = module().FitLimiter(2)
    lock, current, peak = Lock(), 0, 0
    def work(_):
        nonlocal current, peak
        with limiter.acquire(lambda: False) as wait:
            assert wait >= 0
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(.02)
            with lock:
                current -= 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    assert peak == limiter.peak == 2


def test_queued_fit_observes_cancellation_without_consuming_a_slot():
    limiter, stop = module().FitLimiter(1), Event()
    from effusion_runtime import FitCancelled
    with limiter.acquire(lambda: False):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: _acquire(limiter, stop))
            stop.set()
            with pytest.raises(FitCancelled):
                future.result(timeout=1)
    assert limiter.active == 0


def _acquire(limiter, stop):
    with limiter.acquire(stop.is_set):
        pytest.fail('cancelled queued fit acquired a slot')


def test_journal_requires_each_declared_round_for_each_identity(tmp_path):
    d = tmp_path/'data';d.mkdir()
    path, _, _, _ = pack(d)
    data = load(path)
    journal = module().HostJournal(tmp_path/'out', data)
    for cid in data.clients:
        for rnd in range(1, 6):
            if (cid, rnd) != (1, 5):
                journal.record({'event':'fit', 'client_id':cid, 'round':rnd, 'fit_elapsed_s':.1})
    result = journal.finish()
    assert result['status'] == 'incomplete'
    assert result['trained_client_rounds'] == 9
    assert result['completed_client_rounds'] == 0
    assert 'patient_ids' not in json.dumps(result)
    with pytest.raises(FileExistsError):
        module().HostJournal(tmp_path/'out', data)


def test_complete_fits_require_acknowledged_uploads(tmp_path):
    d=tmp_path/'data';d.mkdir()
    path,_,_,_=pack(d)
    journal=module().HostJournal(tmp_path/'out',load(path))
    for cid in journal.data.clients:
        for current in range(1,6):
            journal.record(dict(event='fit',client_id=cid,round=current,fit_elapsed_s=.1))
    assert journal.finish()['status']=='incomplete'
    for cid in journal.data.clients:
        for current in range(1,6):
            journal.record(dict(event='upload',client_id=cid,round=current,accepted=True,protobuf_bytes=10))
    result=journal.finish()
    assert result['status']=='complete' and result['completed_client_rounds']==10


def test_journal_records_owned_process_for_bounded_supervision(tmp_path):
    import os
    d=tmp_path/'data';d.mkdir()
    path,_,_,_=pack(d)
    journal=module().HostJournal(tmp_path/'out',load(path))
    process=json.loads((journal.output/'process.json').read_text())
    assert process['pid']==os.getpid()
    assert process['run_id']=='generated'
    assert process['schema']=='fedlearn.effusion_owned_process.v1'


def test_rpc_measurement_counts_actual_protobuf_messages():
    from fedlearn.communication.generated import fedlearn_pb2 as pb
    events=[]
    class Journal:
        def record(self,event):events.append(event)
    chunks=[pb.ModelChunk(chunk_index=0,total_chunks=1,current_round=2,
                          chunk_data=b'actual-codec-payload',is_final_chunk=True)]
    response=pb.SubmitModelUpdateResponse(received=True)
    class Stub:
        def GetGlobalModelStream(self,request,**kw):return iter(chunks)
        def SubmitModelUpdate(self,request,**kw):return response
    measured=module().MeasuredStub(Stub(),Journal(),7)
    request=pb.GetGlobalModelRequest(client_id='logical-7')
    assert list(measured.GetGlobalModelStream(request))==chunks
    upload=pb.SubmitModelUpdateRequest(client_id='logical-7',trained_on_round=2)
    assert measured.SubmitModelUpdate(upload).received
    assert events[0]['protobuf_bytes']==request.ByteSize()+chunks[0].ByteSize()
    assert events[1]['protobuf_bytes']==upload.ByteSize()+response.ByteSize()
    assert events[0]['payload_bytes']==len(chunks[0].chunk_data)


def test_canonical_entry_dispatches_before_legacy_dependencies(monkeypatch):
    captured={}
    stub=types.ModuleType('effusion_host')
    def cli_main(factory):captured['factory']=factory;return 0
    stub.cli_main=cli_main
    monkeypatch.setitem(sys.modules,'effusion_host',stub)
    monkeypatch.setattr(sys,'argv',['client.py','--effusion-host-plan','generated'])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(__file__).parents[1]/'client.py'),run_name='__main__')
    assert exc.value.code==0
    assert callable(captured['factory'])


def test_host_refuses_shared_process_credentials():
    with pytest.raises(ValueError,match='credential|auth'):
        module().check_host_security({'FEDLEARN_CONNECTION_TOKEN':'test-token'})
    with pytest.raises(ValueError,match='credential|auth'):
        module().check_host_security({'FEDLEARN_REQUIRE_CLIENT_AUTH':'1'})
    module().check_host_security({})
