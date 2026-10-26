"""Bounded host scheduling around the canonical federated client lifecycle."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import resource
from threading import Lock, Semaphore
import time

import torch
import numpy as np
import grpc
import safetensors
import psutil

from effusion_runtime import FitCancelled, load_host_data


def expected_config(data):
    settings, manifest = data.settings, data.manifest
    return {key: str(value) for key, value in dict(
        learning_rate=settings['lr'], local_epochs=settings['local_epochs'],
        batch_size=settings['batch_size'], optimizer=settings['optimizer'],
        run_id=manifest['run_id'], num_clients=manifest['num_clients'],
        seed=manifest['seed'], width=manifest['width'], rounds=settings['rounds']).items()}


def validate_config(data, config):
    if any(config.get(key) != value for key, value in expected_config(data).items()):
        raise ValueError('server configuration differs from the pinned host recipe')
    raw = config.get('server_round')
    if isinstance(raw, bool) or not isinstance(raw, (int, str)) or str(raw) != str(int(raw)):
        raise ValueError('invalid server round')
    current = int(raw)
    if not 1 <= current <= data.settings['rounds']:
        raise ValueError('server round exceeds the declared budget')
    return current


def check_host_security(environment):
    if environment.get('FEDLEARN_CONNECTION_TOKEN') or environment.get('FEDLEARN_REQUIRE_CLIENT_AUTH') == '1':
        raise ValueError('host multiplexing requires separate per-client credentials; shared process auth is refused')


class FitLimiter:
    def __init__(self, limit=2):
        if type(limit) is not int or limit < 1:
            raise ValueError('fit concurrency must be a positive integer')
        self._semaphore, self._lock = Semaphore(limit), Lock()
        self.active = self.peak = 0

    @contextmanager
    def acquire(self, should_stop):
        started = time.perf_counter()
        while True:
            if should_stop():
                raise FitCancelled('queued fit cancelled')
            if self._semaphore.acquire(timeout=.05):
                break
        try:
            if should_stop():
                raise FitCancelled('fit cancelled after queue acquisition')
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                yield time.perf_counter() - started
            finally:
                with self._lock:
                    self.active -= 1
        finally:
            self._semaphore.release()


class HostJournal:
    def __init__(self, output, data):
        self.output, self.data = Path(output), data
        self.output.mkdir(parents=True, exist_ok=False)
        self.events, self._lock = [], Lock()
        self.started = time.perf_counter()
        self.start_load, self.start_available = os.getloadavg(), psutil.virtual_memory().available
        (self.output/'process.json').write_text(json.dumps(dict(
            schema='fedlearn.effusion_owned_process.v1', pid=os.getpid(),
            run_id=data.manifest['run_id']), sort_keys=True)+'\n')

    def record(self, event):
        event = dict(event)
        with self._lock:
            self.events.append(event)
            with (self.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')

    def finish(self, *, peak_fits=None):
        with self._lock:
            events = list(self.events)
        expected = {(cid, rnd) for cid in self.data.clients
                    for rnd in range(1, self.data.settings['rounds']+1)}
        fits = [event for event in events if event['event'] == 'fit']
        actual = {(event['client_id'], event['round']) for event in fits}
        accepted = {(event['client_id'],event['round']) for event in events
                    if event['event']=='upload' and event.get('accepted') is True}
        errors = [event for event in events if event['event'] == 'error']
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(schema='fedlearn.effusion_host_result.v1',
            status='complete' if actual == expected and accepted == expected and not errors else 'incomplete',
            run_id=self.data.manifest['run_id'], host_label=self.data.manifest['host_label'],
            device='cpu', client_ids=list(self.data.clients), trained_client_rounds=len(actual),
            completed_client_rounds=len(actual & accepted),
            planned_client_rounds=len(expected), duplicate_fit_calls=len(fits)-len(actual),
            elapsed_s=time.perf_counter()-self.started, peak_active_fits=peak_fits,
            peak_process_rss_bytes=int(rss if platform.system() == 'Darwin' else rss*1024),
            runtime={'python':platform.python_version(),'torch':torch.__version__,
                     'numpy':np.__version__,'grpc':grpc.__version__,'safetensors':safetensors.__version__,
                     'system':platform.system(),'architecture':platform.machine(),
                     'torch_threads':torch.get_num_threads()},
            events=events, byte_scope='training RPC protobuf messages; excludes registration, heartbeat, gRPC framing, TLS and SSH')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result['system_observations']=dict(start_load_1m=self.start_load[0],end_load_1m=os.getloadavg()[0],
            start_available_memory_bytes=self.start_available,end_available_memory_bytes=psutil.virtual_memory().available,
            process_user_s=usage.ru_utime,process_system_s=usage.ru_stime)
        (self.output/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return result


class MeasuredStub:
    """Observe real protobuf message sizes without changing the transport."""
    def __init__(self, stub, journal, client_id):
        self._stub, self._journal, self._client_id = stub, journal, client_id

    def __getattr__(self, key):
        return getattr(self._stub, key)

    def GetGlobalModelStream(self, request, **kwargs):
        started = time.perf_counter()
        count, payload, current, success = request.ByteSize(), 0, 0, False
        try:
            for chunk in self._stub.GetGlobalModelStream(request, **kwargs):
                count += chunk.ByteSize()
                payload += len(chunk.chunk_data)
                current = chunk.current_round
                yield chunk
            success = True
        finally:
            self._journal.record(dict(event='download',client_id=self._client_id,round=current,
                protobuf_bytes=count,payload_bytes=payload,elapsed_s=time.perf_counter()-started,success=success))

    def SubmitModelUpdate(self, request, **kwargs):
        started = time.perf_counter()
        response = self._stub.SubmitModelUpdate(request, **kwargs)
        self._journal.record(dict(event='upload',client_id=self._client_id,round=request.trained_on_round,
            protobuf_bytes=request.ByteSize()+response.ByteSize(),elapsed_s=time.perf_counter()-started,
            accepted=response.received))
        return response


def run_host(data, client_factory, server_address, output, *, limit=2, start_client_fn=None):
    check_host_security(os.environ)
    torch.set_num_threads(1)
    if start_client_fn is None:
        from fedlearn.client.client import start_client
        start_client_fn = start_client
    journal, limiter = HostJournal(output, data), FitLimiter(limit)
    clients = {cid:client_factory(data,cid,limiter,journal) for cid in data.clients}
    def launch(cid):
        try:
            start_client_fn(server_address,clients[cid],data.clients[cid]['identity'])
        except Exception as error:
            journal.record(dict(event='error',client_id=cid,error_type=type(error).__name__,message=str(error)))
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        list(pool.map(launch, clients))
    return journal.finish(peak_fits=limiter.peak)


def cli_main(client_factory):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effusion-host-plan',type=Path,required=True)
    parser.add_argument('--host-plan-sha256',required=True)
    parser.add_argument('--server-address',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit-concurrency',type=int,default=2,choices=(1,2))
    args = parser.parse_args()
    data = load_host_data(args.effusion_host_plan,args.host_plan_sha256)
    result = run_host(data,client_factory,args.server_address,args.output,limit=args.fit_concurrency)
    print(json.dumps({key:result[key] for key in ('status','host_label','completed_client_rounds','planned_client_rounds')}))
    return 0 if result['status'] == 'complete' else 1
