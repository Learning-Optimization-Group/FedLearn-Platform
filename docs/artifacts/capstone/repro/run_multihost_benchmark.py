"""Frozen CPU benchmark supervisor around the canonical server/client executables.

Launch configuration and outputs are private. Public analysis exports aggregate
measurements only. SSH protects loopback gRPC; protobuf counts exclude SSH bytes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from multihost_manifest import assignments, prepare, verify_prepared, _digest, _json


ENV = dict(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
HISTORICAL_SUMMARY='83f38253f61c5c674d408f4baac141ed7aff2661d304383aa2f8ba8c647f0409'
HISTORICAL_RUNS={'mean512':'d56b38c74c83ca9db769abe16e2a31ae203836639fb250e4b1500c4766019793',
                 'projected128':'740f12ba11e2a970a54b354f6279b7e02e04d3a0d4309b7321c0889d3c509dad'}


def load_launch_config(path):
    value=json.loads(Path(path).read_text())
    if set(value)!={'local_python','hosts'} or set(value['hosts'])!={'g14','orin'}:
        raise ValueError('launch configuration requires local Python and both remote hosts')
    if not isinstance(value['local_python'],str) or not Path(value['local_python']).is_absolute():
        raise ValueError('local interpreter must be absolute')
    for config in value['hosts'].values():
        if (set(config)!={'ssh','python','root'} or not isinstance(config['ssh'],list)
                or len(config['ssh'])<2 or config['ssh'][0]!='ssh'
                or not all(isinstance(arg,str) and arg and '\x00' not in arg for arg in config['ssh'])
                or not all(isinstance(config[k],str) and Path(config[k]).is_absolute() for k in ('python','root'))):
            raise ValueError('invalid remote launch configuration')
    if value['hosts']['g14']['ssh']==value['hosts']['orin']['ssh']:
        raise ValueError('remote hosts require distinct SSH targets')
    return value


def owned_process_matches(argv, executable, plan):
    return (executable in argv and '--effusion-host-plan' in argv
            and argv.index('--effusion-host-plan')+1<len(argv)
            and argv[argv.index('--effusion-host-plan')+1]==plan)


def compare_heads(a,b):
    with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
        if set(left.files)!=set(right.files) or set(left.files)!={'head.weight','head.bias'}:
            raise ValueError('head keys differ')
        differences=[]
        for key in left.files:
            x,y=left[key],right[key]
            if x.shape!=y.shape or x.dtype!=np.float32 or y.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('head layout or finite-value contract differs')
            differences.append(float(np.abs(x-y).max()))
        return max(differences)


def summarize_cell(directory,cell,*,elapsed_s):
    directory=Path(directory)
    server=json.loads((directory/'server/summary.json').read_text())
    rounds=json.loads((directory/'server/rounds.json').read_text())
    receipts=json.loads((directory/'server/receipts.json').read_text())
    expected=list(range(1,cell['rounds']+1))
    ids=[f"{cell['name']}:client-{cid:05d}" for cid in range(cell['num_clients'])]
    if (server['status']!='complete' or server['completed_rounds']!=cell['rounds']
            or [r['round'] for r in rounds]!=expected or [r['round'] for r in receipts]!=expected
            or any(r['client_ids']!=ids or r['client_count']!=cell['num_clients'] for r in receipts)):
        raise ValueError('server did not complete the exact declared cohort')
    compare_heads(directory/'server/head.npz',directory/'server/head.npz')
    from effusion_server import head_sha256
    with np.load(directory/'server/head.npz',allow_pickle=False) as saved:
        if saved['head.weight'].shape!=(2,cell['width']) or saved['head.bias'].shape!=(2,):
            raise ValueError('saved head has the wrong declared shape')
        pin=head_sha256({key:torch.from_numpy(saved[key]) for key in saved.files})
    if server['head_sha256']!=pin or receipts[-1]['head_sha256']!=pin:
        raise ValueError('saved head hash differs from the final aggregation receipt')
    hosts={}
    for host,cids in assignments(cell['num_clients'],cell['placement']).items():
        if not cids:continue
        row=json.loads((directory/host/'summary.json').read_text())
        if (row['status']!='complete' or row['run_id']!=cell['name'] or row['host_label']!=host
                or row['client_ids']!=cids or row['completed_client_rounds']!=len(cids)*cell['rounds']
                or row['planned_client_rounds']!=len(cids)*cell['rounds']
                or row['duplicate_fit_calls']!=0 or not 1<=row['peak_active_fits']<=2):
            raise ValueError('host did not acknowledge every assigned client round exactly once')
        events=row['events']
        hosts[host]=dict(peak_active_fits=row['peak_active_fits'],peak_process_rss_bytes=row['peak_process_rss_bytes'],
            architecture=row['runtime']['architecture'],
            system_observations=row.get('system_observations',{}),
            training_rpc_protobuf_bytes=sum(e.get('protobuf_bytes',0) for e in events),
            fit_elapsed_sum_s=sum(e.get('fit_elapsed_s',0) for e in events if e['event']=='fit'),
            queue_wait_sum_s=sum(e.get('queue_s',0) for e in events if e['event']=='fit'))
    times=[r['round_completed_elapsed_s'] for r in rounds]
    if not np.isfinite(times).all() or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('round completion timers are not monotonic')
    return dict(**{k:cell[k] for k in ('name','num_clients','rounds','placement','width','method','seed','stage')},
        status='complete',completed_client_rounds=cell['num_clients']*cell['rounds'],
        elapsed_s=elapsed_s,first_round_since_strategy_init_s=times[0],
        steady_round_mean_s=float(np.mean(np.diff(times))),selection_auc=rounds[-1]['auc'],hosts=hosts)


def _ssh(config,args,**kwargs):
    return subprocess.run([*config['ssh'],shlex.join([str(a) for a in args])],check=True,**kwargs)


_PROBE = '''import json,platform,torch,numpy,grpc,safetensors
torch.set_num_threads(1)
print(json.dumps(dict(python=platform.python_version(),torch=str(torch.__version__),numpy=numpy.__version__,grpc=grpc.__version__,safetensors=safetensors.__version__,system=platform.system(),architecture=platform.machine(),torch_threads=torch.get_num_threads())))
'''


def probe_launch(config):
    local=subprocess.run([config['local_python'],'-c',_PROBE],env=dict(os.environ,**ENV),
                         capture_output=True,text=True,check=True,timeout=30)
    result={'mac':json.loads(local.stdout)}
    for host,cfg in config['hosts'].items():
        result[host]=json.loads(_ssh(cfg,['env',*[f'{k}={v}' for k,v in ENV.items()],cfg['python'],'-c',_PROBE],
                                   capture_output=True,text=True,timeout=30).stdout)
    return result


_EXTRACT = '''import hashlib,io,pathlib,sys,tarfile
raw=sys.stdin.buffer.read()
if hashlib.sha256(raw).hexdigest()!=sys.argv[2]:raise ValueError("archive hash mismatch")
target=pathlib.Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
 for item in archive.getmembers():
  if not item.isfile() or not (target/item.name).resolve().is_relative_to(target.resolve()):raise ValueError("unsafe archive member")
 archive.extractall(target,filter="data")
print("staged")
'''


def _archive(paths):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for source,name in paths:
            if Path(source).is_symlink() or not Path(source).is_file():raise ValueError('only regular files may be staged')
            archive.add(source,arcname=name,recursive=False)
    return stream.getvalue()


def _stage(config,target,paths):
    raw=_archive(paths)
    _ssh(config,[config['python'],'-c',_EXTRACT,target,hashlib.sha256(raw).hexdigest()],
         input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)


def _fetch(config,path):
    return _ssh(config,[config['python'],'-c','import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',path],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25).stdout


def _stop_remote(config,output,executable,plan):
    # Check the exact run and executable before signalling an owned Linux PID.
    script='''import json,os,pathlib,signal,sys,select
p=pathlib.Path(sys.argv[1])/"process.json"
if not p.exists():raise SystemExit(0)
pid=json.loads(p.read_text())["pid"]
c=pathlib.Path(f"/proc/{pid}/cmdline")
if not c.exists():raise SystemExit(0)
try:fd=os.pidfd_open(pid)
except ProcessLookupError:raise SystemExit(0)
a=c.read_bytes().decode().rstrip("\\0").split("\\0")
if sys.argv[2] not in a or "--effusion-host-plan" not in a or a[a.index("--effusion-host-plan")+1]!=sys.argv[3]:raise ValueError("process ownership mismatch")
signal.pidfd_send_signal(fd,signal.SIGTERM)
if not select.select([fd],[],[],8)[0]:
 signal.pidfd_send_signal(fd,signal.SIGKILL)
 if not select.select([fd],[],[],3)[0]:raise RuntimeError("owned process did not terminate")
os.close(fd)
'''
    _ssh(config,[config['python'],'-c',script,output,executable,plan],
         stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)


def cleanup_remote_jobs(jobs,config,output):
    errors=[]
    for host,remote,executable,plan,_ in jobs:
        try:_stop_remote(config['hosts'][host],str(remote/'output'),str(executable),str(plan))
        except Exception as exc:errors.append(f'{host}: {type(exc).__name__}: {exc}')
        for name in ('summary.json','events.jsonl','process.json'):
            try:
                (output/host).mkdir(exist_ok=True)
                (output/host/name).write_bytes(_fetch(config['hosts'][host],str(remote/'output'/name)))
            except Exception:pass
    return errors


def _environment(source):
    return dict(os.environ,**ENV,PYTHONPATH=f'{source}/framework/src:{source}/fl-runtime')


def _ready(process,port,timeout=30):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if process.poll() is not None:raise RuntimeError('server exited before readiness')
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=.2):return
        except OSError:time.sleep(.05)
    raise TimeoutError('server did not become ready')


def run_cell(prepared,cell,output,config,remote_sources):
    """One attempt; preserve all logs and partial files and never retry a failed fit."""
    prepared,output=Path(prepared),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source=prepared/'sources'
    processes=[];remote_jobs=[]
    started=time.perf_counter()
    try:
        with ExitStack() as stack:
            log=stack.enter_context((output/'server.log').open('w'))
            server=subprocess.Popen([config['local_python'],str(source/'fl-runtime/effusion_server.py'),
                '--server-plan',str(prepared/cell['server_plan_path']),
                '--server-plan-sha256',cell['server_plan_sha256'],'--output',str(output/'server')],
                env=_environment(source),stdout=log,stderr=subprocess.STDOUT)
            processes.append(server)
            port=json.loads((prepared/'plan.json').read_text())['port']
            _ready(server,port)
            for host,ref in cell['hosts'].items():
                log=stack.enter_context((output/f'{host}.log').open('w'))
                if host=='mac':
                    executable=source/'fl-runtime/client.py';plan=prepared/ref['manifest_path']
                    command=[config['local_python'],str(executable),'--effusion-host-plan',str(plan),
                             '--host-plan-sha256',ref['manifest_sha256'],'--server-address',f'127.0.0.1:{port}',
                             '--output',str(output/host)]
                    process=subprocess.Popen(command,env=_environment(source),stdout=log,stderr=subprocess.STDOUT)
                else:
                    cfg=config['hosts'][host];remote=remote_sources[host].parent/'cells'/cell['name']
                    local=(prepared/ref['manifest_path']).parent
                    _stage(cfg,str(remote/'data'),[(p,p.name) for p in sorted(local.iterdir())])
                    executable=remote_sources[host]/'fl-runtime/client.py';plan=remote/'data/host.json'
                    command=['env',*[f'{k}={v}' for k,v in ENV.items()],
                             f'PYTHONPATH={remote_sources[host]}/framework/src:{remote_sources[host]}/fl-runtime',
                             'timeout','--signal=TERM',str(cell['rounds']*600+120),cfg['python'],str(executable),
                             '--effusion-host-plan',str(plan),'--host-plan-sha256',ref['manifest_sha256'],
                             '--server-address',f'127.0.0.1:{port}','--output',str(remote/'output')]
                    process=subprocess.Popen([*cfg['ssh'],shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
                    remote_jobs.append((host,remote,executable,plan,process))
                processes.append(process)
            deadline=time.monotonic()+cell['rounds']*600+120
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('a training process failed')
                if time.monotonic()>deadline:raise TimeoutError('cell exceeded its declared timeout budget')
                time.sleep(.2)
            if any(p.returncode!=0 for p in processes):raise RuntimeError('a training process failed')
            for host,remote,_,_,_ in remote_jobs:
                (output/host).mkdir()
                for name in ('summary.json','events.jsonl','process.json'):
                    (output/host/name).write_bytes(_fetch(config['hosts'][host],str(remote/'output'/name)))
        result=summarize_cell(output,cell,elapsed_s=time.perf_counter()-started)
        _json(output/'result.json',result)
        return result
    except BaseException as error:
        cleanup=cleanup_remote_jobs(remote_jobs,config,output)
        _json(output/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',cleanup_errors=cleanup))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def execute(prepared,launch,output,*,qualification=False):
    torch.set_num_threads(1)
    prepared,output=Path(prepared).absolute(),Path(output).absolute()
    plan=verify_prepared(prepared);config=load_launch_config(launch)
    public=json.loads((prepared/'public-manifest.json').read_text())
    actual_runtimes=probe_launch(config)
    if public['launch_runtimes'] is not None and public['launch_runtimes']!=actual_runtimes:
        raise ValueError('actual selected interpreter runtime changed after preflight')
    if qualification and not plan['synthetic']:raise ValueError('qualification requires generated data')
    if not qualification and plan['synthetic']:raise ValueError('medical execution cannot use a generated bundle')
    output.mkdir(parents=True,exist_ok=False)
    chosen=plan['cells'] if not qualification else [c for c in plan['cells']
        if c['stage']=='systems' and c['num_clients']==3 and c['seed']==14001]
    selected={c['name']:c for c in chosen}
    chosen=[selected[name] for name in public['execution_order'] if name in selected]
    if len(chosen)!=len(selected) or len({c['name'] for c in chosen})!=len(selected):
        raise ValueError('prospective execution order must contain every selected cell exactly once')
    ledger={c['name']:dict(status='unrun') for c in chosen}
    _json(output/'ledger.json',ledger)
    remote_sources={};tunnels=[];logs=[]
    try:
        source_paths=[(prepared/'sources'/name,name) for name in plan['sources']]
        for host,cfg in config['hosts'].items():
            remote=Path(cfg['root'])/output.name/'source'
            _stage(cfg,str(remote),source_paths)
            remote_sources[host]=remote
            # Independent private machine-id/runtime inventory identifies physical contributors.
            raw=_ssh(cfg,[cfg['python'],'-c','import json,pathlib,platform,torch; print(json.dumps(dict(machine_id=pathlib.Path("/etc/machine-id").read_text().strip(),architecture=platform.machine(),python=platform.python_version(),torch=str(torch.__version__),cuda=torch.cuda.is_available())))'],
                     capture_output=True,text=True,timeout=30).stdout
            (output/f'{host}-physical.json').write_text(raw)
            log=(output/f'{host}-tunnel.log').open('w');logs.append(log)
            tunnel=subprocess.Popen([*cfg['ssh'],'-N','-T','-o','ExitOnForwardFailure=yes',
                '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                '-R',f"127.0.0.1:{plan['port']}:127.0.0.1:{plan['port']}"],stdout=log,stderr=subprocess.STDOUT)
            tunnels.append(tunnel)
        physical=[json.loads((output/f'{h}-physical.json').read_text()) for h in ('g14','orin')]
        if physical[0]['machine_id']==physical[1]['machine_id']:raise ValueError('remote physical identities are the same')
        time.sleep(1)
        if any(p.poll() is not None for p in tunnels):raise RuntimeError('SSH forwarding failed')
        for cell in chosen:
            verify_prepared(prepared)
            ledger[cell['name']]=dict(status='running');_json(output/'ledger.json',ledger)
            print(json.dumps(dict(event='cell_started',name=cell['name'])),flush=True)
            try:
                result=run_cell(prepared,cell,output/cell['name'],config,remote_sources)
                for host in cell['hosts']:
                    runtime=json.loads((output/cell['name']/host/'summary.json').read_text())['runtime']
                    if runtime!=actual_runtimes[host]:raise ValueError('host startup runtime differs from its frozen interpreter')
                ledger[cell['name']]=result
            except BaseException as error:
                ledger[cell['name']]=dict(status='failed',error=f'{type(error).__name__}: {error}')
                _json(output/'ledger.json',ledger)
                raise
            _json(output/'ledger.json',ledger)
            print(json.dumps(dict(event='cell_completed',name=cell['name'],elapsed_s=result['elapsed_s'])),flush=True)
    finally:
        for process in tunnels:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
        for log in logs:log.close()
    return ledger


def analyze(prepared,execution,output,*,historical=None):
    plan=verify_prepared(prepared);execution=Path(execution)
    ledger=json.loads((execution/'ledger.json').read_text())
    results=[];pairs=[];replays=[]
    if not plan['synthetic']:
        historical=Path(historical)
        if _digest(historical/'summary.json')!=HISTORICAL_SUMMARY:
            raise ValueError('historical replay summary hash changed')
    for cell in plan['cells']:
        if cell['name'] not in ledger:continue
        state=ledger[cell['name']]
        if state['status']=='complete':
            result=summarize_cell(execution/cell['name'],cell,elapsed_s=state['elapsed_s'])
            server=json.loads((Path(prepared)/cell['server_plan_path']).read_text())
            from effusion_server import rank_auc
            import torch.nn.functional as F
            head=np.load(execution/cell['name']/'server/head.npz',allow_pickle=False)
            x=torch.from_numpy(np.load(server['selection']['x_path'],allow_pickle=False))
            y=np.load(server['selection']['y_path'],allow_pickle=False)
            logits=F.linear(x,torch.from_numpy(head['head.weight']),torch.from_numpy(head['head.bias']))
            auc=rank_auc(y,(logits[:,1].double()-logits[:,0].double()).numpy())
            recorded=json.loads((execution/cell['name']/'server/rounds.json').read_text())[-1]
            loss=float(F.cross_entropy(logits,torch.from_numpy(y)))
            if auc!=result['selection_auc'] or abs(loss-recorded['loss'])>1e-10:
                raise ValueError('independent selection AUROC or loss rescore differs')
            result['selection_rescore_verified']=True;results.append(result)
            if cell['stage']=='anchor' and not plan['synthetic']:
                record=historical/'runs'/f"{cell['method']}_byzfl_clean_clean_s13001.json"
                if _digest(record)!=HISTORICAL_RUNS[cell['method']]:raise ValueError('historical simulator head record hash changed')
                reference=json.loads(record.read_text())['meta']['finalHead']
                difference=max(float(np.max(np.abs(head[key]-np.asarray(reference[key],dtype=np.float32)))) for key in head.files)
                replays.append(dict(name=cell['name'],historical_record_sha256=HISTORICAL_RUNS[cell['method']],
                    max_abs_head_difference=difference,within_prospective_1e_5_tolerance=difference<=1e-5))
            if cell['placement']=='three_hosts':
                counterpart=cell['name'].removesuffix('three_hosts')+'mac_only'
                if ledger.get(counterpart,{}).get('status')=='complete':
                    difference=compare_heads(execution/counterpart/'server/head.npz',execution/cell['name']/'server/head.npz')
                    pairs.append(dict(three_host_cell=cell['name'],max_abs_head_difference=difference,
                        within_prospective_1e_5_tolerance=difference<=1e-5,
                        elapsed_speedup=ledger[counterpart]['elapsed_s']/result['elapsed_s'],
                        steady_round_speedup=ledger[counterpart]['steady_round_mean_s']/result['steady_round_mean_s']))
        else:results.append(dict(name=cell['name'],status=state['status']))
    public=dict(schema='fedlearn.multihost_analysis.v1',synthetic=plan['synthetic'],
        public_manifest_sha256=plan['public_manifest_sha256'],cells=results,placement_pairs=pairs,historical_replays=replays,
        scope=dict(device='CPU',privacy='nonprivate',attacks='none',evaluation='selection only',
                   physical_hosts=3,fit_concurrency='two per host; two total Mac-only, six total three-host',
                   transport='loopback gRPC wrapped in private SSH reverse forwarding',
                   bytes='training RPC protobuf messages; excludes registration, heartbeat, framing, TLS and SSH',
                   timing='supervisor wall time includes process startup/staging/shutdown; steady rounds use server monotonic clock'))
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    _json(output,public)
    return public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','execute','analyze'))
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--bundle',type=Path);parser.add_argument('--bundle-sha256')
    parser.add_argument('--synthetic',action='store_true');parser.add_argument('--port',type=int,default=54051)
    parser.add_argument('--launch',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--execution',type=Path);parser.add_argument('--historical',type=Path)
    parser.add_argument('--qualification',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='preflight':
        runtimes=probe_launch(load_launch_config(args.launch)) if args.launch else None
        prepare(args.bundle,args.bundle_sha256,args.prepared,port=args.port,synthetic=args.synthetic,launch_runtimes=runtimes)
    elif args.mode=='execute':execute(args.prepared,args.launch,args.output,qualification=args.qualification)
    else:analyze(args.prepared,args.execution,args.output,historical=args.historical)


if __name__=='__main__':main()
