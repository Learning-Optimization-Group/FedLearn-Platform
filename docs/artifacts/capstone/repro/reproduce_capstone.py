"""Install isolated CPU dependencies and qualify canonical training on generated data."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3] if len(HERE.parents)>3 else HERE
PLATFORMS={('Darwin','arm64'):'macos-arm64',('Linux','x86_64'):'linux-x86_64',('Linux','aarch64'):'linux-aarch64'}
MODULES={'torch':'torch','numpy':'numpy','scipy':'scipy','grpcio':'grpc','protobuf':'google.protobuf',
         'safetensors':'safetensors','cryptography':'cryptography','pyjwt':'jwt','psutil':'psutil'}


def select_lock(system=None,machine=None):
    key=(system or platform.system(),machine or platform.machine())
    if key not in PLATFORMS:raise ValueError('unsupported CPU qualification platform')
    return HERE/f'requirements-capstone-{PLATFORMS[key]}.lock'


def lock_versions(path):
    return dict(re.findall(r'^([a-zA-Z0-9_-]+)==([^\s\\;]+)',Path(path).read_text(),re.M))


def validate_environment_paths(prefix,search_paths,module_paths):
    prefix=Path(prefix).resolve()
    stdlib=Path(sysconfig.get_path('stdlib')).resolve()
    stdlib_zip=stdlib.parent/f'python{sys.version_info.major}{sys.version_info.minor}.zip'
    roots=() if __file__=='<stdin>' else (HERE,ROOT/'framework/src',ROOT/'fl-runtime')
    for raw in search_paths:
        path=Path(raw).resolve()
        if any(part in ('site-packages','dist-packages') for part in path.parts) and not path.is_relative_to(prefix):
            raise ValueError('external site-packages inherited outside the clean environment')
        if path.is_relative_to(prefix) or path==stdlib_zip or path.is_relative_to(stdlib):continue
        if any(path.is_relative_to(root.resolve()) for root in roots):continue
        raise ValueError('unexpected external import search root could supply lazy dependencies')
    if any(not Path(path).resolve().is_relative_to(prefix) for path in module_paths):
        raise ValueError('package imported from outside the clean environment')


def environment_report(expected=None):
    if sys.version_info[:2]!=(3,12) or sys.prefix==sys.base_prefix:
        raise ValueError('a fresh Python 3.12 virtual environment is required')
    import site
    if site.ENABLE_USER_SITE:raise ValueError('user site packages must be disabled')
    expected=lock_versions(select_lock()) if expected is None else expected
    def normalize(name):return re.sub(r'[-_.]+','-',name).lower()
    installed={normalize(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
    expected={normalize(k):v for k,v in expected.items()}
    if installed!=expected:raise ValueError('installed distributions differ from the complete platform lock')
    origins=[]
    for name,module in MODULES.items():
        value=importlib.import_module(module)
        origins.append(value.__file__)
    validate_environment_paths(sys.prefix,sys.path,origins)
    prefix=Path(sys.prefix).resolve()
    stdlib=Path(sysconfig.get_path('stdlib')).resolve()
    owners=importlib.metadata.packages_distributions()
    project_roots=() if __file__=='<stdin>' else (HERE,ROOT/'framework/src',ROOT/'fl-runtime')
    for name,loaded in list(sys.modules.items()):
        raw=getattr(loaded,'__dict__',{}).get('__file__')
        if not raw or str(raw).startswith('<'):continue
        path=Path(raw).resolve()
        if path.is_relative_to(prefix):continue
        top=name.split('.')[0]
        if path.is_relative_to(stdlib) and not any(p in ('site-packages','dist-packages') for p in path.parts):continue
        if top not in owners and any(path.is_relative_to(root.resolve()) for root in project_roots):continue
        raise ValueError(f'loaded dependency module {name} originated outside the clean environment')
    import torch
    torch.set_num_threads(1)
    if platform.system()=='Linux' and torch.version.cuda is not None:
        raise ValueError('Linux qualification requires a CPU-only Torch wheel')
    return dict(schema='capstone.clean_environment.v1',status='passed',python=platform.python_version(),
        system=platform.system(),architecture=platform.machine(),external_site_packages=False,
        package_origins_inside_environment=True,torch_threads=torch.get_num_threads(),
        loaded_dependency_origins_checked=True,
        cuda_build=torch.version.cuda,packages=installed)


def install_environment(destination,uv='uv'):
    destination=Path(destination).absolute()
    if destination.exists():raise FileExistsError(destination)
    if sys.version_info[:2]!=(3,12):raise ValueError('run installation with Python 3.12')
    lock=select_lock()
    executable=shutil.which(uv) if not Path(uv).is_absolute() else uv
    if not executable:raise ValueError('uv 0.11.9 must be installed before setup')
    version=subprocess.run([executable,'--version'],check=True,capture_output=True,text=True).stdout
    if not version.startswith('uv 0.11.9 '):raise ValueError('setup requires uv 0.11.9')
    env=dict(os.environ,PYTHONNOUSERSITE='1',UV_CACHE_DIR=str(destination.parent/'uv-cache'))
    for key in ('PYTHONPATH','PYTHONHOME','UV_PROJECT_ENVIRONMENT'):env.pop(key,None)
    subprocess.run([executable,'venv','--no-project','--python',sys.executable,str(destination)],env=env,check=True)
    python=destination/'bin/python'
    subprocess.run([executable,'pip','sync','--python',str(python),'--require-hashes','--strict',
                    '--only-binary',':all:','--torch-backend','cpu',str(lock)],env=env,check=True)
    subprocess.run([executable,'pip','check','--python',str(python)],env=env,check=True)
    output=subprocess.run([str(python),str(Path(__file__).resolve()),'doctor'],env=env,
                          check=True,capture_output=True,text=True)
    report=json.loads(output.stdout)
    report['lock_sha256']=hashlib.sha256(lock.read_bytes()).hexdigest()
    (destination/'capstone-environment.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def write_generated_bundle(output):
    import numpy as np
    from fedlearn.simulation.effusion_bundle import write_effusion_bundle
    rng=np.random.default_rng(911)
    def split(n,prefix):
        x=rng.normal(size=(n,512)).astype(np.float32)
        return dict(x=x,y=(x[:,0]>0).astype(np.int64),patient_ids=[str(prefix+i//4) for i in range(n)],
                    image_ids=[f'generated-{prefix}-{i}' for i in range(n)])
    return write_effusion_bundle(output,split(1040,10000),split(80,20000),provenance={'synthetic':True})


def execution_path(output):
    output=Path(output).absolute()
    return output/('execution-'+hashlib.sha256(str(output).encode()).hexdigest()[:16])


def compare_simulator(prepared,execution,selected):
    import numpy as np
    import torch
    from fedlearn.simulation.engine import HeadClient
    from fedlearn.simulation.workloads import FeatureHeadWorkload
    from fedlearn.simulation.behaviors import BehaviorPlan
    from fedlearn.simulation.costs import StageClock
    from fedlearn.simulation.rng import ClientRng,torch_rng_scope
    from fedlearn.server.composed_strategy import ComposedStrategy
    from effusion_runtime import load_host_data
    from multihost_manifest import _digest
    # The real simulator creates its Linear layer before the zero/global head is applied.
    results=[]
    for cell in selected:
        if cell['placement']!='mac_only':continue
        ref=cell['hosts']['mac'];data=load_host_data(prepared/ref['manifest_path'],ref['manifest_sha256'])
        workload=FeatureHeadWorkload.__new__(FeatureHeadWorkload)
        workload.head={'type':'linear'};workload.feat_dim=cell['width'];workload.n_classes=2
        workload.train_x,workload.train_y=data.x,data.y;workload.trigger_train_x=None
        workload.parts=[np.arange(member['start'],member['stop']) for member in data.manifest['clients']]
        config={**dict(data.settings),'algorithm':'fedavg','proximal_mu':0.0}
        clients=[HeadClient(workload,cid,BehaviorPlan([],3,1,2),config,1,{'measured':{cid:0},'analytic':{}},StageClock()) for cid in range(3)]
        parameters=OrderedDict([('head.weight',torch.zeros((2,cell['width']))),('head.bias',torch.zeros(2))])
        strategy=ComposedStrategy(parameters,rule='mean',clip_norm=.05,min_fit_clients=3,clients_per_round=3)
        for current in range(1,6):
            updates=[]
            for cid,client in enumerate(clients):
                with torch_rng_scope(ClientRng(cell['seed'],cid).torch_seed(current)):
                    head,count=client.fit(parameters,{})
                updates.append((cid,head,count))
            parameters=strategy.aggregate_fit(current,updates)
        for candidate in selected:
            if candidate['method']!=cell['method']:continue
            with np.load(execution/candidate['name']/'server/head.npz',allow_pickle=False) as head:
                difference=max(float(np.abs(head[key]-value.numpy()).max()) for key,value in parameters.items())
            if not np.isfinite(difference) or difference>1e-5:raise ValueError('transport differs from the simulator reference')
            results.append(dict(name=candidate['name'],max_abs_head_difference=difference,tolerance=1e-5))
    return results


def qualify(output,launch=None):
    import torch
    torch.set_num_threads(1)
    from multihost_manifest import prepare,verify_prepared,_json,_digest
    from run_multihost_benchmark import load_launch_config,probe_launch,execute,run_cell,analyze,_ssh
    from audit_multihost_results import audit_cell,audit
    environment=environment_report()
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    pinned=[Path(__file__),HERE/'audit_multihost_results.py',HERE/'requirements-capstone-cpu.in',
            *sorted(HERE.glob('requirements-capstone-*.lock'))]
    sources={p.name:_digest(p) for p in pinned}
    _json(output/'environment.json',environment)
    pin=write_generated_bundle(output/'bundle')
    config=load_launch_config(launch) if launch else dict(local_python=sys.executable,hosts={})
    remote_environments={}
    if launch:
        code=Path(__file__).read_bytes()
        for host,cfg in config['hosts'].items():
            # Run the identical doctor from stdin: no installation or remote source mutation.
            inventory=_ssh(cfg,[cfg['python'],'-c','import platform;print(platform.system()+" "+platform.machine())'],capture_output=True,text=True,timeout=20).stdout.split()
            expected=lock_versions(select_lock(*inventory))
            result=_ssh(cfg,[cfg['python'],'-','doctor','--expected-json',json.dumps(expected)],input=code,
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            remote_environments[host]=json.loads(result.stdout)
        # Verify the actual local interpreter in launch configuration as well.
        local=subprocess.run([config['local_python'],str(Path(__file__).resolve()),'doctor'],capture_output=True,text=True,check=True)
        if json.loads(local.stdout)!=environment:raise ValueError('launch local interpreter differs from the qualified environment')
    runtimes=probe_launch(config)
    prepared=output/'prepared';prepare(output/'bundle',pin,prepared,synthetic=True,launch_runtimes=runtimes if launch else None)
    plan=verify_prepared(prepared)
    selected=[c for c in plan['cells'] if c['stage']=='systems' and c['num_clients']==3 and c['seed']==14001 and (launch or c['placement']=='mac_only')]
    execution=execution_path(output)
    if launch:execute(prepared,launch,execution,qualification=True)
    else:
        execution.mkdir()
        ledger={c['name']:dict(status='unrun') for c in selected};_json(execution/'ledger.json',ledger)
        for cell in selected:
            try:ledger[cell['name']]=run_cell(prepared,cell,execution/cell['name'],config,{})
            except BaseException:
                ledger[cell['name']]=dict(status='failed');_json(execution/'ledger.json',ledger);raise
            _json(execution/'ledger.json',ledger)
    comparisons=compare_simulator(prepared,execution,selected)
    analyze(prepared,execution,output/'analysis.json')
    if launch:
        audited=audit(prepared,execution,output/'analysis.json',output/'audit.json',checkpoint='clean-generated-qualification',qualification=True)
        if audited['status']!='passed':raise ValueError('raw journal audit failed')
    else:
        for cell in selected:audit_cell(prepared,execution,cell)
    if sources!={p.name:_digest(p) for p in pinned}:
        raise ValueError('reproduction source changed during qualification')
    receipt=dict(schema='capstone.clean_reproduction.v1',status='passed',synthetic=True,
        cells=len(selected),rounds=5*len(selected),client_rounds=15*len(selected),
        environment=environment,remote_environments=remote_environments,source_sha256=sources,
        generated_bundle_report_sha256=pin,simulator_comparisons=comparisons,
        execution_directory=execution.name,
        public_manifest_sha256=plan['public_manifest_sha256'],scope='Generated CPU qualification only; no medical fitting')
    _json(output/'receipt.json',receipt)
    return receipt


def main():
    if __file__=='<stdin>':sys.path[:]=[entry for entry in sys.path if entry]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('environment','doctor','qualify'))
    p.add_argument('--output',type=Path);p.add_argument('--uv',default='uv');p.add_argument('--launch',type=Path)
    p.add_argument('--expected-json',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.mode!='doctor' and args.output is None:p.error('--output is required for this mode')
    if args.mode=='environment':result=install_environment(args.output,args.uv)
    elif args.mode=='doctor':result=environment_report(json.loads(args.expected_json) if args.expected_json else None)
    else:
        for directory in (ROOT/'framework/src',ROOT/'fl-runtime',HERE):sys.path.insert(0,str(directory))
        result=qualify(args.output,args.launch)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
