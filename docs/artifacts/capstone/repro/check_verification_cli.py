"""Check that scientific verification refuses optimized Python execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,default=Path.cwd())
root=parser.parse_args().source_root.resolve()
rows=[]
for name in ['verify_phase2_study.py','export_phase2_tables.py','plot_phase2_study.py']:
    script=root/'docs/artifacts/capstone/repro'/name
    for mode in ['flag','environment']:
        env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None)
        command=[sys.executable]
        if mode=='flag':command.append('-O')
        else:env['PYTHONOPTIMIZE']='1'
        result=subprocess.run([*command,str(script),'--help'],env=env,capture_output=True,text=True)
        rows.append({'script':name,'mode':mode,'exit':result.returncode,'blocked_with_explanation':result.returncode!=0 and 'assertions' in result.stderr})
    env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None)
    help_result=subprocess.run([sys.executable,str(script),'--help'],env=env,capture_output=True,text=True)
    rows.append({'script':name,'mode':'normal','exit':help_result.returncode,'help_works':help_result.returncode==0 and 'usage:' in help_result.stdout})
print(json.dumps(rows,indent=2))
if not all(r.get('blocked_with_explanation',r.get('help_works',False)) for r in rows):
    raise SystemExit('optimized mode can bypass verification assertions or normal CLI help is broken')
