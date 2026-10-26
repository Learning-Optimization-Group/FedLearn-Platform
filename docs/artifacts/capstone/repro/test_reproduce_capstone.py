"""Reproduction must reject inherited packages and use generated inputs only."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    return importlib.import_module('reproduce_capstone')


def test_unsupported_platform_fails_instead_of_installing_a_different_wheel():
    assert module().select_lock('Darwin','arm64').name=='requirements-capstone-macos-arm64.lock'
    assert module().select_lock('Linux','aarch64').name=='requirements-capstone-linux-aarch64.lock'
    with pytest.raises(ValueError):module().select_lock('Darwin','x86_64')


def test_external_site_packages_are_rejected_even_in_a_venv(tmp_path):
    prefix=tmp_path/'venv';own=prefix/'lib/python3.12/site-packages';own.mkdir(parents=True)
    external=tmp_path/'shared/lib/python3.12/site-packages';external.mkdir(parents=True)
    module().validate_environment_paths(prefix,[own],[own/'torch/__init__.py'])
    with pytest.raises(ValueError,match='outside|external|inherit'):
        module().validate_environment_paths(prefix,[own,external],[own/'torch/__init__.py'])
    with pytest.raises(ValueError,match='outside|external|inherit'):
        module().validate_environment_paths(prefix,[own],[external/'torch/__init__.py'])


def test_symlinked_package_cannot_escape_the_clean_environment(tmp_path):
    prefix=tmp_path/'venv';prefix.mkdir()
    outside=tmp_path/'shared';outside.mkdir()
    (prefix/'package').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):
        module().validate_environment_paths(prefix,[],[prefix/'package/__init__.py'])


def test_generated_bundle_is_deterministic_disjoint_and_refuses_reuse(tmp_path):
    from fedlearn.simulation.effusion_bundle import load_effusion_bundle
    left=tmp_path/'left';right=tmp_path/'right'
    pin=module().write_generated_bundle(left)
    assert module().write_generated_bundle(right)==pin
    data=load_effusion_bundle(left,expected_report_sha256=pin)
    assert data['train']['x'].shape==(1040,512) and data['evaluation']['x'].shape==(80,512)
    assert len(set(data['train']['patient_ids']))==260
    assert not set(data['train']['patient_ids']) & set(data['evaluation']['patient_ids'])
    np.testing.assert_array_equal(data['train']['y'],(data['train']['x'][:,0]>0).astype(np.int64))
    with pytest.raises((ValueError,FileExistsError)):module().write_generated_bundle(left)


def test_environment_install_refuses_existing_directory_before_running_uv(tmp_path):
    dest=tmp_path/'existing';dest.mkdir();(dest/'keep').write_text('preserve')
    with pytest.raises(FileExistsError):module().install_environment(dest,'missing-uv')
    assert (dest/'keep').read_text()=='preserve'


def test_doctor_entry_can_be_sent_over_stdin_from_a_shallow_remote_directory():
    import subprocess
    result=subprocess.run([sys.executable,'-','--help'],input=Path(module().__file__).read_text(),
                          cwd='/',capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_distinct_local_outputs_cannot_reuse_the_same_remote_execution_directory(tmp_path):
    left=module().execution_path(tmp_path/'first'/'run')
    right=module().execution_path(tmp_path/'second'/'run')
    assert left.name!=right.name
    assert left.parent==tmp_path/'first'/'run'


def test_doctor_rejects_a_locked_transitive_package_loaded_from_an_external_directory(tmp_path):
    import importlib.util
    import shutil
    import subprocess
    origin=importlib.util.find_spec('typing_extensions').origin
    shutil.copyfile(origin,tmp_path/'typing_extensions.py')
    script='''import sys
sys.path.insert(0,sys.argv[1])
import typing_extensions
sys.path.insert(0,sys.argv[2])
from reproduce_capstone import environment_report
try:environment_report()
except ValueError as error:
 print(str(error));raise SystemExit(0)
raise SystemExit(3)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path),str(Path(module().__file__).parent)],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'outside' in result.stdout or 'external' in result.stdout


def test_clean_doctor_accepts_platform_generated_standard_library_modules():
    import subprocess
    result=subprocess.run([sys.executable,str(Path(module().__file__)),'doctor'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_stdin_doctor_does_not_treat_the_remote_working_directory_as_project_source(tmp_path):
    import json
    import os
    import subprocess
    (tmp_path/'sitecustomize.py').write_text('import shared_helper\n')
    (tmp_path/'shared_helper.py').write_text('VALUE=1\n')
    expected=module().lock_versions(module().select_lock())
    env=dict(os.environ,PYTHONPATH=str(tmp_path))
    result=subprocess.run([sys.executable,'-','doctor','--expected-json',json.dumps(expected)],
        input=Path(module().__file__).read_text(),cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode!=0,result.stdout
    assert 'external' in result.stderr or 'outside the clean environment' in result.stderr


def test_doctor_rejects_external_search_roots_before_lazy_optimizer_dependencies_load(tmp_path):
    import subprocess
    script='''import sys
sys.path.insert(0,sys.argv[2])
from reproduce_capstone import environment_report
sys.path.insert(0,sys.argv[1])
try:environment_report()
except ValueError as error:
 print(error);raise SystemExit(0)
raise SystemExit(3)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path),str(Path(module().__file__).parent)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'external' in result.stdout
