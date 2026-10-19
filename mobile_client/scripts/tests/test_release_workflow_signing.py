"""Execute the release workflow's signing import and cleanup with private fixtures."""
import base64
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]


class WorkflowSigningTest(unittest.TestCase):
    def setUp(self):
        self.steps = yaml.safe_load((ROOT / '.github/workflows/release-mobile.yml').read_text())['jobs']['build-android']['steps']
        (ROOT / '.temp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='workflow-signing-', dir=ROOT / '.temp')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = dict(os.environ, RUNNER_TEMP=self.temp.name, GITHUB_ENV=str(self.directory / 'job.env'),
                        RELEASE_STORE_BASE64=base64.b64encode(b'private store fixture').decode(),
                        FEDLEARN_RELEASE_STORE_PASSWORD='private-store-password',
                        FEDLEARN_RELEASE_KEY_ALIAS='release', FEDLEARN_RELEASE_KEY_PASSWORD='private-key-password')
        self.original = self.directory / 'retain.txt'
        self.original.write_text('unrelated file')

    def step(self, name):
        found = [step for step in self.steps if step['name'] == name]
        self.assertEqual(len(found), 1, f'Missing workflow step: {name}')
        return found[0]

    def execute(self, name):
        return subprocess.run(['bash', '-c', self.step(name)['run']], env=self.env,
                              text=True, capture_output=True, timeout=10)

    def cleanup_import(self):
        job_env = self.directory / 'job.env'
        if job_env.exists():
            self.env.update(line.split('=', 1) for line in job_env.read_text().splitlines())
        result = self.execute('Remove release signing files')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.original.read_text(), 'unrelated file')

    def test_import_preserves_private_bytes_and_cleanup_is_scoped(self):
        result = self.execute('Import release signing identity')
        self.assertEqual(result.returncode, 0, result.stderr)
        job_env = dict(line.split('=', 1) for line in (self.directory / 'job.env').read_text().splitlines())
        store = Path(job_env['FEDLEARN_RELEASE_STORE_FILE'])
        self.assertEqual(store.read_bytes(), b'private store fixture')
        self.assertEqual(store.stat().st_mode & 0o777, 0o600)
        for name in ('RELEASE_STORE_BASE64', 'FEDLEARN_RELEASE_STORE_PASSWORD', 'FEDLEARN_RELEASE_KEY_PASSWORD'):
            self.assertNotIn(self.env[name], result.stdout + result.stderr)
        self.cleanup_import()
        self.assertFalse(store.exists())

    def test_missing_secret_is_refused_before_a_key_file_is_created(self):
        self.env['FEDLEARN_RELEASE_KEY_PASSWORD'] = ''
        result = self.execute('Import release signing identity')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Release signing secrets are required', result.stdout + result.stderr)
        self.assertFalse(list(self.directory.glob('fedlearn-signing-*')))
        self.cleanup_import()

    def test_invalid_base64_fails_and_partial_output_is_removed(self):
        self.env['RELEASE_STORE_BASE64'] = 'invalid!base64'
        result = self.execute('Import release signing identity')
        self.assertNotEqual(result.returncode, 0)
        self.cleanup_import()
        self.assertFalse(list(self.directory.glob('fedlearn-signing-*')))

    def test_cleanup_runs_after_failure_and_build_receives_credentials(self):
        self.assertEqual(self.step('Remove release signing files')['if'], 'always()')
        build = self.step('Assemble release APK (native FL core)')
        for name in ('FEDLEARN_RELEASE_STORE_PASSWORD', 'FEDLEARN_RELEASE_KEY_ALIAS', 'FEDLEARN_RELEASE_KEY_PASSWORD'):
            self.assertEqual(build['env'][name], '${{ secrets.' + name + ' }}')

    def test_native_include_shim_excludes_bundled_protobuf(self):
        step = self.step('Install CPU torch + prepare header-only shim')
        lines = step['run'].splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith('FULL_INC=')), None)
        self.assertIsNotNone(start, 'Missing header-only isolation before native compilation')
        include = self.directory / 'full include'
        header = include / 'torch/headeronly/macros/cmake_macros.h'
        header.parent.mkdir(parents=True)
        header.write_text('fixture header')
        polluted = include / 'google/protobuf/poison.h'
        polluted.parent.mkdir(parents=True)
        polluted.write_text('must not enter the native include path')
        tools = self.directory / 'bin'
        tools.mkdir()
        python = tools / 'python'
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_TORCH_INCLUDE"\n')
        python.chmod(0o700)
        env = dict(self.env, TEST_TORCH_INCLUDE=str(include), PATH=str(tools) + os.pathsep + os.environ['PATH'])
        # Execute the workflow's copy/isolation logic without installing torch.
        result = subprocess.run(['bash', '-eu', '-c', '\n'.join(lines[start:])], env=env,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        job_env = dict(line.split('=', 1) for line in (self.directory / 'job.env').read_text().splitlines())
        shim = Path(job_env['TORCH_INCLUDE'])
        self.assertEqual((shim / 'torch/headeronly/macros/cmake_macros.h').read_text(), 'fixture header')
        self.assertFalse((shim / 'google').exists())


if __name__ == '__main__':
    unittest.main()
