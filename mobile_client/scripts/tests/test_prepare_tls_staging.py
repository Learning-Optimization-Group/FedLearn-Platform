"""Exercise staging isolation, private outputs and real certificate verification."""
import json
import os
from pathlib import Path
import shlex
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'prepare_tls_staging.py'
ROOT = SCRIPT.parents[2]


class TlsStagingTest(unittest.TestCase):
    def setUp(self):
        (ROOT / '.temp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='tls-staging-test-', dir=ROOT / '.temp')
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'new stage'

    def run_prepare(self, output=None):
        return subprocess.run(
            [os.sys.executable, str(SCRIPT), '--out', str(output or self.output)],
            text=True, capture_output=True, timeout=30,
        )

    def test_preparation_creates_private_distinct_keys_without_claiming_release(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / 'preparation.json').read_text())
        self.assertFalse(report['physical_release_training_qualified'])
        self.assertFalse(report['production_signing_identity_created'])
        self.assertEqual(report['https_trust'], 'self-signed; host preparation only')
        self.assertNotEqual(report['certificates']['api']['sha256'], report['certificates']['grpc']['sha256'])
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        for name in ('api.key', 'grpc.key', 'backend.env'):
            self.assertEqual((self.output / name).stat().st_mode & 0o777, 0o600)
        # Source the generated environment, including paths containing spaces.
        probe = subprocess.run(['bash', '-c', 'set -e; source "$1"; test "$APP_FL_REQUIRE_TLS" = true; '
                               'test "$APP_FL_REQUIRE_CLIENT_AUTH" = true; '
                               'test "$SPRING_PROFILES_ACTIVE" = dev; '
                               'test -f "$FEDLEARN_GRPC_SERVER_KEY"', 'bash', str(self.output / 'backend.env')])
        self.assertEqual(probe.returncode, 0)
        self.assertFalse(report['grpc_listener_is_loopback_isolated'])
        for line in (self.output / 'backend.env').read_text().splitlines():
            if any(name in line for name in ('APP_JWT_SECRET=', 'APP_FL_TOKEN_SECRET=', 'APP_INTERNAL_API_KEY=', 'SPRING_DATASOURCE_PASSWORD=')):
                secret = shlex.split(line.split('=', 1)[1])[0]
                self.assertNotIn(secret, result.stdout + result.stderr + json.dumps(report))
        self.assertFalse((self.output / 'android-release.p12').exists())

    def test_existing_directory_is_refused_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / 'retain.txt'
        marker.write_text('historical evidence')
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)
        self.assertEqual(marker.read_text(), 'historical evidence')
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_output_outside_checkout_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'outside'
            result = self.run_prepare(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('inside the checkout', result.stderr)
            self.assertFalse(output.exists())

    def test_private_environment_cannot_be_written_into_trackable_source(self):
        with tempfile.TemporaryDirectory(prefix='tls-output-test-', dir=ROOT / 'mobile_client') as directory:
            output = Path(directory) / 'unsafe'
            result = self.run_prepare(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ignored .temp', result.stderr)
            self.assertFalse(output.exists())

    def test_generated_certificate_accepts_exact_names_and_refuses_wrong_trust(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        listener.listen()
        listener.settimeout(.1)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(self.output / 'api.crt', self.output / 'api.key')
        stopped = threading.Event()

        def serve():
            while not stopped.is_set():
                try:
                    raw, _ = listener.accept()
                except TimeoutError:
                    continue
                with raw:
                    raw.settimeout(2)
                    try:
                        with server_context.wrap_socket(raw, server_side=True):
                            pass
                    except ssl.SSLError:
                        pass  # Client refusal is expected for the negative controls.

        server = threading.Thread(target=serve)
        server.start()

        def stop_server():
            stopped.set()
            server.join(timeout=3)
            listener.close()
            self.assertFalse(server.is_alive())

        self.addCleanup(stop_server)
        for name in ('localhost', '127.0.0.1'):
            trusted = ssl.create_default_context(cafile=str(self.output / 'api.crt'))
            with socket.create_connection(('127.0.0.1', port), timeout=2) as raw:
                with trusted.wrap_socket(raw, server_hostname=name) as connection:
                    self.assertTrue(connection.getpeercert())
        for cafile, hostname in ((self.output / 'api.crt', 'wrong.invalid'), (self.output / 'grpc.crt', 'localhost'), (None, 'localhost')):
            context = ssl.create_default_context(cafile=str(cafile) if cafile else None)
            with socket.create_connection(('127.0.0.1', port), timeout=2) as raw:
                with self.assertRaises(ssl.SSLCertVerificationError):
                    context.wrap_socket(raw, server_hostname=hostname)

if __name__ == '__main__':
    unittest.main()
