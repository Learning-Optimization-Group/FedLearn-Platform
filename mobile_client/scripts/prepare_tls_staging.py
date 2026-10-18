#!/usr/bin/env python3
"""Prepare private loopback TLS material without starting services or changing a phone."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import ssl
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def private_text(path, text):
    with open(path, 'x', opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(text)


def prepare(output, python):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must remain inside the checkout.')
    if not output.is_relative_to((ROOT / '.temp').resolve()):
        raise ValueError('Private outputs must remain under the checkout\'s ignored .temp directory.')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory to preserve earlier evidence.')
    if any(ord(char) < 32 or char in '$;{}' for char in str(output)):
        raise ValueError('Output path contains unsupported nginx configuration characters.')
    if not output.parent.is_dir():
        raise ValueError('Output parent must already exist.')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Python must name an existing executable.')
    openssl = shutil.which('openssl')
    if not openssl:
        raise ValueError('OpenSSL is required.')
    output.mkdir(mode=0o700)
    certificates = {}
    for name in ('api', 'grpc'):
        config = output / f'{name}.cnf'
        private_text(config, '''[req]
prompt = no
distinguished_name = subject
x509_extensions = server
[subject]
CN = localhost
O = FedLearn local preparation
[server]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:localhost,IP:127.0.0.1
''')
        subprocess.run([
            openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', '7',
            '-config', str(config), '-keyout', str(output / f'{name}.key'),
            '-out', str(output / f'{name}.crt'),
        ], check=True, capture_output=True, timeout=30)
        (output / f'{name}.key').chmod(0o600)
        cert = output / f'{name}.crt'
        cert.chmod(0o644)
        der = ssl.PEM_cert_to_DER_cert(cert.read_text())
        certificates[name] = {'sha256': hashlib.sha256(der).hexdigest(), 'san': ['localhost', '127.0.0.1'], 'validity_days': 7}
    database_password = secrets.token_hex(24)
    for directory in ('bundles', 'artifacts', 'blob-cache', 'client-body', 'proxy-temp', 'fastcgi-temp', 'uwsgi-temp', 'scgi-temp'):
        (output / directory).mkdir(mode=0o700)
    env = {
        'SPRING_PROFILES_ACTIVE': 'dev', 'SERVER_ADDRESS': '127.0.0.1', 'SERVER_PORT': '8086',
        'SPRING_DATASOURCE_URL': 'jdbc:postgresql://127.0.0.1:5546/fedlearn_tls',
        'SPRING_DATASOURCE_USERNAME': 'fedlearn_tls', 'SPRING_DATASOURCE_PASSWORD': database_password,
        'APP_JWT_SECRET': base64.b64encode(secrets.token_bytes(48)).decode(),
        'APP_FL_TOKEN_SECRET': base64.b64encode(secrets.token_bytes(48)).decode(),
        'APP_INTERNAL_API_KEY': secrets.token_hex(32),
        'APP_FL_REQUIRE_TLS': 'true', 'APP_FL_REQUIRE_CLIENT_AUTH': 'true',
        'FEATURE_FL_CLIENT_CERT': 'false', 'FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH': '0',
        'FEDLEARN_GRPC_SERVER_CERT': str(output / 'grpc.crt'),
        'FEDLEARN_GRPC_SERVER_KEY': str(output / 'grpc.key'),
        'FL_SERVER_GRPC_HOST': '127.0.0.1', 'FL_PORT_RANGE_START': '55100', 'FL_PORT_RANGE_END': '55103',
        'APP_BACKEND_INTERNAL_URL': 'http://127.0.0.1:8086',
        'APP_AUTH_COOKIE_SECURE': 'true', 'CORS_ALLOWED_ORIGINS': 'https://localhost:8443',
        'FEDLEARN_PYTHON': str(python), 'MODEL_BUNDLE_PYTHON': str(python),
        'APP_MODEL_BUNDLE_DIR': str(output / 'bundles'),
        'FEDLEARN_ARTIFACT_STORE_ROOT': str(output / 'artifacts'), 'APP_MODEL_BLOB_CACHE_DIR': str(output / 'blob-cache'),
        'FEDLEARN_ROUND_TIMEOUT_S': '900',
    }
    private_text(output / 'backend.env', ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in env.items()))
    private_text(output / 'postgres.env', f'POSTGRES_USER=fedlearn_tls\nPOSTGRES_DB=fedlearn_tls\nPOSTGRES_PASSWORD={database_password}\n')
    private_text(output / 'nginx.conf', f'''pid {json.dumps(str(output / 'nginx.pid'))};
error_log stderr;
events {{}}
http {{
  access_log off;
  client_body_temp_path {json.dumps(str(output / 'client-body'))};
  proxy_temp_path {json.dumps(str(output / 'proxy-temp'))};
  fastcgi_temp_path {json.dumps(str(output / 'fastcgi-temp'))};
  uwsgi_temp_path {json.dumps(str(output / 'uwsgi-temp'))};
  scgi_temp_path {json.dumps(str(output / 'scgi-temp'))};
  map $http_upgrade $connection_upgrade {{ default upgrade; '' close; }}
  server {{
    listen 127.0.0.1:8443 ssl;
    server_name localhost;
    ssl_certificate {json.dumps(str(output / 'api.crt'))};
    ssl_certificate_key {json.dumps(str(output / 'api.key'))};
    ssl_protocols TLSv1.2 TLSv1.3;
    client_max_body_size 128m;
    location / {{
      proxy_pass http://127.0.0.1:8086;
      proxy_http_version 1.1;
      proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection $connection_upgrade;
      proxy_read_timeout 120s;
    }}
  }}
}}
''')
    report = {
        'scope': 'local TLS preparation material; no services started or phone changes',
        'backend_profile': 'dev with explicit generated secrets and required TLS/token authentication',
        'grpc_listener_is_loopback_isolated': False,
        'https_trust': 'self-signed; host preparation only', 'certificates': certificates,
        'api_port': 8086, 'https_port': 8443, 'postgres_port': 5546, 'grpc_ports': [55100, 55103],
        'physical_release_training_qualified': False, 'production_signing_identity_created': False,
        'remaining': ['Android-system-trusted HTTPS endpoint', 'retained Android signing identity',
                      'verified network/firewall isolation for the wildcard gRPC listener',
                      'isolated backend/proxy/database startup and real authenticated training',
                      'physical Release execution and same-key update preserving data'],
    }
    private_text(output / 'preparation.json', json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory under this checkout\'s ignored .temp; parent must exist')
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='Python 3.12 environment with the pinned FL/export dependencies')
    args = parser.parse_args()
    try:
        prepare(args.out, args.python.absolute())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, ValueError):
            parser.exit(1, str(exc) + '\n')
        parser.exit(1, 'TLS preparation failed; any partial output remains private. Choose a fresh directory after inspection.\n')
    print('Private loopback TLS material prepared. No services started, phone changes, or production Release qualification.')


if __name__ == '__main__':
    main()
