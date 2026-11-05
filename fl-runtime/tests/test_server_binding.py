"""Operators can restrict the real training listener to an intended network interface."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from server_binding import grpc_bind_host


@pytest.mark.parametrize('value,expected', [
    (None, '[::]'), ('127.0.0.1', '127.0.0.1'), ('192.0.2.12', '192.0.2.12'),
    ('::1', '[::1]'), ('[::]', '[::]'), (' 127.0.0.1 ', '127.0.0.1'),
])
def test_bind_host_preserves_the_selected_ip_without_a_port(value, expected):
    assert grpc_bind_host(value) == expected


@pytest.mark.parametrize('value', ['', ' ', 'localhost', '127.0.0.1:55100', '[::1]:55100',
                                   'https://127.0.0.1', '127.0.0.1/24', '127.0.0.1;echo bad'])
def test_malformed_explicit_binding_cannot_fall_back_to_all_interfaces(value):
    with pytest.raises(ValueError, match='bind|IP'):
        grpc_bind_host(value)


@pytest.mark.parametrize('configured,expected', [(None, '[::]'), ('127.0.0.1', '127.0.0.1')])
def test_actual_runtime_import_reads_the_operator_binding(configured, expected):
    environment = os.environ.copy()
    environment.pop('FEDLEARN_GRPC_BIND_ADDRESS', None)
    if configured is not None:
        environment['FEDLEARN_GRPC_BIND_ADDRESS'] = configured
    result = subprocess.run([sys.executable, '-c',
        'import fl_server; print("BIND_TEST " + fl_server.bind_address)'],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert [line for line in result.stdout.splitlines() if line.startswith('BIND_TEST ')] == ['BIND_TEST ' + expected]
