"""FR-9 — the framework must not hijack the host application's root logger at import time.

fedlearn/__init__ does `from . import server`, so `import fedlearn` used to run server.py's
module-level `logging.getLogger().setLevel(INFO)` + `root.handlers = [json_handler]`, silently
reconfiguring (and wiping the handlers of) any consumer's logging. Root JSON logging is now set up
explicitly by the FL-server entrypoint (start_server -> configure_logging), not at import.
"""
import importlib
import logging


def test_importing_server_module_does_not_touch_root_logger():
    root = logging.getLogger()
    orig_level, orig_handlers = root.level, list(root.handlers)
    root.setLevel(logging.WARNING)
    root.handlers = []
    try:
        import fedlearn.server.server as srv
        importlib.reload(srv)               # re-run the module body; it must not reconfigure root
        assert root.level == logging.WARNING, "importing the server module must not force root level"
        assert root.handlers == [], "importing the server module must not add/replace root handlers"
    finally:
        root.setLevel(orig_level)
        root.handlers = orig_handlers


def test_configure_logging_sets_up_json_root_logging_when_called():
    from fedlearn.server.server import configure_logging, JSONFormatter
    root = logging.getLogger()
    orig_level, orig_handlers = root.level, list(root.handlers)
    try:
        configure_logging()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
    finally:
        root.setLevel(orig_level)
        root.handlers = orig_handlers
