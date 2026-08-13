"""The FL server's arm filter must actually be reachable at runtime.

Found by a LIVE federation, not by a unit test, and that distinction is the point.

P1-2 added the arm filter to ``fl_server.main()``. It calls ``recipes.validate_arm(...)`` at what
is now line 589. ``recipes`` is imported at module scope (line 15) — but ``main()`` ALSO contains
three ``import recipes`` statements further down, in dataset branches. In Python, any binding of a
name inside a function makes that name local for the *entire* function body, so the use at line 589
referred to an unbound local and raised::

    UnboundLocalError: cannot access local variable 'recipes' where it is not associated with a value

The consequence: **a FROZEN_HEAD server could never start.** It died before binding its port, on
every recipe, every time. Three increments of arm work (P1-2 through P1-5), a full agreement test
between client and server key sets, and every gate green — because every test exercised the
``recipes`` module or ``build_eval_card`` directly, and none ever executed ``main()``.

These tests pin the property statically, since executing ``main()`` in a unit test means binding a
port and loading CIFAR-10. A static check is what this defect needed: it is a *scoping* fact, fully
visible in the AST.
"""

import ast
import os

HERE = os.path.dirname(__file__)
SERVER = os.path.join(HERE, "..", "fl_server.py")


def _main_fn():
    tree = ast.parse(open(SERVER).read())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert fns, "fl_server.main() not found"
    return fns[0]


def _module_level_imports():
    tree = ast.parse(open(SERVER).read())
    names = set()
    for node in tree.body:                      # top level only
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    return names


def test_main_does_not_shadow_a_module_level_import():
    """THE regression, stated generally rather than for `recipes` alone.

    Re-importing a module-level name inside a function is silently fine *until* something uses that
    name earlier in the same function — then it is an UnboundLocalError at runtime, on a path that
    may only execute under one configuration. Any module-level import re-imported inside main() is
    the same latent trap, so this rejects the pattern rather than the one instance of it.
    """
    module_names = _module_level_imports()
    offenders = []
    for node in ast.walk(_main_fn()):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                if bound in module_names:
                    offenders.append(f"line {node.lineno}: import {a.name}"
                                     f"{' as ' + a.asname if a.asname else ''} (bound {bound!r})")
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound = a.asname or a.name
                if bound in module_names:
                    offenders.append(f"line {node.lineno}: from {node.module} import {a.name}")

    assert not offenders, (
        "fl_server.main() re-imports a name already imported at module scope, which makes that "
        "name LOCAL to the whole function and turns every earlier use into an UnboundLocalError:\n  "
        + "\n  ".join(offenders)
        + "\nDrop the inner import — the module-level one already covers it.")


def test_recipes_is_available_at_module_scope():
    """The fix must be 'remove the inner imports', not 'move the arm filter below them'."""
    assert "recipes" in _module_level_imports(), \
        "fl_server must import recipes at module scope; main() relies on it before its own branches"


def test_the_arm_filter_precedes_the_dataset_branches():
    """Ordering matters and should stay this way: the filter must reject an impossible arm BEFORE
    the server spends time loading a dataset, so a bad arm fails fast rather than after minutes."""
    main = _main_fn()
    filter_line = next(
        (n.lineno for n in ast.walk(main)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "validate_arm"), None)
    assert filter_line, "fl_server.main() no longer validates the training arm"

    loads = [n.lineno for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "load_server_test_data"]
    assert loads, "expected server-side dataset loading in main()"
    assert filter_line < min(loads), (
        f"the arm is validated at line {filter_line}, after dataset loading begins at "
        f"{min(loads)} — an invalid arm would fail only after the data is loaded")
