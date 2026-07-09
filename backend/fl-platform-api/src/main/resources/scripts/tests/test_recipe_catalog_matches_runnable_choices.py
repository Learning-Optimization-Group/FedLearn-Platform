# SCRIPTS/tests/test_recipe_catalog_matches_runnable_choices.py
"""The recipe catalog must equal the set of model types fl_server.py/client.py can actually
spawn (SE-10 follow-up).

`recipes.py --describe` feeds both the project-creation picker AND the backend's
`requireModelTypeInCatalog` gate (`FlowerServerManager.java`) — that gate exists specifically so a
project can never be created with a modelType the FL scripts don't support. The guarantee only
holds if the catalog and the scripts' `--model-type` argparse `choices` are the *same set*: if the
catalog ever advertises a key fl_server.py/client.py don't accept, a project sails through the
gate and fl_server.py then crashes with `SystemExit(2)` the moment it is spawned — surfacing to
the browser as a late 502. This happened for real: BLOOD_CNN was catalogued while
fl_server.py/client.py's argparse `choices` still listed only the other five model types.

fl_server.py exposes `build_arg_parser()` for exactly this kind of introspection; client.py's
parser is built inline inside `main()`, so its `--model-type` choices are read directly off the
source text instead of import-and-inspect (importing/running client.py for real requires a live
gRPC server address and immediately starts training).
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402
import fl_server  # noqa: E402

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..")


def _catalog_keys():
    return {r["key"] for r in recipes.RECIPE_METADATA}


def _fl_server_model_type_choices():
    parser = fl_server.build_arg_parser()
    for action in parser._actions:
        if action.option_strings == ["--model-type"]:
            return set(action.choices)
    raise AssertionError("fl_server.py build_arg_parser() has no --model-type argument")


def _client_model_type_choices():
    with open(os.path.join(_SCRIPTS_DIR, "client.py")) as f:
        src = f.read()
    m = re.search(r"--model-type.*?choices=(\[[^\]]+\])", src)
    assert m, "client.py --model-type argument (with choices=[...]) not found"
    return set(ast.literal_eval(m.group(1)))


def test_catalog_matches_fl_server_model_type_choices():
    catalog, choices = _catalog_keys(), _fl_server_model_type_choices()
    assert catalog == choices, (
        f"recipes.py --describe advertises {sorted(catalog)} but fl_server.py's --model-type "
        f"choices are {sorted(choices)} — a key in one but not the other passes the SE-10 "
        f"catalog gate and then crashes the spawned fl_server.py (or silently hides a runnable "
        f"model from the catalog)."
    )


def test_catalog_matches_client_model_type_choices():
    catalog, choices = _catalog_keys(), _client_model_type_choices()
    assert catalog == choices, (
        f"recipes.py --describe advertises {sorted(catalog)} but client.py's --model-type "
        f"choices are {sorted(choices)}."
    )
