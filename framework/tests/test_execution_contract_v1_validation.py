"""Execution contract v1 validation: the Python reader against the shared conformance corpus.

``fixtures/execution_contract_v1/conformance.json`` pairs inputs with the exact issue set every v1
reader must report (``README.md`` beside it states the rules). The Java and TypeScript readers run
the same corpus, so a rule implemented differently in one language fails that language's suite.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os

import pytest

from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract import (ContractIssue, MalformedContractError, parse_contract_binary,
                               parse_contract_json, validate_contract)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "execution_contract_v1")

with open(os.path.join(FIXTURES, "conformance.json"), encoding="utf-8") as _fh:
    CORPUS = json.load(_fh)


def _read_case(case: dict) -> list[ContractIssue]:
    try:
        if "binaryBase64" in case:
            contract = parse_contract_binary(base64.b64decode(case["binaryBase64"]))
        elif "jsonText" in case:
            contract = parse_contract_json(case["jsonText"])
        else:
            contract = parse_contract_json(json.dumps(case["json"]))
    except MalformedContractError:
        return [ContractIssue(code=pb.ISSUE_MALFORMED, path="")]
    context = case.get("context", {})
    return validate_contract(
        contract,
        reader_protocol_version=CORPUS["readerProtocolVersion"],
        expected_run_id=context.get("runId"),
        expected_project_id=context.get("projectId"),
    )


def _rendered(issues) -> list[tuple[str, str]]:
    return sorted((issue.path, pb.ContractIssueCode.Name(issue.code)) for issue in issues)


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["id"] for c in CORPUS["cases"]])
def test_reader_reports_exactly_the_expected_issues(case):
    expected = sorted((issue["path"], issue["code"]) for issue in case["issues"])
    assert _rendered(_read_case(case)) == expected


def test_committed_corpus_is_the_generator_output():
    spec = importlib.util.spec_from_file_location(
        "execution_contract_v1_generate", os.path.join(FIXTURES, "generate.py"))
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    with open(os.path.join(FIXTURES, "conformance.json"), encoding="utf-8") as fh:
        assert generator.to_conformance_text(generator.build_conformance()) == fh.read()


def test_corpus_covers_every_issue_code():
    covered = {issue["code"] for case in CORPUS["cases"] for issue in case["issues"]}
    defined = {name for name in pb.ContractIssueCode.keys() if name != "ISSUE_UNSPECIFIED"}
    assert covered == defined


def test_malformed_bytes_raise_rather_than_returning_a_partial_contract():
    with pytest.raises(MalformedContractError):
        parse_contract_binary(b"\x00")


def test_validation_requires_the_reader_protocol_version():
    with pytest.raises(TypeError):
        validate_contract(pb.ExecutionContract())  # pylint: disable=missing-kwoa
