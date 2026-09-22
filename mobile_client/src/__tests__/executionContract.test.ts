// Execution contract v1: the TypeScript reader against the conformance corpus shared with the Python
// and Java readers (framework/tests/fixtures/execution_contract_v1/), plus the golden fixtures through
// the generated protobuf-es reader. The corpus pairs inputs with the exact issue set every v1 reader
// must report; the README beside it states the rules.
import { equals, toBinary, toJson } from '@bufbuild/protobuf';
import { base64Decode } from '@bufbuild/protobuf/wire';
import {
  ContractIssueCode,
  ContractIssueCodeSchema,
  ExecutionContractSchema,
} from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import {
  type ContractIssue,
  MalformedContractError,
  parseContractBinary,
  parseContractJson,
  validateContract,
} from '@/lib/executionContract';

// The app's TypeScript config carries no Node typings; Jest runs these tests under Node, so the fixture
// reader is typed locally rather than widening the app's global types.
declare const __dirname: string;
type FixtureFs = { readFileSync(path: string, encoding?: 'utf8'): string & Uint8Array };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs: FixtureFs = require('fs');
const FIXTURES = `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1`;

type Case = {
  id: string;
  json?: unknown;
  jsonText?: string;
  binaryBase64?: string;
  context?: { runId?: string; projectId?: string };
  issues: { code: string; path: string }[];
};
type Corpus = { readerProtocolVersion: number; cases: Case[] };

const corpus: Corpus = JSON.parse(fs.readFileSync(`${FIXTURES}/conformance.json`, 'utf8'));
const goldenBytes = new Uint8Array(fs.readFileSync(`${FIXTURES}/golden_tinynet_fedavg.binpb`));
const goldenJson = fs.readFileSync(`${FIXTURES}/golden_tinynet_fedavg.json`, 'utf8');

function issueName(code: ContractIssueCode): string {
  return ContractIssueCodeSchema.value[code]?.name ?? `UNKNOWN_${code}`;
}

function readCase(c: Case): ContractIssue[] {
  let contract;
  try {
    if (c.binaryBase64 !== undefined) {
      contract = parseContractBinary(base64Decode(c.binaryBase64));
    } else if (c.jsonText !== undefined) {
      contract = parseContractJson(c.jsonText);
    } else {
      contract = parseContractJson(JSON.stringify(c.json));
    }
  } catch (e) {
    if (e instanceof MalformedContractError) {
      return [{ code: ContractIssueCode.ISSUE_MALFORMED, path: '' }];
    }
    throw e;
  }
  return validateContract(contract, {
    readerProtocolVersion: corpus.readerProtocolVersion,
    expectedRunId: c.context?.runId,
    expectedProjectId: c.context?.projectId,
  });
}

describe('execution contract v1 conformance', () => {
  it.each(corpus.cases.map(c => [c.id, c] as const))('%s', (_id, c) => {
    const expected = c.issues.map(i => `${i.path} ${i.code}`).sort();
    const actual = readCase(c).map(i => `${i.path} ${issueName(i.code)}`).sort();
    expect(actual).toEqual(expected);
  });
});

describe('execution contract v1 golden fixtures', () => {
  it('decodes the binary and ProtoJSON goldens to the same contract', () => {
    expect(
      equals(ExecutionContractSchema, parseContractBinary(goldenBytes), parseContractJson(goldenJson)),
    ).toBe(true);
  });

  it('reserializes the binary golden to identical bytes', () => {
    expect(toBinary(ExecutionContractSchema, parseContractBinary(goldenBytes))).toEqual(goldenBytes);
  });

  it('renders the committed ProtoJSON document', () => {
    expect(toJson(ExecutionContractSchema, parseContractBinary(goldenBytes))).toEqual(
      JSON.parse(goldenJson),
    );
  });

  it('keeps explicit-presence zeros from both encodings', () => {
    for (const contract of [parseContractBinary(goldenBytes), parseContractJson(goldenJson)]) {
      expect(contract.workload.case).toBe('modelTraining');
      const local =
        contract.workload.case === 'modelTraining' ? contract.workload.value.localTraining : undefined;
      expect(local?.optimizer.case).toBe('sgd');
      const sgd = local?.optimizer.case === 'sgd' ? local.optimizer.value : undefined;
      expect(sgd?.momentum).toBe(0);
      expect(sgd?.nesterov).toBe(false);
      expect(local?.dropLast).toBe(false);
      expect(local?.maxLocalSteps).toBeUndefined();
      expect(contract.security?.secureAggThreshold).toBeUndefined();
      expect(contract.security?.centralDp).toBeUndefined();
    }
  });

  it('accepts the golden with no issues', () => {
    expect(validateContract(parseContractBinary(goldenBytes), { readerProtocolVersion: 2 })).toEqual([]);
  });

  it('throws on malformed bytes instead of returning a partial contract', () => {
    expect(() => parseContractBinary(new Uint8Array([0]))).toThrow(MalformedContractError);
  });
});
