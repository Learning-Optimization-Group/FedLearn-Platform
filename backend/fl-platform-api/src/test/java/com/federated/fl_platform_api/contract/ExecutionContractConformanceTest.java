package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fedlearn.contract.v1.ContractIssueCode;
import com.fedlearn.contract.v1.ExecutionContract;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Execution contract v1: the Java reader against the conformance corpus shared with the Python and
 * TypeScript readers (framework/tests/fixtures/execution_contract_v1/). The corpus pairs inputs with the
 * exact issue set every v1 reader must report; the README beside it states the rules.
 */
class ExecutionContractConformanceTest {

    static final Path FIXTURES =
            Path.of("..", "..", "framework", "tests", "fixtures", "execution_contract_v1");
    private static final ObjectMapper JSON = new ObjectMapper();

    static Stream<Arguments> cases() throws IOException {
        JsonNode corpus = JSON.readTree(FIXTURES.resolve("conformance.json").toFile());
        int readerProtocolVersion = corpus.get("readerProtocolVersion").asInt();
        List<Arguments> cases = new ArrayList<>();
        for (JsonNode c : corpus.get("cases")) {
            cases.add(Arguments.of(c.get("id").asText(), c, readerProtocolVersion));
        }
        return cases.stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void readerReportsExactlyTheExpectedIssues(String id, JsonNode c, int readerProtocolVersion)
            throws IOException {
        List<String> expected = new ArrayList<>();
        for (JsonNode issue : c.get("issues")) {
            expected.add(issue.get("path").asText() + " " + issue.get("code").asText());
        }
        expected.sort(null);
        assertEquals(expected, rendered(read(c, readerProtocolVersion)), id);
    }

    private static List<ContractIssue> read(JsonNode c, int readerProtocolVersion) throws IOException {
        ExecutionContract contract;
        try {
            if (c.has("binaryBase64")) {
                contract = ExecutionContractCodec.parseBinary(
                        Base64.getDecoder().decode(c.get("binaryBase64").asText()));
            } else if (c.has("jsonText")) {
                contract = ExecutionContractCodec.parseJson(c.get("jsonText").asText());
            } else {
                contract = ExecutionContractCodec.parseJson(JSON.writeValueAsString(c.get("json")));
            }
        } catch (MalformedContractException e) {
            return List.of(new ContractIssue(ContractIssueCode.ISSUE_MALFORMED, ""));
        }
        JsonNode context = c.path("context");
        return ExecutionContractValidator.validate(contract, readerProtocolVersion,
                context.hasNonNull("runId") ? context.get("runId").asText() : null,
                context.hasNonNull("projectId") ? context.get("projectId").asText() : null);
    }

    private static List<String> rendered(List<ContractIssue> issues) {
        List<String> out = new ArrayList<>();
        for (ContractIssue issue : issues) {
            out.add(issue.path() + " " + issue.code().name());
        }
        out.sort(null);
        return out;
    }

    static byte[] goldenBytes() throws IOException {
        return Files.readAllBytes(FIXTURES.resolve("golden_tinynet_fedavg.binpb"));
    }
}
