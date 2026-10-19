package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.Sgd;
import com.google.protobuf.util.JsonFormat;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The execution contract v1 golden fixtures through the generated Java reader: both encodings decode
 * to one contract, protobuf bytes round-trip unchanged, the ProtoJSON rendering equals the committed
 * document, and explicit-presence zeros survive.
 */
class ExecutionContractGoldenTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    private static String goldenJson() throws Exception {
        return Files.readString(ExecutionContractConformanceTest.FIXTURES.resolve("golden_tinynet_fedavg.json"));
    }

    @Test
    void binaryAndProtoJsonGoldensDecodeToTheSameContract() throws Exception {
        assertEquals(ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes()),
                ExecutionContractCodec.parseJson(goldenJson()));
    }

    @Test
    void binaryGoldenReserializesToIdenticalBytes() throws Exception {
        byte[] golden = ExecutionContractConformanceTest.goldenBytes();
        assertArrayEquals(golden, ExecutionContractCodec.parseBinary(golden).toByteArray());
    }

    @Test
    void protoJsonRenderingMatchesTheCommittedDocument() throws Exception {
        ExecutionContract contract = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes());
        assertEquals(JSON.readTree(goldenJson()), JSON.readTree(JsonFormat.printer().print(contract)));
    }

    @Test
    void explicitPresenceZerosSurviveBothEncodings() throws Exception {
        for (ExecutionContract contract : new ExecutionContract[] {
                ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes()),
                ExecutionContractCodec.parseJson(goldenJson())}) {
            Sgd sgd = contract.getModelTraining().getLocalTraining().getSgd();
            assertTrue(sgd.hasMomentum() && sgd.getMomentum() == 0.0);
            assertTrue(sgd.hasNesterov() && !sgd.getNesterov());
            assertTrue(contract.getModelTraining().getLocalTraining().hasDropLast());
            assertFalse(contract.getModelTraining().getLocalTraining().hasMaxLocalSteps());
            assertFalse(contract.getSecurity().hasSecureAggThreshold());
            assertFalse(contract.getSecurity().hasCentralDp());
        }
    }

    @Test
    void goldenIsValid() throws Exception {
        ExecutionContract contract = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes());
        assertEquals(java.util.List.of(), ExecutionContractValidator.validate(contract, 2, null, null));
    }

    @Test
    void malformedBytesThrowRatherThanReturningAPartialContract() {
        assertThrows(MalformedContractException.class,
                () -> ExecutionContractCodec.parseBinary(new byte[] {0}));
    }
}
