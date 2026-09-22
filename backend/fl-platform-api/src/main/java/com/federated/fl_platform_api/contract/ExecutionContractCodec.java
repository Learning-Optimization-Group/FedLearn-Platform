package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fedlearn.contract.v1.ExecutionContract;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;

/**
 * Parses execution contract v1 from protobuf bytes or ProtoJSON with the generated reader. A parsed
 * contract is not yet accepted: callers validate it with {@link ExecutionContractValidator}.
 */
public final class ExecutionContractCodec {

    // Unknown fields are ignored (a compatible addition must not break an older reader); an unknown
    // enum name then reads as 0, which validation refuses.
    private static final JsonFormat.Parser JSON_PARSER = JsonFormat.parser().ignoringUnknownFields();

    // JsonFormat reads JSON leniently (a bare NaN literal is accepted), so the document is first checked
    // against strict JSON: standard literals only, a single top-level object, nothing after it.
    private static final ObjectMapper STRICT_JSON = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private ExecutionContractCodec() {
    }

    public static ExecutionContract parseBinary(byte[] data) throws MalformedContractException {
        try {
            return ExecutionContract.parseFrom(data);
        } catch (InvalidProtocolBufferException e) {
            throw new MalformedContractException("contract bytes do not parse", e);
        }
    }

    public static ExecutionContract parseJson(String json) throws MalformedContractException {
        try {
            JsonNode document = STRICT_JSON.readTree(json);
            if (document == null || !document.isObject()) {
                throw new MalformedContractException("contract ProtoJSON is not an object", null);
            }
        } catch (JsonProcessingException e) {
            throw new MalformedContractException("contract ProtoJSON is not strict JSON", e);
        }
        ExecutionContract.Builder builder = ExecutionContract.newBuilder();
        try {
            JSON_PARSER.merge(json, builder);
        } catch (InvalidProtocolBufferException | RuntimeException e) {
            throw new MalformedContractException("contract ProtoJSON does not parse", e);
        }
        return builder.build();
    }
}
