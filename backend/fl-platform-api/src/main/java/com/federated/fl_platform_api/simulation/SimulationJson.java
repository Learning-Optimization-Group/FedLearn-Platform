package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.StreamReadConstraints;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;

final class SimulationJson {
    static final ObjectMapper MAPPER = new ObjectMapper(JsonFactory.builder()
            .enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
            .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(32).maxNumberLength(1000).build())
            .build());

    private SimulationJson() { }

    static JsonNode parse(byte[] bytes, int limit) throws IOException {
        if (bytes == null || bytes.length == 0 || bytes.length > limit) throw new IOException("Invalid JSON size");
        try (var parser = MAPPER.createParser(bytes)) {
            JsonNode value = MAPPER.readTree(parser);
            if (value == null || parser.nextToken() != null) throw new IOException("Trailing JSON");
            return value;
        }
    }

    static JsonNode choices(byte[] bytes) {
        try {
            JsonNode root = parse(bytes, 65536);
            if (!root.isObject() || root.size() != 1 || !root.path("choices").isObject()) throw new IOException("Invalid wrapper");
            return root.get("choices");
        } catch (IOException | RuntimeException error) {
            throw SimulationException.invalid();
        }
    }

    static JsonNode readChoices(InputStream stream) {
        try { return choices(stream.readNBytes(65537)); }
        catch (IOException error) { throw SimulationException.invalid(); }
    }

    static JsonNode stored(String value) {
        if (value == null) return MAPPER.nullNode();
        try { return MAPPER.readTree(value); }
        catch (IOException error) { throw new IllegalStateException("Invalid retained simulation JSON", error); }
    }
}
