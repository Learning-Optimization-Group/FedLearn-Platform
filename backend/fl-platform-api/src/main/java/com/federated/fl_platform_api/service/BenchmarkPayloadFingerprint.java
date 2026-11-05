package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.BenchmarkRoundDto;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/** Canonical accepted metric values, independent of object-key order and omitted nulls. */
final class BenchmarkPayloadFingerprint {
    private static final Set<String> NONNEGATIVE = Set.of("roundDurationMs", "evalDurationMs", "modelSizeMb",
            "paramCount", "clientCount", "samplesEvaluated", "numClasses", "support", "confusionMatrix");

    private BenchmarkPayloadFingerprint() {}

    static String of(BenchmarkRoundDto dto, ObjectMapper mapper) {
        Object payload = canonical(mapper.convertValue(dto, Object.class), "");
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update("fedlearn-benchmark-round-v1\n".getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest.digest(mapper.writeValueAsBytes(payload)));
        } catch (JsonProcessingException invalidPayload) {
            throw new IllegalArgumentException("Benchmark payload cannot be encoded", invalidPayload);
        } catch (NoSuchAlgorithmException unavailable) {
            throw new IllegalStateException("SHA-256 unavailable", unavailable);
        }
    }

    private static Object canonical(Object value, String field) {
        if (value instanceof Number number) {
            if (!Double.isFinite(number.doubleValue())) {
                throw new IllegalArgumentException("Benchmark metrics must be finite or null");
            }
            if (NONNEGATIVE.contains(field) && number.doubleValue() < 0) {
                throw new IllegalArgumentException("Benchmark counts, sizes and durations must be nonnegative");
            }
        } else if (value instanceof Map<?, ?> map) {
            var sorted = new TreeMap<String, Object>();
            for (var entry : map.entrySet()) {
                if (entry.getValue() != null) {
                    String key = (String) entry.getKey();
                    sorted.put(key, canonical(entry.getValue(), key));
                }
            }
            return sorted;
        } else if (value instanceof List<?> list) {
            var ordered = new ArrayList<>();
            for (Object element : list) ordered.add(canonical(element, field));
            return ordered;
        }
        return value;
    }
}
