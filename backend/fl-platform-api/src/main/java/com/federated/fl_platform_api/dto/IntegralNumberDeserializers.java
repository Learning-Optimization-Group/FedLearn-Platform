package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Integral evidence fields must not be truncated from floats or coerced from strings. */
public final class IntegralNumberDeserializers {
    private IntegralNumberDeserializers() {}

    private static void requireInteger(JsonParser parser, DeserializationContext context, Class<?> type) throws IOException {
        if (!parser.hasToken(JsonToken.VALUE_NUMBER_INT)) {
            context.reportInputMismatch(type, "Value must be a JSON integer");
        }
    }

    public static class IntegerValue extends JsonDeserializer<Integer> {
        @Override
        public Integer deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            requireInteger(parser, context, Integer.class);
            return parser.getIntValue();
        }
    }

    public static class LongValue extends JsonDeserializer<Long> {
        @Override
        public Long deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            requireInteger(parser, context, Long.class);
            return parser.getLongValue();
        }
    }

    public static class ConfusionMatrix extends JsonDeserializer<List<List<Integer>>> {
        @Override
        public List<List<Integer>> deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            JsonNode matrix = parser.getCodec().readTree(parser);
            if (!matrix.isArray()) return context.reportInputMismatch(List.class, "Confusion matrix must be an array of count arrays");
            List<List<Integer>> result = new ArrayList<>();
            for (JsonNode row : matrix) {
                if (!row.isArray()) return context.reportInputMismatch(List.class, "Confusion matrix row must be an array");
                List<Integer> counts = new ArrayList<>();
                for (JsonNode count : row) {
                    if (!count.isIntegralNumber() || !count.canConvertToInt() || count.intValue() < 0) {
                        return context.reportInputMismatch(Integer.class, "Confusion matrix counts must be nonnegative JSON integers in range");
                    }
                    counts.add(count.intValue());
                }
                result.add(counts);
            }
            return result;
        }
    }
}
