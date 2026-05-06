package com.federated.fl_platform_api.exception;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.Instant;
import java.util.Map;

/**
 * Standard JSON error response body.
 * Field order is preserved by Jackson based on declaration order.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiError {

    private final Instant timestamp;
    private final int status;
    private final String error;
    private final String message;
    private final String path;
    private final String correlationId;
    private final Map<String, String> fieldErrors;

    private ApiError(Builder b) {
        this.timestamp = b.timestamp;
        this.status = b.status;
        this.error = b.error;
        this.message = b.message;
        this.path = b.path;
        this.correlationId = b.correlationId;
        this.fieldErrors = b.fieldErrors;
    }

    public Instant getTimestamp() { return timestamp; }
    public int getStatus() { return status; }
    public String getError() { return error; }
    public String getMessage() { return message; }
    public String getPath() { return path; }
    public String getCorrelationId() { return correlationId; }
    public Map<String, String> getFieldErrors() { return fieldErrors; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private Instant timestamp = Instant.now();
        private int status;
        private String error;
        private String message;
        private String path;
        private String correlationId;
        private Map<String, String> fieldErrors;

        public Builder status(int status) { this.status = status; return this; }
        public Builder error(String error) { this.error = error; return this; }
        public Builder message(String message) { this.message = message; return this; }
        public Builder path(String path) { this.path = path; return this; }
        public Builder correlationId(String correlationId) { this.correlationId = correlationId; return this; }
        public Builder fieldErrors(Map<String, String> fieldErrors) { this.fieldErrors = fieldErrors; return this; }

        public ApiError build() { return new ApiError(this); }
    }
}
