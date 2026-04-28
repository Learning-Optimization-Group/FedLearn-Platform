package com.federated.fl_platform_api.exception;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.lang.NonNull;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.servlet.NoHandlerFoundException;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Centralised exception handling for all REST controllers.
 *
 * Design contract:
 *   - 4xx responses include a human-readable {@code message}.
 *   - 5xx responses replace the message with a generic line + {@code correlationId};
 *     the full stack trace is logged server-side under that ID so we never leak
 *     internal details to clients but operators can still trace incidents.
 *   - Validation failures additionally carry {@code fieldErrors}.
 */
@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    private static final String GENERIC_5XX_MESSAGE =
            "An unexpected error occurred. Please reference the correlation ID when reporting the issue.";

    // ─── Validation: @Valid on @RequestBody ─────────────────────────────────

    @Override
    protected ResponseEntity<Object> handleMethodArgumentNotValid(
            @NonNull MethodArgumentNotValidException ex,
            @NonNull HttpHeaders headers,
            @NonNull HttpStatusCode status,
            @NonNull WebRequest request) {

        Map<String, String> fieldErrors = new HashMap<>();
        ex.getBindingResult().getFieldErrors().forEach(fe ->
                fieldErrors.put(
                        fe.getField(),
                        fe.getDefaultMessage() != null ? fe.getDefaultMessage() : "Invalid value"
                )
        );

        ApiError body = ApiError.builder()
                .status(HttpStatus.BAD_REQUEST.value())
                .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                .message("Validation failed")
                .path(extractPath(request))
                .fieldErrors(fieldErrors)
                .build();

        return new ResponseEntity<>(body, headers, HttpStatus.BAD_REQUEST);
    }

    // ─── Validation: @Validated on path/query params ────────────────────────

    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ApiError> handleConstraintViolation(
            ConstraintViolationException ex, HttpServletRequest request) {

        Map<String, String> fieldErrors = new HashMap<>();
        ex.getConstraintViolations().forEach(v ->
                fieldErrors.put(v.getPropertyPath().toString(), v.getMessage())
        );

        return badRequest(request, "Constraint violation", fieldErrors);
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ApiError> handleTypeMismatch(
            MethodArgumentTypeMismatchException ex, HttpServletRequest request) {

        String required = ex.getRequiredType() != null ? ex.getRequiredType().getSimpleName() : "expected type";
        String msg = String.format("Parameter '%s' must be a valid %s", ex.getName(), required);
        return badRequest(request, msg, null);
    }

    // ─── Malformed JSON / missing body ──────────────────────────────────────

    @Override
    protected ResponseEntity<Object> handleHttpMessageNotReadable(
            @NonNull HttpMessageNotReadableException ex,
            @NonNull HttpHeaders headers,
            @NonNull HttpStatusCode status,
            @NonNull WebRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.BAD_REQUEST.value())
                .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                .message("Malformed JSON request body")
                .path(extractPath(request))
                .build();

        return new ResponseEntity<>(body, headers, HttpStatus.BAD_REQUEST);
    }

    @Override
    protected ResponseEntity<Object> handleHttpRequestMethodNotSupported(
            @NonNull HttpRequestMethodNotSupportedException ex,
            @NonNull HttpHeaders headers,
            @NonNull HttpStatusCode status,
            @NonNull WebRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.METHOD_NOT_ALLOWED.value())
                .error(HttpStatus.METHOD_NOT_ALLOWED.getReasonPhrase())
                .message(String.format("HTTP method '%s' not supported for this endpoint", ex.getMethod()))
                .path(extractPath(request))
                .build();

        return new ResponseEntity<>(body, headers, HttpStatus.METHOD_NOT_ALLOWED);
    }

    @Override
    protected ResponseEntity<Object> handleNoHandlerFoundException(
            @NonNull NoHandlerFoundException ex,
            @NonNull HttpHeaders headers,
            @NonNull HttpStatusCode status,
            @NonNull WebRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.NOT_FOUND.value())
                .error(HttpStatus.NOT_FOUND.getReasonPhrase())
                .message("No endpoint " + ex.getHttpMethod() + " " + ex.getRequestURL())
                .path(extractPath(request))
                .build();

        return new ResponseEntity<>(body, headers, HttpStatus.NOT_FOUND);
    }

    // ─── Domain exceptions ──────────────────────────────────────────────────

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ApiError> handleResourceNotFound(
            ResourceNotFoundException ex, HttpServletRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.NOT_FOUND.value())
                .error(HttpStatus.NOT_FOUND.getReasonPhrase())
                .message(ex.getMessage())
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.NOT_FOUND);
    }

    @ExceptionHandler(UserAlreadyExistsException.class)
    public ResponseEntity<ApiError> handleUserAlreadyExists(
            UserAlreadyExistsException ex, HttpServletRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.CONFLICT.value())
                .error(HttpStatus.CONFLICT.getReasonPhrase())
                .message(ex.getMessage())
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.CONFLICT);
    }

    @ExceptionHandler(ProjectStateException.class)
    public ResponseEntity<ApiError> handleProjectState(
            ProjectStateException ex, HttpServletRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.CONFLICT.value())
                .error(HttpStatus.CONFLICT.getReasonPhrase())
                .message(ex.getMessage())
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.CONFLICT);
    }

    @ExceptionHandler(ServerProcessException.class)
    public ResponseEntity<ApiError> handleServerProcess(
            ServerProcessException ex, HttpServletRequest request) {

        String correlationId = newCorrelationId();
        log.error("FL/process failure [correlationId={}]", correlationId, ex);

        ApiError body = ApiError.builder()
                .status(HttpStatus.BAD_GATEWAY.value())
                .error(HttpStatus.BAD_GATEWAY.getReasonPhrase())
                .message("Upstream FL process failed. " + GENERIC_5XX_MESSAGE)
                .path(request.getRequestURI())
                .correlationId(correlationId)
                .build();

        return new ResponseEntity<>(body, HttpStatus.BAD_GATEWAY);
    }

    // ─── Auth / authorization ───────────────────────────────────────────────

    @ExceptionHandler({BadCredentialsException.class, UsernameNotFoundException.class})
    public ResponseEntity<ApiError> handleBadCredentials(
            Exception ex, HttpServletRequest request) {
        // Both map to a generic 401 to avoid disclosing which factor failed.
        log.info("Authentication failed: {}", ex.getClass().getSimpleName());

        ApiError body = ApiError.builder()
                .status(HttpStatus.UNAUTHORIZED.value())
                .error(HttpStatus.UNAUTHORIZED.getReasonPhrase())
                .message("Invalid username or password")
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.UNAUTHORIZED);
    }

    @ExceptionHandler(AuthenticationException.class)
    public ResponseEntity<ApiError> handleAuthentication(
            AuthenticationException ex, HttpServletRequest request) {

        log.info("Authentication error: {}", ex.getClass().getSimpleName());

        ApiError body = ApiError.builder()
                .status(HttpStatus.UNAUTHORIZED.value())
                .error(HttpStatus.UNAUTHORIZED.getReasonPhrase())
                .message("Authentication required")
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.UNAUTHORIZED);
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ApiError> handleAccessDenied(
            AccessDeniedException ex, HttpServletRequest request) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.FORBIDDEN.value())
                .error(HttpStatus.FORBIDDEN.getReasonPhrase())
                .message("Access denied")
                .path(request.getRequestURI())
                .build();

        return new ResponseEntity<>(body, HttpStatus.FORBIDDEN);
    }

    // ─── Persistence / data ─────────────────────────────────────────────────

    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ApiError> handleDataIntegrity(
            DataIntegrityViolationException ex, HttpServletRequest request) {

        String correlationId = newCorrelationId();
        log.warn("Data integrity violation [correlationId={}]", correlationId, ex);

        ApiError body = ApiError.builder()
                .status(HttpStatus.CONFLICT.value())
                .error(HttpStatus.CONFLICT.getReasonPhrase())
                .message("Request conflicts with the current data state (duplicate key, foreign key, or constraint).")
                .path(request.getRequestURI())
                .correlationId(correlationId)
                .build();

        return new ResponseEntity<>(body, HttpStatus.CONFLICT);
    }

    // ─── Generic argument issues ────────────────────────────────────────────

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> handleIllegalArgument(
            IllegalArgumentException ex, HttpServletRequest request) {
        // IllegalArgumentException is almost always a caller mistake → 400.
        return badRequest(request, ex.getMessage(), null);
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<ApiError> handleIllegalState(
            IllegalStateException ex, HttpServletRequest request) {

        String correlationId = newCorrelationId();
        log.warn("Illegal state [correlationId={}]: {}", correlationId, ex.getMessage(), ex);

        ApiError body = ApiError.builder()
                .status(HttpStatus.CONFLICT.value())
                .error(HttpStatus.CONFLICT.getReasonPhrase())
                .message("Operation cannot be completed in the current state.")
                .path(request.getRequestURI())
                .correlationId(correlationId)
                .build();

        return new ResponseEntity<>(body, HttpStatus.CONFLICT);
    }

    // ─── Catch-all ──────────────────────────────────────────────────────────

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleGeneric(Exception ex, HttpServletRequest request) {
        String correlationId = newCorrelationId();
        log.error("Unhandled exception [correlationId={}]", correlationId, ex);

        ApiError body = ApiError.builder()
                .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
                .error(HttpStatus.INTERNAL_SERVER_ERROR.getReasonPhrase())
                .message(GENERIC_5XX_MESSAGE)
                .path(request.getRequestURI())
                .correlationId(correlationId)
                .build();

        return new ResponseEntity<>(body, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private ResponseEntity<ApiError> badRequest(
            HttpServletRequest request, String message, Map<String, String> fieldErrors) {

        ApiError body = ApiError.builder()
                .status(HttpStatus.BAD_REQUEST.value())
                .error(HttpStatus.BAD_REQUEST.getReasonPhrase())
                .message(message != null ? message : "Bad request")
                .path(request.getRequestURI())
                .fieldErrors(fieldErrors)
                .build();
        return new ResponseEntity<>(body, HttpStatus.BAD_REQUEST);
    }

    private static String extractPath(WebRequest request) {
        // WebRequest description is "uri=/foo"; strip the prefix for cleaner output.
        String desc = request.getDescription(false);
        return desc != null && desc.startsWith("uri=") ? desc.substring(4) : desc;
    }

    private static String newCorrelationId() {
        return UUID.randomUUID().toString();
    }
}
