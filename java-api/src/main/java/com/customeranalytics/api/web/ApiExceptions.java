package com.customeranalytics.api.web;

/** Domain exceptions mapped to HTTP status + the uniform error body by
 * {@link GlobalExceptionHandler}. */
public final class ApiExceptions {
    private ApiExceptions() {}

    /** -> 404 not_found */
    public static class NotFoundException extends RuntimeException {
        public NotFoundException(String message) {
            super(message);
        }
    }

    /** -> 422 unprocessable_entity */
    public static class UnprocessableException extends RuntimeException {
        public UnprocessableException(String message) {
            super(message);
        }
    }
}
