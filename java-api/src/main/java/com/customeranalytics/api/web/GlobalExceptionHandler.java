package com.customeranalytics.api.web;

import com.customeranalytics.api.model.Models.ErrorResponse;
import com.customeranalytics.api.web.ApiExceptions.NotFoundException;
import com.customeranalytics.api.web.ApiExceptions.UnprocessableException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Renders every error as the same {@code {"error", "detail"}} body the FastAPI
 * service returns, so both APIs are drop-in compatible for the front end.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(NotFoundException.class)
    public ResponseEntity<ErrorResponse> notFound(NotFoundException ex) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ErrorResponse("not_found", ex.getMessage()));
    }

    @ExceptionHandler(UnprocessableException.class)
    public ResponseEntity<ErrorResponse> unprocessable(UnprocessableException ex) {
        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(new ErrorResponse("unprocessable_entity", ex.getMessage()));
    }

    /** A non-numeric query param (e.g. limit=abc) is a client validation error;
     * match FastAPI by returning 422 rather than Spring's default 400. */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ErrorResponse> typeMismatch(MethodArgumentTypeMismatchException ex) {
        String detail = ex.getName() + ": expected a valid "
                + (ex.getRequiredType() != null ? ex.getRequiredType().getSimpleName() : "value");
        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(new ErrorResponse("unprocessable_entity", detail));
    }
}
