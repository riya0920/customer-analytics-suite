package com.customeranalytics.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot port of the FastAPI service in ../api. It serves the same
 * customer-analytics-suite outputs (segmentation, CLV, attribution) over the
 * same REST endpoints and JSON shapes; see java-api/README.md.
 */
@SpringBootApplication
public class ApiApplication {
    public static void main(String[] args) {
        SpringApplication.run(ApiApplication.class, args);
    }
}
