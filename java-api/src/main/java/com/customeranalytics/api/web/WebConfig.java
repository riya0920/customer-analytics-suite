package com.customeranalytics.api.web;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * CORS so the React dashboard (a different origin) can call the API from the
 * browser. Origins are configurable via CAS_CORS_ORIGINS; default is the Vite
 * dev servers, matching the FastAPI service.
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Value("${cas.cors.origins:http://localhost:5173,http://localhost:5177}")
    private String origins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins(origins.split(","))
                .allowedMethods("GET");
    }
}
