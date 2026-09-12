package com.customeranalytics.api;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

/**
 * End-to-end tests over the whole Spring context via MockMvc. The DataRepository
 * bean loads from the committed data (../frontend/public/data or ../bi_export),
 * so the suite runs on a fresh clone and in CI. Mirrors ../tests/test_api.py so
 * both ports are held to the same contract.
 */
@SpringBootTest
@AutoConfigureMockMvc
class AnalyticsApiTests {

    @Autowired
    private MockMvc mvc;

    @Test
    void healthReportsRowCounts() throws Exception {
        mvc.perform(get("/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ok"))
                .andExpect(jsonPath("$.row_counts.segments").value(5))
                .andExpect(jsonPath("$.row_counts.clv_per_customer").value(8000));
    }

    @Test
    void kpisShapeAndValues() throws Exception {
        mvc.perform(get("/api/kpis"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total_customers").value(8000))
                .andExpect(jsonPath("$.n_channels").value(12))
                .andExpect(jsonPath("$.top20pct_value_share").value(0.766));
    }

    @Test
    void segmentsTyped() throws Exception {
        mvc.perform(get("/api/segments"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(5)))
                .andExpect(jsonPath("$[0].customers").isNumber());
    }

    @Test
    void kSelectionPresent() throws Exception {
        mvc.perform(get("/api/segments/k-selection"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.k == 5)]", hasSize(1)));
    }

    @Test
    void clvEndpoints() throws Exception {
        mvc.perform(get("/api/clv/summary")).andExpect(status().isOk());
        mvc.perform(get("/api/clv/channels"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(12)));
        mvc.perform(get("/api/clv/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[*].model", hasItem("BG/NBD + Gamma-Gamma")));
    }

    @Test
    void clvCustomersPagination() throws Exception {
        mvc.perform(get("/api/clv/customers").param("limit", "10").param("offset", "0"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(8000))
                .andExpect(jsonPath("$.limit").value(10))
                .andExpect(jsonPath("$.items", hasSize(10)));
    }

    @Test
    void clvCustomersSegmentFilter() throws Exception {
        mvc.perform(get("/api/clv/customers").param("segment", "Segment 4").param("limit", "50"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total", greaterThan(0)))
                .andExpect(jsonPath("$.items[*].segment", everyItem(is("Segment 4"))));
    }

    @Test
    void clvCustomersUnknownSegment404() throws Exception {
        mvc.perform(get("/api/clv/customers").param("segment", "Segment 99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error").value("not_found"))
                .andExpect(jsonPath("$.detail", containsString("Segment 99")));
    }

    @Test
    void clvCustomersBadLimit422() throws Exception {
        mvc.perform(get("/api/clv/customers").param("limit", "0"))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.error").value("unprocessable_entity"));
        mvc.perform(get("/api/clv/customers").param("limit", "99999"))
                .andExpect(status().isUnprocessableEntity());
    }

    @Test
    void clvCustomersNonNumericLimit422() throws Exception {
        mvc.perform(get("/api/clv/customers").param("limit", "abc"))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.error").value("unprocessable_entity"));
    }

    @Test
    void attributionMethods() throws Exception {
        mvc.perform(get("/api/attribution/methods"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[*].method", hasItem("shapley")));
    }

    @Test
    void attributionCreditFilter() throws Exception {
        mvc.perform(get("/api/attribution/credit").param("method", "shapley"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(12)))
                .andExpect(jsonPath("$[*].method", everyItem(is("shapley"))));
    }

    @Test
    void attributionCreditAllIncludesTruth() throws Exception {
        mvc.perform(get("/api/attribution/credit"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[*].method", hasItem("truth")));
    }

    @Test
    void attributionCreditUnknownMethod404() throws Exception {
        mvc.perform(get("/api/attribution/credit").param("method", "nope"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error").value("not_found"))
                .andExpect(jsonPath("$.detail", containsString("nope")));
    }

    @Test
    void attributionBudget() throws Exception {
        mvc.perform(get("/api/attribution/budget"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[*].allocation", hasItems("shapley", "TRUTH")));
    }
}
