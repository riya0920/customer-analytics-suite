package com.customeranalytics.api.web;

import com.customeranalytics.api.data.DataRepository;
import com.customeranalytics.api.model.Models.*;
import com.customeranalytics.api.web.ApiExceptions.NotFoundException;
import com.customeranalytics.api.web.ApiExceptions.UnprocessableException;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * The REST endpoints, one-for-one with ../api/main.py. Every method returns a
 * typed record (or list), so Jackson emits the same JSON shapes the FastAPI
 * service does.
 */
@RestController
public class AnalyticsController {

    private final DataRepository repo;

    public AnalyticsController(DataRepository repo) {
        this.repo = repo;
    }

    // --- meta -------------------------------------------------------------
    @GetMapping("/health")
    public Health health() {
        return new Health("ok", repo.getDataDir().toString(), repo.rowCounts());
    }

    // --- overview ---------------------------------------------------------
    @GetMapping("/api/kpis")
    public Kpi kpis() {
        return repo.getKpis();
    }

    // --- segmentation -----------------------------------------------------
    @GetMapping("/api/segments")
    public List<Segment> segments() {
        return repo.getSegments();
    }

    @GetMapping("/api/segments/k-selection")
    public List<KSelection> kSelection() {
        return repo.getKSelection();
    }

    // --- CLV --------------------------------------------------------------
    @GetMapping("/api/clv/summary")
    public List<ClvSummary> clvSummary() {
        return repo.getClvSummary();
    }

    @GetMapping("/api/clv/models")
    public List<ClvModel> clvModels() {
        return repo.getClvModels();
    }

    @GetMapping("/api/clv/channels")
    public List<ClvByChannel> clvChannels() {
        return repo.getClvByChannel();
    }

    @GetMapping("/api/clv/customers")
    public Page<ClvPerCustomer> clvCustomers(
            @RequestParam(defaultValue = "100") int limit,
            @RequestParam(defaultValue = "0") int offset,
            @RequestParam(required = false) String segment) {
        if (limit < 1 || limit > 1000) {
            throw new UnprocessableException("limit: must be between 1 and 1000");
        }
        if (offset < 0) {
            throw new UnprocessableException("offset: must be >= 0");
        }
        if (segment != null && !repo.hasSegment(segment)) {
            throw new NotFoundException("Unknown segment '" + segment + "'");
        }
        return repo.customers(limit, offset, segment);
    }

    // --- attribution ------------------------------------------------------
    @GetMapping("/api/attribution/methods")
    public List<MethodScore> attributionMethods() {
        return repo.getMethodScores();
    }

    @GetMapping("/api/attribution/credit")
    public List<AttributionCredit> attributionCredit(
            @RequestParam(required = false) String method) {
        if (method != null && !repo.hasMethod(method)) {
            List<String> known = repo.attributionMethods();
            known.add("truth");
            throw new NotFoundException(
                    "Unknown attribution method '" + method + "'. Known: "
                            + String.join(", ", known));
        }
        return repo.attributionFor(method);
    }

    @GetMapping("/api/attribution/budget")
    public List<BudgetOutcome> attributionBudget() {
        return repo.getBudgetOutcomes();
    }
}
