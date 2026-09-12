package com.customeranalytics.api.data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small, dependency-free CSV reader. The exported files are well formed but
 * include quoted headers with commas (e.g. "Spearman (rank, holdout)" in
 * clv_summary.csv), so a naive split on ',' is wrong. This handles double-quoted
 * fields, escaped quotes ("") and CRLF/LF line endings — the same contract as the
 * hand-written parsers in ../api/data.py and ../frontend/src/data/csv.ts.
 */
public final class Csv {
    private Csv() {}

    /** Parse into rows of string cells (including the header row). */
    public static List<List<String>> parse(String text) {
        List<List<String>> rows = new ArrayList<>();
        List<String> row = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean inQuotes = false;
        int i = 0;
        int n = text.length();
        // Strip a leading UTF-8 BOM if present.
        if (n > 0 && text.charAt(0) == '﻿') {
            i = 1;
        }
        while (i < n) {
            char c = text.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < n && text.charAt(i + 1) == '"') {
                        field.append('"');
                        i += 2;
                        continue;
                    }
                    inQuotes = false;
                    i++;
                    continue;
                }
                field.append(c);
                i++;
                continue;
            }
            if (c == '"') {
                inQuotes = true;
                i++;
            } else if (c == ',') {
                row.add(field.toString());
                field.setLength(0);
                i++;
            } else if (c == '\r') {
                if (i + 1 < n && text.charAt(i + 1) == '\n') {
                    i++;
                }
                row.add(field.toString());
                field.setLength(0);
                rows.add(row);
                row = new ArrayList<>();
                i++;
            } else if (c == '\n') {
                row.add(field.toString());
                field.setLength(0);
                rows.add(row);
                row = new ArrayList<>();
                i++;
            } else {
                field.append(c);
                i++;
            }
        }
        if (field.length() > 0 || !row.isEmpty()) {
            row.add(field.toString());
            rows.add(row);
        }
        return rows;
    }

    /** Parse into a list of column-keyed maps (header -> cell). */
    public static List<Map<String, String>> parseRecords(String text) {
        List<List<String>> rows = parse(text);
        List<Map<String, String>> out = new ArrayList<>();
        if (rows.isEmpty()) {
            return out;
        }
        List<String> header = rows.get(0);
        for (int r = 1; r < rows.size(); r++) {
            List<String> cells = rows.get(r);
            if (cells.size() == 1 && cells.get(0).isEmpty()) {
                continue; // skip blank line
            }
            Map<String, String> rec = new LinkedHashMap<>();
            for (int c = 0; c < header.size(); c++) {
                rec.put(header.get(c).trim(), c < cells.size() ? cells.get(c) : "");
            }
            out.add(rec);
        }
        return out;
    }
}
