import { describe, expect, it } from "vitest";
import { parseCsv, parseRows } from "./csv";

describe("parseCsv", () => {
  it("splits simple rows", () => {
    expect(parseCsv("a,b,c\n1,2,3")).toEqual([
      ["a", "b", "c"],
      ["1", "2", "3"],
    ]);
  });

  it("handles quoted fields containing commas (as in clv_summary.csv)", () => {
    const rows = parseCsv('metric,value\n"Spearman (rank, holdout)",0.7366');
    expect(rows[1]).toEqual(["Spearman (rank, holdout)", "0.7366"]);
  });

  it("handles escaped double quotes", () => {
    expect(parseCsv('a\n"he said ""hi"""')).toEqual([["a"], ['he said "hi"']]);
  });

  it("handles CRLF line endings", () => {
    expect(parseCsv("a,b\r\n1,2\r\n")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("strips a leading UTF-8 BOM", () => {
    expect(parseCsv("﻿a,b\n1,2")[0]).toEqual(["a", "b"]);
  });

  it("does not emit a trailing empty row for a final newline", () => {
    expect(parseCsv("a\n1\n")).toEqual([["a"], ["1"]]);
  });
});

describe("parseRows", () => {
  const text = "name,score\nalice,0.9\nbob,0.8\n";

  it("maps cells by header with a numeric coercer", () => {
    const rows = parseRows(text, (g, n) => ({
      name: g("name"),
      score: n("score"),
    }));
    expect(rows).toEqual([
      { name: "alice", score: 0.9 },
      { name: "bob", score: 0.8 },
    ]);
  });

  it("throws on a missing column", () => {
    expect(() => parseRows(text, (g) => g("nope"))).toThrow(/missing column/);
  });

  it("throws when a numeric column is not numeric", () => {
    expect(() =>
      parseRows("v\nabc", (_g, n) => n("v")),
    ).toThrow(/not numeric/);
  });

  it("returns NaN for a blank numeric cell (e.g. missing incremental_cac)", () => {
    const rows = parseRows("channel,x\nretargeting,\n", (g, n) => ({
      channel: g("channel"),
      x: n("x"),
    }));
    expect(rows[0].channel).toBe("retargeting");
    expect(Number.isNaN(rows[0].x)).toBe(true);
  });
});
