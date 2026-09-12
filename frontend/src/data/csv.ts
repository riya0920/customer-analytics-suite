// A small, dependency-free CSV parser. The exported files are well-formed but do
// contain quoted headers with commas (e.g. "Spearman (rank, holdout)" in
// clv_summary.csv), so a naive split(",") is wrong. This handles double-quoted
// fields, escaped quotes ("") and CRLF/LF line endings. It is deliberately tiny
// and fully unit-tested (see csv.test.ts) rather than pulling in a CSV library.

export type Cell = string;
export type Row = Cell[];

/** Parse CSV text into an array of string rows (including the header row). */
export function parseCsv(text: string): Row[] {
  const rows: Row[] = [];
  let field = "";
  let row: Row = [];
  let inQuotes = false;
  let i = 0;
  // Strip a leading UTF-8 BOM if present.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);

  const pushField = () => {
    row.push(field);
    field = "";
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
  };

  while (i < text.length) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += ch;
      i++;
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (ch === ",") {
      pushField();
      i++;
      continue;
    }
    if (ch === "\r") {
      // Handle CRLF and a lone CR.
      if (text[i + 1] === "\n") i++;
      pushRow();
      i++;
      continue;
    }
    if (ch === "\n") {
      pushRow();
      i++;
      continue;
    }
    field += ch;
    i++;
  }
  // Flush the final field/row unless the file ended on a newline (no trailing
  // empty row).
  if (field.length > 0 || row.length > 0) pushRow();
  return rows;
}

/**
 * Parse CSV text into an array of objects keyed by header, applying a per-row
 * mapper that coerces the raw string cells into the typed shape. The mapper gets
 * a `num` helper that throws on non-numeric input, so a malformed export fails
 * loudly instead of producing NaN rows.
 */
export function parseRows<T>(
  text: string,
  map: (get: (col: string) => string, num: (col: string) => number) => T,
): T[] {
  const rows = parseCsv(text);
  if (rows.length === 0) return [];
  const header = rows[0];
  const index = new Map<string, number>();
  header.forEach((h, i) => index.set(h.trim(), i));

  const out: T[] = [];
  for (let r = 1; r < rows.length; r++) {
    const cells = rows[r];
    if (cells.length === 1 && cells[0] === "") continue; // skip blank line
    const get = (col: string): string => {
      const idx = index.get(col);
      if (idx === undefined) throw new Error(`CSV missing column "${col}"`);
      return cells[idx] ?? "";
    };
    const num = (col: string): number => {
      const raw = get(col).trim();
      if (raw === "") return NaN; // missing numeric (e.g. incremental_cac blank)
      const n = Number(raw);
      if (Number.isNaN(n)) {
        throw new Error(`CSV column "${col}" is not numeric: "${raw}"`);
      }
      return n;
    };
    out.push(map(get, num));
  }
  return out;
}
