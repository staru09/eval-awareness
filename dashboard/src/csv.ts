// Minimal RFC4180 CSV parser, scanning character-by-character so quoted fields
// containing literal newlines (the model's own raw_response text) parse correctly.
export function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cur = "";
  let inQuotes = false;
  let i = 0;
  const n = text.length;
  while (i < n) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          cur += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      cur += c;
      i++;
      continue;
    }
    if (c === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (c === ",") {
      row.push(cur);
      cur = "";
      i++;
      continue;
    }
    if (c === "\r") {
      i++;
      continue;
    }
    if (c === "\n") {
      row.push(cur);
      cur = "";
      rows.push(row);
      row = [];
      i++;
      continue;
    }
    cur += c;
    i++;
  }
  if (cur.length > 0 || row.length > 0) {
    row.push(cur);
    rows.push(row);
  }
  if (rows.length === 0) return [];
  const header = rows[0];
  return rows
    .slice(1)
    .filter((r) => !(r.length === 1 && r[0] === ""))
    .map((cells) => {
      const obj: Record<string, string> = {};
      header.forEach((h, idx) => (obj[h] = cells[idx] ?? ""));
      return obj;
    });
}
