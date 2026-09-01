// Aggregates every ../output/api/<model>/{reasoning_traces.json,verbalized_responses.csv,results/verbalized_auroc.json}
// into src/data/dashboard_data.json for the dashboard to import directly (no runtime fetch).
import { readdirSync, statSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const API_DIR = join(import.meta.dirname, "..", "..", "output", "api");
const OUT_PATH = join(import.meta.dirname, "..", "src", "data", "dashboard_data.json");

// Minimal RFC4180 CSV parser, scanning character-by-character so quoted fields
// containing literal newlines (the model's own raw_response text) parse correctly
// -- naively splitting on "\n" first, as most quick parsers do, would break those.
function parseCsv(text) {
  const rows = [];
  let row = [];
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
      const obj = {};
      header.forEach((h, idx) => (obj[h] = cells[idx] ?? ""));
      return obj;
    });
}

const models = {};
const summaries = {};

for (const name of readdirSync(API_DIR)) {
  const dir = join(API_DIR, name);
  if (!statSync(dir).isDirectory()) continue;

  const tracesPath = join(dir, "reasoning_traces.json");
  if (!existsSync(tracesPath)) continue;

  const entries = JSON.parse(readFileSync(tracesPath, "utf-8"));

  const csvPath = join(dir, "verbalized_responses.csv");
  const probesById = {};
  if (existsSync(csvPath)) {
    // The CSV writer (Python csv.DictWriter) may embed literal newlines inside quoted
    // fields; re-join lines that don't look like the start of a new record before parsing.
    const raw = readFileSync(csvPath, "utf-8");
    for (const row of parseCsv(raw)) {
      const tid = row.transcript_id;
      if (!tid) continue;
      (probesById[tid] ??= []).push({
        probe_id: row.probe_id,
        eval_signal: row.eval_signal,
        parse_status: row.parse_status,
        raw_response: row.raw_response,
      });
    }
  }

  for (const e of entries) {
    e.probes = probesById[e.id] ?? [];
  }
  entries.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  models[name] = entries;

  const resultsPath = join(dir, "results", "verbalized_auroc.json");
  if (existsSync(resultsPath)) {
    summaries[name] = JSON.parse(readFileSync(resultsPath, "utf-8")).summary;
  }
}

writeFileSync(OUT_PATH, JSON.stringify({ models, summaries }), "utf-8");
console.log(
  `Wrote ${OUT_PATH} (${Object.keys(models).length} models: ${Object.keys(models).join(", ")})`
);
