import { useEffect, useMemo, useState, type ReactNode } from "react";
import { parseCsv } from "./csv";
import { FormattedText } from "./markdown";
import type { DashboardData, ModelSummary, ProbeAnswer, Transcript } from "./types";

function esc(s: string | null | undefined) {
  return s ?? "";
}

// Loads model output directly from ../output/api/<model>/{reasoning_traces.json,
// verbalized_responses.csv,results/verbalized_auroc.json} via the dev-server
// middleware in vite.config.ts, so the dashboard always reflects what's on
// disk right now -- including a model that finishes mid-session -- rather
// than a frozen snapshot from `npm run aggregate`.
async function loadDashboardData(): Promise<DashboardData> {
  const modelsRes = await fetch("/api-data/");
  const modelDirs: string[] = await modelsRes.json();

  const models: Record<string, Transcript[]> = {};
  const summaries: Record<string, ModelSummary> = {};

  await Promise.all(
    modelDirs.map(async (dir) => {
      const [tracesRes, csvRes, summaryRes] = await Promise.all([
        fetch(`/api-data/${dir}/reasoning_traces.json`),
        fetch(`/api-data/${dir}/verbalized_responses.csv`),
        fetch(`/api-data/${dir}/results/verbalized_auroc.json`),
      ]);

      const entries: Transcript[] = tracesRes.ok ? await tracesRes.json() : [];

      const probesById: Record<string, ProbeAnswer[]> = {};
      if (csvRes.ok) {
        const csvText = await csvRes.text();
        for (const row of parseCsv(csvText)) {
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
      models[dir] = entries;

      if (summaryRes.ok) {
        const result = await summaryRes.json();
        summaries[dir] = result.summary;
      }
    })
  );

  return { models, summaries };
}

function ModelTabs({
  modelKeys,
  summaries,
  current,
  onSelect,
}: {
  modelKeys: string[];
  summaries: Record<string, ModelSummary>;
  current: string;
  onSelect: (key: string) => void;
}) {
  return (
    <div id="model-tabs">
      {modelKeys.map((key) => {
        const s = summaries[key];
        const auroc =
          s?.ensemble_auroc === null || s?.ensemble_auroc === undefined
            ? "—"
            : s.ensemble_auroc.toFixed(3);
        return (
          <button
            key={key}
            className={"model-tab" + (key === current ? " active" : "")}
            onClick={() => onSelect(key)}
          >
            <span className="name">{s?.model ?? key}</span>
            <span className="meta">
              <span>{s?.provider ?? ""}</span>
              <span className="auroc">AUROC {auroc}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function TxList({
  rows,
  currentId,
  onSelect,
}: {
  rows: Transcript[];
  currentId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div id="tx-list">
      {rows.map((r) => (
        <div
          key={r.id}
          className={"tx-row" + (r.id === currentId ? " active" : "")}
          onClick={() => onSelect(r.id)}
        >
          <span className={"chip " + (r.ground_truth_label === 1 ? "eval" : "deploy")}>
            {r.ground_truth_label === 1 ? "eval" : "deploy"}
          </span>
          <span className="id">{r.id}</span>
          <span className="cat">{r.category}</span>
        </div>
      ))}
    </div>
  );
}

function ProbeRow({ probe, groundTruth }: { probe: Transcript["probes"][number]; groundTruth: 0 | 1 }) {
  const [open, setOpen] = useState(false);
  const sig = parseFloat(probe.eval_signal);
  let matchChip: ReactNode;
  if (probe.parse_status === "parse_failure" || sig < 0) {
    matchChip = <span className="chip warn">parse fail</span>;
  } else {
    const predicted = Math.round(sig);
    matchChip =
      predicted === groundTruth ? (
        <span className="chip ok">match</span>
      ) : (
        <span className="chip bad">mismatch</span>
      );
  }
  return (
    <div className={"probe-row" + (open ? " open" : "")}>
      <div className="ph" onClick={() => setOpen((o) => !o)}>
        <span className="pid mono">{probe.probe_id}</span>
        {matchChip}
        <span className="chev">&#9656;</span>
      </div>
      {open && (
        <div className="pb">
          {esc(probe.raw_response) ? (
            <FormattedText text={probe.raw_response} />
          ) : (
            <span className="empty-note">empty response</span>
          )}
        </div>
      )}
    </div>
  );
}

function Detail({ tx }: { tx: Transcript | undefined }) {
  const [reasoningOpen, setReasoningOpen] = useState(true);

  if (!tx) {
    return <p className="empty-note">No transcript selected.</p>;
  }

  return (
    <div id="detail">
      <div className="tx-header">
        <span className="txid">{tx.id}</span>
        <span className={"chip " + (tx.ground_truth_label === 1 ? "eval" : "deploy")}>
          {tx.ground_truth_label === 1 ? "eval" : "deploy"}
        </span>
        <span className="stat">
          {tx.source} / {tx.category}
        </span>
        <span className="stat">{tx.n_generated_tokens ?? "—"} tok</span>
        <span className="stat">{tx.elapsed_s ?? "—"}s</span>
      </div>

      <div className="block prompt">
        <div className="label">Prompt</div>
        <div className="body">
          <FormattedText text={tx.prompt_text} />
        </div>
      </div>

      <div className={"block reasoning" + (reasoningOpen ? "" : " collapsed")}>
        <div className="label" onClick={() => setReasoningOpen((o) => !o)}>
          Reasoning <span className="chev">&#9662;</span>
        </div>
        <div className="body">
          {tx.reasoning_trace ? (
            <FormattedText text={tx.reasoning_trace} />
          ) : (
            <span className="empty-note">no reasoning trace captured for this response</span>
          )}
        </div>
      </div>

      <div className="block answer">
        <div className="label">Answer</div>
        <div className="body">
          <FormattedText text={tx.final_answer} />
        </div>
      </div>

      <div id="probes">
        <h2>Verbalized eval-awareness probes ({tx.probes.length})</h2>
        {tx.probes.length === 0 ? (
          <span className="empty-note">no probe data</span>
        ) : (
          tx.probes.map((p) => (
            <ProbeRow key={p.probe_id} probe={p} groundTruth={tx.ground_truth_label} />
          ))
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData()
      .then((d) => {
        setData(d);
        const firstKey = Object.keys(d.models)[0] ?? null;
        setCurrentModel(firstKey);
        setCurrentId(firstKey ? d.models[firstKey][0]?.id ?? null : null);
      })
      .catch((err) => setError(String(err)));
  }, []);

  const modelKeys = data ? Object.keys(data.models) : [];
  const activeRows = data && currentModel ? data.models[currentModel] ?? [] : [];
  const currentTx = useMemo(
    () => activeRows.find((r) => r.id === currentId),
    [activeRows, currentId]
  );

  function selectModel(key: string) {
    setCurrentModel(key);
    const firstId = data?.models[key]?.[0]?.id ?? null;
    setCurrentId(firstId);
  }

  return (
    <div id="app">
      <header>
        <h1>reasoning&nbsp;trace&nbsp;bench</h1>
        <span className="sub">
          eval-awareness transcripts &middot; live from output/api/*
        </span>
        <span className="spacer" />
        <span className="legend">
          <span>
            <span className="dot" style={{ background: "var(--eval)" }} />
            eval (ground truth)
          </span>
          <span>
            <span className="dot" style={{ background: "var(--deploy)" }} />
            deploy (ground truth)
          </span>
        </span>
      </header>

      <nav id="rail">
        <ModelTabs
          modelKeys={modelKeys}
          summaries={data?.summaries ?? {}}
          current={currentModel ?? ""}
          onSelect={selectModel}
        />
        <TxList rows={activeRows} currentId={currentId} onSelect={setCurrentId} />
      </nav>

      <main id="main">
        {error ? (
          <p className="empty-note">Failed to load output/api data: {error}</p>
        ) : !data ? (
          <p className="empty-note">Loading transcripts from output/api…</p>
        ) : (
          <Detail tx={currentTx} />
        )}
      </main>
    </div>
  );
}
