import { Boxes, CheckCircle2, Cpu, Gauge, Network, Play, RefreshCw, Server, TerminalSquare } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api";
import type { HPCStatus, ModelRecord } from "../../types";
import { PanelEmpty } from "./PanelEmpty";

export function ModelingPanel({ sessionId, view }: { sessionId: string | null; view: string }) {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [hpc, setHpc] = useState<HPCStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedConfig, setSelectedConfig] = useState("analysis/oled_gat/configs/campaign_gat.yaml");
  const [runName, setRunName] = useState("optovlab-campaign-gat");
  const [confirmed, setConfirmed] = useState(false);
  const [job, setJob] = useState<Record<string, unknown> | null>(null);

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      const [nextModels, nextHpc] = await Promise.all([api.models(), api.hpcStatus()]);
      setModels(nextModels);
      setHpc(nextHpc);
      if (nextModels.length && !nextModels.some((model) => model.config_path === selectedConfig)) setSelectedConfig(nextModels[0].config_path);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { void refresh(); }, []);

  if (!sessionId) return <PanelEmpty icon={Network} title="No modeling session" detail="Create a chat before preparing an experiment." />;
  if (view === "models") return <ModelRegistry models={models} busy={busy} refresh={refresh} />;
  if (view === "hpc") return <HPCPanel hpc={hpc} busy={busy} refresh={refresh} />;

  async function prepare(submit: boolean) {
    setBusy(true);
    setError("");
    try {
      setJob(await api.prepareTraining(sessionId!, {
        run_name: runName,
        config_path: selectedConfig,
        partition: "rtx5880",
        gpus: 1,
        time_limit: "08:00:00",
        confirm_submit: submit
      }));
      if (submit) setConfirmed(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="training-panel">
      <div className="panel-heading"><div><small>CONTROLLED EXECUTION</small><h3>OLED-GAT training</h3></div><TerminalSquare size={20} /></div>
      <label className="control-field"><span>Run name</span><input value={runName} onChange={(event) => setRunName(event.target.value.replace(/[^A-Za-z0-9_.-]/g, "-"))} /></label>
      <label className="control-field"><span>Frozen configuration</span><select value={selectedConfig} onChange={(event) => setSelectedConfig(event.target.value)}>{models.map((model) => <option key={model.id} value={model.config_path}>{model.name} · {model.split_mode}</option>)}</select></label>
      <div className="training-resources"><div><Cpu size={17} /><span>GPU</span><b>1 x RTX 5880</b></div><div><Gauge size={17} /><span>Limit</span><b>08:00:00</b></div></div>
      <button className="secondary-action full-button" onClick={() => void prepare(false)} disabled={busy}><TerminalSquare size={16} /> Prepare script</button>
      <label className="confirm-submit"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>I reviewed the configuration and authorize one Slurm GPU job.</span></label>
      <button className="primary-action full-button" onClick={() => void prepare(true)} disabled={busy || !confirmed}><Play size={16} /> Submit to Slurm</button>
      {error && <div className="inline-error">{error}</div>}
      {job && <div className="job-result"><CheckCircle2 size={19} /><div><strong>{String(job.status)}</strong><span>{job.scheduler_job_id ? `Slurm job ${job.scheduler_job_id}` : String(job.script_path)}</span></div></div>}
    </div>
  );
}

function ModelRegistry({ models, busy, refresh }: { models: ModelRecord[]; busy: boolean; refresh: () => Promise<void> }) {
  return (
    <div className="model-registry">
      <div className="panel-heading"><div><small>VERSIONED MODELS</small><h3>OLED-GAT registry</h3></div><button className="plain-icon" onClick={() => void refresh()}><RefreshCw size={17} className={busy ? "spin" : ""} /></button></div>
      {models.map((model) => <article key={model.id}><header><Network size={19} /><div><strong>{model.name}</strong><span>{model.architecture || "Graph neural network"}</span></div></header><dl><div><dt>Split</dt><dd>{model.split_mode || "unknown"}</dd></div><div><dt>Quantiles</dt><dd>{model.quantiles.join(" / ") || "point model"}</dd></div><div><dt>Runs</dt><dd>{model.metrics.length}</dd></div></dl><code>{model.config_path}</code></article>)}
    </div>
  );
}

function HPCPanel({ hpc, busy, refresh }: { hpc: HPCStatus | null; busy: boolean; refresh: () => Promise<void> }) {
  return (
    <div className="hpc-panel">
      <div className="panel-heading"><div><small>LIVE INFRASTRUCTURE</small><h3>Slurm and GPUs</h3></div><button className="plain-icon" onClick={() => void refresh()}><RefreshCw size={17} className={busy ? "spin" : ""} /></button></div>
      {!hpc ? <PanelEmpty icon={Server} title="Loading cluster status" detail="Querying sinfo, squeue, and nvidia-smi." /> : <>
        <div className="gpu-grid">{hpc.gpus.map((gpu) => <div key={String(gpu.index)}><span>GPU {String(gpu.index)}</span><strong>{Number(gpu.memory_free_mb).toLocaleString()} MB free</strong><i><b style={{ width: `${100 - Number(gpu.utilization_percent)}%` }} /></i><small>{String(gpu.name)}</small></div>)}</div>
        <h4><Boxes size={16} /> Partitions</h4>
        <div className="data-table">{hpc.partitions.map((row, index) => <div key={index}><b>{row.partition}</b><span>{row.availability}</span><span>{row.gres}</span></div>)}</div>
        <h4><Server size={16} /> Visible jobs</h4>
        <div className="data-table">{hpc.jobs.map((row, index) => <div key={index}><b>{row.job_id}</b><span>{row.name}</span><span>{row.state}</span></div>)}</div>
      </>}
    </div>
  );
}
