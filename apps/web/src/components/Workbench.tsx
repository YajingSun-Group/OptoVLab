import {
  BarChart3,
  BookOpenCheck,
  Boxes,
  Database,
  FileCheck2,
  FlaskConical,
  Gauge,
  History,
  Layers3,
  Network,
  RefreshCw,
  Wrench,
  X
} from "lucide-react";
import type { AgentProfile } from "./AgentWorkspace";
import type { SessionWorkspace } from "../types";
import { AnalysisPanel } from "./workbench/AnalysisPanel";
import { ExperimentPanel } from "./workbench/ExperimentPanel";
import { MiningReviewPanel } from "./workbench/MiningReviewPanel";
import { ModelingPanel } from "./workbench/ModelingPanel";
import { ToolsPanel } from "./workbench/ToolsPanel";
import { PanelEmpty } from "./workbench/PanelEmpty";

interface WorkbenchProps {
  profile: AgentProfile;
  sessionId: string | null;
  workspace: SessionWorkspace | null;
  activeView: string;
  onViewChange: (view: string) => void;
  onClose: () => void;
  onRefresh: () => void;
}

const navigation = {
  data_mining: [
    { id: "overview", label: "Overview", icon: Gauge },
    { id: "results", label: "Results", icon: FileCheck2 },
    { id: "analysis", label: "Analysis", icon: BarChart3 },
    { id: "tools", label: "Tools", icon: Wrench }
  ],
  device_modeling: [
    { id: "overview", label: "Overview", icon: Gauge },
    { id: "models", label: "Models", icon: Network },
    { id: "hpc", label: "HPC", icon: Boxes },
    { id: "training", label: "Training", icon: Layers3 },
    { id: "tools", label: "Tools", icon: Wrench }
  ],
  experimental_design: [
    { id: "overview", label: "Overview", icon: Gauge },
    { id: "evidence", label: "Evidence", icon: BookOpenCheck },
    { id: "tools", label: "Tools", icon: Wrench }
  ]
};

export function Workbench(props: WorkbenchProps) {
  const { profile, activeView, onViewChange, onClose, onRefresh } = props;
  const items = navigation[profile.type];
  const selected = items.some((item) => item.id === activeView) ? activeView : "overview";
  return (
    <aside className="workbench">
      <header className="workbench-header">
        <div><small>{profile.shortName}</small><strong>Workbench</strong></div>
        <div>
          <button className="plain-icon" onClick={onRefresh} title="Refresh"><RefreshCw size={17} /></button>
          <button className="plain-icon" onClick={onClose} title="Close workbench"><X size={18} /></button>
        </div>
      </header>
      <nav className="workbench-nav">
        {items.map(({ id, label, icon: Icon }) => (
          <button key={id} className={selected === id ? "active" : ""} onClick={() => onViewChange(id)} title={label}>
            <Icon size={17} /><span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="workbench-content">
        {selected === "overview" && <WorkbenchOverview {...props} />}
        {profile.type === "data_mining" && selected === "results" && (
          <MiningReviewPanel sessionId={props.sessionId} workspace={props.workspace} onRefresh={props.onRefresh} />
        )}
        {profile.type === "data_mining" && selected === "analysis" && (
          <AnalysisPanel sessionId={props.sessionId} workspace={props.workspace} />
        )}
        {profile.type === "device_modeling" && ["models", "hpc", "training"].includes(selected) && (
          <ModelingPanel sessionId={props.sessionId} view={selected} />
        )}
        {profile.type === "experimental_design" && selected === "evidence" && (
          <ExperimentPanel workspace={props.workspace} />
        )}
        {selected === "tools" && <ToolsPanel workspace={props.workspace} />}
      </div>
    </aside>
  );
}

function WorkbenchOverview({ profile, workspace, sessionId, onViewChange }: WorkbenchProps) {
  const linked = workspace?.linked_workspaces || [];
  const jobs = linked.flatMap((item) => item.jobs);
  const completed = jobs.filter((job) => job.status === "completed").length;
  const resultCount = linked.filter((item) => item.result).length;
  if (!sessionId) {
    return <PanelEmpty icon={History} title="Start a conversation" detail="Create a chat to activate this workbench." />;
  }
  if (profile.type === "data_mining") {
    return (
      <div className="workbench-overview">
        <MetricStrip values={[
          [String(linked.length), "PDFs"],
          [String(completed), "Completed"],
          [String(resultCount), "Results"]
        ]} />
        <SectionIntro icon={FileCheck2} title="Evidence-backed review" text="Each PDF runs independently. Open Results to review devices, layers, performance, and paper materials before saving edits." />
        <button className="primary-action" onClick={() => onViewChange("results")}>Open extracted results</button>
        {jobs.length > 0 && (
          <div className="compact-job-list">
            {linked.map((item) => {
              const job = item.jobs.at(-1);
              return <div key={item.session.session_id}><span>{item.session.title}</span><b>{job?.status || item.session.status}</b><small>{job?.current_step || "Awaiting run"}</small></div>;
            })}
          </div>
        )}
      </div>
    );
  }
  if (profile.type === "device_modeling") {
    return (
      <div className="workbench-overview">
        <MetricStrip values={[["4,228", "Model-ready devices"], ["0.81", "Reported test R2"], ["Q10-Q90", "Prediction interval"]]} />
        <SectionIntro icon={Network} title="OLED device graph modeling" text="Inspect frozen configurations and metrics, then prepare a versioned Slurm script. Submission remains a separate explicit action." />
        <button className="primary-action" onClick={() => onViewChange("models")}>Inspect model registry</button>
      </div>
    );
  }
  return (
    <div className="workbench-overview">
      <MetricStrip values={[["19,175", "Reported OLED devices"], ["DOI", "Provenance"], ["Human", "Final approval"]]} />
      <SectionIntro icon={FlaskConical} title="Grounded experiment planning" text="The agent retrieves device-level precedents first and separates observed evidence from hypotheses and proposed experiments." />
      <button className="primary-action" onClick={() => onViewChange("evidence")}>Inspect retrieved evidence</button>
    </div>
  );
}

function MetricStrip({ values }: { values: Array<[string, string]> }) {
  return <div className="metric-strip">{values.map(([value, label]) => <div key={label}><strong>{value}</strong><span>{label}</span></div>)}</div>;
}

function SectionIntro({ icon: Icon, title, text }: { icon: typeof Database; title: string; text: string }) {
  return <section className="section-intro"><Icon size={22} /><div><h3>{title}</h3><p>{text}</p></div></section>;
}
