import { BarChart3, Download, Play, Table2 } from "lucide-react";
import { useState } from "react";
import { api } from "../../api";
import type { AnalysisResult, SessionWorkspace } from "../../types";
import { PanelEmpty } from "./PanelEmpty";

const skills = [
  { id: "dataset_summary", label: "Dataset summary" },
  { id: "data_quality_profile", label: "Data quality profile" },
  { id: "univariate_distribution", label: "Univariate distribution" },
  { id: "bivariate_relationship", label: "Bivariate relationship" },
  { id: "group_comparison", label: "Group comparison" },
  { id: "correlation_matrix", label: "Correlation matrix" }
];

const numericFields = [
  "eqe_max",
  "ce_max",
  "pe_max",
  "luminance_max",
  "turn_on_voltage",
  "layer_count",
  "material_count",
  "el_peak",
  "fwhm",
  "lifetime",
  "year"
];

const groupFields = ["emission_color", "fabrication_method", "device_type", "final_emitter_class", "journal", "quality_tier"];

export function AnalysisPanel({ sessionId, workspace }: { sessionId: string | null; workspace: SessionWorkspace | null }) {
  const [skillId, setSkillId] = useState("univariate_distribution");
  const [xField, setXField] = useState("layer_count");
  const [yField, setYField] = useState("eqe_max");
  const [metric, setMetric] = useState("eqe_max");
  const [groupField, setGroupField] = useState("emission_color");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (!sessionId) return <PanelEmpty icon={BarChart3} title="No active session" detail="Create a mining conversation before running analysis skills." />;

  const linkedResults = (workspace?.linked_workspaces || []).filter((item) => item.result).length;

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await api.analyze(sessionId!, { skill_id: skillId, x_field: xField, y_field: yField, metric, group_field: groupField }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="analysis-panel">
      <div className="panel-heading"><div><small>VALIDATED SKILLS</small><h3>Analyze mined devices</h3></div><span>{linkedResults || "All"} paper results</span></div>
      <label className="control-field"><span>Analysis skill</span><select value={skillId} onChange={(event) => setSkillId(event.target.value)}>{skills.map((skill) => <option value={skill.id} key={skill.id}>{skill.label}</option>)}</select></label>
      {skillId === "univariate_distribution" && <SelectField label="Numeric field" value={metric} options={numericFields} onChange={setMetric} />}
      {skillId === "bivariate_relationship" && <div className="two-controls"><SelectField label="X field" value={xField} options={numericFields} onChange={setXField} /><SelectField label="Y field" value={yField} options={numericFields} onChange={setYField} /></div>}
      {skillId === "group_comparison" && <div className="two-controls"><SelectField label="Group" value={groupField} options={groupFields} onChange={setGroupField} /><SelectField label="Metric" value={metric} options={numericFields} onChange={setMetric} /></div>}
      <button className="primary-action run-skill-button" onClick={() => void run()} disabled={busy}><Play size={16} /> {busy ? "Analyzing..." : "Run skill"}</button>
      {error && <div className="inline-error">{error}</div>}
      {result && (
        <div className="analysis-result">
          <section><small>INTERPRETATION</small><p>{result.summary}</p></section>
          <details><summary><Table2 size={15} /> Statistics</summary><pre>{JSON.stringify(result.statistics, null, 2)}</pre></details>
          <div className="artifact-gallery">
            {result.artifacts.map((artifact) => artifact.mime_type.startsWith("image/") ? (
              <figure key={artifact.artifact_id}><img src={artifact.url} alt={artifact.title} /><figcaption>{artifact.title}</figcaption></figure>
            ) : (
              <a key={artifact.artifact_id} href={artifact.url} download><Download size={16} /><span>{artifact.title}</span></a>
            ))}
          </div>
        </div>
      )}
      {!result && workspace?.artifacts.length ? (
        <div className="previous-artifacts"><small>PREVIOUS ARTIFACTS</small>{workspace.artifacts.slice(0, 4).map((artifact) => <a key={artifact.artifact_id} href={artifact.url}><BarChart3 size={15} />{artifact.title}</a>)}</div>
      ) : null}
    </div>
  );
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="control-field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option value={option} key={option}>{option}</option>)}</select></label>;
}
