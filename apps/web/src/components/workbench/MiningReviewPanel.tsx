import {
  Check,
  ChevronLeft,
  ChevronRight,
  FileText,
  FlaskConical,
  Layers3,
  Play,
  Save,
  TestTube2
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import type { ExternalMiningWorkspace, SessionWorkspace } from "../../types";
import { PanelEmpty } from "./PanelEmpty";

const PdfPreview = lazy(() =>
  import("../PdfPreview").then((module) => ({ default: module.PdfPreview }))
);

type JsonRecord = Record<string, unknown>;

export function MiningReviewPanel({
  sessionId,
  workspace,
  onRefresh
}: {
  sessionId: string | null;
  workspace: SessionWorkspace | null;
  onRefresh: () => void;
}) {
  const linked = workspace?.linked_workspaces || [];
  const [paperIndex, setPaperIndex] = useState(0);
  const selected = linked[paperIndex] || null;
  const [draft, setDraft] = useState<JsonRecord | null>(null);
  const [deviceIndex, setDeviceIndex] = useState(0);
  const [tab, setTab] = useState<"devices" | "materials" | "evidence" | "paper">("devices");
  const [saving, setSaving] = useState(false);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState("");
  const [progressNotice, setProgressNotice] = useState("");

  useEffect(() => {
    if (paperIndex >= linked.length) setPaperIndex(Math.max(0, linked.length - 1));
  }, [linked.length, paperIndex]);

  useEffect(() => {
    const result = selected?.result;
    setDraft(result ? clone(asRecord(hasKeys(result.reviewed_result) ? result.reviewed_result : result.raw_result)) : null);
    setDeviceIndex(0);
    setNotice("");
  }, [selected?.result?.result_id, selected?.result?.review_status, selected?.session.session_id]);

  const devices = useMemo(() => asArray(draft?.devices).map(asRecord), [draft]);
  const materialsKey = Array.isArray(draft?.paper_materials) ? "paper_materials" : "materials";
  const materials = useMemo(() => asArray(draft?.[materialsKey]).map(asRecord), [draft, materialsKey]);
  const evidence = useMemo(() => asArray(draft?.evidence).map(asRecord), [draft]);

  if (!sessionId || !workspace) {
    return <PanelEmpty icon={FileText} title="No mining workspace" detail="Start a chat and upload a PDF first." />;
  }
  if (!linked.length) {
    return <PanelEmpty icon={FileText} title="Upload a PDF" detail="Attach one or more PDF papers in the chat composer, then request OLED mining." />;
  }

  const job = selected?.jobs.at(-1);
  if (!selected?.result || !draft) {
    const ready = !job && selected?.session.status === "ready_to_run";
    const failed = job?.status === "failed";
    const finalizing = job?.status === "completed";
    const heading = ready
      ? "Ready to run"
      : failed
        ? "Mining needs attention"
        : finalizing
          ? "Finalizing result"
          : "Mining in progress";
    const detail = ready
      ? "The OLED plan and PDF are validated. Start the approved pipeline when ready."
      : job?.current_step || selected?.session.status || "Waiting for the OLED pipeline to start";

    async function startMining() {
      if (!sessionId) return;
      setStarting(true);
      setProgressNotice("");
      try {
        const response = await api.startMining(sessionId);
        const started = Array.isArray(response.started) ? response.started.length : 0;
        setProgressNotice(started ? `${started} mining job(s) queued` : "No pending PDFs required a new job");
        onRefresh();
      } catch (reason) {
        setProgressNotice(reason instanceof Error ? reason.message : String(reason));
      } finally {
        setStarting(false);
      }
    }

    return (
      <div className="mining-progress-panel">
        <PaperSelector linked={linked} selected={paperIndex} setSelected={setPaperIndex} />
        <div className="progress-hero">
          <span className={failed ? "failed" : "running"}><FlaskConical size={25} /></span>
          <h3>{heading}</h3>
          <p>{detail}</p>
          {job && <div className="progress-track"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>}
          {job?.error_message && <pre>{job.error_message}</pre>}
        </div>
        <div className="progress-actions">
          {(ready || failed) && (
            <button className="primary-action" onClick={() => void startMining()} disabled={starting}>
              <Play size={16} /> {starting ? "Starting..." : failed ? "Retry mining" : "Start OLED mining"}
            </button>
          )}
          <button className="secondary-action" onClick={onRefresh}>Refresh status</button>
        </div>
        {progressNotice && <p className="progress-notice">{progressNotice}</p>}
      </div>
    );
  }

  const device = devices[deviceIndex] || null;

  async function save() {
    if (!sessionId || !selected || !draft) return;
    setSaving(true);
    setNotice("");
    try {
      await api.updateMiningResult(sessionId, selected.session.session_id, draft);
      setNotice("Reviewed values saved");
      onRefresh();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  function update(path: Array<string | number>, value: unknown) {
    setDraft((current) => {
      if (!current) return current;
      const next = clone(current);
      setAtPath(next, path, value);
      return next;
    });
  }

  return (
    <div className="mining-review-panel">
      <PaperSelector linked={linked} selected={paperIndex} setSelected={setPaperIndex} />
      <div className="review-tabbar">
        <button className={tab === "devices" ? "active" : ""} onClick={() => setTab("devices")}><Layers3 size={15} /> Devices <b>{devices.length}</b></button>
        <button className={tab === "materials" ? "active" : ""} onClick={() => setTab("materials")}><TestTube2 size={15} /> Materials <b>{materials.length}</b></button>
        <button className={tab === "evidence" ? "active" : ""} onClick={() => setTab("evidence")}><Check size={15} /> Evidence <b>{evidence.length}</b></button>
        <button className={tab === "paper" ? "active" : ""} onClick={() => setTab("paper")}><FileText size={15} /> Paper</button>
      </div>

      {tab === "devices" && (
        <div className="review-tab-content">
          {devices.length ? (
            <>
              <div className="device-switcher">
                <button disabled={deviceIndex === 0} onClick={() => setDeviceIndex((index) => index - 1)}><ChevronLeft size={17} /></button>
                <div><small>DEVICE {deviceIndex + 1} OF {devices.length}</small><strong>{text(device?.device_label) || `Device ${deviceIndex + 1}`}</strong></div>
                <button disabled={deviceIndex >= devices.length - 1} onClick={() => setDeviceIndex((index) => index + 1)}><ChevronRight size={17} /></button>
              </div>
              {device && <DeviceEditor device={device} deviceIndex={deviceIndex} update={update} />}
            </>
          ) : (
            <PanelEmpty icon={Layers3} title="No OLED devices extracted" detail="The paper may not report an eligible OLED device architecture." />
          )}
        </div>
      )}

      {tab === "materials" && (
        <div className="review-tab-content material-editor-list">
          {materials.length ? materials.map((material, index) => (
            <details className="review-section" key={text(material.paper_material_id) || index} open={index === 0}>
              <summary><span>{text(material.paper_material_id) || `M${index + 1}`}</span><strong>{materialName(material)}</strong></summary>
              <div className="review-field-grid">
                {materialFields.map(([key, label]) => (
                  <EditableField key={key} label={label} value={material[key]} onChange={(value) => update([materialsKey, index, key], value)} />
                ))}
              </div>
              <ResolvedStructure material={material} bundle={selected.material_bundle} />
            </details>
          )) : <PanelEmpty icon={TestTube2} title="No device-used materials" detail="Material records appear after device extraction." />}
        </div>
      )}

      {tab === "evidence" && (
        <div className="review-tab-content evidence-list">
          {evidence.length ? evidence.map((item, index) => (
            <article key={text(item.evidence_id) || index}><header><b>{text(item.evidence_id) || `E${index + 1}`}</b><span>Page {text(item.page_id) || "?"}</span></header><p>{text(item.source_text) || text(item.quote) || "No source text"}</p></article>
          )) : <PanelEmpty icon={Check} title="Evidence is stored separately" detail="Field-level evidence anchors are available in the candidate bundle and remain auditable in the full review platform." />}
        </div>
      )}

      {tab === "paper" && <PaperSummary workspace={selected} />}

      <footer className="review-savebar">
        <span>{notice}</span>
        <button className="primary-action" onClick={() => void save()} disabled={saving}><Save size={16} /> {saving ? "Saving..." : "Save reviewed result"}</button>
      </footer>
    </div>
  );
}

function PaperSelector({ linked, selected, setSelected }: { linked: ExternalMiningWorkspace[]; selected: number; setSelected: (index: number) => void }) {
  return (
    <div className="paper-selector">
      <button disabled={selected === 0} onClick={() => setSelected(selected - 1)}><ChevronLeft size={17} /></button>
      <div><small>PDF {selected + 1} OF {linked.length}</small><strong>{linked[selected]?.session.title || "Paper"}</strong></div>
      <button disabled={selected >= linked.length - 1} onClick={() => setSelected(selected + 1)}><ChevronRight size={17} /></button>
    </div>
  );
}

function DeviceEditor({ device, deviceIndex, update }: { device: JsonRecord; deviceIndex: number; update: (path: Array<string | number>, value: unknown) => void }) {
  const base: Array<string | number> = ["devices", deviceIndex];
  const fabrication = asRecord(device.fabrication);
  const layers = asArray(device.layers).map(asRecord);
  const performance = asArray(device.performance).map(asRecord);
  return (
    <div className="device-editor">
      <details className="review-section" open>
        <summary><span>01</span><strong>Device identity and architecture</strong></summary>
        <div className="review-field-grid">
          <EditableField label="Device label" value={device.device_label} onChange={(value) => update([...base, "device_label"], value)} />
          <EditableField label="Control or target" value={device.control_or_target} onChange={(value) => update([...base, "control_or_target"], value)} />
          <EditableField label="Device type" value={device.device_type} onChange={(value) => update([...base, "device_type"], value)} />
          <EditableField label="Emission color" value={device.emission_color} onChange={(value) => update([...base, "emission_color"], value)} />
          <EditableField wide multiline label="Architecture text" value={device.architecture_text || device.architecture} onChange={(value) => update([...base, "architecture_text"], value)} />
        </div>
      </details>
      <details className="review-section">
        <summary><span>02</span><strong>Fabrication</strong></summary>
        <div className="review-field-grid">
          <EditableField label="Method" value={fabrication.method} onChange={(value) => update([...base, "fabrication", "method"], value)} />
          <EditableField label="Encapsulation" value={fabrication.encapsulation} onChange={(value) => update([...base, "fabrication", "encapsulation"], value)} />
          <EditableField label="Device area" value={fabrication.device_area} onChange={(value) => update([...base, "fabrication", "device_area"], value)} />
        </div>
      </details>
      {layers.map((layer, layerIndex) => (
        <LayerEditor key={`${layerIndex}-${text(layer.layer_name)}`} layer={layer} path={[...base, "layers", layerIndex]} index={layerIndex} update={update} />
      ))}
      <details className="review-section">
        <summary><span>P</span><strong>Performance ({performance.length})</strong></summary>
        <div className="performance-editor-list">
          {performance.map((metric, index) => {
            const metricKey = "metric_name" in metric ? "metric_name" : "metric_family";
            const valueKey = "normalized_value" in metric ? "normalized_value" : "value" in metric ? "value" : "raw_value";
            const unitKey = "normalized_unit" in metric ? "normalized_unit" : "unit" in metric ? "unit" : "raw_unit";
            return (
            <div className="performance-editor" key={index}>
              <EditableField label="Metric" value={metric[metricKey]} onChange={(value) => update([...base, "performance", index, metricKey], value)} />
              <EditableField label="Statistic" value={metric.statistic} onChange={(value) => update([...base, "performance", index, "statistic"], value)} />
              <EditableField label="Value" value={metric[valueKey]} onChange={(value) => update([...base, "performance", index, valueKey], value)} />
              <EditableField label="Unit" value={metric[unitKey]} onChange={(value) => update([...base, "performance", index, unitKey], value)} />
            </div>
          )})}
        </div>
      </details>
    </div>
  );
}

function LayerEditor({ layer, path, index, update }: { layer: JsonRecord; path: Array<string | number>; index: number; update: (path: Array<string | number>, value: unknown) => void }) {
  const thickness = asRecord(layer.thickness);
  const components = asArray(layer.components).map(asRecord);
  return (
    <details className="review-section layer-review-section">
      <summary><span>{String(index + 1).padStart(2, "0")}</span><strong>{text(layer.layer_role) || "Unknown layer"}</strong><em>{text(layer.layer_name)}</em></summary>
      <div className="review-field-grid">
        <EditableField label="Layer role" value={layer.layer_role} onChange={(value) => update([...path, "layer_role"], value)} />
        <EditableField label="Layer name" value={layer.layer_name} onChange={(value) => update([...path, "layer_name"], value)} />
        <EditableField label="Thickness" value={thickness.value} onChange={(value) => update([...path, "thickness", "value"], value)} />
        <EditableField label="Unit" value={thickness.unit} onChange={(value) => update([...path, "thickness", "unit"], value)} />
      </div>
      <div className="component-list">
        {components.map((component, componentIndex) => (
          <div className="component-row" key={componentIndex}>
            <EditableField label="Material" value={component.material_mention} onChange={(value) => update([...path, "components", componentIndex, "material_mention"], value)} />
            <EditableField label="Component role" value={component.component_role} onChange={(value) => update([...path, "components", componentIndex, "component_role"], value)} />
            <EditableField label="Material ID" value={component.paper_material_id} onChange={(value) => update([...path, "components", componentIndex, "paper_material_id"], value)} />
          </div>
        ))}
      </div>
    </details>
  );
}

function EditableField({ label, value, onChange, wide = false, multiline = false }: { label: string; value: unknown; onChange: (value: unknown) => void; wide?: boolean; multiline?: boolean }) {
  const display = scalar(value);
  const className = wide ? "review-field review-field-wide" : "review-field";
  return (
    <label className={className}><span>{label}</span>{multiline ? <textarea value={display} rows={3} onChange={(event) => onChange(coerce(event.target.value, value))} /> : <input value={display} onChange={(event) => onChange(coerce(event.target.value, value))} />}</label>
  );
}

function ResolvedStructure({ material, bundle }: { material: JsonRecord; bundle: Record<string, unknown> | null }) {
  const paperMaterialId = text(material.paper_material_id);
  const links = asArray(bundle?.links).map(asRecord);
  const globals = asArray(bundle?.global_materials).map(asRecord);
  const link = links.find((item) => text(item.paper_material_id) === paperMaterialId);
  const global = globals.find((item) => text(item.global_material_id) === text(link?.global_material_id));
  if (!global) return <div className="structure-summary unresolved"><span>Structure</span><b>Not resolved</b></div>;
  return <div className="structure-summary"><span>Resolved structure</span><b>{text(global.canonical_name) || materialName(material)}</b><code>{text(global.canonical_smiles) || "No SMILES"}</code><small>{text(global.source)} · {text(link?.match_status)}</small></div>;
}

function PaperSummary({ workspace }: { workspace: ExternalMiningWorkspace }) {
  const paper = workspace.paper || {};
  const paperId = workspace.session.paper_id;
  return (
    <div className="review-tab-content paper-summary">
      <small>PAPER METADATA</small>
      <h3>{text(paper.title) || workspace.session.title}</h3>
      <dl>
        <div><dt>DOI</dt><dd>{text(paper.doi) || "Not resolved"}</dd></div>
        <div><dt>Journal</dt><dd>{text(paper.journal) || "Not resolved"}</dd></div>
        <div><dt>Publisher</dt><dd>{text(paper.publisher) || "Not resolved"}</dd></div>
        <div><dt>Year</dt><dd>{text(paper.year) || "Not resolved"}</dd></div>
      </dl>
      {paperId && (
        <Suspense fallback={<div className="pdf-loading-state">Loading PDF viewer</div>}>
          <PdfPreview url={`/api/papers/${encodeURIComponent(paperId)}/pdf`} title={workspace.session.title} />
        </Suspense>
      )}
    </div>
  );
}

const materialFields: Array<[string, string]> = [
  ["paper_material_id", "Stable material ID"],
  ["abbreviation", "Abbreviation"],
  ["full_name_in_paper", "Full name in paper"],
  ["normalized_name", "Normalized name"],
  ["material_class", "Material class"]
];

function materialName(material: JsonRecord) {
  return text(material.mention) || text(material.abbreviation) || text(material.mention_list) || text(material.normalized_name) || text(material.paper_material_id) || "Unnamed material";
}

function hasKeys(value: Record<string, unknown>) {
  return Object.keys(value || {}).length > 0;
}

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map(text).filter(Boolean).join(", ");
  return "";
}

function scalar(value: unknown) {
  if (value === null || value === undefined) return "";
  if (["string", "number", "boolean"].includes(typeof value)) return String(value);
  return JSON.stringify(value);
}

function coerce(value: string, original: unknown): unknown {
  if (typeof original === "number") {
    const number = Number(value);
    return Number.isFinite(number) ? number : value;
  }
  if (typeof original === "boolean") return value.toLowerCase() === "true";
  return value || null;
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

function setAtPath(root: JsonRecord, path: Array<string | number>, value: unknown) {
  let current: unknown = root;
  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index];
    const nextKey = path[index + 1];
    const container = current as Record<string | number, unknown>;
    if (container[key] === null || container[key] === undefined || typeof container[key] !== "object") {
      container[key] = typeof nextKey === "number" ? [] : {};
    }
    current = container[key];
  }
  (current as Record<string | number, unknown>)[path[path.length - 1]] = value;
}
