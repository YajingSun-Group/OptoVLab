import { Check, Clock3, Wrench, X } from "lucide-react";
import type { SessionWorkspace } from "../../types";
import { PanelEmpty } from "./PanelEmpty";

export function ToolsPanel({ workspace }: { workspace: SessionWorkspace | null }) {
  const local = workspace?.tool_events || [];
  const linked = (workspace?.linked_workspaces || []).flatMap((item) =>
    item.events.map((event) => ({
      event_id: event.event_id,
      tool_name: event.stage,
      status: event.status,
      title: event.title,
      detail: event.detail,
      created_at: event.created_at,
      payload: {}
    }))
  );
  const events = [...local, ...linked].sort((a, b) => b.created_at.localeCompare(a.created_at));
  if (!events.length) {
    return <PanelEmpty icon={Wrench} title="No tool calls yet" detail="Tool activity will appear here with status and provenance." />;
  }
  return (
    <div className="tools-panel">
      <div className="panel-heading"><div><small>AUDIT TRAIL</small><h3>Tool executions</h3></div><span>{events.length}</span></div>
      {events.map((event) => (
        <details className="tool-log-row" key={event.event_id}>
          <summary>
            <ToolStatus status={event.status} />
            <div><strong>{event.title}</strong><span>{event.tool_name.replaceAll("_", " ")}</span></div>
            <time>{new Date(event.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
          </summary>
          <div className="tool-log-detail">
            {event.detail && <p>{event.detail}</p>}
            {Object.keys(event.payload).length > 0 && <pre>{JSON.stringify(event.payload, null, 2)}</pre>}
          </div>
        </details>
      ))}
    </div>
  );
}

function ToolStatus({ status }: { status: string }) {
  if (["completed", "success", "ready"].includes(status)) return <span className="tool-status success"><Check size={13} /></span>;
  if (["failed", "error"].includes(status)) return <span className="tool-status error"><X size={13} /></span>;
  return <span className="tool-status running"><Clock3 size={13} /></span>;
}
