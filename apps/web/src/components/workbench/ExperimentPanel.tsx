import { BookOpenCheck, ExternalLink, Search, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../../api";
import type { RAGHit, SessionWorkspace } from "../../types";
import { PanelEmpty } from "./PanelEmpty";

export function ExperimentPanel({ workspace }: { workspace: SessionWorkspace | null }) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<RAGHit[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const eventHits = useMemo(() => {
    const event = workspace?.tool_events.find((item) => item.tool_name === "oled_device_rag" && Array.isArray(item.payload.hits));
    return (event?.payload.hits || []) as RAGHit[];
  }, [workspace]);
  const visibleHits = hits.length ? hits : eventHits;

  async function search() {
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      setHits((await api.ragSearch(query, 10)).hits);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="experiment-panel">
      <div className="panel-heading"><div><small>DEVICE-LEVEL RAG</small><h3>Retrieved precedents</h3></div><BookOpenCheck size={20} /></div>
      <div className="rag-search"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && query.trim() && void search()} placeholder="Emitter, architecture, color, process..." /><button onClick={() => void search()} disabled={busy || !query.trim()}>{busy ? "..." : "Search"}</button></div>
      {error && <div className="inline-error">{error}</div>}
      {!visibleHits.length ? <PanelEmpty icon={Sparkles} title="No evidence retrieved yet" detail="Ask the agent for an experimental recommendation or search device precedents here." /> : (
        <div className="rag-hit-list">{visibleHits.map((hit) => <article key={`${hit.device_id}-${hit.rank}`}><header><span>#{hit.rank}</span><div><strong>{hit.final_emitter || hit.device_label || "OLED device"}</strong><small>score {hit.score.toFixed(3)}</small></div>{hit.eqe_max !== null && <b>{hit.eqe_max}% EQE</b>}</header><p>{hit.architecture || "Architecture not reported"}</p><footer><code>{hit.device_id}</code>{hit.doi && <a href={`https://doi.org/${hit.doi}`} target="_blank" rel="noreferrer"><ExternalLink size={13} /> {hit.doi}</a>}</footer></article>)}</div>
      )}
    </div>
  );
}
