import {
  AppWindow,
  ArrowUp,
  BarChart3,
  Bot,
  BrainCircuit,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Clock3,
  Download,
  FileText,
  FlaskConical,
  Folder,
  Home,
  LoaderCircle,
  MessageSquarePlus,
  Network,
  Paperclip,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Wrench,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, RefObject, SetStateAction } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { navigate } from "../App";
import { api } from "../api";
import type {
  AgentType,
  ConversationTurn,
  ExternalMiningEvent,
  Message,
  SessionSummary,
  SessionWorkspace,
  ToolEvent
} from "../types";
import { Brand } from "./Brand";
import { Workbench } from "./Workbench";

export interface AgentProfile {
  type: AgentType;
  name: string;
  shortName: string;
  headline: string;
  placeholder: string;
  accent: string;
  icon: typeof Bot;
  suggested: string[];
}

type OptimisticTurn = {
  message: Message;
  knownMessageIds: string[];
};

export const profiles: Record<AgentType, AgentProfile> = {
  data_mining: {
    type: "data_mining",
    name: "Data Mining Agent",
    shortName: "Mining",
    headline: "What literature should we understand?",
    placeholder: "Upload PDFs or ask for OLED mining and analysis...",
    accent: "#0a9c92",
    icon: FileText,
    suggested: [
      "Mine OLED device and material data from the uploaded papers",
      "Plot the EQE_max distribution for these results",
      "Check extraction completeness before analysis"
    ]
  },
  device_modeling: {
    type: "device_modeling",
    name: "Device Modeling Agent",
    shortName: "Modeling",
    headline: "What device model should we build?",
    placeholder: "Ask about OLED-GAT, data splits, code, or HPC training...",
    accent: "#3277b3",
    icon: Network,
    suggested: [
      "Inspect the current OLED-GAT model registry",
      "Show current Slurm and GPU status",
      "Design a leakage-safe device graph experiment"
    ]
  },
  experimental_design: {
    type: "experimental_design",
    name: "Experimental Design Agent",
    shortName: "Experiments",
    headline: "What experiment should we plan next?",
    placeholder: "Describe an emitter, target, bottleneck, or device architecture...",
    accent: "#d2733f",
    icon: FlaskConical,
    suggested: [
      "Find high-EQE solution-processed green TADF precedents",
      "Suggest controls for a 4CzIPN hole-blocking-layer study",
      "Compare related architectures and propose a minimal experiment matrix"
    ]
  }
};

export function AgentWorkspace({ agentType }: { agentType: AgentType }) {
  const profile = profiles[agentType];
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [workspace, setWorkspace] = useState<SessionWorkspace | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() =>
    window.matchMedia("(max-width: 760px)").matches
  );
  const [workbenchOpen, setWorkbenchOpen] = useState(false);
  const [workbenchView, setWorkbenchView] = useState("overview");
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [pendingTurn, setPendingTurn] = useState<OptimisticTurn | null>(null);
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [libraryOpen, setLibraryOpen] = useState<"skills" | "apps" | "search" | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const thread = useRef<HTMLDivElement | null>(null);
  const threadEnd = useRef<HTMLDivElement | null>(null);
  const workspaceRequest = useRef(0);

  const refreshWorkspace = useCallback(async (sessionId: string, quiet = false) => {
    const requestId = ++workspaceRequest.current;
    if (!quiet) setError("");
    try {
      const next = await api.workspace(sessionId);
      if (requestId !== workspaceRequest.current) return;
      setWorkspace(next);
      setSessions((current) =>
        current.map((item) => (item.session_id === next.session.session_id ? next.session : item))
      );
    } catch (reason) {
      if (!quiet && requestId === workspaceRequest.current) setError(messageFor(reason));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void api
      .sessions(agentType)
      .then((items) => {
        if (cancelled) return;
        setSessions(items);
        const remembered = window.localStorage.getItem(`optovlab-session-${agentType}`);
        const selected = items.find((item) => item.session_id === remembered) || items[0];
        if (selected) setActiveSessionId(selected.session_id);
      })
      .catch((reason: unknown) => !cancelled && setError(messageFor(reason)));
    return () => {
      cancelled = true;
    };
  }, [agentType]);

  useEffect(() => {
    if (!activeSessionId) {
      setWorkspace(null);
      return;
    }
    setWorkspace((current) =>
      current?.session.session_id === activeSessionId ? current : null
    );
    window.localStorage.setItem(`optovlab-session-${agentType}`, activeSessionId);
    void refreshWorkspace(activeSessionId);
  }, [activeSessionId, agentType, refreshWorkspace]);

  const hasRunningMining = useMemo(
    () =>
      workspace?.linked_workspaces.some((linked) =>
        linked.jobs.some((job) => job.status === "queued" || job.status === "running")
      ) ?? false,
    [workspace]
  );

  useEffect(() => {
    if (!activeSessionId || !hasRunningMining) return;
    const timer = window.setInterval(() => void refreshWorkspace(activeSessionId, true), 3000);
    return () => window.clearInterval(timer);
  }, [activeSessionId, hasRunningMining, refreshWorkspace]);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [workspace?.messages.length, pendingTurn, busy]);

  async function createSession() {
    setBusy(true);
    setError("");
    try {
      const session = await api.createSession(agentType);
      setSessions((current) => [session, ...current]);
      setActiveSessionId(session.session_id);
      setLibraryOpen(null);
    } catch (reason) {
      setError(messageFor(reason));
    } finally {
      setBusy(false);
    }
  }

  async function deleteSession(session: SessionSummary) {
    setDeletingSessionId(session.session_id);
    setError("");
    try {
      await api.deleteSession(session.session_id);
      const deletedIndex = sessions.findIndex((item) => item.session_id === session.session_id);
      const remaining = sessions.filter((item) => item.session_id !== session.session_id);
      setSessions(remaining);
      if (activeSessionId === session.session_id) {
        workspaceRequest.current += 1;
        setWorkspace(null);
        const next = remaining[Math.min(Math.max(deletedIndex, 0), remaining.length - 1)] || null;
        setActiveSessionId(next?.session_id ?? null);
        if (next) {
          window.localStorage.setItem(`optovlab-session-${agentType}`, next.session_id);
        } else {
          window.localStorage.removeItem(`optovlab-session-${agentType}`);
        }
      }
      setPendingDelete(null);
    } catch (reason) {
      setError(messageFor(reason));
      setPendingDelete(null);
    } finally {
      setDeletingSessionId(null);
    }
  }

  async function submit(messageOverride?: string) {
    const content = (messageOverride ?? input).trim();
    if (busy || (!content && files.length === 0)) return;
    const effectiveMessage =
      content || "Mine OLED device and material data from the uploaded paper(s).";
    const optimisticMessage: Message = {
      message_id: `optimistic-${crypto.randomUUID()}`,
      session_id: activeSessionId ?? "pending-session",
      role: "user",
      message_type: "text",
      content: effectiveMessage,
      metadata: { optimistic: true },
      created_at: new Date().toISOString()
    };
    const optimisticTurn: OptimisticTurn = {
      message: optimisticMessage,
      knownMessageIds: workspace?.messages.map((message) => message.message_id) ?? []
    };

    setBusy(true);
    setError("");
    setPendingTurn(optimisticTurn);
    setInput("");
    setFiles([]);
    try {
      let sessionId = activeSessionId;
      if (!sessionId) {
        const created = await api.createSession(agentType);
        setSessions((current) => [created, ...current]);
        setActiveSessionId(created.session_id);
        sessionId = created.session_id;
      }
      setPendingTurn((current) =>
        current?.message.message_id === optimisticMessage.message_id
          ? { ...current, message: { ...current.message, session_id: sessionId } }
          : current
      );
      for (const file of files) {
        await api.uploadPdf(sessionId, file);
      }
      const turn = await api.message(sessionId, effectiveMessage);
      applyConversationTurn(turn, setWorkspace, setSessions);
      setPendingTurn(null);
      await refreshWorkspace(sessionId, true);
      if (agentType === "data_mining") {
        openWorkbench("overview");
      }
    } catch (reason) {
      setError(messageFor(reason));
      setInput(effectiveMessage);
      setPendingTurn(null);
    } finally {
      setBusy(false);
    }
  }

  function openWorkbench(view: string) {
    setWorkbenchView(view);
    setWorkbenchOpen(true);
    setLibraryOpen(null);
    if (window.matchMedia("(max-width: 1180px)").matches) setSidebarCollapsed(true);
  }

  function toggleWorkbench() {
    const next = !workbenchOpen;
    setWorkbenchOpen(next);
    if (next && window.matchMedia("(max-width: 1180px)").matches) setSidebarCollapsed(true);
  }

  function selectSession(sessionId: string) {
    if (sessionId === activeSessionId) return;
    workspaceRequest.current += 1;
    setWorkspace(null);
    setWorkbenchView("overview");
    setActiveSessionId(sessionId);
    setLibraryOpen(null);
    if (workbenchOpen && window.matchMedia("(max-width: 1180px)").matches) {
      setSidebarCollapsed(true);
    }
    window.requestAnimationFrame(() => thread.current?.scrollTo({ top: 0 }));
  }

  function locateMessage(messageId: string) {
    setLibraryOpen(null);
    window.requestAnimationFrame(() => {
      const element = document.querySelector<HTMLElement>(`[data-message-id="${messageId}"]`);
      if (!element) return;
      element.scrollIntoView({ block: "center" });
      element.classList.add("message-located");
      window.setTimeout(() => element.classList.remove("message-located"), 1600);
    });
  }

  const messages = useMemo(() => conversationMessages(workspace), [workspace]);
  const showOptimisticMessage = Boolean(
    pendingTurn && !messages.some(
      (message) =>
        message.role === "user"
        && message.content === pendingTurn.message.content
        && !pendingTurn.knownMessageIds.includes(message.message_id)
    )
  );
  const eventTimeline = mergeEvents(workspace);
  const isEmpty = messages.length <= 1 && !pendingTurn;

  return (
    <div className="agent-desktop" style={{ "--agent-accent": profile.accent } as CSSProperties}>
      <div className={`agent-window ${workbenchOpen ? "with-workbench" : ""}`}>
        <aside className={`agent-sidebar ${sidebarCollapsed ? "agent-sidebar-collapsed" : ""}`}>
          <div className="agent-sidebar-top">
            <Brand compact={sidebarCollapsed} />
            <button
              className="plain-icon"
              onClick={() => setSidebarCollapsed((value) => !value)}
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? <ChevronRight size={19} /> : <ChevronLeft size={19} />}
            </button>
          </div>
          <nav className="agent-primary-nav">
            <SidebarButton icon={MessageSquarePlus} label="New chat" collapsed={sidebarCollapsed} onClick={createSession} />
            <SidebarButton icon={Search} label="Search" collapsed={sidebarCollapsed} onClick={() => setLibraryOpen("search")} />
            <SidebarButton icon={AppWindow} label="Apps" collapsed={sidebarCollapsed} onClick={() => setLibraryOpen("apps")} />
            <SidebarButton icon={BrainCircuit} label="Skills" collapsed={sidebarCollapsed} onClick={() => setLibraryOpen("skills")} />
            <SidebarButton icon={BarChart3} label="Workbench" collapsed={sidebarCollapsed} onClick={() => openWorkbench("overview")} />
          </nav>
          {!sidebarCollapsed && (
            <div className="session-list">
              <span className="session-list-label">RECENT</span>
              {sessions.length ? (
                sessions.map((session) => (
                  <div className="session-row" key={session.session_id}>
                    <button
                      className={`session-select ${session.session_id === activeSessionId ? "session-active" : ""}`}
                      onClick={() => selectSession(session.session_id)}
                    >
                      <span>{session.title}</span>
                      <small>{relativeTime(session.updated_at)}</small>
                    </button>
                    <button
                      className="session-delete"
                      title={`Delete ${session.title}`}
                      aria-label={`Delete ${session.title}`}
                      onClick={() => setPendingDelete(session)}
                      disabled={deletingSessionId === session.session_id || (busy && activeSessionId === session.session_id)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))
              ) : (
                <p>No sessions yet.</p>
              )}
            </div>
          )}
          <div className="agent-sidebar-footer">
            <SidebarButton icon={Home} label="OptoVLab Apps" collapsed={sidebarCollapsed} onClick={() => navigate("/")} />
            {!sidebarCollapsed && <span className="connection-dot" title="Backend connected" />}
          </div>
        </aside>

        <main className="agent-main">
          <header className="agent-main-header">
            <div className="agent-identity">
              <span className="agent-avatar"><profile.icon size={20} /></span>
              <div><strong>{profile.name}</strong><small>Microsoft Agent Framework</small></div>
            </div>
            <div className="agent-header-actions">
              {activeSessionId && (
                <button className="plain-icon" onClick={() => void refreshWorkspace(activeSessionId)} title="Refresh">
                  <RefreshCw size={18} className={hasRunningMining ? "spin" : ""} />
                </button>
              )}
              <button className="workbench-toggle" onClick={toggleWorkbench}>
                <Wrench size={17} /> {workbenchOpen ? "Close" : "Workbench"}
              </button>
            </div>
          </header>

          {libraryOpen ? (
            <LibraryView
              view={libraryOpen}
              profile={profile}
              workspace={workspace}
              close={() => setLibraryOpen(null)}
              openWorkbench={openWorkbench}
              locateMessage={locateMessage}
            />
          ) : (
            <div ref={thread} className={`agent-thread ${isEmpty ? "agent-thread-empty" : ""}`}>
              {isEmpty ? (
                <EmptyConversation profile={profile} onSelect={(value) => void submit(value)} />
              ) : (
                <div className="message-stack">
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.message_id}
                      message={message}
                      profile={profile}
                      onOpenResults={() => openWorkbench("results")}
                    />
                  ))}
                  {showOptimisticMessage && pendingTurn && (
                    <ChatMessage
                      message={pendingTurn.message}
                      profile={profile}
                      onOpenResults={() => openWorkbench("results")}
                    />
                  )}
                  {eventTimeline.length > 0 && <ToolTimeline events={eventTimeline.slice(-12)} />}
                  {pendingTurn && <AssistantLoading profile={profile} />}
                  <div ref={threadEnd} />
                </div>
              )}
              <Composer
                profile={profile}
                value={input}
                files={files}
                busy={busy}
                onChange={setInput}
                onSubmit={() => void submit()}
                onFiles={(selected) => setFiles((current) => [...current, ...selected])}
                removeFile={(index) => setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                fileInput={fileInput}
              />
              {error && <div className="agent-error"><CircleAlert size={16} /> {error}</div>}
            </div>
          )}
        </main>

        {workbenchOpen && (
          <Workbench
            key={activeSessionId ?? "empty-session"}
            profile={profile}
            sessionId={activeSessionId}
            workspace={workspace}
            activeView={workbenchView}
            onViewChange={setWorkbenchView}
            onClose={() => setWorkbenchOpen(false)}
            onRefresh={() => activeSessionId && void refreshWorkspace(activeSessionId)}
          />
        )}
      </div>
      {pendingDelete && (
        <ConfirmDeleteSession
          session={pendingDelete}
          deleting={deletingSessionId === pendingDelete.session_id}
          onCancel={() => setPendingDelete(null)}
          onConfirm={() => void deleteSession(pendingDelete)}
        />
      )}
    </div>
  );
}

function ConfirmDeleteSession({
  session,
  deleting,
  onCancel,
  onConfirm
}: {
  session: SessionSummary;
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !deleting) onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [deleting, onCancel]);

  return (
    <div className="confirm-dialog-scrim" role="presentation" onMouseDown={() => !deleting && onCancel()}>
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-session-heading"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <span className="confirm-dialog-icon"><Trash2 size={20} /></span>
        <div>
          <h2 id="delete-session-heading">Delete session?</h2>
          <p><strong>{session.title}</strong> and its chat history and generated analysis files will be removed.</p>
          <small>Completed mining results and scientific database records are preserved.</small>
        </div>
        <footer>
          <button className="dialog-cancel" onClick={onCancel} disabled={deleting} autoFocus>Cancel</button>
          <button className="dialog-delete" onClick={onConfirm} disabled={deleting}>
            <Trash2 size={15} /> {deleting ? "Deleting..." : "Delete"}
          </button>
        </footer>
      </section>
    </div>
  );
}

function SidebarButton({
  icon: Icon,
  label,
  collapsed,
  onClick
}: {
  icon: typeof Home;
  label: string;
  collapsed: boolean;
  onClick?: () => void;
}) {
  return (
    <button onClick={onClick} title={collapsed ? label : undefined}>
      <Icon size={19} /> {!collapsed && <span>{label}</span>}
    </button>
  );
}

function EmptyConversation({ profile, onSelect }: { profile: AgentProfile; onSelect: (value: string) => void }) {
  return (
    <section className="empty-conversation">
      <span className="empty-agent-icon"><profile.icon size={29} /></span>
      <h1>{profile.headline}</h1>
      <p>{profile.name} combines domain tools with an evidence-first conversation.</p>
      <div className="suggestion-list">
        {profile.suggested.map((suggestion) => (
          <button key={suggestion} onClick={() => onSelect(suggestion)}>
            <Sparkles size={15} /> {suggestion}
          </button>
        ))}
      </div>
    </section>
  );
}

function ChatMessage({
  message,
  profile,
  onOpenResults
}: {
  message: Message;
  profile: AgentProfile;
  onOpenResults: () => void;
}) {
  if (message.role === "system" || message.role === "tool") return null;
  if (message.message_type === "file") {
    const filename = stringMetadata(message.metadata, "filename") || message.content;
    const pageCount = numberMetadata(message.metadata, "page_count");
    const sizeBytes = numberMetadata(message.metadata, "size_bytes");
    const details = [
      "PDF document",
      pageCount ? `${pageCount} pages` : null,
      sizeBytes ? formatBytes(sizeBytes) : null
    ].filter(Boolean).join(" · ");
    return (
      <article className="chat-message chat-message-user chat-message-file" data-message-id={message.message_id}>
        <div className="pdf-attachment-card">
          <span><FileText size={24} /></span>
          <div><strong>{filename}</strong><small>{details}</small></div>
        </div>
      </article>
    );
  }
  const analysis = analysisMessageMetadata(message.metadata);
  const mining = miningMessageMetadata(message.metadata);
  return (
    <article className={`chat-message chat-message-${message.role}`} data-message-id={message.message_id}>
      {message.role === "assistant" && <span className="message-avatar"><profile.icon size={17} /></span>}
      <div>
        <span className="message-author">{message.role === "user" ? "You" : profile.name}</span>
        <div className="message-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
        {mining && <ChatMiningResult mining={mining} onOpenResults={onOpenResults} />}
        {analysis && <ChatAnalysisResult analysis={analysis} />}
      </div>
    </article>
  );
}

type ChatMining = {
  status: string;
  filename: string;
  pageCount: number | null;
  deviceCount: number;
  materialCount: number;
  evidenceCount: number;
  reviewStatus: string;
  pipeline: string[];
};

function ChatMiningResult({
  mining,
  onOpenResults
}: {
  mining: ChatMining;
  onOpenResults: () => void;
}) {
  return (
    <section className="chat-mining-result">
      <header>
        <span><Check size={18} /></span>
        <div><small>DATA MINING WORKFLOW · {mining.status}</small><strong>OLED device v1</strong></div>
      </header>
      <div className="chat-mining-file"><FileText size={17} /><span>{mining.filename}</span>{mining.pageCount ? <small>{mining.pageCount} pages</small> : null}</div>
      <div className="chat-mining-metrics">
        <div><strong>{mining.deviceCount}</strong><span>Devices</span></div>
        <div><strong>{mining.materialCount}</strong><span>Materials</span></div>
        <div><strong>{mining.evidenceCount}</strong><span>Evidence</span></div>
      </div>
      <div className="chat-mining-pipeline">
        {mining.pipeline.map((stage) => <div key={stage}><Check size={13} /><span>{stage}</span></div>)}
      </div>
      <footer>
        <span>Review status: {mining.reviewStatus.replaceAll("_", " ")}</span>
        <button onClick={onOpenResults}>Open extracted results</button>
      </footer>
    </section>
  );
}

function miningMessageMetadata(metadata: Record<string, unknown>): ChatMining | null {
  const value = metadata.mining;
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const pipeline = Array.isArray(record.pipeline)
    ? record.pipeline.filter((stage): stage is string => typeof stage === "string")
    : [];
  return {
    status: typeof record.status === "string" ? record.status : "completed",
    filename: typeof record.filename === "string" ? record.filename : "Uploaded PDF",
    pageCount: typeof record.page_count === "number" ? record.page_count : null,
    deviceCount: typeof record.device_count === "number" ? record.device_count : 0,
    materialCount: typeof record.material_count === "number" ? record.material_count : 0,
    evidenceCount: typeof record.evidence_count === "number" ? record.evidence_count : 0,
    reviewStatus: typeof record.review_status === "string" ? record.review_status : "pending_review",
    pipeline
  };
}

type ChatAnalysis = {
  skillId: string;
  scope: string;
  statistics: Record<string, unknown>;
  artifacts: Array<Record<string, unknown>>;
};

function ChatAnalysisResult({ analysis }: { analysis: ChatAnalysis }) {
  const images = analysis.artifacts.filter((artifact) =>
    String(artifact.mime_type || "").startsWith("image/")
  );
  const downloads = analysis.artifacts.filter((artifact) => !images.includes(artifact));
  const metrics = ["devices", "papers", "emitters", "eqe_reported"]
    .flatMap((key) => analysis.statistics[key] == null ? [] : [[key, analysis.statistics[key]]]);
  return (
    <section className="chat-analysis-result">
      <header>
        <span><BarChart3 size={18} /></span>
        <div>
          <small>ANALYSIS SKILL · {analysis.scope === "catalog" ? "FULL OLED DATABASE" : "CURRENT SESSION"}</small>
          <strong>{analysis.skillId.replaceAll("_", " ")}</strong>
        </div>
      </header>
      {metrics.length > 0 && (
        <div className="chat-analysis-metrics">
          {metrics.map(([label, value]) => <div key={String(label)}><strong>{String(value)}</strong><span>{String(label).replaceAll("_", " ")}</span></div>)}
        </div>
      )}
      {images.map((artifact) => (
        <figure key={String(artifact.artifact_id)}>
          <img src={String(artifact.url)} alt={String(artifact.title || "Analysis chart")} />
          <figcaption>{String(artifact.title || "Analysis chart")}</figcaption>
        </figure>
      ))}
      {downloads.map((artifact) => (
        <a key={String(artifact.artifact_id)} href={String(artifact.url)} download>
          <Download size={15} /><span>{String(artifact.title || "Download analysis data")}</span>
        </a>
      ))}
    </section>
  );
}

function analysisMessageMetadata(metadata: Record<string, unknown>): ChatAnalysis | null {
  const value = metadata.analysis;
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const artifacts = Array.isArray(record.artifacts)
    ? record.artifacts.filter(
        (artifact): artifact is Record<string, unknown> =>
          Boolean(artifact) && typeof artifact === "object" && !Array.isArray(artifact)
      )
    : [];
  const statistics = record.statistics;
  return {
    skillId: typeof record.skill_id === "string" ? record.skill_id : "analysis",
    scope: typeof record.scope === "string" ? record.scope : "auto",
    statistics: statistics && typeof statistics === "object" && !Array.isArray(statistics)
      ? statistics as Record<string, unknown>
      : {},
    artifacts
  };
}

function conversationMessages(workspace: SessionWorkspace | null): Message[] {
  if (!workspace) return [];
  const existingResourceIds = new Set(
    workspace.messages
      .filter((message) => message.message_type === "file")
      .map((message) => stringMetadata(message.metadata, "resource_id"))
      .filter(Boolean)
  );
  const existingFilenames = new Set(
    workspace.messages
      .filter((message) => message.message_type === "file")
      .map((message) => stringMetadata(message.metadata, "filename"))
      .filter(Boolean)
  );
  const attachments = workspace.resources.flatMap((resource): Message[] => {
    if (resource.resource_type !== "data_mining_session") return [];
    if (existingResourceIds.has(resource.resource_id) || existingFilenames.has(resource.label)) return [];
    const linked = workspace.linked_workspaces.find(
      (item) => item.session.session_id === resource.resource_id
    );
    const linkedFile = linked?.messages?.find((message) => message.message_type === "file");
    const metadata = linkedFile?.metadata ?? {};
    return [{
      message_id: `resource-${resource.link_id}`,
      session_id: resource.session_id,
      role: "user",
      message_type: "file",
      content: resource.label,
      metadata: {
        ...metadata,
        resource_id: resource.resource_id,
        filename: resource.label,
        mime_type: "application/pdf",
        size_bytes: numberMetadata(metadata, "size_bytes")
          ?? numberMetadata(linked?.paper ?? {}, "pdf_size_bytes")
          ?? numberMetadata(resource.metadata, "size_bytes"),
        page_count: numberMetadata(metadata, "page_count")
          ?? numberMetadata(resource.metadata, "page_count")
      },
      created_at: resource.created_at
    }];
  });
  return [...workspace.messages, ...attachments].sort((left, right) =>
    left.created_at.localeCompare(right.created_at) || left.message_id.localeCompare(right.message_id)
  );
}

function applyConversationTurn(
  turn: ConversationTurn,
  setWorkspace: Dispatch<SetStateAction<SessionWorkspace | null>>,
  setSessions: Dispatch<SetStateAction<SessionSummary[]>>
) {
  setWorkspace((current) => {
    const base = current?.session.session_id === turn.session.session_id
      ? current
      : {
          session: turn.session,
          messages: [],
          tool_events: [],
          artifacts: [],
          resources: [],
          linked_workspaces: []
        };
    return {
      ...base,
      session: turn.session,
      messages: upsertById(
        base.messages,
        [turn.user_message, turn.assistant_message],
        (message) => message.message_id
      ),
      tool_events: upsertById(
        base.tool_events,
        turn.tool_events,
        (event) => event.event_id
      ),
      artifacts: upsertById(
        base.artifacts,
        turn.artifacts,
        (artifact) => artifact.artifact_id
      )
    };
  });
  setSessions((current) =>
    current.map((session) =>
      session.session_id === turn.session.session_id ? turn.session : session
    )
  );
}

function upsertById<T>(current: T[], incoming: T[], idFor: (item: T) => string): T[] {
  const merged = new Map(current.map((item) => [idFor(item), item]));
  incoming.forEach((item) => merged.set(idFor(item), item));
  return Array.from(merged.values());
}

function stringMetadata(metadata: Record<string, unknown>, key: string): string | null {
  const value = metadata[key];
  return typeof value === "string" && value ? value : null;
}

function numberMetadata(metadata: Record<string, unknown>, key: string): number | null {
  const value = metadata[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

type TimelineEvent = {
  id: string;
  status: string;
  title: string;
  detail?: string | null;
  tool: string;
  createdAt: string;
};

function mergeEvents(workspace: SessionWorkspace | null): TimelineEvent[] {
  if (!workspace) return [];
  const local = workspace.tool_events.map((event: ToolEvent) => ({
    id: event.event_id,
    status: event.status,
    title: event.title,
    detail: event.detail,
    tool: event.tool_name,
    createdAt: event.created_at
  }));
  const linked = workspace.linked_workspaces.flatMap((item) =>
    item.events.map((event: ExternalMiningEvent) => ({
      id: event.event_id,
      status: event.status,
      title: event.title,
      detail: event.detail,
      tool: event.stage,
      createdAt: event.created_at
    }))
  );
  return [...local, ...linked].sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}

function ToolTimeline({ events }: { events: TimelineEvent[] }) {
  return (
    <section className="tool-timeline">
      <header><Wrench size={15} /> Tool activity</header>
      {events.map((event) => (
        <div className="tool-event" key={`${event.tool}-${event.id}`}>
          <StatusIcon status={event.status} />
          <div><strong>{event.title}</strong><span>{event.tool.replaceAll("_", " ")}</span>{event.detail && <p>{event.detail}</p>}</div>
        </div>
      ))}
    </section>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (["completed", "ready", "success"].includes(status)) return <Check size={15} className="status-success" />;
  if (["failed", "error"].includes(status)) return <X size={15} className="status-error" />;
  return <Clock3 size={15} className="status-running" />;
}

function AssistantLoading({ profile }: { profile: AgentProfile }) {
  return (
    <article className="chat-message chat-message-assistant chat-message-loading" role="status" aria-live="polite">
      <span className="message-avatar"><profile.icon size={17} /></span>
      <div>
        <span className="message-author">{profile.name}</span>
        <div className="assistant-loading"><LoaderCircle size={17} className="spin" /><span>Thinking...</span></div>
      </div>
    </article>
  );
}

function Composer({
  profile,
  value,
  files,
  busy,
  onChange,
  onSubmit,
  onFiles,
  removeFile,
  fileInput
}: {
  profile: AgentProfile;
  value: string;
  files: File[];
  busy: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onFiles: (files: File[]) => void;
  removeFile: (index: number) => void;
  fileInput: RefObject<HTMLInputElement | null>;
}) {
  return (
    <div className="composer-wrap">
      <div className="composer">
        {files.length > 0 && (
          <div className="composer-files">
            {files.map((file, index) => (
              <span key={`${file.name}-${index}`}><FileText size={14} /> {file.name}<button onClick={() => removeFile(index)}><X size={13} /></button></span>
            ))}
          </div>
        )}
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSubmit();
            }
          }}
          placeholder={profile.placeholder}
          rows={3}
          disabled={busy}
        />
        <div className="composer-controls">
          <div>
            {profile.type === "data_mining" && (
              <>
                <button className="composer-add" onClick={() => fileInput.current?.click()} title="Attach PDF"><Plus size={19} /></button>
                <input
                  ref={fileInput}
                  type="file"
                  accept="application/pdf,.pdf"
                  multiple
                  hidden
                  onChange={(event) => {
                    onFiles(Array.from(event.target.files || []));
                    event.target.value = "";
                  }}
                />
              </>
            )}
            <span className="tool-access"><Wrench size={14} /> Research tools</span>
          </div>
          <div>
            <span className="model-pill"><Bot size={14} /> deepseek-v4-flash</span>
            <button className="send-button" disabled={busy || (!value.trim() && !files.length)} onClick={onSubmit} title="Send"><ArrowUp size={18} /></button>
          </div>
        </div>
        {profile.type === "data_mining" && <div className="composer-project"><Folder size={15} /> OLED device v1 <span>MinerU / DeepSeek / DECIMER</span></div>}
      </div>
    </div>
  );
}

function LibraryView({
  view,
  profile,
  workspace,
  close,
  openWorkbench,
  locateMessage
}: {
  view: "skills" | "apps" | "search";
  profile: AgentProfile;
  workspace: SessionWorkspace | null;
  close: () => void;
  openWorkbench: (view: string) => void;
  locateMessage: (messageId: string) => void;
}) {
  const skillWorkbenchView =
    profile.type === "data_mining"
      ? "analysis"
      : profile.type === "device_modeling"
        ? "models"
        : "evidence";
  return (
    <section className="library-view">
      <header><div><small>{profile.name}</small><h2>{view[0].toUpperCase() + view.slice(1)}</h2></div><button className="plain-icon" onClick={close}><X size={19} /></button></header>
      {view === "apps" && (
        <div className="library-grid">
          {Object.values(profiles).map((item) => <button key={item.type} onClick={() => navigate(`/agents/${item.type.replaceAll("_", "-")}`)}><item.icon /><strong>{item.name}</strong><span>{item.headline}</span></button>)}
          <button onClick={() => navigate("/database")}><AppWindow /><strong>Device Database</strong><span>OLED, OFET, and OPV records</span></button>
        </div>
      )}
      {view === "skills" && (
        <div className="library-grid">
          {(profile.type === "data_mining"
            ? ["Dataset summary", "Data quality profile", "Univariate distribution", "Bivariate relationship", "Group comparison", "Correlation matrix"]
            : profile.type === "device_modeling"
              ? ["Dataset inspection", "OLED-GAT coding", "Leakage-safe evaluation", "HPC training", "Model registry"]
              : ["OLED device RAG", "Precedent comparison", "Experiment matrix", "Control selection", "Uncertainty audit"]
          ).map((skill) => <button key={skill} onClick={() => openWorkbench(skillWorkbenchView)}><BrainCircuit /><strong>{skill}</strong><span>Validated OptoVLab skill</span></button>)}
        </div>
      )}
      {view === "search" && (
        <SessionSearchView
          workspace={workspace}
          locateMessage={locateMessage}
          openTools={() => openWorkbench("tools")}
        />
      )}
    </section>
  );
}

function SessionSearchView({
  workspace,
  locateMessage,
  openTools
}: {
  workspace: SessionWorkspace | null;
  locateMessage: (messageId: string) => void;
  openTools: () => void;
}) {
  const [query, setQuery] = useState("");
  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized || !workspace) return [];
    const messages = workspace.messages
      .filter((message) => message.content.toLowerCase().includes(normalized))
      .map((message) => ({
        id: message.message_id,
        kind: "message" as const,
        title: message.role === "user" ? "You" : "Agent response",
        detail: message.content
      }));
    const tools = workspace.tool_events
      .filter((event) =>
        [event.title, event.detail, event.tool_name]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(normalized))
      )
      .map((event) => ({
        id: event.event_id,
        kind: "tool" as const,
        title: event.title,
        detail: event.detail || event.tool_name.replaceAll("_", " ")
      }));
    return [...messages, ...tools].slice(0, 40);
  }, [query, workspace]);
  const searched = Boolean(query.trim());
  return (
    <div className="session-search-view">
      <label className="session-search-box">
        <Search size={22} />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search messages and tool activity"
          autoFocus
        />
      </label>
      <p>{workspace?.messages.length || 0} messages and {workspace?.tool_events.length || 0} tool events in this workspace.</p>
      {searched && (
        <div className="session-search-results" aria-live="polite">
          {results.length ? results.map((result) => (
            <button
              key={`${result.kind}-${result.id}`}
              onClick={() => result.kind === "message" ? locateMessage(result.id) : openTools()}
            >
              <span>{result.kind}</span>
              <strong>{result.title}</strong>
              <p>{result.detail}</p>
            </button>
          )) : <div className="session-search-empty">No matching messages or tool activity.</div>}
        </div>
      )}
    </div>
  );
}

function relativeTime(value: string) {
  const delta = Date.now() - new Date(value).getTime();
  if (delta < 60_000) return "now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h`;
  return `${Math.floor(delta / 86_400_000)}d`;
}

function messageFor(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason);
}
