import type {
  AgentType,
  AnalysisResult,
  AppSummary,
  ConversationTurn,
  HPCStatus,
  ModelRecord,
  RAGSearchResult,
  SessionSummary,
  SessionWorkspace
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status fallback.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

const jsonRequest = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
});

export const api = {
  status: () => request<Record<string, unknown>>("/api/optovlab/status"),
  apps: () => request<AppSummary[]>("/api/optovlab/apps"),
  skills: () => request<Array<Record<string, string>>>("/api/optovlab/skills"),
  sessions: (agentType: AgentType) =>
    request<SessionSummary[]>(
      `/api/optovlab/sessions?agent_type=${encodeURIComponent(agentType)}`
    ),
  createSession: (agentType: AgentType, title?: string) =>
    request<SessionSummary>(
      "/api/optovlab/sessions",
      jsonRequest("POST", { agent_type: agentType, title })
    ),
  deleteSession: (sessionId: string) =>
    request<{ session_id: string; deleted: boolean; preserved_linked_resources: number }>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    ),
  workspace: (sessionId: string) =>
    request<SessionWorkspace>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/workspace`
    ),
  message: (sessionId: string, content: string) =>
    request<ConversationTurn>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/messages`,
      jsonRequest("POST", { content })
    ),
  uploadPdf: async (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("pdf", file);
    return request<Record<string, unknown>>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/pdf`,
      { method: "POST", body: form }
    );
  },
  startMining: (sessionId: string) =>
    request<Record<string, unknown>>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/mining/start`,
      { method: "POST" }
    ),
  updateMiningResult: (
    sessionId: string,
    miningSessionId: string,
    reviewedResult: Record<string, unknown>
  ) =>
    request<Record<string, unknown>>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/mining/result`,
      jsonRequest("PUT", {
        mining_session_id: miningSessionId,
        reviewed_result: reviewedResult,
        message: "Edited in OptoVLab"
      })
    ),
  analyze: (
    sessionId: string,
    payload: {
      skill_id: string;
      x_field?: string;
      y_field?: string;
      group_field?: string;
      metric?: string;
      scope?: "auto" | "session" | "catalog";
    }
  ) =>
    request<AnalysisResult>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/analysis`,
      jsonRequest("POST", payload)
    ),
  ragSearch: (query: string, topK = 8, filters: Record<string, unknown> = {}) =>
    request<RAGSearchResult>(
      "/api/optovlab/rag/search",
      jsonRequest("POST", { query, top_k: topK, filters })
    ),
  models: () => request<ModelRecord[]>("/api/optovlab/models"),
  hpcStatus: () => request<HPCStatus>("/api/optovlab/hpc/status"),
  prepareTraining: (
    sessionId: string,
    payload: {
      run_name: string;
      config_path?: string;
      partition?: string;
      gpus?: number;
      time_limit?: string;
      confirm_submit: boolean;
    }
  ) =>
    request<Record<string, unknown>>(
      `/api/optovlab/sessions/${encodeURIComponent(sessionId)}/training`,
      jsonRequest("POST", payload)
    )
};
