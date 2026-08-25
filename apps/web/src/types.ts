export type AgentType = "data_mining" | "device_modeling" | "experimental_design";

export interface AppSummary {
  app_id: string;
  name: string;
  category: string;
  description: string;
  route: string;
  status: string;
  metrics: Record<string, unknown>;
}

export interface SessionSummary {
  session_id: string;
  agent_type: AgentType;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  message_id: string;
  session_id: string;
  role: "user" | "assistant" | "system" | "tool";
  message_type: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ToolEvent {
  event_id: string;
  session_id: string;
  job_id: string | null;
  tool_name: string;
  status: string;
  title: string;
  detail: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface Artifact {
  artifact_id: string;
  session_id: string;
  artifact_type: string;
  title: string;
  filename: string;
  mime_type: string;
  url: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ResourceLink {
  link_id: string;
  session_id: string;
  resource_type: string;
  resource_id: string;
  label: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ExternalMiningJob {
  job_id: string;
  status: string;
  current_step: string;
  progress: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ExternalMiningEvent {
  event_id: string;
  stage: string;
  status: string;
  title: string;
  detail: string | null;
  created_at: string;
}

export interface ExternalMiningResult {
  result_id: string;
  session_id: string;
  paper_id: string;
  raw_result: Record<string, unknown>;
  reviewed_result: Record<string, unknown>;
  review_status: string;
}

export interface ExternalMiningWorkspace {
  session: {
    session_id: string;
    title: string;
    status: string;
    paper_id: string | null;
    plan_status: string;
  };
  messages?: Message[];
  jobs: ExternalMiningJob[];
  events: ExternalMiningEvent[];
  paper: Record<string, unknown> | null;
  result: ExternalMiningResult | null;
  candidate_bundle: Record<string, unknown> | null;
  material_bundle: Record<string, unknown> | null;
}

export interface SessionWorkspace {
  session: SessionSummary;
  messages: Message[];
  tool_events: ToolEvent[];
  artifacts: Artifact[];
  resources: ResourceLink[];
  linked_workspaces: ExternalMiningWorkspace[];
}

export interface ConversationTurn {
  session: SessionSummary;
  user_message: Message;
  assistant_message: Message;
  tool_events: ToolEvent[];
  artifacts: Artifact[];
}

export interface AnalysisResult {
  skill_id: string;
  summary: string;
  statistics: Record<string, unknown>;
  artifacts: Artifact[];
}

export interface RAGHit {
  rank: number;
  score: number;
  device_id: string;
  doi: string | null;
  title: string | null;
  journal: string | null;
  device_label: string | null;
  architecture: string | null;
  final_emitter: string | null;
  eqe_max: number | null;
  record: Record<string, unknown>;
}

export interface RAGSearchResult {
  query: string;
  total_devices: number;
  hits: RAGHit[];
}

export interface HPCStatus {
  scheduler_available: boolean;
  partitions: Array<Record<string, string>>;
  jobs: Array<Record<string, string>>;
  gpus: Array<Record<string, string | number>>;
  errors: string[];
}

export interface ModelRecord {
  id: string;
  name: string;
  config_path: string;
  architecture: string | null;
  split_mode: string | null;
  quantiles: number[];
  output_dir: string;
  metrics: Array<Record<string, unknown>>;
}
