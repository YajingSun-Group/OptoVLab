from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo


def now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


class Database:
    _init_lock = Lock()
    _initialized_paths: set[Path] = set()

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 60000")
        return conn

    def init_db(self) -> None:
        cache_key = self.path.resolve()
        if cache_key in self._initialized_paths and self.path.exists():
            return
        with self._init_lock:
            if cache_key in self._initialized_paths and self.path.exists():
                return
            self._init_db_once()
            self._initialized_paths.add(cache_key)

    def _init_db_once(self) -> None:
        with self.connect() as conn:
            # Stage 3 performs concurrent network work with short, bounded write bursts.
            # WAL lets readers continue while the single SQLite writer commits a batch.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                  paper_id TEXT PRIMARY KEY,
                  doi TEXT NOT NULL UNIQUE,
                  title TEXT,
                  journal TEXT,
                  publisher TEXT,
                  year INTEGER,
                  pdf_path TEXT NOT NULL,
                  pdf_sha256 TEXT NOT NULL,
                  pdf_size_bytes INTEGER NOT NULL,
                  source TEXT NOT NULL,
                  download_status TEXT NOT NULL,
                  parse_status TEXT NOT NULL DEFAULT 'pending',
                  mining_status TEXT NOT NULL DEFAULT 'pending',
                  review_status TEXT NOT NULL DEFAULT 'not_started',
                  review_reason TEXT,
                  domain TEXT NOT NULL DEFAULT 'unknown',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_jobs (
                  job_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL UNIQUE,
                  doi TEXT NOT NULL,
                  source_pdf_path TEXT,
                  inbox_pdf_path TEXT NOT NULL,
                  pdf_sha256 TEXT NOT NULL,
                  pdf_size_bytes INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  current_stage TEXT NOT NULL,
                  last_completed_stage TEXT,
                  retry_count INTEGER NOT NULL DEFAULT 0,
                  max_retries INTEGER NOT NULL DEFAULT 2,
                  error_message TEXT,
                  stage_timings_json TEXT NOT NULL DEFAULT '{}',
                  stage_errors_json TEXT NOT NULL DEFAULT '{}',
                  options_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  started_at TEXT,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_status
                  ON batch_jobs(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_checksum
                  ON batch_jobs(pdf_sha256);
                CREATE TABLE IF NOT EXISTS device_records_reviewed (
                  record_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  device_label TEXT,
                  architecture TEXT,
                  notes TEXT,
                  substrate TEXT,
                  anode TEXT,
                  hil TEXT,
                  htl TEXT,
                  ebl TEXT,
                  eml_host TEXT,
                  eml_dopant TEXT,
                  eml_emitter TEXT,
                  hbl TEXT,
                  etl TEXT,
                  eil TEXT,
                  cathode TEXT,
                  layer_thicknesses TEXT,
                  eqe_max TEXT,
                  ce_max TEXT,
                  pe_max TEXT,
                  luminance_max TEXT,
                  turn_on_voltage TEXT,
                  cie_x TEXT,
                  cie_y TEXT,
                  el_peak TEXT,
                  fwhm TEXT,
                  lifetime TEXT,
                  evidence_text TEXT,
                  evidence_page INTEGER,
                  review_status TEXT NOT NULL DEFAULT 'in_progress',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  confirmed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                  event_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  record_id TEXT,
                  event_type TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  message TEXT,
                  before_json TEXT,
                  after_json TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(record_id) REFERENCES device_records_reviewed(record_id)
                );
                CREATE TABLE IF NOT EXISTS document_blocks (
                  paper_id TEXT NOT NULL,
                  block_id TEXT NOT NULL,
                  page_id INTEGER NOT NULL,
                  block_index INTEGER NOT NULL,
                  block_type TEXT NOT NULL,
                  text TEXT NOT NULL,
                  bbox_json TEXT NOT NULL,
                  source TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (paper_id, block_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS extraction_runs (
                  run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  extractor_name TEXT NOT NULL,
                  extractor_version TEXT NOT NULL,
                  status TEXT NOT NULL,
                  input_block_count INTEGER NOT NULL DEFAULT 0,
                  raw_record_count INTEGER NOT NULL DEFAULT 0,
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS device_records_raw (
                  raw_record_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  device_label TEXT,
                  architecture TEXT,
                  notes TEXT,
                  substrate TEXT,
                  anode TEXT,
                  hil TEXT,
                  htl TEXT,
                  ebl TEXT,
                  eml_host TEXT,
                  eml_dopant TEXT,
                  eml_emitter TEXT,
                  hbl TEXT,
                  etl TEXT,
                  eil TEXT,
                  cathode TEXT,
                  layer_thicknesses TEXT,
                  eqe_max TEXT,
                  ce_max TEXT,
                  pe_max TEXT,
                  luminance_max TEXT,
                  turn_on_voltage TEXT,
                  cie_x TEXT,
                  cie_y TEXT,
                  el_peak TEXT,
                  fwhm TEXT,
                  lifetime TEXT,
                  evidence_text TEXT,
                  evidence_page INTEGER,
                  evidence_block_ids_json TEXT NOT NULL,
                  field_evidence_json TEXT NOT NULL,
                  confidence_json TEXT NOT NULL,
                  raw_payload_json TEXT NOT NULL,
                  review_status TEXT NOT NULL DEFAULT 'pending',
                  reviewed_record_id TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES extraction_runs(run_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(reviewed_record_id) REFERENCES device_records_reviewed(record_id)
                );
                CREATE TABLE IF NOT EXISTS evidence_anchors (
                  evidence_anchor_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  page_id INTEGER,
                  block_id TEXT,
                  bbox_json TEXT NOT NULL,
                  source_text TEXT,
                  source_type TEXT NOT NULL DEFAULT 'text',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS candidate_field_values (
                  candidate_field_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  record_scope TEXT NOT NULL DEFAULT 'device',
                  record_id TEXT NOT NULL,
                  field_name TEXT NOT NULL,
                  field_label TEXT NOT NULL,
                  mined_value TEXT,
                  reviewed_value TEXT,
                  confidence REAL,
                  confidence_json TEXT NOT NULL,
                  evidence_anchor_id TEXT,
                  extractor_name TEXT,
                  extractor_version TEXT,
                  field_status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(evidence_anchor_id) REFERENCES evidence_anchors(evidence_anchor_id)
                );
                CREATE TABLE IF NOT EXISTS oled_devices_final (
                  final_device_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  device_label TEXT,
                  architecture TEXT,
                  notes TEXT,
                  substrate TEXT,
                  anode TEXT,
                  hil TEXT,
                  htl TEXT,
                  ebl TEXT,
                  eml_host TEXT,
                  eml_dopant TEXT,
                  eml_emitter TEXT,
                  hbl TEXT,
                  etl TEXT,
                  eil TEXT,
                  cathode TEXT,
                  layer_thicknesses TEXT,
                  eqe_max TEXT,
                  ce_max TEXT,
                  pe_max TEXT,
                  luminance_max TEXT,
                  turn_on_voltage TEXT,
                  cie_x TEXT,
                  cie_y TEXT,
                  el_peak TEXT,
                  fwhm TEXT,
                  lifetime TEXT,
                  evidence_text TEXT,
                  evidence_page INTEGER,
                  source_candidate_ids_json TEXT NOT NULL,
                  confirmed_by TEXT NOT NULL,
                  confirmed_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS candidate_ingestion_runs (
                  candidate_run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  template_version TEXT NOT NULL,
                  source_name TEXT NOT NULL,
                  source_version TEXT,
                  status TEXT NOT NULL,
                  validation_report_json TEXT NOT NULL,
                  mining_result_json TEXT NOT NULL,
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS candidate_entities (
                  candidate_entity_id TEXT PRIMARY KEY,
                  candidate_run_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  entity_type TEXT NOT NULL,
                  entity_path TEXT NOT NULL,
                  entity_label TEXT,
                  parent_entity_id TEXT,
                  sort_order INTEGER NOT NULL DEFAULT 0,
                  source_json TEXT NOT NULL,
                  review_status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(parent_entity_id) REFERENCES candidate_entities(candidate_entity_id)
                );
                CREATE INDEX IF NOT EXISTS idx_candidate_entities_run
                  ON candidate_entities(candidate_run_id, entity_path);
                CREATE TABLE IF NOT EXISTS candidate_values (
                  candidate_value_id TEXT PRIMARY KEY,
                  candidate_run_id TEXT NOT NULL,
                  candidate_entity_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  template_field_path TEXT NOT NULL,
                  concrete_path TEXT NOT NULL,
                  field_label TEXT NOT NULL,
                  data_type TEXT NOT NULL,
                  value_json TEXT NOT NULL,
                  reviewed_value_json TEXT,
                  display_value TEXT,
                  evidence_anchor_ids_json TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(candidate_entity_id) REFERENCES candidate_entities(candidate_entity_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_candidate_values_run
                  ON candidate_values(candidate_run_id, concrete_path);
                CREATE TABLE IF NOT EXISTS candidate_value_review_events (
                  event_id TEXT PRIMARY KEY,
                  candidate_value_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  candidate_entity_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  template_field_path TEXT NOT NULL,
                  concrete_path TEXT NOT NULL,
                  action TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  message TEXT,
                  original_value_json TEXT NOT NULL,
                  before_reviewed_value_json TEXT,
                  after_reviewed_value_json TEXT,
                  before_status TEXT NOT NULL,
                  after_status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(candidate_value_id) REFERENCES candidate_values(candidate_value_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(candidate_entity_id) REFERENCES candidate_entities(candidate_entity_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_candidate_value_review_events_run
                  ON candidate_value_review_events(candidate_run_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_candidate_value_review_events_value
                  ON candidate_value_review_events(candidate_value_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS candidate_final_records (
                  final_record_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  template_version TEXT NOT NULL,
                  final_json TEXT NOT NULL,
                  source_candidate_value_ids_json TEXT NOT NULL,
                  confirmed_by TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'confirmed',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  confirmed_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_candidate_final_records_paper
                  ON candidate_final_records(paper_id, template_id);
                CREATE TABLE IF NOT EXISTS mineru_parse_runs (
                  mineru_run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  task_id TEXT,
                  status TEXT NOT NULL,
                  service_base_url TEXT NOT NULL,
                  parser_version TEXT,
                  content_item_count INTEGER NOT NULL DEFAULT 0,
                  result_path TEXT,
                  content_list_path TEXT,
                  markdown_path TEXT,
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_mineru_parse_runs_paper
                  ON mineru_parse_runs(paper_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS llm_mining_runs (
                  llm_run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  template_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  status TEXT NOT NULL,
                  source_parser TEXT NOT NULL,
                  input_item_count INTEGER NOT NULL DEFAULT 0,
                  prompt_path TEXT,
                  raw_response_path TEXT,
                  mining_result_path TEXT,
                  validation_report_path TEXT,
                  candidate_run_id TEXT,
                  error_message TEXT,
                  token_usage_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_llm_mining_runs_paper
                  ON llm_mining_runs(paper_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS materials_global (
                  global_material_id TEXT PRIMARY KEY,
                  canonical_name TEXT,
                  material_class TEXT NOT NULL DEFAULT 'unknown',
                  representation_type TEXT NOT NULL DEFAULT 'unknown',
                  raw_smiles TEXT,
                  canonical_smiles TEXT,
                  isomeric_smiles TEXT,
                  inchi TEXT,
                  inchi_key TEXT,
                  formula TEXT,
                  molecular_weight REAL,
                  source TEXT NOT NULL DEFAULT 'manual',
                  source_detail_json TEXT NOT NULL DEFAULT '{}',
                  confidence REAL,
                  review_status TEXT NOT NULL DEFAULT 'candidate',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  confirmed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_materials_global_inchi_key
                  ON materials_global(inchi_key);
                CREATE TABLE IF NOT EXISTS material_aliases (
                  alias_id TEXT PRIMARY KEY,
                  global_material_id TEXT NOT NULL,
                  alias_text TEXT NOT NULL,
                  normalized_alias TEXT NOT NULL,
                  alias_type TEXT NOT NULL DEFAULT 'unknown',
                  source_paper_id TEXT,
                  source TEXT NOT NULL DEFAULT 'manual',
                  confidence REAL,
                  review_status TEXT NOT NULL DEFAULT 'candidate',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(global_material_id) REFERENCES materials_global(global_material_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_aliases_normalized
                  ON material_aliases(normalized_alias);
                CREATE TABLE IF NOT EXISTS paper_material_links (
                  paper_material_link_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  global_material_id TEXT,
                  match_method TEXT NOT NULL DEFAULT 'none',
                  match_confidence REAL,
                  match_status TEXT NOT NULL DEFAULT 'unresolved',
                  evidence_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  confirmed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(global_material_id) REFERENCES materials_global(global_material_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_material_links_unique
                  ON paper_material_links(candidate_run_id, paper_material_id);
                CREATE TABLE IF NOT EXISTS material_resolution_tasks (
                  task_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  material_mentions_json TEXT NOT NULL,
                  material_context_json TEXT NOT NULL,
                  priority TEXT NOT NULL DEFAULT 'normal',
                  status TEXT NOT NULL DEFAULT 'pending',
                  assigned_strategy TEXT NOT NULL DEFAULT 'unresolved',
                  current_stage TEXT NOT NULL DEFAULT 'unresolved',
                  next_action TEXT NOT NULL DEFAULT 'resolve',
                  retry_count INTEGER NOT NULL DEFAULT 0,
                  stage_timings_json TEXT NOT NULL DEFAULT '{}',
                  stage_errors_json TEXT NOT NULL DEFAULT '{}',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_material_resolution_tasks_unique
                  ON material_resolution_tasks(candidate_run_id, paper_material_id);
                CREATE TABLE IF NOT EXISTS paper_material_name_reviews (
                  review_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  reviewed_name TEXT,
                  reviewed_full_name_in_paper TEXT,
                  reviewed_abbreviation TEXT,
                  reviewed_normalized_name TEXT,
                  reviewed_canonical_name TEXT,
                  review_status TEXT NOT NULL DEFAULT 'corrected',
                  actor TEXT NOT NULL DEFAULT 'local_user',
                  message TEXT,
                  source TEXT NOT NULL DEFAULT 'manual_review',
                  source_detail_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_material_name_reviews_unique
                  ON paper_material_name_reviews(candidate_run_id, paper_material_id);
                CREATE TABLE IF NOT EXISTS paper_material_name_suggestions (
                  suggestion_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  agent_name TEXT NOT NULL DEFAULT 'material_name_agent_v1',
                  original_name TEXT,
                  suggested_name TEXT NOT NULL,
                  suggested_full_name_in_paper TEXT,
                  suggested_abbreviation TEXT,
                  suggested_normalized_name TEXT,
                  suggested_canonical_name TEXT,
                  confidence REAL,
                  reason TEXT,
                  evidence_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'suggested',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_material_name_suggestions_unique
                  ON paper_material_name_suggestions(
                    candidate_run_id, paper_material_id, agent_name, suggested_name
                  );
                CREATE INDEX IF NOT EXISTS idx_paper_material_name_suggestions_run
                  ON paper_material_name_suggestions(candidate_run_id, paper_material_id, status);
                CREATE TABLE IF NOT EXISTS material_structure_candidates (
                  structure_candidate_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  resolver_name TEXT NOT NULL,
                  query_text TEXT NOT NULL,
                  query_type TEXT NOT NULL DEFAULT 'name',
                  source_identifier TEXT,
                  source_url TEXT,
                  canonical_name TEXT,
                  material_class TEXT NOT NULL DEFAULT 'unknown',
                  representation_type TEXT NOT NULL DEFAULT 'small_molecule',
                  raw_smiles TEXT,
                  canonical_smiles TEXT,
                  isomeric_smiles TEXT,
                  inchi TEXT,
                  inchi_key TEXT,
                  formula TEXT,
                  molecular_weight REAL,
                  synonyms_json TEXT NOT NULL DEFAULT '[]',
                  evidence_json TEXT NOT NULL DEFAULT '{}',
                  confidence REAL,
                  status TEXT NOT NULL DEFAULT 'pending_review',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_material_structure_candidates_unique
                  ON material_structure_candidates(
                    candidate_run_id, paper_material_id, provider, source_identifier
                  );
                CREATE INDEX IF NOT EXISTS idx_material_structure_candidates_run
                  ON material_structure_candidates(candidate_run_id, paper_material_id);
                CREATE TABLE IF NOT EXISTS material_identity_judgments (
                  judgment_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  structure_candidate_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  verdict TEXT NOT NULL DEFAULT 'insufficient_evidence',
                  confidence REAL,
                  supporting_evidence_json TEXT NOT NULL DEFAULT '[]',
                  conflicts_json TEXT NOT NULL DEFAULT '[]',
                  recommended_action TEXT NOT NULL DEFAULT 'manual_review',
                  deterministic_checks_json TEXT NOT NULL DEFAULT '{}',
                  input_context_json TEXT NOT NULL DEFAULT '{}',
                  raw_response_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'completed',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(structure_candidate_id)
                    REFERENCES material_structure_candidates(structure_candidate_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_identity_judgments_run
                  ON material_identity_judgments(candidate_run_id, paper_material_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_material_identity_judgments_candidate
                  ON material_identity_judgments(structure_candidate_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS material_identity_evidence_runs (
                  evidence_run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  trigger_judgment_id TEXT,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  strategy TEXT NOT NULL,
                  query_plan_json TEXT NOT NULL DEFAULT '[]',
                  status TEXT NOT NULL DEFAULT 'running',
                  generated_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
                  recommended_next_action TEXT NOT NULL DEFAULT 'manual_review',
                  raw_response_json TEXT NOT NULL DEFAULT '{}',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(trigger_judgment_id)
                    REFERENCES material_identity_judgments(judgment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_identity_evidence_runs_material
                  ON material_identity_evidence_runs(
                    candidate_run_id, paper_material_id, created_at DESC
                  );
                CREATE TABLE IF NOT EXISTS material_identity_evidence_items (
                  evidence_item_id TEXT PRIMARY KEY,
                  evidence_run_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  source_type TEXT NOT NULL DEFAULT 'web_search',
                  source_tier TEXT NOT NULL DEFAULT 'C',
                  source_title TEXT,
                  source_url TEXT,
                  query_text TEXT,
                  excerpt TEXT,
                  alias TEXT,
                  full_name TEXT,
                  cas_number TEXT,
                  pubchem_cid TEXT,
                  explicitly_linked INTEGER NOT NULL DEFAULT 0,
                  confidence REAL,
                  extraction_json TEXT NOT NULL DEFAULT '{}',
                  raw_source_json TEXT NOT NULL DEFAULT '{}',
                  review_status TEXT NOT NULL DEFAULT 'pending_review',
                  reviewed_by TEXT,
                  review_note TEXT,
                  reviewed_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(evidence_run_id)
                    REFERENCES material_identity_evidence_runs(evidence_run_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_identity_evidence_items_material
                  ON material_identity_evidence_items(
                    candidate_run_id, paper_material_id, created_at DESC
                  );
                CREATE INDEX IF NOT EXISTS idx_material_identity_evidence_items_run
                  ON material_identity_evidence_items(evidence_run_id, created_at ASC);
                CREATE TABLE IF NOT EXISTS material_review_events (
                  event_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  structure_candidate_id TEXT,
                  global_material_id TEXT,
                  action TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  message TEXT,
                  before_candidate_status TEXT,
                  after_candidate_status TEXT,
                  before_link_json TEXT,
                  after_link_json TEXT,
                  before_task_json TEXT,
                  after_task_json TEXT,
                  before_candidate_json TEXT,
                  after_candidate_json TEXT,
                  created_global_material_id TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(structure_candidate_id)
                    REFERENCES material_structure_candidates(structure_candidate_id),
                  FOREIGN KEY(global_material_id) REFERENCES materials_global(global_material_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_review_events_run
                  ON material_review_events(candidate_run_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_material_review_events_candidate
                  ON material_review_events(structure_candidate_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS material_property_candidates (
                  property_candidate_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  global_material_id TEXT,
                  property_name TEXT NOT NULL,
                  property_category TEXT NOT NULL DEFAULT 'unknown',
                  value_numeric REAL,
                  value_text TEXT,
                  value_raw TEXT,
                  unit TEXT,
                  normalized_value_numeric REAL,
                  normalized_unit TEXT,
                  condition_json TEXT NOT NULL DEFAULT '{}',
                  method TEXT,
                  source_type TEXT NOT NULL DEFAULT 'unknown',
                  evidence_text TEXT,
                  llm_evidence_text TEXT,
                  source_block_text TEXT,
                  evidence_anchor_json TEXT NOT NULL DEFAULT '{}',
                  provider TEXT NOT NULL DEFAULT 'unknown',
                  model TEXT,
                  prompt_version TEXT NOT NULL DEFAULT 'material_property_miner_v1',
                  confidence REAL,
                  status TEXT NOT NULL DEFAULT 'pending_review',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(global_material_id) REFERENCES materials_global(global_material_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_property_candidates_run
                  ON material_property_candidates(candidate_run_id, paper_material_id, status);
                CREATE INDEX IF NOT EXISTS idx_material_property_candidates_property
                  ON material_property_candidates(property_name, status);
                CREATE TABLE IF NOT EXISTS material_property_reviews (
                  review_id TEXT PRIMARY KEY,
                  property_candidate_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  decision TEXT NOT NULL,
                  reviewed_property_name TEXT NOT NULL,
                  reviewed_value_numeric REAL,
                  reviewed_value_text TEXT,
                  reviewed_unit TEXT,
                  reviewed_condition_json TEXT NOT NULL DEFAULT '{}',
                  reviewed_evidence_anchor_json TEXT NOT NULL DEFAULT '{}',
                  actor TEXT NOT NULL DEFAULT 'local_user',
                  message TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(property_candidate_id)
                    REFERENCES material_property_candidates(property_candidate_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_property_reviews_run
                  ON material_property_reviews(candidate_run_id, paper_material_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_material_property_reviews_candidate
                  ON material_property_reviews(property_candidate_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS material_property_review_events (
                  event_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  paper_material_id TEXT NOT NULL,
                  property_candidate_id TEXT,
                  event_type TEXT NOT NULL,
                  before_json TEXT,
                  after_json TEXT,
                  actor TEXT NOT NULL DEFAULT 'local_user',
                  message TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(property_candidate_id)
                    REFERENCES material_property_candidates(property_candidate_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_property_review_events_run
                  ON material_property_review_events(candidate_run_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_material_property_review_events_candidate
                  ON material_property_review_events(property_candidate_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS chemical_figure_blocks (
                  figure_block_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  mineru_run_id TEXT NOT NULL,
                  content_index INTEGER NOT NULL,
                  content_type TEXT NOT NULL,
                  sub_type TEXT,
                  page_idx INTEGER,
                  page_id INTEGER,
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  img_path TEXT,
                  resolved_img_path TEXT,
                  image_exists INTEGER NOT NULL DEFAULT 0,
                  caption TEXT,
                  nearby_text TEXT,
                  heuristic_tags_json TEXT NOT NULL DEFAULT '[]',
                  confidence REAL,
                  status TEXT NOT NULL DEFAULT 'pending_classification',
                  source_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(mineru_run_id) REFERENCES mineru_parse_runs(mineru_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chemical_figure_blocks_run_content
                  ON chemical_figure_blocks(mineru_run_id, content_index);
                CREATE INDEX IF NOT EXISTS idx_chemical_figure_blocks_paper
                  ON chemical_figure_blocks(paper_id, mineru_run_id, content_index);
                CREATE TABLE IF NOT EXISTS material_agent_runs (
                  agent_run_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  strategy TEXT NOT NULL DEFAULT 'foundation',
                  source_candidate_run_id TEXT,
                  mineru_run_id TEXT,
                  material_count INTEGER NOT NULL DEFAULT 0,
                  visual_block_count INTEGER NOT NULL DEFAULT 0,
                  tool_summary_json TEXT NOT NULL DEFAULT '{}',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(source_candidate_run_id)
                    REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(mineru_run_id) REFERENCES mineru_parse_runs(mineru_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_material_agent_runs_paper
                  ON material_agent_runs(paper_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS document_visual_blocks (
                  visual_block_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  mineru_run_id TEXT NOT NULL,
                  collected_by_agent_run_id TEXT,
                  content_index INTEGER NOT NULL,
                  content_type TEXT NOT NULL,
                  sub_type TEXT,
                  page_idx INTEGER,
                  page_id INTEGER,
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  img_path TEXT,
                  resolved_img_path TEXT,
                  image_exists INTEGER NOT NULL DEFAULT 0,
                  caption TEXT,
                  nearby_text TEXT,
                  source_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(mineru_run_id) REFERENCES mineru_parse_runs(mineru_run_id),
                  FOREIGN KEY(collected_by_agent_run_id)
                    REFERENCES material_agent_runs(agent_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_document_visual_blocks_run_content
                  ON document_visual_blocks(mineru_run_id, content_index);
                CREATE INDEX IF NOT EXISTS idx_document_visual_blocks_paper
                  ON document_visual_blocks(paper_id, mineru_run_id, content_index);
                CREATE TABLE IF NOT EXISTS figure_triage_results (
                  triage_result_id TEXT PRIMARY KEY,
                  agent_run_id TEXT NOT NULL,
                  visual_block_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  contains_molecular_structures INTEGER NOT NULL DEFAULT 0,
                  image_role TEXT NOT NULL DEFAULT 'unknown',
                  has_clean_structure_depictions INTEGER NOT NULL DEFAULT 0,
                  has_orbital_overlay INTEGER NOT NULL DEFAULT 0,
                  has_energy_level_diagram INTEGER NOT NULL DEFAULT 0,
                  has_device_stack INTEGER NOT NULL DEFAULT 0,
                  should_run_decimer_segmentation INTEGER NOT NULL DEFAULT 0,
                  label_candidates_json TEXT NOT NULL DEFAULT '[]',
                  related_paper_material_ids_json TEXT NOT NULL DEFAULT '[]',
                  confidence REAL,
                  reason TEXT,
                  raw_response_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'completed',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(agent_run_id) REFERENCES material_agent_runs(agent_run_id),
                  FOREIGN KEY(visual_block_id) REFERENCES document_visual_blocks(visual_block_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_figure_triage_results_run_block
                  ON figure_triage_results(agent_run_id, visual_block_id);
                CREATE INDEX IF NOT EXISTS idx_figure_triage_results_paper
                  ON figure_triage_results(paper_id, agent_run_id, visual_block_id);
                CREATE TABLE IF NOT EXISTS molecule_crops (
                  crop_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  agent_run_id TEXT NOT NULL,
                  triage_result_id TEXT NOT NULL,
                  visual_block_id TEXT NOT NULL,
                  segment_index INTEGER NOT NULL,
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  source_image_path TEXT NOT NULL,
                  crop_path TEXT NOT NULL,
                  width INTEGER,
                  height INTEGER,
                  segmentation_confidence REAL,
                  validation_json TEXT NOT NULL DEFAULT '{}',
                  raw_segment_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'pending_validation',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(agent_run_id) REFERENCES material_agent_runs(agent_run_id),
                  FOREIGN KEY(triage_result_id) REFERENCES figure_triage_results(triage_result_id),
                  FOREIGN KEY(visual_block_id) REFERENCES document_visual_blocks(visual_block_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_molecule_crops_triage_segment
                  ON molecule_crops(triage_result_id, segment_index);
                CREATE INDEX IF NOT EXISTS idx_molecule_crops_paper
                  ON molecule_crops(paper_id, agent_run_id, visual_block_id);
                CREATE TABLE IF NOT EXISTS molecule_crop_validations (
                  validation_id TEXT PRIMARY KEY,
                  crop_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  agent_run_id TEXT NOT NULL,
                  visual_block_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  is_molecular_depiction INTEGER NOT NULL DEFAULT 0,
                  is_single_molecule INTEGER NOT NULL DEFAULT 0,
                  is_complete_structure INTEGER NOT NULL DEFAULT 0,
                  has_benign_highlight INTEGER NOT NULL DEFAULT 0,
                  is_ocsr_readable INTEGER NOT NULL DEFAULT 0,
                  has_blocking_interference INTEGER NOT NULL DEFAULT 0,
                  has_orbital_overlay INTEGER NOT NULL DEFAULT 0,
                  has_excess_annotation INTEGER NOT NULL DEFAULT 0,
                  has_multiple_structures INTEGER NOT NULL DEFAULT 0,
                  has_reaction_arrow INTEGER NOT NULL DEFAULT 0,
                  has_non_structural_graphics INTEGER NOT NULL DEFAULT 0,
                  should_run_ocsr INTEGER NOT NULL DEFAULT 0,
                  confidence REAL,
                  reason TEXT,
                  raw_response_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'completed',
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(crop_id) REFERENCES molecule_crops(crop_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(agent_run_id) REFERENCES material_agent_runs(agent_run_id),
                  FOREIGN KEY(visual_block_id) REFERENCES document_visual_blocks(visual_block_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_crop_validations_crop_provider_model
                  ON molecule_crop_validations(crop_id, provider, model);
                CREATE INDEX IF NOT EXISTS idx_crop_validations_paper
                  ON molecule_crop_validations(paper_id, agent_run_id, crop_id);
                CREATE TABLE IF NOT EXISTS molecule_label_bindings (
                  binding_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  candidate_run_id TEXT NOT NULL,
                  agent_run_id TEXT NOT NULL,
                  crop_id TEXT NOT NULL,
                  visual_block_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  source_figure_path TEXT NOT NULL,
                  highlighted_source_figure_path TEXT NOT NULL,
                  crop_path TEXT NOT NULL,
                  caption_text TEXT,
                  nearby_text TEXT,
                  triage_label_candidates_json TEXT NOT NULL DEFAULT '[]',
                  candidate_materials_json TEXT NOT NULL DEFAULT '[]',
                  model_observed_label TEXT,
                  model_label_source TEXT NOT NULL DEFAULT 'unknown',
                  model_proposed_paper_material_id TEXT,
                  model_alternative_paper_material_ids_json TEXT NOT NULL DEFAULT '[]',
                  model_decision TEXT NOT NULL DEFAULT 'failed',
                  model_confidence REAL,
                  model_reason TEXT,
                  raw_response_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'completed',
                  error_message TEXT,
                  reviewed_paper_material_id TEXT,
                  reviewed_observed_label TEXT,
                  review_status TEXT NOT NULL DEFAULT 'pending_review',
                  reviewed_by TEXT,
                  reviewed_at TEXT,
                  review_note TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(candidate_run_id) REFERENCES candidate_ingestion_runs(candidate_run_id),
                  FOREIGN KEY(agent_run_id) REFERENCES material_agent_runs(agent_run_id),
                  FOREIGN KEY(crop_id) REFERENCES molecule_crops(crop_id),
                  FOREIGN KEY(visual_block_id) REFERENCES document_visual_blocks(visual_block_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_label_bindings_crop_provider_model
                  ON molecule_label_bindings(crop_id, provider, model);
                CREATE INDEX IF NOT EXISTS idx_label_bindings_paper
                  ON molecule_label_bindings(paper_id, agent_run_id, review_status);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_label_bindings_one_reviewed_crop
                  ON molecule_label_bindings(crop_id)
                  WHERE review_status IN (
                    'confirmed', 'corrected', 'unresolved', 'material_missing',
                    'not_target_material', 'invalid_crop'
                  );
                CREATE TABLE IF NOT EXISTS molecule_label_binding_review_events (
                  event_id TEXT PRIMARY KEY,
                  binding_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  crop_id TEXT NOT NULL,
                  action TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  message TEXT,
                  before_reviewed_paper_material_id TEXT,
                  after_reviewed_paper_material_id TEXT,
                  before_observed_label TEXT,
                  after_observed_label TEXT,
                  before_review_status TEXT NOT NULL,
                  after_review_status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(binding_id) REFERENCES molecule_label_bindings(binding_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(crop_id) REFERENCES molecule_crops(crop_id)
                );
                CREATE INDEX IF NOT EXISTS idx_label_binding_review_events_binding
                  ON molecule_label_binding_review_events(binding_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_label_binding_review_events_paper
                  ON molecule_label_binding_review_events(paper_id, created_at);
                CREATE TABLE IF NOT EXISTS vlm_call_logs (
                  vlm_call_id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL,
                  agent_run_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  input_entity_type TEXT NOT NULL,
                  input_entity_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  prompt_version TEXT NOT NULL,
                  input_image_paths_json TEXT NOT NULL DEFAULT '[]',
                  input_context_json TEXT NOT NULL DEFAULT '{}',
                  parsed_response_json TEXT NOT NULL DEFAULT '{}',
                  usage_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'running',
                  error_message TEXT,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  duration_ms INTEGER,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
                  FOREIGN KEY(agent_run_id) REFERENCES material_agent_runs(agent_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_vlm_call_logs_paper_run
                  ON vlm_call_logs(paper_id, agent_run_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_vlm_call_logs_stage_status
                  ON vlm_call_logs(stage, status, started_at);
                """
            )
            _ensure_column(conn, "material_property_candidates", "llm_evidence_text", "TEXT")
            _ensure_column(conn, "material_property_candidates", "source_block_text", "TEXT")
            _ensure_column(conn, "papers", "review_reason", "TEXT")
            _ensure_column(
                conn,
                "material_resolution_tasks",
                "current_stage",
                "TEXT NOT NULL DEFAULT 'unresolved'",
            )
            _ensure_column(
                conn,
                "material_resolution_tasks",
                "next_action",
                "TEXT NOT NULL DEFAULT 'resolve'",
            )
            _ensure_column(
                conn,
                "material_resolution_tasks",
                "retry_count",
                "INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                conn,
                "material_resolution_tasks",
                "stage_timings_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            _ensure_column(
                conn,
                "material_resolution_tasks",
                "stage_errors_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            _ensure_column(conn, "material_review_events", "before_candidate_json", "TEXT")
            _ensure_column(conn, "material_review_events", "after_candidate_json", "TEXT")
            _ensure_column(
                conn,
                "molecule_crop_validations",
                "has_benign_highlight",
                "INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                conn, "molecule_crop_validations", "is_ocsr_readable", "INTEGER NOT NULL DEFAULT 0"
            )
            _ensure_column(
                conn,
                "molecule_crop_validations",
                "has_blocking_interference",
                "INTEGER NOT NULL DEFAULT 0",
            )
            conn.execute("DROP INDEX IF EXISTS idx_label_bindings_one_reviewed_crop")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_label_bindings_one_reviewed_crop
                  ON molecule_label_bindings(crop_id)
                  WHERE review_status IN (
                    'confirmed', 'corrected', 'unresolved', 'material_missing',
                    'not_target_material', 'invalid_crop'
                  )
                """
            )


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in existing:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
