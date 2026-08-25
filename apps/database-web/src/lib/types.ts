export type DatasetKey = "oled" | "ofet" | "opv";

export type DataRecord = Record<string, unknown>;

export interface DatasetManifest {
  key: DatasetKey;
  label: string;
  record_count: number;
  paper_count: number;
  file: string;
  compressed_bytes: number;
  sha256: string;
  source: string;
  source_url: string | null;
  license: string | null;
  quality_counts: Record<string, number> | null;
}

export interface Catalog {
  schema_version: string;
  generated_at: string;
  datasets: DatasetManifest[];
}

export interface Filters {
  search: string;
  primary: string;
  secondary: string;
  metricMin: string;
  metricMax: string;
  yearMin: string;
  yearMax: string;
}

export interface SortState {
  key: string;
  ascending: boolean;
}

export interface Column {
  key: string;
  label: string;
  numeric?: boolean;
  className?: string;
  render: (record: DataRecord) => React.ReactNode;
  sortValue?: (record: DataRecord) => string | number | null | undefined;
}

export interface DatasetOption {
  value: string;
  label: string;
}
