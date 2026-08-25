"use client";

import {
  Activity,
  ArrowDownAZ,
  Atom,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  Database,
  Download,
  ExternalLink,
  FileText,
  Filter,
  FlaskConical,
  Layers3,
  Lightbulb,
  LoaderCircle,
  RotateCcw,
  Search,
  Sun,
  X,
} from "lucide-react";
import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import { MoleculeCanvas } from "@/components/MoleculeCanvas";
import { loadCatalog, loadDataset } from "@/lib/data";
import {
  csvEscape,
  doiUrl,
  formatBytes,
  formatInteger,
  formatNumber,
  numberValue,
  percent,
  recordText,
  shortText,
  text,
  uniqueValues,
} from "@/lib/format";
import type {
  Catalog,
  Column,
  DataRecord,
  DatasetKey,
  DatasetManifest,
  Filters,
  SortState,
} from "@/lib/types";

const PER_PAGE = 75;
const EMPTY_FILTERS: Filters = {
  search: "",
  primary: "",
  secondary: "",
  metricMin: "",
  metricMax: "",
  yearMin: "",
  yearMax: "",
};

const TAB_META: Record<
  DatasetKey,
  {
    short: string;
    title: string;
    description: string;
    icon: typeof Lightbulb;
    metricLabel: string;
    metricKey: string;
    metricUnit: string;
  }
> = {
  oled: {
    short: "OLED",
    title: "Organic Light-Emitting Diodes",
    description:
      "Variable-layer device architectures, emitting materials, and reported performance.",
    icon: Lightbulb,
    metricLabel: "EQE max",
    metricKey: "eqe_max",
    metricUnit: "%",
  },
  ofet: {
    short: "OFET",
    title: "Organic Field-Effect Transistors",
    description:
      "Semiconductor identity, device geometry, fabrication conditions, and mobility.",
    icon: Activity,
    metricLabel: "Mobility",
    metricKey: "highest_mobility",
    metricUnit: "cm² V⁻¹ s⁻¹",
  },
  opv: {
    short: "OPV",
    title: "Organic Photovoltaics",
    description:
      "Donor–acceptor systems, device processing, photovoltaic metrics, and benchmark tiers.",
    icon: Sun,
    metricLabel: "PCE",
    metricKey: "pce",
    metricUnit: "%",
  },
};

export function DatabaseExplorer() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [active, setActive] = useState<DatasetKey>("oled");
  const [recordsByDataset, setRecordsByDataset] = useState<
    Partial<Record<DatasetKey, DataRecord[]>>
  >({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draftFilters, setDraftFilters] = useState<Filters>(EMPTY_FILTERS);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sort, setSort] = useState<SortState>({
    key: "year",
    ascending: false,
  });
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<DataRecord | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadCatalog(controller.signal)
      .then(setCatalog)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : String(reason));
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, []);

  const manifest = useMemo(
    () => catalog?.datasets.find((dataset) => dataset.key === active) ?? null,
    [active, catalog],
  );

  useEffect(() => {
    if (!manifest || recordsByDataset[active]) {
      if (manifest) setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    loadDataset(active, manifest.file, controller.signal)
      .then((records) => {
        setRecordsByDataset((current) => ({ ...current, [active]: records }));
        setLoading(false);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : String(reason));
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [active, manifest, recordsByDataset]);

  const records = recordsByDataset[active] ?? [];
  const filterMeta = useMemo(() => buildFilterMeta(active, records), [active, records]);
  const columns = useMemo(() => columnsFor(active), [active]);

  const filtered = useMemo(() => {
    const result = records.filter((record) =>
      matchesFilters(active, record, filters),
    );
    const column = columns.find((item) => item.key === sort.key);
    if (!column) return result;
    return result.sort((left, right) => {
      const leftValue = column.sortValue
        ? column.sortValue(left)
        : scalarValue(left[column.key]);
      const rightValue = column.sortValue
        ? column.sortValue(right)
        : scalarValue(right[column.key]);
      const leftMissing =
        leftValue === null || leftValue === undefined || leftValue === "";
      const rightMissing =
        rightValue === null || rightValue === undefined || rightValue === "";
      if (leftMissing && rightMissing) return 0;
      if (leftMissing) return 1;
      if (rightMissing) return -1;
      const comparison = compareValues(leftValue, rightValue);
      return sort.ascending ? comparison : -comparison;
    });
  }, [active, columns, filters, records, sort]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PER_PAGE));
  const safePage = Math.min(page, pageCount);
  const visibleRecords = filtered.slice(
    (safePage - 1) * PER_PAGE,
    safePage * PER_PAGE,
  );

  const changeTab = useCallback((key: DatasetKey) => {
    setActive(key);
    setDraftFilters(EMPTY_FILTERS);
    setFilters(EMPTY_FILTERS);
    setPage(1);
    setSelected(null);
    setSort({ key: key === "opv" ? "pce" : key === "ofet" ? "highest_mobility" : "year", ascending: false });
  }, []);

  const applyFilters = () => {
    setFilters(draftFilters);
    setPage(1);
  };

  const clearFilters = () => {
    setDraftFilters(EMPTY_FILTERS);
    setFilters(EMPTY_FILTERS);
    setPage(1);
  };

  const toggleSort = (key: string) => {
    setSort((current) =>
      current.key === key
        ? { key, ascending: !current.ascending }
        : { key, ascending: !isNumericColumn(columns, key) },
    );
  };

  return (
    <div className={`site-shell theme-${active}`}>
      <header className="site-header">
        <div className="header-main">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true">
              <Atom size={27} />
            </div>
            <div>
              <h1>EvoOptoDB</h1>
              <p>Organic optoelectronic device database</p>
            </div>
          </div>
          <div className="header-status">
            <span className="release-badge">
              <Database size={14} />
              Static research release
            </span>
            {catalog ? (
              <span className="release-date">
                Generated {new Date(catalog.generated_at).toLocaleDateString()}
              </span>
            ) : null}
          </div>
        </div>
        <nav className="dataset-tabs" aria-label="Device database">
          {(Object.keys(TAB_META) as DatasetKey[]).map((key) => {
            const item = TAB_META[key];
            const Icon = item.icon;
            const itemManifest = catalog?.datasets.find(
              (dataset) => dataset.key === key,
            );
            return (
              <button
                key={key}
                type="button"
                className={active === key ? "active" : ""}
                aria-current={active === key ? "page" : undefined}
                onClick={() => changeTab(key)}
              >
                <Icon size={17} aria-hidden="true" />
                <span>{item.short}</span>
                <small>
                  {itemManifest
                    ? itemManifest.record_count.toLocaleString()
                    : "…"}
                </small>
              </button>
            );
          })}
        </nav>
      </header>

      <main className="content">
        <DatasetHeading active={active} manifest={manifest} />

        {loading ? (
          <LoadingPanel active={active} manifest={manifest} />
        ) : error ? (
          <section className="error-panel" role="alert">
            <strong>Dataset could not be loaded</strong>
            <span>{error}</span>
          </section>
        ) : (
          <>
            <StatsStrip active={active} records={records} manifest={manifest} />
            <FilterPanel
              active={active}
              draft={draftFilters}
              meta={filterMeta}
              onChange={setDraftFilters}
              onApply={applyFilters}
              onClear={clearFilters}
              onExport={() => exportCsv(active, filtered)}
            />
            <SummaryBands active={active} records={records} />
            <section className="records-panel">
              <div className="panel-heading">
                <div>
                  <h2>Device records</h2>
                  <p>
                    Showing {filtered.length.toLocaleString()} of{" "}
                    {records.length.toLocaleString()} records
                  </p>
                </div>
                <span className="result-badge">{visibleRecords.length} on page</span>
              </div>
              <RecordTable
                columns={columns}
                records={visibleRecords}
                sort={sort}
                onSort={toggleSort}
                onSelect={setSelected}
              />
              <Pagination
                page={safePage}
                pageCount={pageCount}
                total={filtered.length}
                onChange={setPage}
              />
            </section>
          </>
        )}
      </main>

      <SiteFooter manifest={manifest} />
      {selected ? (
        <DetailDrawer
          active={active}
          record={selected}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </div>
  );
}

function DatasetHeading({
  active,
  manifest,
}: {
  active: DatasetKey;
  manifest: DatasetManifest | null;
}) {
  const meta = TAB_META[active];
  return (
    <section className="dataset-heading">
      <div>
        <span className="eyebrow">{meta.short} database</span>
        <h2>{meta.title}</h2>
        <p>{meta.description}</p>
      </div>
      {manifest ? (
        <div className="source-note">
          <FileText size={17} aria-hidden="true" />
          <div>
            <strong>{manifest.source}</strong>
            <span>
              {manifest.paper_count.toLocaleString()} source papers ·{" "}
              {formatBytes(manifest.compressed_bytes)} compressed
            </span>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function LoadingPanel({
  active,
  manifest,
}: {
  active: DatasetKey;
  manifest: DatasetManifest | null;
}) {
  return (
    <section className="loading-panel" aria-live="polite">
      <LoaderCircle className="spin" size={25} aria-hidden="true" />
      <div>
        <strong>Loading {TAB_META[active].short} records</strong>
        <span>
          {manifest
            ? `Downloading ${formatBytes(manifest.compressed_bytes)} compressed data…`
            : "Reading release catalog…"}
        </span>
      </div>
    </section>
  );
}

function StatsStrip({
  active,
  records,
  manifest,
}: {
  active: DatasetKey;
  records: DataRecord[];
  manifest: DatasetManifest | null;
}) {
  const stats = statsFor(active, records, manifest);
  return (
    <section className="stats-strip" aria-label={`${active.toUpperCase()} summary`}>
      {stats.map((stat, index) => (
        <div className={`stat-block tone-${index % 6}`} key={stat.label}>
          <span>{stat.label}</span>
          <strong>{stat.value}</strong>
          <small>{stat.note}</small>
        </div>
      ))}
    </section>
  );
}

function FilterPanel({
  active,
  draft,
  meta,
  onChange,
  onApply,
  onClear,
  onExport,
}: {
  active: DatasetKey;
  draft: Filters;
  meta: ReturnType<typeof buildFilterMeta>;
  onChange: (filters: Filters) => void;
  onApply: () => void;
  onClear: () => void;
  onExport: () => void;
}) {
  const device = TAB_META[active];
  const update = (key: keyof Filters, value: string) =>
    onChange({ ...draft, [key]: value });
  return (
    <section className="filter-panel">
      <div className="panel-heading compact">
        <div>
          <h2>
            <Filter size={16} aria-hidden="true" />
            Filters
          </h2>
          <p>Search and narrow the static release.</p>
        </div>
        <div className="panel-actions">
          <button className="button secondary" type="button" onClick={onExport}>
            <Download size={15} aria-hidden="true" />
            Filtered CSV
          </button>
          <button className="button ghost" type="button" onClick={onClear}>
            <RotateCcw size={15} aria-hidden="true" />
            Clear
          </button>
        </div>
      </div>
      <div className="filter-grid">
        <label className="filter-field search-field">
          <span>Search</span>
          <div className="input-with-icon">
            <Search size={15} aria-hidden="true" />
            <input
              type="search"
              value={draft.search}
              placeholder={meta.searchPlaceholder}
              onChange={(event) => update("search", event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && onApply()}
            />
          </div>
        </label>
        <SelectField
          label={meta.primaryLabel}
          value={draft.primary}
          options={meta.primaryOptions}
          onChange={(value) => update("primary", value)}
        />
        <SelectField
          label={meta.secondaryLabel}
          value={draft.secondary}
          options={meta.secondaryOptions}
          onChange={(value) => update("secondary", value)}
        />
        <label className="filter-field">
          <span>{device.metricLabel} min</span>
          <input
            type="number"
            value={draft.metricMin}
            placeholder="Any"
            onChange={(event) => update("metricMin", event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onApply()}
          />
        </label>
        <label className="filter-field">
          <span>{device.metricLabel} max</span>
          <input
            type="number"
            value={draft.metricMax}
            placeholder="Any"
            onChange={(event) => update("metricMax", event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onApply()}
          />
        </label>
        <label className="filter-field">
          <span>Year from</span>
          <input
            type="number"
            value={draft.yearMin}
            placeholder="Any"
            onChange={(event) => update("yearMin", event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onApply()}
          />
        </label>
        <label className="filter-field">
          <span>Year to</span>
          <input
            type="number"
            value={draft.yearMax}
            placeholder="Any"
            onChange={(event) => update("yearMax", event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onApply()}
          />
        </label>
        <button className="button primary apply-button" type="button" onClick={onApply}>
          <Search size={15} aria-hidden="true" />
          Apply
        </button>
      </div>
    </section>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function SummaryBands({
  active,
  records,
}: {
  active: DatasetKey;
  records: DataRecord[];
}) {
  const [leftTitle, leftRows, rightTitle, rightRows] = summaryFor(active, records);
  return (
    <section className="summary-layout">
      <SummaryBand title={leftTitle} rows={leftRows} />
      <SummaryBand title={rightTitle} rows={rightRows} />
    </section>
  );
}

function SummaryBand({
  title,
  rows,
}: {
  title: string;
  rows: Array<{ label: string; count: number; total: number; tone?: string }>;
}) {
  return (
    <section className="summary-band">
      <h2>{title}</h2>
      <div className="bar-list">
        {rows.map((row) => {
          const width = row.total ? Math.min(100, (row.count / row.total) * 100) : 0;
          return (
            <div className="bar-row" key={row.label}>
              <span title={row.label}>{row.label}</span>
              <div className="bar-track">
                <div
                  className={`bar-fill ${row.tone ?? ""}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <strong>{row.count.toLocaleString()}</strong>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function RecordTable({
  columns,
  records,
  sort,
  onSort,
  onSelect,
}: {
  columns: Column[];
  records: DataRecord[];
  sort: SortState;
  onSort: (key: string) => void;
  onSelect: (record: DataRecord) => void;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                className={`${column.numeric ? "numeric" : ""} ${column.className ?? ""}`}
                aria-sort={
                  sort.key === column.key
                    ? sort.ascending
                      ? "ascending"
                      : "descending"
                    : "none"
                }
              >
                <button type="button" onClick={() => onSort(column.key)}>
                  {column.label}
                  <ArrowDownAZ
                    size={13}
                    className={sort.key === column.key ? "active" : ""}
                    aria-hidden="true"
                  />
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.length ? (
            records.map((record, index) => (
              <tr
                key={text(record.id) || `${text(record.doi)}-${index}`}
                tabIndex={0}
                onClick={() => onSelect(record)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(record);
                  }
                }}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={`${column.numeric ? "numeric" : ""} ${column.className ?? ""}`}
                  >
                    {column.render(record)}
                  </td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td className="empty-row" colSpan={columns.length}>
                No records match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function Pagination({
  page,
  pageCount,
  total,
  onChange,
}: {
  page: number;
  pageCount: number;
  total: number;
  onChange: (page: number) => void;
}) {
  const pages: number[] = [];
  for (let value = Math.max(1, page - 2); value <= Math.min(pageCount, page + 2); value += 1) {
    pages.push(value);
  }
  return (
    <div className="pagination">
      <span>
        Page {page.toLocaleString()} of {pageCount.toLocaleString()} ·{" "}
        {total.toLocaleString()} records
      </span>
      <div>
        <button
          type="button"
          title="Previous page"
          disabled={page <= 1}
          onClick={() => onChange(Math.max(1, page - 1))}
        >
          <ChevronLeft size={16} aria-hidden="true" />
        </button>
        {pages.map((value) => (
          <button
            type="button"
            key={value}
            className={value === page ? "active" : ""}
            onClick={() => onChange(value)}
          >
            {value}
          </button>
        ))}
        <button
          type="button"
          title="Next page"
          disabled={page >= pageCount}
          onClick={() => onChange(Math.min(pageCount, page + 1))}
        >
          <ChevronRight size={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

function DetailDrawer({
  active,
  record,
  onClose,
}: {
  active: DatasetKey;
  record: DataRecord;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.classList.add("drawer-open");
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("drawer-open");
    };
  }, [onClose]);

  const title =
    active === "oled"
      ? text(record.device_label) || "OLED device"
      : active === "ofet"
        ? text(record.semiconductor) || "OFET record"
        : `${text(record.donor) || "Unknown donor"} / ${text(record.acceptor) || "Unknown acceptor"}`;
  const url = doiUrl(record.doi);

  return (
    <div className="drawer-overlay" role="presentation" onMouseDown={onClose}>
      <aside
        className="detail-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`${TAB_META[active].short} record details`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="drawer-header">
          <div>
            <span>{TAB_META[active].short} record</span>
            <h2>{title}</h2>
            {url ? (
              <a href={url} target="_blank" rel="noopener noreferrer">
                {text(record.doi)}
                <ExternalLink size={13} aria-hidden="true" />
              </a>
            ) : null}
          </div>
          <button className="icon-button" type="button" title="Close details" onClick={onClose}>
            <X size={20} aria-hidden="true" />
          </button>
        </header>
        <div className="drawer-content">
          {active === "oled" ? <OledDetail record={record} /> : null}
          {active === "ofet" ? <OfetDetail record={record} /> : null}
          {active === "opv" ? <OpvDetail record={record} /> : null}
        </div>
      </aside>
    </div>
  );
}

function OledDetail({ record }: { record: DataRecord }) {
  const layers = arrayOfObjects(record.layers);
  const performance = arrayOfObjects(record.performance);
  const materials = arrayOfObjects(record.materials);
  return (
    <>
      <DetailSection title="Paper and device">
        <DetailGrid
          items={[
            ["Quality tier", qualityPill(record.quality_tier)],
            ["Title", text(record.title) || "—"],
            ["Journal", text(record.journal) || "—"],
            ["Year", text(record.year) || "—"],
            ["Device type", humanize(record.device_type)],
            ["Emission color", humanize(record.emission_color)],
            ["Mechanism", humanize(record.emission_mechanism)],
            ["Fabrication", humanize(record.fabrication_method)],
            ["Final emitter", text(record.final_emitter) || "—"],
            ["Layers / materials", `${formatInteger(record.layer_count)} / ${formatInteger(record.material_count)}`],
          ]}
        />
        <div className="architecture-block">
          <span>Device architecture</span>
          <p>{text(record.architecture) || "Not reported"}</p>
        </div>
      </DetailSection>
      <DetailSection title="Key performance">
        <MetricRow
          items={[
            ["EQE max", record.eqe_max, "%"],
            ["CE max", record.ce_max, "cd/A"],
            ["PE max", record.pe_max, "lm/W"],
            ["Luminance max", record.luminance_max, "cd/m²"],
            ["Turn-on", record.turn_on_voltage, "V"],
            ["EL peak", record.el_peak, "nm"],
            ["FWHM", record.fwhm, "nm"],
          ]}
        />
      </DetailSection>
      <DetailSection title={`Layer stack (${layers.length})`}>
        <div className="layer-stack">
          {layers.map((layer, index) => (
            <div className="layer-row" key={`${text(layer.layer_index)}-${index}`}>
              <div className="layer-order">{text(layer.layer_index) || index + 1}</div>
              <div>
                <strong>{text(layer.layer_role) || "Unknown layer"}</strong>
                <span>{text(layer.layer_name) || "—"}</span>
              </div>
              <div className="layer-materials">
                {arrayOfObjects(layer.components)
                  .map((component) => {
                    const ratio = objectValue(component.ratio);
                    return [
                      text(component.material_mention),
                      ratio ? text(ratio.raw) || `${text(ratio.value)} ${text(ratio.unit)}` : "",
                    ]
                      .filter(Boolean)
                      .join(" · ");
                  })
                  .filter(Boolean)
                  .join("; ") || "—"}
              </div>
              <div className="layer-thickness">
                {measurement(layer.thickness)}
              </div>
            </div>
          ))}
        </div>
      </DetailSection>
      <DetailSection title={`Reported performance (${performance.length})`}>
        <div className="data-table-wrap">
          <table className="detail-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Statistic</th>
                <th>Value</th>
                <th>Condition</th>
              </tr>
            </thead>
            <tbody>
              {performance.map((item, index) => (
                <tr key={`${text(item.metric_name)}-${index}`}>
                  <td>{text(item.metric_name) || text(item.metric_family) || "—"}</td>
                  <td>{humanize(item.statistic)}</td>
                  <td>
                    {text(item.normalized_value) || text(item.raw_value) || "—"}{" "}
                    {text(item.normalized_unit) || text(item.raw_unit)}
                  </td>
                  <td>{conditionText(item.condition)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DetailSection>
      <DetailSection title={`Device materials (${materials.length})`}>
        <div className="material-list">
          {materials.map((material, index) => (
            <div className="material-row" key={`${text(material.paper_material_id)}-${index}`}>
              <div>
                <strong>
                  {text(material.canonical_name) ||
                    text(material.full_name) ||
                    text(material.mention) ||
                    text(material.abbreviation) ||
                    text(material.paper_material_id)}
                </strong>
                <span>
                  {[text(material.paper_material_id), humanize(material.material_class)]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </div>
              {material.canonical_smiles || material.isomeric_smiles ? (
                <span className="status-pill success">
                  <CheckCircle2 size={13} />
                  Structure resolved
                </span>
              ) : (
                <span className="status-pill muted">Identity only</span>
              )}
            </div>
          ))}
        </div>
      </DetailSection>
      <MoleculeCanvas
        label={`${text(record.final_emitter) || "Final emitter"} structure`}
        smiles={text(record.final_emitter_smiles) || null}
      />
    </>
  );
}

function OfetDetail({ record }: { record: DataRecord }) {
  return (
    <>
      <MoleculeCanvas
        label={text(record.semiconductor) || "Organic semiconductor"}
        smiles={text(record.smiles) || null}
      />
      <DetailSection title="Semiconductor and performance">
        <DetailGrid
          items={[
            ["Material", text(record.semiconductor) || "—"],
            ["Category", text(record.major_category) || "—"],
            ["Sub-category", text(record.sub_category) || "—"],
            ["Conduction", text(record.conduction_type) || "—"],
            ["Mobility reported", text(record.mobility) || "—"],
            ["Highest mobility", `${formatNumber(record.highest_mobility, 4)} cm² V⁻¹ s⁻¹`],
            ["On/off ratio", text(record.on_off_ratio) || "—"],
            ["Threshold voltage", text(record.threshold_voltage) || "—"],
          ]}
        />
      </DetailSection>
      <DetailSection title="Device and fabrication">
        <DetailGrid
          items={[
            ["Geometry", text(record.device_geometry) || "—"],
            ["Fabrication", text(record.fabrication_method) || "—"],
            ["Organic layer thickness", text(record.organic_layer_thickness) || "—"],
            ["Dielectric", text(record.dielectric_layer) || "—"],
            ["Dielectric thickness", unitValue(record.dielectric_thickness, "nm")],
            ["Source / drain", [text(record.source_electrode), text(record.drain_electrode)].filter(Boolean).join(" / ") || "—"],
            ["Gate electrode", text(record.gate_electrode) || "—"],
            ["Atmosphere", text(record.test_atmosphere) || "—"],
          ]}
        />
        <div className="architecture-block">
          <span>Fabrication details</span>
          <p>{text(record.fabrication_details) || "Not reported"}</p>
        </div>
      </DetailSection>
      <DetailSection title="Publication">
        <DetailGrid
          items={[
            ["Journal", text(record.journal) || "—"],
            ["Publisher", text(record.publisher) || "—"],
            ["Year", text(record.year) || "—"],
            ["Dataset file", text(record.pdf) || "—"],
          ]}
        />
      </DetailSection>
    </>
  );
}

function OpvDetail({ record }: { record: DataRecord }) {
  return (
    <>
      <div className="molecule-grid">
        <MoleculeCanvas
          label={`Donor · ${text(record.donor) || "Unknown"}`}
          smiles={text(record.donor_smiles) || null}
        />
        <MoleculeCanvas
          label={`Acceptor · ${text(record.acceptor) || "Unknown"}`}
          smiles={text(record.acceptor_smiles) || null}
        />
      </div>
      <DetailSection title="Photovoltaic performance">
        <MetricRow
          items={[
            ["PCE", record.pce, "%"],
            ["Voc", record.voc, "V"],
            ["Jsc", record.jsc, "mA/cm²"],
            ["FF", record.ff, "%"],
            ["PCE recomputed", record.pce_recomputed, "%"],
            ["PCE error", record.pce_relative_error_percent, "%"],
          ]}
        />
      </DetailSection>
      <DetailSection title="Materials and processing">
        <DetailGrid
          items={[
            ["Benchmark", qualityPill(record.primary_layer)],
            ["Donor", text(record.donor) || "—"],
            ["Acceptor", text(record.acceptor) || "—"],
            ["D:A ratio", text(record.d_a_ratio) || "—"],
            ["Solvent", text(record.solvent) || "—"],
            ["Additive", [text(record.additive), text(record.additive_ratio)].filter(Boolean).join(" · ") || "—"],
            ["Active-layer thickness", unitValue(record.active_layer_thickness, "nm")],
            ["Annealing", unitValue(record.annealing_temp, "°C")],
            ["ETL", text(record.etl) || "—"],
            ["HTL", text(record.htl) || "—"],
          ]}
        />
        <div className="architecture-block">
          <span>Device structure</span>
          <p>{text(record.device_structure) || "Not reported"}</p>
        </div>
      </DetailSection>
      <DetailSection title="Energy levels">
        <DetailGrid
          items={[
            ["Donor HOMO / LUMO", `${formatNumber(record.homo_d)} / ${formatNumber(record.lumo_d)} eV`],
            ["Donor Eg", unitValue(record.eg_d, "eV")],
            ["Acceptor HOMO / LUMO", `${formatNumber(record.homo_a)} / ${formatNumber(record.lumo_a)} eV`],
            ["Acceptor Eg", unitValue(record.eg_a, "eV")],
          ]}
        />
      </DetailSection>
    </>
  );
}

function DetailSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="detail-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function DetailGrid({
  items,
}: {
  items: Array<[string, React.ReactNode]>;
}) {
  return (
    <dl className="detail-grid">
      {items.map(([label, value]) => (
        <div key={label} className={text(value).length > 90 ? "wide" : ""}>
          <dt>{label}</dt>
          <dd>{value || "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

function MetricRow({
  items,
}: {
  items: Array<[string, unknown, string]>;
}) {
  return (
    <div className="metric-row">
      {items.map(([label, value, unit]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{formatNumber(value)}</strong>
          <small>{numberValue(value) === null ? "" : unit}</small>
        </div>
      ))}
    </div>
  );
}

function SiteFooter({
  manifest,
}: {
  manifest: DatasetManifest | null;
}) {
  return (
    <footer className="site-footer">
      <div>
        <strong>EvoOptoDB</strong>
        <span>
          Static device browser. Source DOIs are retained for record-level provenance.
        </span>
      </div>
      <p>
        {manifest?.source}
        {manifest?.source_url ? (
          <>
            {" · "}
            <a
              href={manifest.source_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Source
            </a>
          </>
        ) : null}
        {manifest?.license ? ` · ${manifest.license}` : null}
      </p>
    </footer>
  );
}

function statsFor(
  active: DatasetKey,
  records: DataRecord[],
  manifest: DatasetManifest | null,
) {
  if (active === "oled") {
    const finalized = records.filter(
      (record) => record.quality_tier === "human_finalized",
    ).length;
    const withEqe = records.filter(
      (record) => numberValue(record.eqe_max) !== null,
    ).length;
    const withStructure = records.filter((record) =>
      arrayOfObjects(record.materials).some(
        (material) => material.canonical_smiles || material.isomeric_smiles,
      ),
    ).length;
    const maxEqe = maxNumber(records, "eqe_max");
    return [
      {
        label: "Device records",
        value: records.length.toLocaleString(),
        note: "variable-layer OLEDs",
      },
      {
        label: "Source papers",
        value: (manifest?.paper_count ?? 0).toLocaleString(),
        note: "confirmed archive",
      },
      {
        label: "Human finalized",
        value: finalized.toLocaleString(),
        note: percent(finalized, records.length),
      },
      {
        label: "Auto reviewed",
        value: (records.length - finalized).toLocaleString(),
        note: "materials completed",
      },
      {
        label: "EQE coverage",
        value: withEqe.toLocaleString(),
        note: percent(withEqe, records.length),
      },
      {
        label: "Top EQE",
        value: maxEqe === null ? "—" : `${maxEqe.toFixed(1)}%`,
        note: `${withStructure.toLocaleString()} devices with structures`,
      },
    ];
  }
  if (active === "ofet") {
    const withSmiles = countTruthy(records, "has_smiles");
    const pType = records.filter((record) =>
      text(record.conduction_category).toLowerCase().includes("p-type"),
    ).length;
    const nType = records.filter((record) =>
      text(record.conduction_category).toLowerCase().includes("n-type"),
    ).length;
    const maxMobility = maxNumber(records, "highest_mobility");
    return [
      {
        label: "Device records",
        value: records.length.toLocaleString(),
        note: "OFET measurements",
      },
      {
        label: "Source DOIs",
        value: uniqueCount(records, "doi").toLocaleString(),
        note: `${uniqueCount(records, "journal")} journals`,
      },
      {
        label: "Structures",
        value: withSmiles.toLocaleString(),
        note: percent(withSmiles, records.length),
      },
      {
        label: "p-type",
        value: pType.toLocaleString(),
        note: percent(pType, records.length),
      },
      {
        label: "n-type",
        value: nType.toLocaleString(),
        note: percent(nType, records.length),
      },
      {
        label: "Top mobility",
        value: maxMobility === null ? "—" : maxMobility.toLocaleString(),
        note: "cm² V⁻¹ s⁻¹",
      },
    ];
  }
  const strictPerformance = countTruthy(records, "strict_performance_benchmark");
  const strictMolecular = countTruthy(records, "strict_molecular_benchmark");
  const allMetrics = countTruthy(records, "has_all_four_metrics");
  const smiles = countTruthy(records, "has_both_smiles");
  return [
    {
      label: "Full archive",
      value: records.length.toLocaleString(),
      note: "publication-facing records",
    },
    {
      label: "Source DOIs",
      value: (manifest?.paper_count ?? 0).toLocaleString(),
      note: "record-level provenance",
    },
    {
      label: "Strict performance",
      value: strictPerformance.toLocaleString(),
      note: percent(strictPerformance, records.length),
    },
    {
      label: "Strict molecular",
      value: strictMolecular.toLocaleString(),
      note: percent(strictMolecular, records.length),
    },
    {
      label: "All four metrics",
      value: allMetrics.toLocaleString(),
      note: percent(allMetrics, records.length),
    },
    {
      label: "Paired SMILES",
      value: smiles.toLocaleString(),
      note: percent(smiles, records.length),
    },
  ];
}

function summaryFor(
  active: DatasetKey,
  records: DataRecord[],
): [
  string,
  Array<{ label: string; count: number; total: number; tone?: string }>,
  string,
  Array<{ label: string; count: number; total: number; tone?: string }>,
] {
  const total = records.length;
  if (active === "oled") {
    const colors = topCounts(records, "emission_color", 5);
    return [
      "Review tiers",
      [
        {
          label: "Auto-reviewed archive",
          count: records.filter((record) => record.quality_tier === "auto_reviewed").length,
          total,
          tone: "teal",
        },
        {
          label: "Human-finalized benchmark",
          count: records.filter((record) => record.quality_tier === "human_finalized").length,
          total,
          tone: "green",
        },
        {
          label: "Architecture reported",
          count: countFilled(records, "architecture"),
          total,
          tone: "blue",
        },
        {
          label: "EQE max reported",
          count: countFilled(records, "eqe_max"),
          total,
          tone: "amber",
        },
      ],
      "Emission colors",
      colors.map(([label, count], index) => ({
        label: humanize(label),
        count,
        total: colors[0]?.[1] ?? total,
        tone: ["blue", "teal", "green", "amber", "coral"][index],
      })),
    ];
  }
  if (active === "ofet") {
    const categories = topCounts(records, "major_category", 5);
    return [
      "Conduction types",
      topCounts(records, "conduction_category", 5).map(([label, count], index) => ({
        label,
        count,
        total,
        tone: ["teal", "blue", "amber", "green", "coral"][index],
      })),
      "Data completeness",
      [
        { label: "Molecular structure", count: countFilled(records, "smiles"), total, tone: "green" },
        { label: "Numeric mobility", count: countFilled(records, "highest_mobility"), total, tone: "teal" },
        { label: "Device geometry", count: countFilled(records, "device_geometry"), total, tone: "blue" },
        { label: "Fabrication method", count: countFilled(records, "fabrication_method"), total, tone: "amber" },
        { label: categories[0]?.[0] ?? "Top category", count: categories[0]?.[1] ?? 0, total, tone: "coral" },
      ],
    ];
  }
  return [
    "Benchmark layers",
    [
      { label: "Full archive", count: total, total, tone: "blue" },
      { label: "All four metrics", count: countTruthy(records, "has_all_four_metrics"), total, tone: "teal" },
      { label: "Strict performance", count: countTruthy(records, "strict_performance_benchmark"), total, tone: "green" },
      { label: "Strict molecular", count: countTruthy(records, "strict_molecular_benchmark"), total, tone: "amber" },
    ],
    "Validation and completeness",
    [
      {
        label: "PCE error ≤2%",
        count: records.filter((record) => record.pce_error_bucket === "<=2%").length,
        total,
        tone: "green",
      },
      { label: "Both SMILES", count: countTruthy(records, "has_both_smiles"), total, tone: "teal" },
      { label: "Device stack", count: countFilled(records, "device_structure"), total, tone: "blue" },
      { label: "Solvent", count: countFilled(records, "solvent"), total, tone: "amber" },
      { label: "Additive", count: countFilled(records, "additive"), total, tone: "coral" },
    ],
  ];
}

function buildFilterMeta(active: DatasetKey, records: DataRecord[]) {
  if (active === "oled") {
    return {
      searchPlaceholder: "DOI, title, journal, emitter, material, device stack",
      primaryLabel: "Review tier",
      primaryOptions: [
        { value: "human_finalized", label: "Human finalized" },
        { value: "auto_reviewed", label: "Auto reviewed" },
      ],
      secondaryLabel: "Emission color",
      secondaryOptions: uniqueValues(records, "emission_color").map((value) => ({
        value,
        label: humanize(value),
      })),
    };
  }
  if (active === "ofet") {
    return {
      searchPlaceholder: "DOI, semiconductor, journal, fabrication, dielectric",
      primaryLabel: "Conduction",
      primaryOptions: uniqueValues(records, "conduction_category").map((value) => ({
        value,
        label: value,
      })),
      secondaryLabel: "Geometry",
      secondaryOptions: uniqueValues(records, "geometry_category").map((value) => ({
        value,
        label: value,
      })),
    };
  }
  return {
    searchPlaceholder: "DOI, donor, acceptor, SMILES, solvent, device stack",
    primaryLabel: "Benchmark",
    primaryOptions: [
      { value: "strict_performance", label: "Strict performance" },
      { value: "strict_molecular", label: "Strict molecular" },
      { value: "full_archive_only", label: "Full archive only" },
    ],
    secondaryLabel: "SMILES",
    secondaryOptions: [
      { value: "both", label: "Donor and acceptor" },
      { value: "missing", label: "Missing one or both" },
    ],
  };
}

function matchesFilters(
  active: DatasetKey,
  record: DataRecord,
  filters: Filters,
): boolean {
  const query = filters.search.trim().toLocaleLowerCase();
  if (query && !searchText(active, record).includes(query)) return false;
  const year = numberValue(active === "opv" ? record.publish_year : record.year);
  const yearMin = numberValue(filters.yearMin);
  const yearMax = numberValue(filters.yearMax);
  if (yearMin !== null && (year === null || year < yearMin)) return false;
  if (yearMax !== null && (year === null || year > yearMax)) return false;
  const metric = numberValue(record[TAB_META[active].metricKey]);
  const metricMin = numberValue(filters.metricMin);
  const metricMax = numberValue(filters.metricMax);
  if (metricMin !== null && (metric === null || metric < metricMin)) return false;
  if (metricMax !== null && (metric === null || metric > metricMax)) return false;

  if (active === "oled") {
    if (filters.primary && record.quality_tier !== filters.primary) return false;
    if (filters.secondary && record.emission_color !== filters.secondary) return false;
  } else if (active === "ofet") {
    if (filters.primary && record.conduction_category !== filters.primary) return false;
    if (filters.secondary && record.geometry_category !== filters.secondary) return false;
  } else {
    if (filters.primary === "strict_performance" && !record.strict_performance_benchmark) return false;
    if (filters.primary === "strict_molecular" && !record.strict_molecular_benchmark) return false;
    if (filters.primary === "full_archive_only" && record.strict_performance_benchmark) return false;
    if (filters.secondary === "both" && !record.has_both_smiles) return false;
    if (filters.secondary === "missing" && record.has_both_smiles) return false;
  }
  return true;
}

function searchText(active: DatasetKey, record: DataRecord): string {
  if (active === "oled") {
    const materials = arrayOfObjects(record.materials)
      .flatMap((material) => [
        material.mention,
        material.full_name,
        material.canonical_name,
        material.abbreviation,
        material.canonical_smiles,
      ])
      .map(text)
      .join(" ");
    return `${recordText(record, [
      "doi",
      "title",
      "journal",
      "device_label",
      "architecture",
      "final_emitter",
      "emission_color",
      "emission_mechanism",
    ])} ${materials.toLocaleLowerCase()}`;
  }
  if (active === "ofet") {
    return recordText(record, [
      "doi",
      "semiconductor",
      "smiles",
      "journal",
      "publisher",
      "fabrication_method",
      "fabrication_details",
      "dielectric_layer",
      "device_geometry",
      "major_category",
      "sub_category",
    ]);
  }
  return recordText(record, [
    "doi",
    "donor",
    "acceptor",
    "donor_canonical",
    "acceptor_canonical",
    "donor_smiles",
    "acceptor_smiles",
    "device_structure",
    "solvent",
    "additive",
  ]);
}

function columnsFor(active: DatasetKey): Column[] {
  if (active === "oled") {
    return [
      {
        key: "quality_tier",
        label: "Review tier",
        render: (record) => qualityPill(record.quality_tier),
      },
      {
        key: "doi",
        label: "DOI",
        className: "doi-column",
        render: (record) => <DoiCell record={record} />,
      },
      {
        key: "year",
        label: "Year",
        numeric: true,
        render: (record) => text(record.year) || "—",
      },
      {
        key: "device_label",
        label: "Device",
        render: (record) => (
          <span title={text(record.device_label)}>
            {shortText(record.device_label, 20) || "—"}
          </span>
        ),
      },
      {
        key: "final_emitter",
        label: "Final emitter",
        render: (record) => (
          <span title={text(record.final_emitter)}>
            {shortText(record.final_emitter, 20) || "—"}
          </span>
        ),
      },
      {
        key: "emission_mechanism",
        label: "Mechanism",
        render: (record) => shortText(humanize(record.emission_mechanism), 22),
      },
      {
        key: "emission_color",
        label: "Color",
        render: (record) =>
          record.emission_color ? (
            <span className={`color-pill color-${slug(text(record.emission_color))}`}>
              {humanize(record.emission_color)}
            </span>
          ) : (
            "—"
          ),
      },
      {
        key: "eqe_max",
        label: "EQE max",
        numeric: true,
        render: (record) => metricCell(record.eqe_max, "%"),
      },
      {
        key: "layer_count",
        label: "Layers",
        numeric: true,
        render: (record) => formatInteger(record.layer_count),
      },
      {
        key: "architecture",
        label: "Architecture",
        className: "wide-column",
        render: (record) => (
          <span title={text(record.architecture)}>
            {shortText(record.architecture, 46) || "—"}
          </span>
        ),
      },
    ];
  }
  if (active === "ofet") {
    return [
      {
        key: "doi",
        label: "DOI",
        className: "doi-column",
        render: (record) => <DoiCell record={record} />,
      },
      {
        key: "year",
        label: "Year",
        numeric: true,
        render: (record) => text(record.year) || "—",
      },
      {
        key: "semiconductor",
        label: "Semiconductor",
        className: "wide-column",
        render: (record) => (
          <span title={text(record.semiconductor)}>
            {shortText(record.semiconductor, 35) || "—"}
          </span>
        ),
      },
      {
        key: "conduction_category",
        label: "Conduction",
        render: (record) => (
          <span className="status-pill neutral">
            {text(record.conduction_category) || "Unknown"}
          </span>
        ),
      },
      {
        key: "highest_mobility",
        label: "Mobility",
        numeric: true,
        render: (record) => metricCell(record.highest_mobility, ""),
      },
      {
        key: "geometry_category",
        label: "Geometry",
        render: (record) => text(record.geometry_category) || "—",
      },
      {
        key: "fabrication_category",
        label: "Fabrication",
        render: (record) => (
          <span title={text(record.fabrication_method)}>
            {shortText(record.fabrication_category || record.fabrication_method, 23) ||
              "—"}
          </span>
        ),
      },
      {
        key: "dielectric_layer",
        label: "Dielectric",
        render: (record) => shortText(record.dielectric_layer, 18) || "—",
      },
      {
        key: "test_atmosphere",
        label: "Atmosphere",
        render: (record) => shortText(record.test_atmosphere, 18) || "—",
      },
      {
        key: "journal",
        label: "Journal",
        className: "wide-column",
        render: (record) => (
          <span title={text(record.journal)}>
            {shortText(record.journal, 30) || "—"}
          </span>
        ),
      },
    ];
  }
  return [
    {
      key: "primary_layer",
      label: "Benchmark",
      render: (record) => qualityPill(record.primary_layer),
    },
    {
      key: "doi",
      label: "DOI",
      className: "doi-column",
      render: (record) => <DoiCell record={record} />,
    },
    {
      key: "publish_year",
      label: "Year",
      numeric: true,
      render: (record) => text(record.publish_year) || "—",
    },
    {
      key: "donor",
      label: "Donor",
      render: (record) => shortText(record.donor, 20) || "—",
    },
    {
      key: "acceptor",
      label: "Acceptor",
      render: (record) => shortText(record.acceptor, 20) || "—",
    },
    {
      key: "pce",
      label: "PCE",
      numeric: true,
      render: (record) => metricCell(record.pce, "%"),
    },
    {
      key: "voc",
      label: "Voc",
      numeric: true,
      render: (record) => metricCell(record.voc, "V", 3),
    },
    {
      key: "jsc",
      label: "Jsc",
      numeric: true,
      render: (record) => metricCell(record.jsc, ""),
    },
    {
      key: "ff",
      label: "FF",
      numeric: true,
      render: (record) => metricCell(record.ff, "%"),
    },
    {
      key: "pce_error_bucket",
      label: "PCE error",
      render: (record) => errorPill(record.pce_error_bucket),
    },
    {
      key: "has_both_smiles",
      label: "SMILES",
      render: (record) =>
        record.has_both_smiles ? (
          <span className="status-pill success">
            <CheckCircle2 size={13} />
            Both
          </span>
        ) : (
          <span className="status-pill muted">Missing</span>
        ),
    },
    {
      key: "device_type",
      label: "Device",
      render: (record) => shortText(record.device_type || record.device_structure, 18) || "—",
    },
    {
      key: "solvent",
      label: "Solvent",
      render: (record) => shortText(record.solvent, 16) || "—",
    },
  ];
}

function DoiCell({ record }: { record: DataRecord }) {
  const url = doiUrl(record.doi);
  return url ? (
    <a
      href={url}
      title={text(record.doi)}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(event) => event.stopPropagation()}
    >
      {shortText(record.doi, 27)}
    </a>
  ) : (
    "—"
  );
}

function metricCell(value: unknown, unit: string, digits = 2) {
  const rendered = formatNumber(value, digits);
  return rendered === "—" ? "—" : (
    <strong className="metric-cell">
      {rendered}
      {unit ? <small>{unit}</small> : null}
    </strong>
  );
}

function qualityPill(value: unknown) {
  const rendered = text(value);
  if (["human_finalized", "strict_molecular_benchmark", "strict_molecular"].includes(rendered)) {
    return (
      <span className="status-pill success">
        <CheckCircle2 size={13} />
        {rendered === "human_finalized" ? "Human final" : "Strict molecular"}
      </span>
    );
  }
  if (["auto_reviewed", "strict_performance_benchmark", "strict_performance"].includes(rendered)) {
    return (
      <span className="status-pill info">
        <CircleGauge size={13} />
        {rendered === "auto_reviewed" ? "Auto reviewed" : "Strict performance"}
      </span>
    );
  }
  return <span className="status-pill muted">{humanize(rendered) || "Full archive"}</span>;
}

function errorPill(value: unknown) {
  const rendered = text(value) || "Unchecked";
  const tone =
    rendered === "<=2%"
      ? "success"
      : rendered === "2-5%"
        ? "info"
        : rendered === "5-10%"
          ? "warning"
          : rendered === ">10%"
            ? "danger"
            : "muted";
  return <span className={`status-pill ${tone}`}>{rendered}</span>;
}

function exportCsv(active: DatasetKey, records: DataRecord[]) {
  const keys =
    active === "oled"
      ? [
          "id",
          "doi",
          "title",
          "journal",
          "year",
          "quality_tier",
          "device_label",
          "device_type",
          "architecture",
          "emission_color",
          "emission_mechanism",
          "final_emitter",
          "eqe_max",
          "ce_max",
          "pe_max",
          "luminance_max",
          "turn_on_voltage",
          "el_peak",
          "fwhm",
          "layers",
          "performance",
          "materials",
        ]
      : active === "ofet"
        ? [
            "id",
            "doi",
            "year",
            "journal",
            "semiconductor",
            "smiles",
            "conduction_type",
            "mobility",
            "highest_mobility",
            "device_geometry",
            "fabrication_method",
            "organic_layer_thickness",
            "dielectric_layer",
            "test_atmosphere",
          ]
        : [
            "id",
            "doi",
            "publish_year",
            "primary_layer",
            "donor",
            "acceptor",
            "donor_smiles",
            "acceptor_smiles",
            "voc",
            "jsc",
            "ff",
            "pce",
            "device_structure",
            "device_type",
            "solvent",
            "additive",
          ];
  const content = [
    keys.join(","),
    ...records.map((record) => keys.map((key) => csvEscape(record[key])).join(",")),
  ].join("\n");
  const blob = new Blob([`\ufeff${content}`], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `EvoOptoDB_${active.toUpperCase()}_filtered.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function arrayOfObjects(value: unknown): DataRecord[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is DataRecord =>
          typeof item === "object" && item !== null && !Array.isArray(item),
      )
    : [];
}

function objectValue(value: unknown): DataRecord | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as DataRecord)
    : null;
}

function humanize(value: unknown): string {
  const rendered = text(value);
  return rendered
    ? rendered
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase())
    : "—";
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function measurement(value: unknown): string {
  const item = objectValue(value);
  if (!item) return "—";
  return [text(item.value), text(item.unit)].filter(Boolean).join(" ") || "—";
}

function unitValue(value: unknown, unit: string): string {
  const rendered = text(value);
  return rendered ? `${rendered} ${unit}` : "—";
}

function conditionText(value: unknown): string {
  const condition = objectValue(value);
  if (!condition) return "—";
  const items = Object.entries(condition).map(([name, raw]) => {
    const measurementValue = objectValue(raw);
    return `${humanize(name)}: ${
      measurementValue
        ? [text(measurementValue.value), text(measurementValue.unit)]
            .filter(Boolean)
            .join(" ")
        : text(raw)
    }`;
  });
  return items.filter(Boolean).join("; ") || "—";
}

function scalarValue(value: unknown): string | number | null {
  const number = numberValue(value);
  return number ?? (value === null || value === undefined ? null : text(value).toLowerCase());
}

function compareValues(
  left: string | number | null | undefined,
  right: string | number | null | undefined,
): number {
  if (typeof left === "number" && typeof right === "number") return left - right;
  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function isNumericColumn(columns: Column[], key: string): boolean {
  return Boolean(columns.find((column) => column.key === key)?.numeric);
}

function countFilled(records: DataRecord[], key: string): number {
  return records.filter((record) => {
    const value = record[key];
    return value !== null && value !== undefined && text(value).trim() !== "";
  }).length;
}

function countTruthy(records: DataRecord[], key: string): number {
  return records.filter((record) => Boolean(record[key])).length;
}

function uniqueCount(records: DataRecord[], key: string): number {
  return new Set(records.map((record) => text(record[key])).filter(Boolean)).size;
}

function maxNumber(records: DataRecord[], key: string): number | null {
  const values = records
    .map((record) => numberValue(record[key]))
    .filter((value): value is number => value !== null);
  return values.length ? Math.max(...values) : null;
}

function topCounts(
  records: DataRecord[],
  key: string,
  limit: number,
): Array<[string, number]> {
  const counts = new Map<string, number>();
  records.forEach((record) => {
    const value = text(record[key]).trim();
    if (value) counts.set(value, (counts.get(value) ?? 0) + 1);
  });
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit);
}
