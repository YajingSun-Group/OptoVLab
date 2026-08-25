import type { DataRecord } from "@/lib/types";

export function text(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(text).filter(Boolean).join(", ");
  return String(value);
}

export function shortText(value: unknown, limit = 28): string {
  const rendered = text(value);
  return rendered.length > limit ? `${rendered.slice(0, limit - 1)}…` : rendered;
}

export function numberValue(value: unknown): number | null {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatNumber(
  value: unknown,
  digits = 2,
  fallback = "—",
): string {
  const parsed = numberValue(value);
  return parsed === null ? fallback : parsed.toFixed(digits);
}

export function formatInteger(value: unknown): string {
  const parsed = numberValue(value);
  return parsed === null ? "—" : Math.round(parsed).toLocaleString();
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

export function doiUrl(value: unknown): string | null {
  const doi = text(value).trim();
  return doi ? `https://doi.org/${encodeURIComponent(doi).replaceAll("%2F", "/")}` : null;
}

export function recordText(record: DataRecord, keys: string[]): string {
  return keys.map((key) => text(record[key])).join(" ").toLocaleLowerCase();
}

export function uniqueValues(
  records: DataRecord[],
  key: string,
  limit = 60,
): string[] {
  const counts = new Map<string, number>();
  records.forEach((record) => {
    const raw = record[key];
    const values = Array.isArray(raw) ? raw : [raw];
    values.forEach((value) => {
      const rendered = text(value).trim();
      if (rendered) counts.set(rendered, (counts.get(rendered) ?? 0) + 1);
    });
  });
  return [...counts.keys()]
    .sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0) || a.localeCompare(b))
    .slice(0, limit);
}

export function percent(value: number, total: number): string {
  return total ? `${((value / total) * 100).toFixed(1)}%` : "0.0%";
}

export function csvEscape(value: unknown): string {
  let rendered =
    typeof value === "object" && value !== null ? JSON.stringify(value) : text(value);
  if (/^[=+\-@]/.test(rendered)) rendered = `'${rendered}`;
  return /[",\n\r]/.test(rendered)
    ? `"${rendered.replaceAll('"', '""')}"`
    : rendered;
}
