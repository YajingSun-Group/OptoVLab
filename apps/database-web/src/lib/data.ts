import { gunzipSync, strFromU8 } from "fflate";

import type { Catalog, DataRecord, DatasetKey } from "@/lib/types";

const datasetCache = new Map<DatasetKey, DataRecord[]>();

export async function loadCatalog(signal?: AbortSignal): Promise<Catalog> {
  const response = await fetch("/data/catalog.json", {
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    throw new Error(`Catalog request failed (${response.status})`);
  }
  return (await response.json()) as Catalog;
}

export async function loadDataset(
  key: DatasetKey,
  file: string,
  signal?: AbortSignal,
): Promise<DataRecord[]> {
  const cached = datasetCache.get(key);
  if (cached) {
    return cached;
  }
  const response = await fetch(`/${file}`, {
    cache: "force-cache",
    signal,
  });
  if (!response.ok) {
    throw new Error(`${key.toUpperCase()} data request failed (${response.status})`);
  }
  const compressed = new Uint8Array(await response.arrayBuffer());
  let jsonText: string;
  try {
    jsonText = strFromU8(gunzipSync(compressed));
  } catch {
    // Some static hosts transparently decode .gz assets before fetch returns them.
    jsonText = strFromU8(compressed);
  }
  const payload = JSON.parse(jsonText) as unknown;
  if (!Array.isArray(payload)) {
    throw new Error(`${key.toUpperCase()} data is not a JSON array`);
  }
  const records = payload as DataRecord[];
  datasetCache.set(key, records);
  return records;
}
