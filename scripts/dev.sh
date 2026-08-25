#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runtime/optovlab/dev-logs"
mkdir -p "$LOG_DIR"

for command in uv npm; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$command" >&2
    exit 1
  fi
done

if [[ ! -d "$ROOT/apps/web/node_modules" || ! -d "$ROOT/apps/database-web/node_modules" ]]; then
  printf 'Install frontend dependencies first; see README.md.\n' >&2
  exit 1
fi

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(
  cd "$ROOT"
  exec uv run mining-platform serve-api --host 127.0.0.1 --port 8000
) >"$LOG_DIR/api.log" 2>&1 &
pids+=("$!")

(
  cd "$ROOT/apps/database-web"
  exec npm run dev -- --port 3000
) >"$LOG_DIR/database.log" 2>&1 &
pids+=("$!")

(
  cd "$ROOT/apps/web"
  exec npm run dev
) >"$LOG_DIR/web.log" 2>&1 &
pids+=("$!")

printf 'OptoVLab services started:\n'
printf '  Agent workbench  http://127.0.0.1:5175\n'
printf '  Database browser http://127.0.0.1:3000\n'
printf '  API docs          http://127.0.0.1:8000/docs\n'
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Press Ctrl+C to stop.\n'

wait -n "${pids[@]}"
