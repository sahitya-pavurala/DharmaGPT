#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/dharmagpt/.env"

args=()
if [[ -f "$ENV_FILE" ]]; then
  args=(--env-file "$ENV_FILE")
fi

cd "$REPO_ROOT"
docker compose "${args[@]}" --profile tunnel down
