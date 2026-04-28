#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/dharmagpt/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  echo "Create it with: cp dharmagpt/.env.example dharmagpt/.env" >&2
  exit 1
fi

if ! grep -Eq '^POSTGRES_PASSWORD=.{16,}' "$ENV_FILE"; then
  echo "POSTGRES_PASSWORD must be set in dharmagpt/.env and should be at least 16 characters." >&2
  exit 1
fi

args=(--env-file "$ENV_FILE")
compose_args=(up -d)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tunnel)
      if ! grep -Eq '^CLOUDFLARED_TUNNEL_TOKEN=.' "$ENV_FILE"; then
        echo "CLOUDFLARED_TUNNEL_TOKEN must be set in dharmagpt/.env when using --tunnel." >&2
        exit 1
      fi
      args+=(--profile tunnel)
      shift
      ;;
    --build)
      compose_args+=(--build)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--build] [--tunnel]" >&2
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"
docker compose "${args[@]}" "${compose_args[@]}"
docker compose "${args[@]}" ps
