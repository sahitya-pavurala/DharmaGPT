#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <postgres-dump.sql> [knowledge.tar.gz]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/dharmagpt/.env"
DB_DUMP="$1"
KNOWLEDGE_ARCHIVE="${2:-}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi
if [[ ! -f "$DB_DUMP" ]]; then
  echo "Missing database dump: $DB_DUMP" >&2
  exit 1
fi

cd "$REPO_ROOT"

docker compose --env-file "$ENV_FILE" up -d db
docker compose --env-file "$ENV_FILE" exec -T db sh -c 'dropdb -U "$POSTGRES_USER" --if-exists "$POSTGRES_DB"'
docker compose --env-file "$ENV_FILE" exec -T db sh -c 'createdb -U "$POSTGRES_USER" "$POSTGRES_DB"'
docker compose --env-file "$ENV_FILE" exec -T db sh -c 'psql -U "$POSTGRES_USER" "$POSTGRES_DB"' < "$DB_DUMP"

if [[ -n "$KNOWLEDGE_ARCHIVE" ]]; then
  if [[ ! -f "$KNOWLEDGE_ARCHIVE" ]]; then
    echo "Missing knowledge archive: $KNOWLEDGE_ARCHIVE" >&2
    exit 1
  fi
  tar -xzf "$KNOWLEDGE_ARCHIVE" -C "$REPO_ROOT/dharmagpt"
fi

docker compose --env-file "$ENV_FILE" up -d
