#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/dharmagpt/.env"
BACKUP_DIR="${1:-$REPO_ROOT/backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

cd "$REPO_ROOT"

db_backup="$BACKUP_DIR/dharmagpt-postgres-$STAMP.sql"
knowledge_backup="$BACKUP_DIR/dharmagpt-knowledge-$STAMP.tar.gz"

docker compose --env-file "$ENV_FILE" exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' > "$db_backup"
tar -czf "$knowledge_backup" -C "$REPO_ROOT/dharmagpt" knowledge

echo "Wrote:"
echo "  $db_backup"
echo "  $knowledge_backup"
