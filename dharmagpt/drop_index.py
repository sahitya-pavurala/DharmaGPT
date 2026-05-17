import psycopg
from core.postgres_db import database_url

with psycopg.connect(database_url()) as conn:
    print("Dropping index idx_chunk_store_embedding...")
    conn.execute("DROP INDEX IF EXISTS idx_chunk_store_embedding")
    conn.commit()
    print("Dropped.")
