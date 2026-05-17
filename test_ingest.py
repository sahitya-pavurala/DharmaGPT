import os
import requests
import json
from pathlib import Path

url = "http://localhost:8000/admin/vector/upload"
raw_dir = Path("dharmagpt/knowledge/raw/valmiki_ramayanam")

for filepath in raw_dir.glob("*.jsonl"):
    print(f"Uploading {filepath.name}...")
    with open(filepath, "rb") as f:
        files = {"file": (filepath.name, f, "application/octet-stream")}
        
        # Derive section name from filename (e.g., bala_chunks -> Bala Kanda)
        section_prefix = filepath.name.split("_")[0]
        section_name = section_prefix.capitalize() + " Kanda"
        
        data = {
            "vector_db": "pgvector",
            "index_name": "pgvector",
            "source": f"valmiki_ramayana_{section_prefix}",
            "source_title": "Valmiki Ramayana",
            "section": section_name,
            "language": "en"
        }
        
        response = requests.post(url, files=files, data=data)
        
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"Failed to decode JSON. Status code: {response.status_code}")
            print(response.text)

