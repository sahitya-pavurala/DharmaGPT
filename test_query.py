import os
import sys

# Ensure dharmagpt folder is in PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "dharmagpt"))

from core.backends.embedding import LocalHashEmbeddings
from core.backends.rag import _PgVectorRetriever

# Instantiate the local hash embedder
embedder = LocalHashEmbeddings(dims=3072)

# Instantiate the PgVector retriever (top_k=3)
retriever = _PgVectorRetriever(embedder=embedder, top_k=3)

# Test query
query_str = "Rama went to the forest"
print(f"Query: {query_str}")
docs = retriever.get_relevant_documents(query_str)

print(f"\nRetrieved {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    score = doc.metadata.get('score')
    score_str = f"{score:.4f}" if score is not None else "N/A"
    print(f"\n--- Document {i} (Score: {score_str}) ---")
    print(f"Citation: {doc.metadata.get('citation')}")
    print(f"Section: {doc.metadata.get('section')}")
    content = doc.page_content[:300].encode('ascii', errors='replace').decode('ascii')
    print(f"Content Preview: {content}...")
