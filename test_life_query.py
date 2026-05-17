import os
import sys

# Ensure dharmagpt folder is in PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "dharmagpt"))

from core.backends.embedding import LocalHashEmbeddings
from core.backends.rag import _PgVectorRetriever

# 1. Initialize our retriever with the same local_hash embedding model used for ingestion
embedder = LocalHashEmbeddings(dims=3072)
retriever = _PgVectorRetriever(embedder=embedder, top_k=3)

# 2. Define the basic life question
life_question = "Anger and greed should be abandoned"

print("=" * 60)
print(f"LIFE QUESTION: '{life_question}'")
print("=" * 60)

# 3. Retrieve relevant passages
docs = retriever.get_relevant_documents(life_question)

print(f"\nFound {len(docs)} matching passages in Valmiki Ramayana:\n")
for i, doc in enumerate(docs, 1):
    score = doc.metadata.get('score')
    score_str = f"{score:.4f}" if score is not None else "N/A"
    
    print(f"[{i}] Citation: {doc.metadata.get('citation')} - {doc.metadata.get('section')}")
    print(f"    Similarity Score: {score_str}")
    
    # Format and print the text securely (encoding safe for Windows shell)
    text_content = doc.page_content.encode('ascii', errors='replace').decode('ascii')
    
    # Format to look like a book passage
    print("    Excerpt:")
    lines = text_content.split('. ')
    for line in lines[:3]:  # Print first 3 sentences
        if line.strip():
            print(f"      * {line.strip()}")
    if len(lines) > 3:
        print("      * ...")
    print("-" * 60)

print("\n[EXPLANATION] How similarity is calculated:")
print("- The life question was converted into a 3,072-dimensional vector.")
print("- PostgreSQL calculated the cosine distance (<=>) between the question vector and all 10,503 database chunks.")
print("- The exact math is: Similarity = 1 - (Question_Vector <=> Chunk_Vector).")
print("- A score closer to 1.0 indicates higher lexical/overlap similarity!")
