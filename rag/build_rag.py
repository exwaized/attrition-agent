# ============================================================
# build_rag.py — Embed HR Policy Docs into ChromaDB
# ============================================================
# PURPOSE: Chunks policy docs → embeds via sentence-transformers
#          → stores in ChromaDB for retrieval by agent
# FLOW: policy txts → chunks → embeddings → chroma collection
# CONNECTED TO: policy docs (input) → retriever.py (output)
# ============================================================

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import yaml
import uuid

# --- Step 1: Load config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# --- Step 2: Initialize ChromaDB ---
# Persistent storage — embeddings survive across sessions
chroma_path = cfg["paths"]["chroma_db"]
Path(chroma_path).mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=chroma_path)

# Delete existing collection if rebuilding
# Prevents duplicate embeddings on re-run
try:
    client.delete_collection("hr_policies")
    print("Existing collection deleted — rebuilding")
except:
    print("No existing collection — creating fresh")

collection = client.create_collection(
    name="hr_policies",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
)

# --- Step 3: Load embedding model ---
# all-MiniLM-L6-v2: fast, accurate, 384 dimensions
# Downloads ~80MB on first run — cached after that
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

# --- Step 4: Load and chunk policy documents ---
# Chunk size 300 chars with 50 char overlap
# Overlap prevents context loss at chunk boundaries
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks.
    Simple forward-only iteration — no infinite loop risk.
    """
    text   = text.replace("\x00", "").strip()
    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        # Always move forward by (chunk_size - overlap)
        # Never go backwards — fixes infinite loop
        start += (chunk_size - overlap)

    return chunks

policy_dir = Path("rag/policies")
all_chunks = []
all_ids    = []
all_metas  = []

for policy_file in policy_dir.glob("*.txt"):
    # Explicit UTF-8 encoding — prevents MemoryError from encoding issues
    # errors="ignore" skips any undecodable bytes
    text     = policy_file.read_text(encoding="utf-8", errors="ignore")
    chunks   = chunk_text(text)
    doc_name = policy_file.stem  # filename without .txt

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{doc_name}_{i}_{str(uuid.uuid4())[:8]}")
        all_metas.append({
            "source":       doc_name,
            "chunk_idx":    i,
            "total_chunks": len(chunks)
        })

    print(f"  {doc_name}: {len(chunks)} chunks")

print(f"\nTotal chunks to embed: {len(all_chunks)}")

# Safety check — if no chunks found, something is wrong with files
if len(all_chunks) == 0:
    raise ValueError("No chunks generated — check policy files are not empty")

# --- Step 5: Embed and store ---
# Batch embedding = faster than one by one
print("Embedding chunks...")
embeddings = embedder.encode(
    all_chunks,
    show_progress_bar=True,
    batch_size=32           # smaller batch = less memory pressure
).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=all_ids,
    metadatas=all_metas
)

print(f"\n✅ RAG built successfully")
print(f"Collection: hr_policies")
print(f"Total chunks stored: {collection.count()}")

# --- Step 6: Test retrieval ---
# Verify RAG works before agent tries to use it
print("\nTesting retrieval...")
test_queries = [
    "employee no promotion 18 months high performer",
    "salary below market correction MAP program",
    "peer attrition contagion team members leaving",
    "post appraisal resignation retention window"
]

for query in test_queries:
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    print(f"\nQuery: '{query}'")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  Source: {meta['source']} | Chunk: {meta['chunk_idx']}")
        print(f"  Text: {doc[:120]}...")

print("\n✅ RAG retrieval working correctly")