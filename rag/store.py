"""Embed the interview experiences into Chroma and retrieve the relevant ones.

    python -m rag.store                    # (re)build the index from rag/docs/
    python -m rag.store Amazon OS DBMS     # test a query: company, then weak concepts

How retrieval works: each experience is turned into a 384-number vector by a sentence
embedding model, such that texts with similar meaning get vectors pointing in similar
directions. A query is embedded the same way and Chroma returns the stored experiences
whose vectors are closest to it (cosine similarity), restricted to the chosen company.
"""
import sys
from functools import lru_cache
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
DOCS_DIR = RAG_DIR / "docs"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION = "experiences"
# small (about 90 MB) and fast on CPU; good enough for a few hundred short documents
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# the query says what each short concept name means, since "OS" or "CN" alone embeds poorly
CONCEPT_DESCRIPTIONS = {
    "DSA": "data structures and algorithms, arrays, trees, graphs, dynamic programming, coding problems",
    "OS": "operating systems, processes, threads, deadlock, memory management, scheduling",
    "DBMS": "databases, SQL queries, indexes, normalisation, transactions",
    "CN": "computer networks, TCP, HTTP, DNS, sockets, load balancing",
    "OOP": "object oriented programming, classes, inheritance, polymorphism, design patterns, low level design",
    "ML": "machine learning, models, overfitting, evaluation metrics",
}


def companies():
    """Company names, taken from the file names in rag/docs/."""
    return sorted(p.stem for p in DOCS_DIR.glob("*.txt"))


def load_documents():
    """Parse rag/docs/*.txt into a list of {"id", "company", "role", "year", "concepts", "text"}."""
    documents = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        blocks = [b.strip() for b in path.read_text(encoding="utf8").split("\n---\n")]
        for i, block in enumerate(b for b in blocks if b):
            header, text = {}, []
            for line in block.splitlines():
                key, _, value = line.partition(":")
                if not text and key in ("role", "year", "concepts"):
                    header[key] = value.strip()
                else:
                    text.append(line)
            documents.append({
                "id": f"{path.stem}-{i + 1}",
                "company": path.stem,
                "role": header.get("role", ""),
                "year": header.get("year", ""),
                "concepts": header.get("concepts", ""),
                "text": " ".join(text).strip(),
            })
    return documents


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts):
    # normalised vectors make cosine similarity a plain dot product
    return _embedder().encode(texts, normalize_embeddings=True).tolist()


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build():
    """Rebuild the collection from scratch so deleted or edited documents do not linger."""
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    documents = load_documents()
    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        embeddings=embed([d["text"] for d in documents]),
        metadatas=[{k: d[k] for k in ("company", "role", "year", "concepts")} for d in documents],
    )
    return len(documents)


def retrieve(company, concepts, k=4):
    """The k experiences from this company most relevant to the given concepts.

    Returns [{"text", "company", "role", "year", "concepts", "distance"}], closest first.
    The index is built on first use if it does not exist yet.
    """
    client = _client()
    if COLLECTION not in [c.name for c in client.list_collections()]:
        build()
    collection = client.get_collection(COLLECTION)

    topics = "; ".join(CONCEPT_DESCRIPTIONS.get(c, c) for c in concepts)
    query = f"{company} interview questions on {topics}"
    result = collection.query(query_embeddings=embed([query]), n_results=k, where={"company": company})
    return [
        {"text": text, **metadata, "distance": round(distance, 3)}
        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


if __name__ == "__main__":
    if len(sys.argv) > 2:
        for hit in retrieve(sys.argv[1], sys.argv[2:]):
            print(f"[{hit['distance']}] {hit['role']} ({hit['concepts']}): {hit['text'][:150]}...")
    else:
        print(f"indexed {build()} experiences from {len(companies())} companies into {CHROMA_DIR}")
