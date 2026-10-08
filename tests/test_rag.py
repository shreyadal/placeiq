from rag.store import companies, load_documents, retrieve


def test_documents_parse():
    documents = load_documents()
    assert len(documents) >= 50
    assert {d["company"] for d in documents} == set(companies())
    for d in documents:
        assert d["role"] and d["concepts"] and len(d["text"]) > 100
        # header lines must not leak into the text that gets embedded
        assert not d["text"].startswith(("role:", "year:", "concepts:"))


def test_retrieval_stays_in_company_and_finds_the_topic():
    hits = retrieve("Oracle", ["DBMS"], k=3)
    assert len(hits) == 3
    assert all(h["company"] == "Oracle" for h in hits)
    assert "DBMS" in hits[0]["concepts"]
    # closest first
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)
