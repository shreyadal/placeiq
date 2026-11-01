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


def test_gemini_generates_company_question_and_returns_none_on_failure(monkeypatch):
    from google import genai

    import rag.generate

    class FakeModels:
        reply = ('{"question": "How does Oracle store an index?", "reference_answer": "As a B+ tree.",'
                 ' "key_points": ["B+ tree", "leaf nodes linked"]}')

        def generate_content(self, model, contents, config):
            assert "Company: Oracle" in contents and "<interview_notes>" in contents
            if FakeModels.reply is None:
                raise ConnectionError("no network")
            return type("Response", (), {"text": FakeModels.reply})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setenv("PLACEIQ_GENERATOR", "gemini")
    monkeypatch.setattr(genai, "Client", FakeClient)
    question = rag.generate.generate_question("Oracle", "DBMS", 2)
    assert question["concept"] == "DBMS" and question["difficulty"] == 2
    assert question["key_points"] == ["B+ tree", "leaf nodes linked"]

    FakeModels.reply = None
    assert rag.generate.generate_question("Oracle", "DBMS", 2) is None
