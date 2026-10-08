# rag/ — company-style questions from interview experiences

```
python -m rag.store                  # embed rag/docs/ into Chroma (rag/chroma/)
python -m rag.store Amazon OS DBMS   # see what is retrieved for a company and concepts
python -m rag.generate Amazon OS     # generate one question (needs Ollama running)
```

## Pipeline

1. **Documents.** `docs/<Company>.txt` holds interview experiences tagged with role, year
   and concepts. The format, and where the bundled samples came from, is in
   [docs/README.md](docs/README.md).
2. **Embed.** [store.py](store.py) turns each experience into a 384-dimensional vector with
   the `all-MiniLM-L6-v2` sentence-transformers model and stores it in a Chroma collection
   with the company as metadata.
3. **Retrieve.** Given a company and the user's weak concepts, the query
   `"<company> interview questions on <concept descriptions>"` is embedded the same way
   and Chroma returns the nearest experiences by cosine distance, filtered to that company.
4. **Generate.** [generate.py](generate.py) gives the retrieved experiences to a local
   model served by Ollama and asks for one new question in that company's style on the concept, with a reference
   answer and key points so it can be graded like a bank question.

## Where it plugs in

`GET /sessions/{id}/next` in the backend first picks a concept and difficulty from
knowledge tracing. If the session has a company, it then calls `generate_question` for
that concept. If generation is unavailable (Ollama not running, model not pulled), the question
from the bank is used, so the interview never stalls.

`GET /companies/{company}/experiences` returns the retrieved notes themselves; the results
page shows them for the user's weak concepts. That part does not need Ollama.
