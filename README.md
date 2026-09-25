# YouTube RAG

Ask questions about any YouTube video. The app downloads the video's transcript, stores it in a local vector database, and answers your questions with a Groq-hosted LLM, using only what was said in the video.

It comes with a small web page: paste a link, ask a question, and see the answer along with the transcript excerpts it was based on.

## How it works

```
YouTube link
   │
   ▼
1. Ingest    fetch transcript ─► split into chunks ─► embed ─► store in ChromaDB
   │
   ▼
2. Retrieve  embed the question ─► find the 5 most similar chunks
   │
   ▼
3. Generate  send question + chunks to the LLM ─► answer grounded in the transcript
```

| Step | Tool |
|---|---|
| Transcript | `youtube-transcript-api` via LangChain's `YoutubeLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Vector store | ChromaDB, cosine similarity, saved to `database/` |
| LLM | `openai/gpt-oss-120b` on Groq |
| Web server | FastAPI + uvicorn |

## Setup

Requires **Python 3.13** and a free [Groq API key](https://console.groq.com/keys).

**1. Install dependencies**

With [uv](https://docs.astral.sh/uv/):
```bash
uv sync --no-install-project
```

Or with pip:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a file named `.env` in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
`.env` is already in `.gitignore`, so the key won't be committed.

## Usage

### Web app

From the project root:
```bash
uv run --no-project uvicorn server:app --reload
```
Open http://127.0.0.1:8000, then:

1. Paste a YouTube link (watch, youtu.be, shorts or embed) or an 11-character video ID, and click **Load transcript**.
2. Type a question and click **Ask**.

The page shows the answer, the transcript chunks used as sources with their similarity scores, and a console with the server's log output.

The first start takes a little while because the embedding model has to load.

### Command line

```bash
python main.py
```
It asks for a video ID, loads the transcript, then asks for your question and prints the answer.

## API

The web page talks to two endpoints. You can also call them directly.

**`POST /ingest`**: load a video. This replaces whatever video was loaded before.
```json
// request
{ "video_id": "qbLc5a9jdXo" }

// response
{ "chunks": 57, "logs": ["..."] }
```

**`POST /ask`**: answer a question about the loaded video.
```json
// request
{ "question": "What is the instructor's name?" }

// response
{
  "answer": "The instructor's name is Caleb.",
  "sources": [
    { "rank": 1, "similarity_score": 0.22, "content": "...", "metadata": {}, "id": "...", "distance": 0.78 }
  ],
  "logs": ["..."]
}
```

On failure, both return an HTTP error with `{ "detail": "reason" }`.

## Project structure

```
.
├── server.py                 FastAPI app: /ingest, /ask, serves the web page
├── main.py                   Command-line version of the same pipeline
├── frontend/
│   └── index.html            Web page
├── rag/
│   ├── Ingestion/
│   │   ├── YTranscriptLoader.py   Fetches the transcript
│   │   └── chunking.py            Splits it into chunks
│   ├── Embedding/
│   │   └── embedding_manager.py   Turns text into vectors
│   ├── vectorstore/
│   │   └── ChromaDB.py            Stores and clears vectors
│   ├── retrieval/
│   │   └── rag_retrieve.py        Finds the chunks closest to a question
│   └── Generation/
│       └── LLM.py                 Builds the prompt and calls Groq
├── database/                 ChromaDB data (created on first run)
├── requirements.txt
└── pyproject.toml
```

## Notes and limitations

- **One video at a time.** Loading a new video clears the database first.
- **English transcripts only.** Videos without English captions can't be loaded.
- **Low similarity scores are normal.** A short question compared with a 1000-character chunk usually scores 0.1–0.3, so the score threshold is left at 0.
- **Changing the model.** Pass a different Groq model name to `RAGGenerator(retriever, model_name="...")` in `server.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Create `.env` in the folder you start the server from (see Setup). |
| `OpenBLAS error: Memory allocation still failed` | Set `OPENBLAS_NUM_THREADS=1` before starting the server. |
| `uv` fails with `Expected a Python module at: src\rag\__init__.py` | Add `--no-install-project` to the `uv` command. The code lives in `rag/`, not `src/rag/`. |
| `VideoUnavailable` or no transcript found | The video is private, removed, or has no English captions. |
| Page loads but buttons do nothing | Start the server from the project root so it can find `frontend/`. |
