# RAG Architecture Notes

## Vector store choice

Decided to use LanceDB instead of a hosted vector DB (Pinecone, Weaviate)
because it's embedded — no server to run, just a directory on disk. Good
fit for a personal tool that should work offline.

## Embeddings

Using sentence-transformers `all-MiniLM-L6-v2` locally instead of an
embeddings API. Keeps the project runnable without a second API key, and
384 dimensions is plenty for a few thousand personal notes.

## Agentic loop

The agent should decide *when* to search, not search on every message.
General knowledge questions (e.g. "what's a transformer?") shouldn't
trigger a vault search — only things that sound personal or project
specific.
