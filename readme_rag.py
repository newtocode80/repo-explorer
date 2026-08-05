"""
RAG tool that fetches a GitHub repo's README live and indexes it in an
in-memory Chroma vector store, then retrieves relevant chunks for a query.

Pipeline:
  repo name
    → GitHub API (raw README text)
    → chunk into overlapping segments
    → embed via OpenAI text-embedding-3-small
    → store in Chroma (in-memory, per-repo collection)
    → retrieve top-k chunks for a query
    → return as context for the agent to answer from

One collection is built per repo and cached for the session, so repeated
questions about the same repo don't re-fetch and re-embed each time.
"""

import os
import requests
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Lazy init: don't create the client at import time so unit tests that mock
# network calls don't fail just because OPENAI_API_KEY isn't set in the
# test environment. Client is created on first actual embed call.
_openai_client = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75
GITHUB_API_BASE = "https://api.github.com"

# Cache: repo_full_name -> chroma collection
# Avoids re-fetching + re-embedding the same README for repeated questions
_collection_cache: dict = {}
_chroma_client = chromadb.Client()


def fetch_readme(repo_full_name: str) -> str | None:
    """
    Fetch raw README text from GitHub. Returns None on any failure so the
    caller can return a clean error message rather than raising an exception.
    Uses the same error-handling shape as api_tools.py for consistency.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/readme"
    headers = {"Accept": "application/vnd.github.raw+json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.ConnectionError:
        return None

    if response.status_code == 404:
        return None
    if response.status_code == 403:
        return None
    if response.status_code != 200:
        return None

    return response.text


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, same pattern as resume_rag.py."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI. Same function as resume_rag.py."""
    response = _get_openai_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_index(repo_full_name: str) -> tuple[chromadb.Collection, int] | tuple[None, str]:
    """
    Fetch README and build a Chroma collection for this repo.
    Returns (collection, chunk_count) on success, or (None, error_message) on failure.
    """
    readme_text = fetch_readme(repo_full_name)

    if readme_text is None:
        return None, (
            f"Could not fetch README for '{repo_full_name}'. "
            "The repo may not exist, have no README, or the GitHub API rate limit may be exceeded."
        )

    chunks = chunk_text(readme_text)
    if not chunks:
        return None, f"README for '{repo_full_name}' appears to be empty."

    embeddings = embed_texts(chunks)

    # Use a sanitized collection name (Chroma requires alphanumeric + hyphens)
    collection_name = repo_full_name.replace("/", "--").replace("_", "-")[:60]

    # Delete if already exists (handles re-indexing the same repo in one session)
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = _chroma_client.create_collection(collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    return collection, len(chunks)


def get_or_build_index(repo_full_name: str) -> tuple[chromadb.Collection, None] | tuple[None, str]:
    """Return cached collection if available, otherwise build it."""
    if repo_full_name not in _collection_cache:
        collection, result = build_index(repo_full_name)
        if collection is None:
            return None, result  # result is an error message here
        _collection_cache[repo_full_name] = collection
        print(f"  [readme_rag] indexed README for {repo_full_name} ({result} chunks)")

    return _collection_cache[repo_full_name], None


def readme_lookup(repo_full_name: str, query: str, n_results: int = 3) -> str:
    """
    The tool function the agent calls. Retrieves the most relevant README
    chunks for the query and returns them as context. Builds + caches the
    index on first call for a given repo.
    """
    collection, error = get_or_build_index(repo_full_name)
    if error:
        return f"Error: {error}"

    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    matched_chunks = results["documents"][0]
    if not matched_chunks:
        return "No relevant information found in the README."

    return "\n---\n".join(matched_chunks)


# Tool schema for agent.py
README_TOOL_SCHEMA = {
    "name": "readme_lookup",
    "description": (
        "Search a GitHub repo's README for information to answer a question "
        "about how the project works, its features, installation, usage, or "
        "architecture. Use this when the user asks a detailed question about "
        "a repo's content, not just its metadata (stars, language, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_full_name": {
                "type": "string",
                "description": "Repo in 'owner/repo' format, e.g. 'anthropics/anthropic-sdk-python'",
            },
            "query": {
                "type": "string",
                "description": "What to search for in the README",
            },
        },
        "required": ["repo_full_name", "query"],
    },
}


if __name__ == "__main__":
    # Standalone test — runs the full pipeline against a real repo
    repo = "anthropics/anthropic-sdk-python"
    print(f"Testing readme_lookup against {repo}...\n")

    result = readme_lookup(repo, "how do I install this library?")
    print("Query: how do I install this library?")
    print(result)
    print()

    result = readme_lookup(repo, "what models are supported?")
    print("Query: what models are supported?")
    print(result)
