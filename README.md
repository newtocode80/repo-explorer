![tests](https://github.com/newtocode80/repo-explorer/actions/workflows/tests.yml/badge.svg)

# Repo Explorer (Step 1: Live API Tools)

An agentic assistant with two REAL external API tools â€” no fake data, no
API keys beyond Anthropic's. This is step 1 of a larger build:

1. **Live API tools** (this step) â€” GitHub + weather, real network calls
2. RAG over public docs â€” index an open-source project's README
3. Repo Q&A tool â€” combine steps 1+2 to answer questions about any public repo
4. Multi-agent system â€” split into a "fetcher" agent and a "Q&A" agent with a router

## Files

| File | Purpose |
|---|---|
| `api_tools.py` | GitHub + weather API calls, with real error handling |
| `agent.py` | Agent loop using the two live tools |
| `test_api_tools.py` | Unit tests with mocked network calls |

## Setup

```bash
pip install anthropic python-dotenv requests pytest
copy .env.example .env
# edit .env, add your real ANTHROPIC_API_KEY, save
```

## Run

```bash
# Free, fast, no network calls (mocked) â€” run this first
python -m pytest test_api_tools.py -v

# Live agent â€” costs a few cents in API usage
python agent.py
```

## Why this step matters

Most LLM tutorials use either zero tools or fake/local data. Two things
change once a tool hits a **real** external API:

1. **Errors are real, not contrived.** GitHub's unauthenticated rate limit
   (60 requests/hour) actually got hit while building this â€” so the error
   handling for that case isn't theoretical, it's tested against a failure
   that genuinely happened.
2. **You can't assert exact responses in tests**, because live data changes
   (star counts go up, weather changes). The unit tests here mock the
   network layer with `unittest.mock.patch`, so they test *your error
   handling logic*, not whether the internet is currently working. This is
   the standard professional pattern for testing any code that calls an
   external API â€” also worth name-dropping in interviews.

## Known limitation (by design, for this step)

The weather tool only supports a fixed list of cities (Montreal, NYC, SF,
London, Toronto, Paris) since it skips a geocoding API to keep this step
focused on the tool-use mechanics rather than chaining a third API. Real
production version would add a geocoding lookup first.

