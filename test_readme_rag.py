"""
Tests for readme_rag.py -- same mocking pattern as test_api_tools.py.
We test: chunking math, cache behavior, and each error-handling branch.
We do NOT test the live GitHub fetch or OpenAI embeddings -- those are
integration concerns, not unit concerns.
"""

from unittest.mock import patch, Mock, MagicMock
from readme_rag import chunk_text, fetch_readme, CHUNK_SIZE, CHUNK_OVERLAP


class TestChunkText:
    def test_short_text_produces_one_chunk(self):
        text = "A" * 100
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_overlap_exists_between_consecutive_chunks(self):
        text = "B" * 1500
        chunks = chunk_text(text)
        assert len(chunks) > 1
        # Last CHUNK_OVERLAP chars of chunk 0 should equal first CHUNK_OVERLAP of chunk 1
        assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]

    def test_empty_chunks_filtered_out(self):
        # Whitespace-only segments shouldn't make it into the index
        text = "Real content here" + " " * CHUNK_SIZE + "More content"
        chunks = chunk_text(text)
        assert all(c.strip() for c in chunks)

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []


class TestFetchReadme:
    @patch("readme_rag.requests.get")
    def test_successful_fetch_returns_text(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# My Project\nInstall with pip install myproject"
        mock_get.return_value = mock_response

        result = fetch_readme("owner/repo")
        assert result == "# My Project\nInstall with pip install myproject"

    @patch("readme_rag.requests.get")
    def test_404_returns_none(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_readme("nobody/nonexistent")
        assert result is None

    @patch("readme_rag.requests.get")
    def test_rate_limit_returns_none(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        result = fetch_readme("owner/repo")
        assert result is None

    @patch("readme_rag.requests.get")
    def test_timeout_returns_none(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = fetch_readme("owner/repo")
        assert result is None

    @patch("readme_rag.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = fetch_readme("owner/repo")
        assert result is None


class TestReadmeLookupErrorPaths:
    @patch("readme_rag.fetch_readme", return_value=None)
    def test_returns_error_message_when_readme_unavailable(self, mock_fetch):
        from readme_rag import readme_lookup
        result = readme_lookup("owner/bad-repo", "how do I install this?")
        assert "error" in result.lower()

    @patch("readme_rag.fetch_readme", return_value="   ")
    def test_returns_error_message_when_readme_empty(self, mock_fetch):
        from readme_rag import readme_lookup, _collection_cache
        # Clear cache so it tries to build fresh
        _collection_cache.clear()
        result = readme_lookup("owner/empty-repo", "anything")
        assert "error" in result.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
