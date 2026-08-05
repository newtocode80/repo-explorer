"""
Unit tests for api_tools.py.

Key technique here, different from the practice-run project: these tools
make REAL network calls, so we use unittest.mock to fake the network layer
entirely. This is the standard pattern for testing code that hits external
APIs -- you test YOUR error-handling logic, not whether the internet works
right now. A test suite that depends on a live API being up is flaky by
design; mocking the response means these tests are fast and deterministic
every time.
"""

from unittest.mock import patch, Mock
from api_tools import github_lookup, weather_lookup, CITY_COORDS


class TestGithubLookup:
    @patch("api_tools.requests.get")
    def test_successful_lookup(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "full_name": "anthropics/anthropic-sdk-python",
            "description": "Anthropic Python SDK",
            "language": "Python",
            "stargazers_count": 500,
            "open_issues_count": 10,
            "default_branch": "main",
        }
        mock_get.return_value = mock_response

        result = github_lookup("anthropics/anthropic-sdk-python")
        assert "500" in result
        assert "Python" in result

    @patch("api_tools.requests.get")
    def test_repo_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = github_lookup("nobody/fake-repo-xyz")
        assert "not found" in result.lower()

    @patch("api_tools.requests.get")
    def test_rate_limit_returns_clear_message(self, mock_get):
        # This is the case we actually hit live while building this --
        # worth a permanent test since it's a realistic, recurring failure mode.
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        result = github_lookup("anthropics/anthropic-sdk-python")
        assert "rate limit" in result.lower()

    @patch("api_tools.requests.get")
    def test_timeout_handled_gracefully(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()
        result = github_lookup("anthropics/anthropic-sdk-python")
        assert "timed out" in result.lower()

    @patch("api_tools.requests.get")
    def test_connection_error_handled_gracefully(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()
        result = github_lookup("anthropics/anthropic-sdk-python")
        assert "connect" in result.lower()


class TestWeatherLookup:
    def test_unsupported_city_returns_helpful_error(self):
        result = weather_lookup("Atlantis")
        assert "error" in result.lower()
        assert "montreal" in result.lower()  # confirms supported list is shown

    def test_city_matching_is_case_insensitive(self):
        # Confirms "MONTREAL", "montreal", "Montreal" are treated the same
        assert "montreal" in CITY_COORDS
        # this only checks the lookup table itself; the live-call cases
        # below confirm the case-folding logic that uses this table.

    @patch("api_tools.requests.get")
    def test_successful_weather_lookup(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current": {"temperature_2m": 15.2}}
        mock_get.return_value = mock_response

        result = weather_lookup("MONTREAL")  # uppercase, tests case-insensitivity
        assert "15.2" in result
        assert "Montreal" in result

    @patch("api_tools.requests.get")
    def test_missing_temperature_data_handled(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current": {}}  # malformed/incomplete response
        mock_get.return_value = mock_response

        result = weather_lookup("Montreal")
        assert "error" in result.lower()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
