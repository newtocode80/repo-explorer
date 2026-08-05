"""
Tests for step 3 additions: context repo tracking and conversation history.
These are pure logic tests — no API calls, no mocking needed.
"""

from agent import extract_repo_from_text, build_system_prompt


class TestExtractRepo:
    def test_extracts_owner_repo_format(self):
        assert extract_repo_from_text("tell me about anthropics/anthropic-sdk-python") \
            == "anthropics/anthropic-sdk-python"

    def test_extracts_from_url_style_input(self):
        assert extract_repo_from_text("facebook/react looks interesting") \
            == "facebook/react"

    def test_returns_none_when_no_repo_present(self):
        assert extract_repo_from_text("what's the weather in Montreal?") is None

    def test_returns_none_for_plain_question(self):
        assert extract_repo_from_text("how do I install it?") is None

    def test_picks_first_repo_when_multiple_present(self):
        result = extract_repo_from_text("compare django/django and pallets/flask")
        assert result == "django/django"


class TestBuildSystemPrompt:
    def test_no_context_repo_returns_base_prompt(self):
        prompt = build_system_prompt(None)
        assert "context repo" not in prompt.lower()

    def test_context_repo_injected_into_prompt(self):
        prompt = build_system_prompt("anthropics/anthropic-sdk-python")
        assert "anthropics/anthropic-sdk-python" in prompt

    def test_context_repo_prompt_mentions_follow_up_guidance(self):
        prompt = build_system_prompt("django/django")
        assert "this repo" in prompt or "the repo" in prompt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
