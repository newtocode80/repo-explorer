"""
Two real, live external API tools -- no API key required for either.

GitHub REST API: public repo metadata. Unauthenticated requests are capped
at 60/hour per IP, which is realistic production behavior to design around
(rate limits, not just happy-path responses).

Open-Meteo: free weather API, no signup, no key. Good for demos because it
removes "go get another API key" friction while still being a real,
production-grade external service.

This is the step-1 building block in repo_explorer's incremental build:
later steps will use github_lookup as part of indexing a full repo, and
the error-handling patterns here (timeouts, non-200 status, rate limits)
are reused in the more complex tools.
"""

import requests

GITHUB_API_BASE = "https://api.github.com"
WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast"

# Small fixed lookup so the weather tool doesn't need a geocoding API too.
# Real production version would call a geocoding endpoint; kept simple here
# since the point of this exercise is the tool-use pattern, not geocoding.
CITY_COORDS = {
    "montreal": (45.50, -73.57),
    "new york": (40.71, -74.01),
    "san francisco": (37.77, -122.42),
    "london": (51.51, -0.13),
    "toronto": (43.65, -79.38),
    "paris": (48.85, 2.35),
}


def github_lookup(repo_full_name: str) -> str:
    """
    Fetch public metadata for a GitHub repo, e.g. 'anthropics/anthropic-sdk-python'.
    Handles the realistic failure modes: repo not found, rate limit hit,
    network timeout -- each returns a distinct, informative message instead
    of crashing, so the agent can relay something useful to the user.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        return "Error: GitHub API request timed out. Try again shortly."
    except requests.exceptions.ConnectionError:
        return "Error: could not connect to GitHub API. Check network connectivity."

    if response.status_code == 404:
        return f"Error: repo '{repo_full_name}' not found. Check the owner/name spelling."

    if response.status_code == 403:
        # GitHub returns 403 for rate limiting, not 429 -- a real API quirk
        # worth knowing rather than assuming standard HTTP status conventions.
        return (
            "Error: GitHub API rate limit exceeded for this IP. "
            "Unauthenticated requests are capped at 60/hour. "
            "Try again later, or use an authenticated token for higher limits."
        )

    if response.status_code != 200:
        return f"Error: GitHub API returned unexpected status {response.status_code}."

    data = response.json()
    return (
        f"Repo: {data.get('full_name')}\n"
        f"Description: {data.get('description') or 'No description'}\n"
        f"Language: {data.get('language') or 'Not specified'}\n"
        f"Stars: {data.get('stargazers_count')}\n"
        f"Open issues: {data.get('open_issues_count')}\n"
        f"Default branch: {data.get('default_branch')}"
    )


def weather_lookup(city: str) -> str:
    """
    Get current weather for a small set of supported cities via Open-Meteo.
    Same error-handling shape as github_lookup -- distinct messages for
    each realistic failure mode rather than a generic try/except swallow.
    """
    city_key = city.strip().lower()
    if city_key not in CITY_COORDS:
        supported = ", ".join(sorted(CITY_COORDS.keys()))
        return f"Error: '{city}' isn't in the supported city list ({supported})."

    lat, lon = CITY_COORDS[city_key]
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius",
    }

    try:
        response = requests.get(WEATHER_API_BASE, params=params, timeout=10)
    except requests.exceptions.Timeout:
        return "Error: weather API request timed out. Try again shortly."
    except requests.exceptions.ConnectionError:
        return "Error: could not connect to weather API. Check network connectivity."

    if response.status_code != 200:
        return f"Error: weather API returned unexpected status {response.status_code}."

    data = response.json()
    current = data.get("current", {})
    temp = current.get("temperature_2m")
    if temp is None:
        return "Error: weather API response was missing expected data."

    return f"Current temperature in {city.title()}: {temp}°C"


# --- Tool schemas, in the same shape agent.py already expects ---

GITHUB_TOOL_SCHEMA = {
    "name": "github_lookup",
    "description": "Look up public metadata for a GitHub repository "
    "(stars, description, language, open issues). Requires the repo's "
    "full name in 'owner/repo' format, e.g. 'anthropics/anthropic-sdk-python'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_full_name": {
                "type": "string",
                "description": "Repo in 'owner/repo' format",
            }
        },
        "required": ["repo_full_name"],
    },
}

WEATHER_TOOL_SCHEMA = {
    "name": "weather_lookup",
    "description": "Get the current temperature for a supported city "
    "(Montreal, New York, San Francisco, London, Toronto, Paris).",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
}


if __name__ == "__main__":
    print(github_lookup("anthropics/anthropic-sdk-python"))
    print()
    print(github_lookup("this-repo/does-not-exist-xyz123"))
    print()
    print(weather_lookup("Montreal"))
    print()
    print(weather_lookup("Atlantis"))
