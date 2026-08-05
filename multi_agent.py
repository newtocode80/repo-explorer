import os
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from api_tools import GITHUB_TOOL_SCHEMA, WEATHER_TOOL_SCHEMA, github_lookup, weather_lookup
from readme_rag import README_TOOL_SCHEMA, readme_lookup

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4.6"

def route(user_question: str) -> str:
    response = client.messages.create(
        model=MODEL,    
        max_tokens=10,
        system="You are a router. Respond with only one word: 'repo' if the question is about a GitHub repository, or 'weather' if it is about the weather."
        messages=[{"role": "user", "content": user_question}]
    )
    return response.content[0].text.strip().lower()


