"""
Step 3 of the Repo Explorer build: interactive, general-purpose Repo Q&A.

The agent can now answer questions about ANY public GitHub repo, not just
hardcoded examples. Key additions over step 2:

1. Interactive REPL loop -- user types questions, agent responds, conversation
   continues until the user types 'quit'.
2. Persistent conversation history -- context carries across turns, so "what
   about its license?" after a repo question works without restating the repo.
3. Context repo tracking -- once a repo is mentioned, subsequent questions
   assume the same repo until a new one is specified. The system prompt
   informs the model of the current context repo each turn.
4. run_agent now takes existing history as input and returns the updated
   history, enabling stateful multi-turn conversations.
"""

import os
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from api_tools import GITHUB_TOOL_SCHEMA, WEATHER_TOOL_SCHEMA, github_lookup, weather_lookup
from readme_rag import README_TOOL_SCHEMA, readme_lookup

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

BASE_SYSTEM_PROMPT = """You are a helpful assistant that can answer questions about
GitHub repositories and current weather. You have three tools:

- github_lookup: repo metadata (stars, language, open issues). Use for
  questions about a repo's stats or existence.
- readme_lookup: search a repo's README content. Use for detailed questions
  about how a project works, its features, installation, or usage.
- weather_lookup: current temperature for supported cities
  (Montreal, New York, San Francisco, London, Toronto, Paris).

Guidelines:
- For broad repo questions, use BOTH github_lookup (stats) AND readme_lookup
  (content) in the same step.
- If the user mentions a repo without the full owner/name format, ask for
  clarification or infer from context.
- If a tool returns an error, explain it clearly.
- Keep answers concise but complete."""

TOOLS = [GITHUB_TOOL_SCHEMA, WEATHER_TOOL_SCHEMA, README_TOOL_SCHEMA]

TOOL_IMPLEMENTATIONS = {
    "github_lookup": lambda inp: github_lookup(inp["repo_full_name"]),
    "weather_lookup": lambda inp: weather_lookup(inp["city"]),
    "readme_lookup": lambda inp: readme_lookup(inp["repo_full_name"], inp["query"]),
}

# Pattern to extract owner/repo from a user message or tool call
REPO_PATTERN = re.compile(r'\b([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)\b')


def extract_repo_from_text(text: str) -> str | None:
    """Pull the first owner/repo pattern out of a string, if any."""
    match = REPO_PATTERN.search(text)
    return match.group(1) if match else None


def build_system_prompt(context_repo: str | None) -> str:
    """Inject the current context repo into the system prompt each turn."""
    if context_repo:
        return (
            BASE_SYSTEM_PROMPT
            + f"\n\nCurrent context repo: {context_repo}. If the user asks about "
            f"'it', 'this repo', 'the repo', or similar, assume they mean {context_repo}."
        )
    return BASE_SYSTEM_PROMPT


def run_turn(
    user_message: str,
    history: list,
    context_repo: str | None,
    max_steps: int = 8,
    verbose: bool = True,
) -> tuple[str, list, str | None]:
    """
    Run one conversational turn.

    Takes existing history + new user message, returns:
      (assistant_answer, updated_history, updated_context_repo)

    Keeping history outside run_turn (passed in, returned out) means the
    REPL loop owns the state -- cleaner than hiding it inside the function
    as a global, and easier to test or extend later.
    """
    # Check if the user mentioned a new repo in this message
    mentioned_repo = extract_repo_from_text(user_message)
    if mentioned_repo:
        context_repo = mentioned_repo

    messages = history + [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=build_system_prompt(context_repo),
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  [step {step}] model says: {block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"  [step {step}] model calls: {block.name}({block.input})")

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")

            # Check if the model's tool calls mentioned a repo we should track
            for block in response.content:
                if block.type == "tool_use" and "repo_full_name" in block.input:
                    context_repo = block.input["repo_full_name"]

            # Update history with this full turn
            updated_history = messages + [
                {"role": "assistant", "content": response.content}
            ]
            return final_text.strip(), updated_history, context_repo

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Track repos mentioned in tool calls
                if "repo_full_name" in block.input:
                    context_repo = block.input["repo_full_name"]

                impl = TOOL_IMPLEMENTATIONS.get(block.name)
                result = impl(block.input) if impl else "Error: unknown tool"
                if verbose:
                    preview = result[:150].replace('\n', ' ')
                    print(f"  [step {step}] tool result: {preview}...")
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
        messages.append({"role": "user", "content": tool_results})

    return "Error: max steps reached.", messages, context_repo


def run_repl():
    """
    Interactive REPL loop. Maintains conversation history and context repo
    across turns. Type 'quit' or 'exit' to stop, 'reset' to clear context.
    """
    print("Repo Explorer — ask me about any public GitHub repo or the weather.")
    print("Examples:")
    print("  'Tell me about anthropics/anthropic-sdk-python'")
    print("  'How do I install it?'  (follow-up, same repo)")
    print("  'What's the weather in Montreal?'")
    print("Type 'reset' to clear conversation, 'quit' to exit.\n")

    history = []
    context_repo = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            history = []
            context_repo = None
            print("Conversation reset.\n")
            continue

        answer, history, context_repo = run_turn(
            user_input, history, context_repo, verbose=True
        )
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    run_repl()
