# Long-term memory — extract, store, and inject

import json
import anthropic

client = anthropic.Anthropic()
# ── Extract facts from a conversation ────────────────────────
def extract_facts(conversation: list) -> dict:
    prompt = f"""
    Extract key facts from this conversation.

    Return ONLY valid JSON with these fields:
    {{
      "name": string or null,
      "preferences": [list of stated preferences],
      "expertise_level": "beginner/intermediate/expert" or null,
      "topics_discussed": [list of topics],
      "decisions_made": [list of decisions or conclusions]
    }}

    Conversation:
    {json.dumps(conversation)}
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.content[0].text)


# ── Save to database (DynamoDB, Postgres, SQLite — anything) ──
def save_user_memory(user_id: str, facts: dict):
    # In production: write to DynamoDB, Postgres, Redis, etc.
    with open(f"memory_{user_id}.json", "w") as f:
        json.dump(facts, f)


# ── Load at session start ─────────────────────────────────────
def load_user_memory(user_id: str) -> dict:
    try:
        with open(f"memory_{user_id}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Inject into system prompt ─────────────────────────────────
def build_system_with_memory(user_id: str) -> str:
    memory = load_user_memory(user_id)

    if not memory:
        return "You are a helpful AI assistant."

    return f"""
You are a helpful AI assistant.

What you know about this user:
- Name: {memory.get('name', 'Unknown')}
- Expertise: {memory.get('expertise_level', 'Unknown')}
- Preferences: {', '.join(memory.get('preferences', []))}
- Previously discussed: {', '.join(memory.get('topics_discussed', []))}

Use this context to personalise your responses.
"""
