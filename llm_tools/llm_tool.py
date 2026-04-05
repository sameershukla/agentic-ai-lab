import anthropic
import json

client = anthropic.Anthropic()

# ── Step 1: Define the tool ───────────────────────────────────
get_weather_tool = {
    "name": "get_current_weather",
    "description": (
        "Get the current weather for a city. Use this when "
        "the user asks about weather, temperature, or conditions "
        "in a specific location. Returns temperature in Celsius "
        "and a conditions description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. London, Tokyo"
            }
        },
        "required": ["city"]
    }
}

# ── Step 2: Your actual function (the real logic) ─────────────
# This is what Claude "calls" — but YOU run it, not Claude.
def get_current_weather(city: str) -> dict:
    # In production this calls a real weather API
    # For learning we simulate a response
    return {
        "city": city,
        "temperature_celsius": 18,
        "conditions": "Partly cloudy with light breeze"
    }

# ── Step 3: Send message + tool to Claude ─────────────────────
messages = [
    {"role": "user", "content": "What's the weather in London?"}
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    tools=[get_weather_tool],  # ← tools available to Claude
    messages=messages
)

# ── Step 4: Check if Claude wants to use a tool ───────────────
if response.stop_reason == "tool_use":
    # Extract the tool call Claude made
    tool_block = next(b for b in response.content if b.type == "tool_use")

    tool_name = tool_block.name          # "get_current_weather"
    tool_args = tool_block.input         # {"city": "London"}
    tool_id = tool_block.id              # unique ID for this call

    print(f"Claude wants to call: {tool_name}({tool_args})")

    # ── Step 5: YOU run the function ──────────────────────────
    result = get_current_weather(**tool_args)

    # ── Step 6: Send result back to Claude ────────────────────
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,  # links result to the call
                "content": json.dumps(result)  # your function's output
            }
        ]
    })

    # ── Step 7: Claude reads result and answers ───────────────
    final_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        tools=[get_weather_tool],
        messages=messages
    )

    print(final_response.content[0].text)
    # Example output:
    # "The current weather in London is 18°C with partly cloudy
    # skies and a light breeze."
