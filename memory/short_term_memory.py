# Short-term memory — the messages list IS the memory
import anthropic

messages = []
client = anthropic.Anthropic()

def chat(user_input: str) -> str:
    # Add user message to history
    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=messages  # ← entire history sent every call
    )

    reply = response.content[0].text

    # Add assistant response to history
    messages.append({
        "role": "assistant",
        "content": reply
    })

    return reply


# The model remembers everything said in THIS session

chat("My name is Sameer")
chat("What is RAG?")
chat("Give me an example relevant to my background")

# ↑ Model knows "my background" = Sameer's context from turn 1
