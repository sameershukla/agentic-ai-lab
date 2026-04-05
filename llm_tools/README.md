# Function Calling with Anthropic

This example shows how tool use works with Anthropic models.

The key idea is simple:

The model does not run your Python function directly.  
It only decides which tool should be used and what arguments should be passed.  
Your Python code then runs the actual function, sends the result back to the model, and the model uses that result to generate the final answer.

In this example, the model is given access to a weather tool called `get_current_weather`.

---

## What this example demonstrates

This code explains the full flow of function calling:

1. Define a tool and describe what it does
2. Send the tool definition to Claude
3. Let Claude decide whether the tool should be used
4. Extract the tool call from the response
5. Run the actual Python function yourself
6. Send the tool result back to Claude
7. Get the final natural language answer

---

## File purpose

This script is a simple learning example for understanding:

- tool use
- function calling
- tool definitions
- tool execution flow
- how Claude interacts with external functions

---

## Code flow explained

### Step 1: Define the tool

You define a tool in Python as a dictionary.

It includes:

- `name`  
  The tool name the model will use

- `description`  
  A clear explanation of what the tool does, when it should be used, and what it returns

- `input_schema`  
  A JSON schema that tells the model what arguments the tool accepts

Example:

```python
get_weather_tool = {
    "name": "get_current_weather",
    "description": "...",
    "input_schema": {...}
}
