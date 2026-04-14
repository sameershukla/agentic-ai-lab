"""
Supervisor Multi-Agent — LangGraph
=====================================
Book companion: "From Tokens to Agents" — Part 5 (Multi-Agent Systems)

Pattern: A Supervisor LLM reads the user's request and delegates sub-tasks
to specialised Worker agents.  Workers report back; the Supervisor decides
whether to call another worker or compose the final answer.

Agents in this example (travel planning domain):
  ┌─────────────────────────────────────────────┐
  │              SUPERVISOR                     │
  │  "Which agent should handle this next?"     │
  └──────┬──────────────┬───────────────┬───────┘
         │              │               │
    [flight_agent] [hotel_agent] [weather_agent]
    searches &     finds hotels   checks weather
    books flights  and prices     for destination

Flow:
    User query
        ↓
    [supervisor] ── picks worker ──► [worker]
        ▲                                 │
        └──── worker result sent back ────┘
        ↓ (when supervisor says FINISH)
       END
"""

from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# ── 1. State ─────────────────────────────────────────────────────────────────
#  Shared across every node in the graph.

class MultiAgentState(TypedDict):
    messages:   Annotated[list, add_messages]  # full conversation history
    next_agent: str                            # supervisor's routing decision


# ── 2. Tools  (one focused toolset per worker) ───────────────────────────────

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights between two cities."""
    return (
        f"Flights found for {origin}→{destination} on {date}:\n"
        f"  • AI-101  08:00  $320  (non-stop)\n"
        f"  • AI-205  14:30  $275  (1 stop, +3h)"
    )

@tool
def book_flight(flight_id: str) -> str:
    """Book a specific flight by its ID."""
    return f"Flight {flight_id} booked successfully. PNR: {abs(hash(flight_id)) % 99999:05d}"

@tool
def search_hotels(city: str, check_in: str, check_out: str) -> str:
    """Search hotels in a city for given dates."""
    return (
        f"Hotels in {city} ({check_in} → {check_out}):\n"
        f"  • Grand Plaza    ★★★★   $180/night  (city centre)\n"
        f"  • Harbour Inn    ★★★    $110/night  (2km from centre)"
    )

@tool
def book_hotel(hotel_name: str, city: str) -> str:
    """Book a hotel by name."""
    return f"{hotel_name} in {city} confirmed. Booking ref: HTL-{abs(hash(hotel_name)) % 9999:04d}"

@tool
def get_weather(city: str, date: str) -> str:
    """Get the weather forecast for a city on a specific date."""
    return f"{city} on {date}: 24°C, light breeze, no rain expected — great travel weather!"


# ── 3. Structured output: how the supervisor signals routing ─────────────────

WORKERS = ["flight_agent", "hotel_agent", "weather_agent"]

class SupervisorDecision(BaseModel):
    """Supervisor's routing decision at each step."""
    next: Literal["flight_agent", "hotel_agent", "weather_agent", "FINISH"]
    reason: str   # short explanation — useful for tracing / book demos


# ── 4. LLMs ──────────────────────────────────────────────────────────────────

base_llm = ChatAnthropic(model="claude-3-5-haiku-20241022")

# Supervisor uses structured output so we can reliably extract .next
supervisor_llm = base_llm.with_structured_output(SupervisorDecision)

# Each worker binds only its own tools (principle of least privilege)
flight_llm  = base_llm.bind_tools([search_flights, book_flight])
hotel_llm   = base_llm.bind_tools([search_hotels,  book_hotel])
weather_llm = base_llm.bind_tools([get_weather])


# ── 5. Supervisor node ───────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are a travel planning coordinator managing a team of specialist agents.

Your team:
  - flight_agent  : searches and books flights
  - hotel_agent   : searches and books hotels
  - weather_agent : retrieves weather forecasts

Given the conversation so far, decide:
  1. Which agent should act next (if any work remains), OR
  2. "FINISH" if all requested tasks are complete and you can summarise.

Always delegate before finishing. Only say FINISH when every sub-task is done."""

def supervisor_node(state: MultiAgentState) -> dict:
    """
    The supervisor reads the full conversation and decides what happens next.
    It uses structured output so the routing is always machine-readable.
    """
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]
    decision: SupervisorDecision = supervisor_llm.invoke(messages)

    print(f"\n[Supervisor] → {decision.next}  |  reason: {decision.reason}")

    return {"next_agent": decision.next}


# ── 6. Worker node factory ───────────────────────────────────────────────────

def make_worker(name: str, system_prompt: str, llm) -> callable:
    """
    Returns a node function for a worker agent.
    Each worker: reads the conversation, acts with its tools, reports back.
    """
    def worker_node(state: MultiAgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        # If the LLM called a tool, execute it and append the result
        tool_messages = []
        if isinstance(response, AIMessage) and response.tool_calls:
            from langchain_core.messages import ToolMessage
            from langchain_core.tools import BaseTool

            tools_map = {t.name: t for t in llm.kwargs["tools"]}   # bound tools
            for tc in response.tool_calls:
                tool_fn = tools_map.get(tc["name"])
                if tool_fn:
                    result  = tool_fn.invoke(tc["args"])
                    tool_messages.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )

        # Tag the final message so the supervisor knows who reported
        final_content = response.content or ""
        if tool_messages:
            # Summarise tool results in a follow-up AI message
            summary = base_llm.invoke(
                messages + [response] + tool_messages +
                [HumanMessage(content="Summarise what you just did in one sentence.")]
            )
            final_content = summary.content

        tagged = AIMessage(content=f"[{name}]: {final_content}")
        print(f"\n[{name}]: {final_content[:120]}{'…' if len(final_content) > 120 else ''}")
        return {"messages": [tagged]}

    worker_node.__name__ = name
    return worker_node


# Build the three workers
flight_agent = make_worker(
    "flight_agent",
    "You are a flight specialist. Search and book flights as requested. "
    "Always confirm the booking reference when done.",
    flight_llm
)

hotel_agent = make_worker(
    "hotel_agent",
    "You are a hotel specialist. Search and book hotels as requested. "
    "Always confirm the booking reference when done.",
    hotel_llm
)

weather_agent = make_worker(
    "weather_agent",
    "You are a weather specialist. Retrieve and report weather forecasts clearly.",
    weather_llm
)


# ── 7. Graph ──────────────────────────────────────────────────────────────────

def route_from_supervisor(state: MultiAgentState) -> str:
    """Map the supervisor's decision string to the next graph node."""
    return state["next_agent"]   # "flight_agent" | "hotel_agent" | "weather_agent" | "FINISH"


builder = StateGraph(MultiAgentState)

# Add all nodes
builder.add_node("supervisor",    supervisor_node)
builder.add_node("flight_agent",  flight_agent)
builder.add_node("hotel_agent",   hotel_agent)
builder.add_node("weather_agent", weather_agent)

# Entry point → supervisor always goes first
builder.add_edge(START, "supervisor")

# Supervisor routes to a worker OR ends
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "flight_agent":  "flight_agent",
        "hotel_agent":   "hotel_agent",
        "weather_agent": "weather_agent",
        "FINISH":        END,
    }
)

# Every worker reports back to the supervisor after acting
builder.add_edge("flight_agent",  "supervisor")
builder.add_edge("hotel_agent",   "supervisor")
builder.add_edge("weather_agent", "supervisor")

graph = builder.compile()


# ── 8. Runner ─────────────────────────────────────────────────────────────────

def run(query: str) -> str:
    """Invoke the multi-agent graph and return the final supervisor summary."""
    print(f"\n{'═'*60}")
    print(f"User: {query}")
    print(f"{'═'*60}")

    final_state = graph.invoke({
        "messages":   [HumanMessage(content=query)],
        "next_agent": ""
    })

    # The last AI message in state is the supervisor's final summary
    last_ai = next(
        (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    answer = last_ai.content if last_ai else "No response."
    print(f"\n{'─'*60}")
    print(f"Final answer:\n{answer}")
    return answer


# ── 9. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Requires flight_agent + hotel_agent + weather_agent
    run(
        "I need to fly from Mumbai to Paris on 2025-08-10, "
        "book a hotel for 5 nights, and check what the weather will be like."
    )
