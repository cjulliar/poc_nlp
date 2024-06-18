import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AnyMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.messages.ai import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from rich import print as rprint  # To not overwrite built-in print
from typing_extensions import TypedDict


# Source environment
load_dotenv()

# Define AgentState
class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], operator.add]


# Define Agent
class Agent:
  def __init__(self, model, tools: list, checkpointer, system: str = "") -> None:
    # Store system prompt
    self.system = system
    # Build and compile graph to be stored as attribute
    graph_builder = StateGraph(AgentState)
    ## Nodes
    graph_builder.add_node("llm", self.call_openai)
    graph_builder.add_node("action", self.take_action)
    ## Edges
    graph_builder.add_conditional_edges(
      "llm",  # Initial edge
      self.exists_action,  # Decision function
      # Map self.exists_action output and destination node
      {True: "action", False: END}
    )
    graph_builder.add_edge("action", "llm")  # Feed llm with action result
    ## Set entry point, compile and store
    graph_builder.set_entry_point("llm")
    self.graph = graph_builder.compile(checkpointer=checkpointer)
    self.tools = {tool.name: tool for tool in tools}
    self.model = model.bind_tools(tools)
    
  def call_openai(self, state: AgentState) -> AgentState:
    """Defines the function within the 'llm' node"""
    messages = state["messages"]
    if self.system:
      # prepend system message
      messages = [SystemMessage(content=self.system)] + messages
    message = self.model.invoke(messages)
    return {"messages": [message]}
    
  def exists_action(self, state: AgentState) -> AgentState:
    """Checks if an action is to take according to presence of tool calls"""
    last_message = state["messages"][-1]
    return len(last_message.tool_calls) > 0
    
  def take_action(self, state: AgentState) -> AgentState:
    """Defines the function within the 'action' node"""
    rprint("[bold red]To asnwer this, I need to browse the Web...[/bold red]")
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tool in tool_calls:
      rprint(f"Calling: {tool}")
      # Retrieve tool result
      tool_result = self.tools[tool["name"]].invoke(tool["args"])
      # Transmit corresponding ToolMessage
      results.append(
        ToolMessage(
          tool_call_id = tool["id"],
          name = tool["name"],
          content = str(tool_result)
        )
      )
    rprint("[bold blue]Tool results transmitted to the model[/bold blue]")
    return {"messages": results}


# Set arguments, then instanciate
model = ChatOpenAI(model="gpt-3.5-turbo")
tools = [TavilySearchResults(max_results=4)]
memory = SqliteSaver.from_conn_string(":memory:")
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

agent = Agent(model, tools, system=prompt, checkpointer=memory)


if __name__ == "__main__":
  rprint(
    "[bold yellow]Conversational Chatbot - When done type quit, exit or q, then ENTER[/bold yellow]")
  while True:
    user_input = input("User: ")
    if user_input.lower() in ("quit", "exit", "q"):
        rprint("[bold yellow]Goodbye![/bold yellow]")
        break
    for event in agent.graph.stream(
      {"messages": [HumanMessage(content=user_input)]},
      {"configurable": {"thread_id": "1"}}
    ):
        for value in event.values():
          message = value["messages"][-1]
          if isinstance(message, AIMessage) and message.content:
            rprint(message.content)
