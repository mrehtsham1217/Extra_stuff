# compulsor for adding extra tools
# tool condition-->if LLM doesnt have knowledge of query it should give this query to tools-->google wikipedia

from langgraph.prebuilt import ToolNode, tools_condition

# tools for performing retrievals
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# tool for searching in tavily
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# for adding nodes in Graph nodes start end
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Stategraph is used to manage the state and START-->Starting Node and Ending node
# message history is added and state will get change and keep the track of all things

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)


Search = TavilySearchAPIWrapper(tavily_api_key=os.getenv('TAVILY_API_KEY'))
tavily_tool = TavilySearchResults(api_wrapper=Search)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq LLM with your GroQ API key
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="Gemma2-9b-It"
)
tools = [wikipedia_tool,
         arxiv_tool,
         tavily_tool]

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
graph_builder


def chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}


graph_builder.add_node("Chatbot", chatbot)
graph_builder.add_edge(START, "Chatbot")
graph_builder.add_edge("Chatbot", END)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition

)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    for events in graph.stream({"messages": ("user", user_input)}):
        print(events.values())
