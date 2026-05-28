import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

model = ChatOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],

    base_url="https://openrouter.ai/api/v1",
    name="mistralai/codestral-2508"
)


@tool
def demo_tool():
    """Dummy tool"""
    print("tool call")
    return "Demo tool, I do nothing"


agent = create_agent(
    model=model,
    tools=[demo_tool]
)
