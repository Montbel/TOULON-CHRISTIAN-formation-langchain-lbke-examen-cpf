import os

from langgraph_sdk import get_client
import asyncio
# Va se connecter à l'agent en local
# Vous pouvez aussi tester avec l'adresse Render
client = get_client(url=os.environ.get(
    "LANGGRAPH_URL", "http://127.0.0.1:2024"))
# Get stream


async def call():
    print("Available assistants: ", await client.assistants.search())
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving chunk (type: {chunk.event}):")
        print(chunk.data)
        print("\n\n")
if __name__ == "__main__":
    asyncio.run(call())
