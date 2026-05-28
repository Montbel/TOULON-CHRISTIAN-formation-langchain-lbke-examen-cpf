# Where we connect to MCP servers as client
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

dir_path = os.path.dirname(os.path.realpath(__file__))

mcp_client = MultiServerMCPClient(
    {
        # Ici le client déclenche le serveur lui-même,
        # mais le serveur
        "math": {
            "transport": "stdio",  # Local subprocess communication
            "command": "python",
            # Absolute path to your math_server.py file
            "args": [os.path.join(dir_path, "mcp_stdio_math_server.py")],
        },
        #  "weather": {
        #      "transport": "http",  # HTTP-based remote server
        #      # Ensure you start your weather server on port 8000
        #      "url": "http://localhost:8000/mcp",
        #  }
    }
)
