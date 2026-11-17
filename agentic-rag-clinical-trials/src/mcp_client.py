import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import MCP_SERVER_URL, MCP_TRANSPORT, REPO_ROOT

if MCP_TRANSPORT == "http":
    MCP_SERVERS = {
        "tools": {
            "transport": "streamable_http",
            "url": MCP_SERVER_URL,
        }
    }
else:
    MCP_SERVERS = {
        "tools": {
            "command": sys.executable,
            "args": ["-m", "src.mcp_server"],
            "cwd": str(REPO_ROOT),
            "transport": "stdio",
        }
    }


async def get_tools():
    """spins up the mcp client and fetches every tool the server exposes -
    retrieval plus all six AACT tools. returns them ready to hand straight
    to create_agent."""
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()
