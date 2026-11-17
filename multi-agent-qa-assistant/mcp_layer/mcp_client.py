"""
Thin wrapper around the MCP client SDK so any node in the graph (see
graph.py) can call a tool on mcp-server without repeating the connection
boilerplate every time.

This file runs inside the `backend` container (see Dockerfile.backend) --
it is the actual MCP *client* in this system, reaching out over the
network to the `mcp-server` container rather than spawning it as a local
subprocess. MCP_SERVER_URL defaults to the local port mapping in
docker-compose.yml, but inside Compose itself it's overridden to
http://mcp-server:8000/mcp so the two containers find each other by
service name instead of localhost.
"""

import os
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


@asynccontextmanager
async def mcp_session():
    async with streamable_http_client(MCP_SERVER_URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(tool_name: str, **kwargs) -> str:
    """Call a single MCP tool on mcp-server by name and return its text
    result. Opens a fresh session per call -- simple and correct, at the
    cost of a small per-call connection overhead instead of a pooled,
    reused session."""
    async with mcp_session() as session:
        result = await session.call_tool(tool_name, arguments=kwargs)
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )
