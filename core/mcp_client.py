"""Async client helpers for talking to the local MCP filesystem server."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Absolute path to the MCP server script that sits next to this package.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "mcp_server.py")


@asynccontextmanager
async def mcp_session():
    """Start the MCP server subprocess and yield an initialized session."""
    params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(session: ClientSession, name: str, arguments: dict) -> Any:
    """Call an MCP tool and return its decoded JSON payload.

    Raises RuntimeError with a readable message when the tool reports an error
    or returns output that is not valid JSON (instead of a cryptic JSON error).
    """
    result = await session.call_tool(name, arguments)
    text = ""
    for chunk in result.content:
        if getattr(chunk, "type", None) == "text":
            text += chunk.text
    text = text.strip()

    if getattr(result, "isError", False):
        raise RuntimeError(f"tool '{name}' failed: {text or 'unknown error'}")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"tool '{name}' returned non-JSON output: {text[:300]}")


async def open_path(path: str) -> dict:
    """Open a file or folder on the host via the MCP server. Returns the result dict."""
    result = None
    completed = False
    try:
        async with mcp_session() as session:
            result = await call_tool(session, "open_path", {"path": path})
            completed = True
    except BaseException:
        # Ignore MCP subprocess teardown noise once the call has completed.
        if not completed:
            raise
    return result or {}
