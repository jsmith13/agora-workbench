"""Tests for direct code-execution MCP tool registration."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client

from .. import CodeExecutionServer, ServerConfig
from ..auth import create_noop_auth_config
from ..code_execution_models import CodeExecutionResult, ToolCallRecord


class _SessionResourceOperation:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_tool_has_no_duplicate_structured_content():
    server = CodeExecutionServer(
        server_config=ServerConfig(
            name="test_direct",
            description="Test direct code execution",
            type="uv",
            dependency_file="# empty",
        ),
        auth_config=create_noop_auth_config(),
    )
    tool_name = server.get_tool_name()
    tool = await server.mcp.get_tool(tool_name)
    assert tool.output_schema is None

    session = SimpleNamespace(
        session_id="test-session",
        extensions={},
        data_manager=MagicMock(),
    )
    server._get_or_create_session = AsyncMock(return_value=session)
    server._inject_tool_proxies = AsyncMock()
    server._execute_code_with_tracing = AsyncMock(
        return_value=CodeExecutionResult(
            stdout="hello\n",
            tool_calls=[
                ToolCallRecord(
                    tool_name="successful_tool",
                    args={"large": "input"},
                    result={"large": "output"},
                )
            ],
        ),
    )
    server.session_manager.session_resource_operation = MagicMock(return_value=_SessionResourceOperation())
    server.session_manager.update_session = MagicMock()
    server.activity_publisher.publish_nowait = MagicMock()

    async with Client(server.mcp) as client:
        result = await client.call_tool(tool_name, {"code": "print('hello')"})

    payload = json.loads(result.content[0].text)
    assert payload["stdout"] == "hello\n"
    assert "tool_calls" not in payload
    assert "failed_tool_calls" not in payload
    assert result.structured_content is None
    activity_event = server.activity_publisher.publish_nowait.call_args.args[0]
    assert activity_event["tool_calls"][0]["result"] == {"large": "output"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_tool_omits_failed_internal_tool_calls():
    server = CodeExecutionServer(
        server_config=ServerConfig(
            name="test_direct",
            description="Test direct code execution",
            type="uv",
            dependency_file="# empty",
        ),
        auth_config=create_noop_auth_config(),
    )
    tool_name = server.get_tool_name()
    session = SimpleNamespace(
        session_id="test-session",
        extensions={},
        data_manager=MagicMock(),
    )
    server._get_or_create_session = AsyncMock(return_value=session)
    server._inject_tool_proxies = AsyncMock()
    server._execute_code_with_tracing = AsyncMock(
        return_value=CodeExecutionResult(
            stdout="continued after a caught error\n",
            success=True,
            tool_calls=[
                ToolCallRecord(
                    tool_name="successful_tool",
                    args={"large": "input"},
                    result={"large": "output"},
                    duration_ms=10,
                ),
                ToolCallRecord(
                    tool_name="failed_tool",
                    args={"secret": "not returned"},
                    result={"large": "not returned"},
                    duration_ms=20,
                    success=False,
                    error="ValueError: invalid input",
                ),
            ],
        )
    )
    server.session_manager.session_resource_operation = MagicMock(return_value=_SessionResourceOperation())
    server.session_manager.update_session = MagicMock()
    server.activity_publisher.publish_nowait = MagicMock()

    async with Client(server.mcp) as client:
        result = await client.call_tool(tool_name, {"code": "run_tools()"})

    payload = json.loads(result.content[0].text)
    assert payload["success"] is True
    assert "tool_calls" not in payload
    assert "failed_tool_calls" not in payload
