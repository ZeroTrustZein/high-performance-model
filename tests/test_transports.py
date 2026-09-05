"""Comprehensive test suite for Model Context Protocol transports (Memory, Stdio, SSE)."""

from __future__ import annotations

import asyncio

import httpx

from high_performance_model.protocol.jsonrpc import parse_message
from high_performance_model.protocol.server import MCPServer
from high_performance_model.protocol.transports import (
    MemoryTransport,
    SSEServer,
    SSETransport,
)


class TestMemoryTransport:
    """Test in-memory bidirectional queue transport."""

    def test_memory_transport_flow(self) -> None:
        async def _run():
            transport = MemoryTransport()
            await transport.feed_input('{"jsonrpc": "2.0", "id": 1, "method": "ping"}')
            msg = await transport.read_message()
            assert msg is not None
            req, err = parse_message(msg)
            assert err is None
            assert req is not None
            assert req.method == "ping"

            await transport.write_message('{"jsonrpc": "2.0", "id": 1, "result": {}}')
            out = await transport.read_output(timeout=0.5)
            assert out is not None
            assert '"result": {}' in out

            await transport.close()
            assert await transport.read_message() is None

        asyncio.run(_run())


class TestSSETransport:
    """Test Server-Sent Events MCP transport."""

    def test_sse_transport_events(self) -> None:
        async def _run():
            transport = SSETransport(session_id="test-session-123")
            assert transport.session_id == "test-session-123"

            # Post a message from client
            await transport.post_message('{"jsonrpc": "2.0", "method": "ping"}')
            msg = await transport.read_message()
            assert msg == '{"jsonrpc": "2.0", "method": "ping"}'

            # Write a response from server
            await transport.write_message('{"jsonrpc": "2.0", "result": "pong"}')

            # Stream events generator
            event_iter = transport.stream_events()
            endpoint_evt = await anext(event_iter)
            assert "event: endpoint" in endpoint_evt
            assert "/messages?sessionId=test-session-123" in endpoint_evt

            msg_evt = await anext(event_iter)
            assert "event: message" in msg_evt
            assert '"result": "pong"' in msg_evt

            await transport.close()

        asyncio.run(_run())


class TestSSEServerIntegration:
    """Test lightweight HTTP/SSE server endpoint routing."""

    def test_sse_server_http_lifecycle(self) -> None:
        async def _run():
            server = MCPServer(name="test-mcp")
            sse_srv = SSEServer(mcp_server=server, host="127.0.0.1", port=18765)
            await sse_srv.start()

            try:
                async with httpx.AsyncClient(base_url="http://127.0.0.1:18765") as client:
                    # 1. 404 on unknown endpoint
                    resp_404 = await client.get("/unknown")
                    assert resp_404.status_code == 404

                    # 2. 400 on POST /messages without sessionId
                    resp_bad = await client.post("/messages", content="{}")
                    assert resp_bad.status_code == 400

                    # 3. Connect to /sse stream
                    async with client.stream("GET", "/sse") as sse_stream:
                        assert sse_stream.status_code == 200
                        assert "text/event-stream" in sse_stream.headers.get("content-type", "")

                        line_iter = sse_stream.aiter_lines()

                        # Read initial endpoint event
                        lines = []
                        async for line in line_iter:
                            lines.append(line)
                            if line == "":
                                break

                        full_evt = "\n".join(lines)
                        assert "event: endpoint" in full_evt
                        assert "sessionId=" in full_evt

                        # Extract session_id
                        sid = full_evt.split("sessionId=")[1].split("\n")[0].strip()

                        # 4. POST initialize request to /messages?sessionId=...
                        init_payload = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "clientInfo": {"name": "test-client", "version": "1.0"},
                                "capabilities": {},
                            },
                        }
                        post_resp = await client.post(
                            f"/messages?sessionId={sid}",
                            json=init_payload,
                        )
                        assert post_resp.status_code == 202

                        # Read server response over SSE stream using same line_iter
                        msg_lines = []
                        async for line in line_iter:
                            msg_lines.append(line)
                            if line == "":
                                break

                        msg_evt = "\n".join(msg_lines)
                        assert "event: message" in msg_evt
                        assert "protocolVersion" in msg_evt

            finally:
                await sse_srv.stop()

        asyncio.run(_run())
