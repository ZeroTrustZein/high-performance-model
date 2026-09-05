"""Server-Sent Events (SSE) MCP Transport and HTTP server integration."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from high_performance_model.protocol.transports import Transport

if TYPE_CHECKING:
    from high_performance_model.protocol.server import MCPServer

logger = logging.getLogger("high_performance_model.protocol.sse")


class SSETransport(Transport):
    """MCP transport using Server-Sent Events (SSE) for server->client and HTTP POST for client->server."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or uuid4().hex
        self.incoming: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.outgoing_events: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._closed = False

    async def read_message(self) -> Optional[str]:
        """Read message posted by client."""
        if self._closed and self.incoming.empty():
            return None
        return await self.incoming.get()

    async def write_message(self, message: str) -> None:
        """Enqueue JSON-RPC message to be sent as SSE event."""
        if self._closed:
            return
        sse_event = f"event: message\ndata: {message}\n\n"
        await self.outgoing_events.put(sse_event)

    async def post_message(self, message: str) -> None:
        """Called by HTTP handler when client POSTs a message."""
        if not self._closed:
            await self.incoming.put(message)

    async def stream_events(self) -> AsyncIterator[str]:
        """Yield SSE formatted events for the client HTTP response stream."""
        # Initial endpoint event per MCP SSE spec
        endpoint_event = f"event: endpoint\ndata: /messages?sessionId={self.session_id}\n\n"
        yield endpoint_event

        while not self._closed:
            event = await self.outgoing_events.get()
            if event is None:
                break
            yield event

    async def close(self) -> None:
        """Close session transport."""
        if not self._closed:
            self._closed = True
            await self.incoming.put(None)
            await self.outgoing_events.put(None)


class SSEServer:
    """Lightweight asynchronous HTTP server serving MCP over SSE and HTTP POST."""

    def __init__(
        self,
        mcp_server: MCPServer,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.mcp_server = mcp_server
        self.host = host
        self.port = port
        self.sessions: Dict[str, SSETransport] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start accepting connections on specified host and port."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        logger.info("SSE MCP Server listening on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop server and clean up sessions."""
        for transport in list(self.sessions.values()):
            await transport.close()
        self.sessions.clear()

        for t in self._tasks:
            t.cancel()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("SSE MCP Server stopped.")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Parse HTTP/1.1 request and route to SSE or POST message handler."""
        try:
            req_line_bytes = await reader.readline()
            if not req_line_bytes:
                writer.close()
                return

            req_line = req_line_bytes.decode("utf-8", errors="ignore").strip()
            parts = req_line.split(" ")
            if len(parts) < 2:
                writer.close()
                return

            method, raw_path = parts[0].upper(), parts[1]

            # Parse headers
            headers: Dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n" or line == b"\n":
                    break
                header_line = line.decode("utf-8", errors="ignore").strip()
                if ":" in header_line:
                    k, v = header_line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            parsed_url = urlparse(raw_path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)

            if method == "GET" and path == "/sse":
                await self._handle_sse_stream(writer)
            elif method == "POST" and path == "/messages":
                await self._handle_post_message(reader, writer, headers, query_params)
            else:
                writer.write(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found"
                )
                await writer.drain()
                writer.close()

        except Exception as exc:
            logger.warning("HTTP connection error: %s", exc)
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_sse_stream(self, writer: asyncio.StreamWriter) -> None:
        """Handle incoming SSE connection request."""
        transport = SSETransport()
        self.sessions[transport.session_id] = transport

        # Launch MCP server runner on this session
        task = asyncio.create_task(self.mcp_server.run(transport))
        self._tasks.append(task)

        # Send SSE HTTP response headers
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n"
        )
        writer.write(headers.encode("utf-8"))
        await writer.drain()

        try:
            async for event in transport.stream_events():
                writer.write(event.encode("utf-8"))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            await transport.close()
            if transport.session_id in self.sessions:
                del self.sessions[transport.session_id]
            writer.close()

    async def _handle_post_message(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: Dict[str, str],
        query_params: Dict[str, list[str]],
    ) -> None:
        """Handle incoming POST message from client to active session."""
        session_ids = query_params.get("sessionId") or query_params.get("session_id")
        session_id = session_ids[0] if session_ids else None

        if not session_id or session_id not in self.sessions:
            writer.write(
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nMissing or invalid sessionId\n"
            )
            await writer.drain()
            writer.close()
            return

        content_length = int(headers.get("content-length", 0))
        if content_length <= 0:
            writer.write(
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nEmpty body\n"
            )
            await writer.drain()
            writer.close()
            return

        body_bytes = await reader.readexactly(content_length)
        body = body_bytes.decode("utf-8")

        transport = self.sessions[session_id]
        await transport.post_message(body)

        writer.write(
            b"HTTP/1.1 202 Accepted\r\nContent-Type: text/plain\r\nContent-Length: 8\r\n\r\nAccepted"
        )
        await writer.drain()
        writer.close()
