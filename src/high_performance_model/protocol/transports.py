"""Transport abstractions for Model Context Protocol communication."""

from __future__ import annotations

import abc
import asyncio
import sys
from typing import AsyncIterator, Optional


class Transport(abc.ABC):
    """Abstract MCP transport interface."""

    @abc.abstractmethod
    async def read_message(self) -> Optional[str]:
        """Read single serialized message line."""
        raise NotImplementedError

    @abc.abstractmethod
    async def write_message(self, message: str) -> None:
        """Write single serialized message line."""
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Close transport resources."""
        raise NotImplementedError

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over incoming messages until transport closes."""
        while True:
            msg = await self.read_message()
            if msg is None:
                break
            yield msg


class MemoryTransport(Transport):
    """In-memory bidirectional queue transport for testing and local benchmarking."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.outgoing: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._closed = False

    async def read_message(self) -> Optional[str]:
        if self._closed and self.incoming.empty():
            return None
        return await self.incoming.get()

    async def write_message(self, message: str) -> None:
        await self.outgoing.put(message)

    async def feed_input(self, message: str) -> None:
        """Helper to feed an input request into the server transport."""
        await self.incoming.put(message)

    async def feed_eof(self) -> None:
        """Signal end of incoming stream without preventing final response writes."""
        await self.incoming.put(None)

    async def read_output(self, timeout: float = 2.0) -> Optional[str]:
        """Helper to read outgoing response from the server transport."""
        try:
            return await asyncio.wait_for(self.outgoing.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        self._closed = True
        await self.incoming.put(None)


class StdioTransport(Transport):
    """Asynchronous standard I/O transport using line-delimited JSON-RPC."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False

    async def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    async def read_message(self) -> Optional[str]:
        if self._closed:
            return None
        loop = await self._get_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return None
        return line.strip()

    async def write_message(self, message: str) -> None:
        if self._closed:
            return
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    async def close(self) -> None:
        self._closed = True


from high_performance_model.protocol.sse import SSEServer, SSETransport  # noqa: E402

__all__ = ["Transport", "MemoryTransport", "StdioTransport", "SSETransport", "SSEServer"]
