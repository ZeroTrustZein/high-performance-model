"""High-performance Model Context Protocol (MCP) server dispatcher."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from high_performance_model.protocol.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    make_error_response,
    make_success_response,
    parse_message,
    serialize_response,
)
from high_performance_model.protocol.models import (
    CallToolResult,
    GetPromptResult,
    Implementation,
    InitializeResult,
    JsonRpcRequest,
    JsonRpcResponse,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ServerCapabilities,
    TextContent,
    Tool,
)
from high_performance_model.protocol.transports import Transport

logger = logging.getLogger("high_performance_model.protocol.server")

HandlerFunc = Callable[..., Awaitable[Any]]


class MCPServer:
    """Core Model Context Protocol server."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self._tool_handlers: Dict[str, HandlerFunc] = {}
        self.resources: Dict[str, Resource] = {}
        self._resource_handlers: Dict[str, HandlerFunc] = {}
        self.prompts: Dict[str, Prompt] = {}
        self._prompt_handlers: Dict[str, HandlerFunc] = {}
        self._is_initialized = False

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable[[HandlerFunc], HandlerFunc]:
        """Decorator to register a tool handler."""

        def decorator(func: HandlerFunc) -> HandlerFunc:
            tool_name = name or func.__name__
            doc = description or (inspect.getdoc(func) or f"Tool {tool_name}")
            schema = input_schema or {"type": "object", "properties": {}}

            tool_obj = Tool(name=tool_name, description=doc, inputSchema=schema)
            self.tools[tool_name] = tool_obj
            self._tool_handlers[tool_name] = func
            return func

        return decorator

    def resource(
        self,
        uri: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: str = "application/json",
    ) -> Callable[[HandlerFunc], HandlerFunc]:
        """Decorator to register a resource read handler."""

        def decorator(func: HandlerFunc) -> HandlerFunc:
            res_name = name or uri
            res_obj = Resource(
                uri=uri,
                name=res_name,
                description=description or inspect.getdoc(func),
                mimeType=mime_type,
            )
            self.resources[uri] = res_obj
            self._resource_handlers[uri] = func
            return func

        return decorator

    def prompt(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[HandlerFunc], HandlerFunc]:
        """Decorator to register a prompt template handler."""

        def decorator(func: HandlerFunc) -> HandlerFunc:
            p_name = name or func.__name__
            p_obj = Prompt(
                name=p_name,
                description=description or inspect.getdoc(func),
            )
            self.prompts[p_name] = p_obj
            self._prompt_handlers[p_name] = func
            return func

        return decorator

    async def handle_request(self, request: JsonRpcRequest) -> Optional[JsonRpcResponse]:
        """Route and execute JSON-RPC request."""
        method = request.method
        req_id = request.id
        params = request.params or {}

        try:
            if method == "initialize":
                res = InitializeResult(
                    capabilities=ServerCapabilities(),
                    serverInfo=Implementation(name=self.name, version=self.version),
                )
                self._is_initialized = True
                return make_success_response(req_id, res.model_dump())

            if method == "notifications/initialized":
                # Notification - no response required
                return None

            if method == "ping":
                return make_success_response(req_id, {})

            if method == "tools/list":
                tool_list = list(self.tools.values())
                return make_success_response(req_id, ListToolsResult(tools=tool_list).model_dump())

            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                if not tool_name or tool_name not in self._tool_handlers:
                    return make_error_response(
                        req_id,
                        INVALID_PARAMS,
                        f"Unknown tool: {tool_name}",
                    )

                handler = self._tool_handlers[tool_name]
                raw_result = await handler(**arguments)
                if isinstance(raw_result, CallToolResult):
                    return make_success_response(req_id, raw_result.model_dump())
                return make_success_response(
                    req_id,
                    CallToolResult(content=[TextContent(text=str(raw_result))]).model_dump(),
                )

            if method == "resources/list":
                res_list = list(self.resources.values())
                return make_success_response(
                    req_id, ListResourcesResult(resources=res_list).model_dump()
                )

            if method == "resources/read":
                uri = params.get("uri")
                if not uri or uri not in self._resource_handlers:
                    return make_error_response(
                        req_id,
                        INVALID_PARAMS,
                        f"Resource not found: {uri}",
                    )
                handler = self._resource_handlers[uri]
                data = await handler(uri=uri)
                if isinstance(data, ReadResourceResult):
                    return make_success_response(req_id, data.model_dump())
                return make_success_response(
                    req_id,
                    ReadResourceResult(
                        contents=[{"uri": uri, "mimeType": "application/json", "text": str(data)}]
                    ).model_dump(),
                )

            if method == "prompts/list":
                p_list = list(self.prompts.values())
                return make_success_response(req_id, ListPromptsResult(prompts=p_list).model_dump())

            if method == "prompts/get":
                p_name = params.get("name")
                if not p_name or p_name not in self._prompt_handlers:
                    return make_error_response(
                        req_id,
                        INVALID_PARAMS,
                        f"Unknown prompt: {p_name}",
                    )
                p_handler = self._prompt_handlers[p_name]
                p_result = await p_handler(**params.get("arguments", {}))
                if isinstance(p_result, GetPromptResult):
                    return make_success_response(req_id, p_result.model_dump())
                return make_success_response(req_id, p_result)

            return make_error_response(
                req_id,
                METHOD_NOT_FOUND,
                f"Method not found: {method}",
            )

        except Exception as exc:
            logger.exception("Error executing MCP method %s", method)
            return make_error_response(
                req_id,
                INTERNAL_ERROR,
                f"Internal server error: {exc}",
            )

    async def run(self, transport: Transport) -> None:
        """Run MCP server event loop on provided transport."""
        logger.info("Starting MCP server '%s' v%s", self.name, self.version)
        try:
            async for raw_line in transport:
                request, error_resp = parse_message(raw_line)
                if error_resp:
                    await transport.write_message(serialize_response(error_resp))
                    continue
                if request is None:
                    continue

                response = await self.handle_request(request)
                if response is not None:
                    await transport.write_message(serialize_response(response))
        except asyncio.CancelledError:
            pass
        finally:
            await transport.close()
            logger.info("MCP server stopped.")
