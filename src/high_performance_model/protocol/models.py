"""Model Context Protocol (MCP) data contracts and schema definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

LATEST_PROTOCOL_VERSION = "2024-11-05"


class Implementation(BaseModel):
    """Client or server metadata."""

    name: str
    version: str


class ServerCapabilities(BaseModel):
    """Server-supported MCP capabilities."""

    tools: Dict[str, Any] = Field(default_factory=lambda: {"listChanged": True})
    resources: Dict[str, Any] = Field(default_factory=lambda: {"subscribe": True, "listChanged": True})
    prompts: Dict[str, Any] = Field(default_factory=lambda: {"listChanged": True})
    logging: Dict[str, Any] = Field(default_factory=dict)


class InitializeParams(BaseModel):
    """Parameters for initialize request."""

    protocolVersion: str = LATEST_PROTOCOL_VERSION
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    clientInfo: Implementation


class InitializeResult(BaseModel):
    """Result of initialize request."""

    protocolVersion: str = LATEST_PROTOCOL_VERSION
    capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    serverInfo: Implementation


class Tool(BaseModel):
    """Executable MCP tool descriptor."""

    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ListToolsResult(BaseModel):
    """Result returned for tools/list."""

    tools: List[Tool] = Field(default_factory=list)


class TextContent(BaseModel):
    """Standard MCP text content block."""

    type: Literal["text"] = "text"
    text: str


class CallToolRequest(BaseModel):
    """Parameters for tools/call."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class CallToolResult(BaseModel):
    """Output returned by a tool invocation."""

    content: List[TextContent] = Field(default_factory=list)
    isError: bool = False


class Resource(BaseModel):
    """MCP resource metadata descriptor."""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = "application/json"


class ListResourcesResult(BaseModel):
    """Result returned for resources/list."""

    resources: List[Resource] = Field(default_factory=list)


class ReadResourceResult(BaseModel):
    """Resource contents returned for resources/read."""

    contents: List[Dict[str, Any]] = Field(default_factory=list)


class PromptArgument(BaseModel):
    """Argument specification for an MCP prompt template."""

    name: str
    description: Optional[str] = None
    required: bool = False


class Prompt(BaseModel):
    """MCP prompt template descriptor."""

    name: str
    description: Optional[str] = None
    arguments: List[PromptArgument] = Field(default_factory=list)


class ListPromptsResult(BaseModel):
    """Result returned for prompts/list."""

    prompts: List[Prompt] = Field(default_factory=list)


class PromptMessage(BaseModel):
    """Message returned within a getPrompt call."""

    role: Literal["user", "assistant", "system"]
    content: TextContent


class GetPromptResult(BaseModel):
    """Result returned for prompts/get."""

    description: Optional[str] = None
    messages: List[PromptMessage] = Field(default_factory=list)


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request or notification message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[int, str]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[int, str]] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
