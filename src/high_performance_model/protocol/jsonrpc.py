"""JSON-RPC 2.0 protocol encoder, decoder, and standard error codes."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Union

from high_performance_model.protocol.models import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
)

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def make_error_response(
    request_id: Optional[Union[int, str]],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> JsonRpcResponse:
    """Construct standard JSON-RPC error response."""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message, data=data),
    )


def make_success_response(
    request_id: Optional[Union[int, str]],
    result: Any,
) -> JsonRpcResponse:
    """Construct standard JSON-RPC success response."""
    return JsonRpcResponse(
        id=request_id,
        result=result,
    )


def parse_message(raw_line: str) -> Tuple[Optional[JsonRpcRequest], Optional[JsonRpcResponse]]:
    """Parse incoming string line to JsonRpcRequest, returning error response if invalid."""
    if not raw_line.strip():
        return None, None

    try:
        data: Dict[str, Any] = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        return None, make_error_response(None, PARSE_ERROR, f"Parse error: {exc}")

    if not isinstance(data, dict):
        return None, make_error_response(None, INVALID_REQUEST, "Invalid Request: expected JSON object")

    if data.get("jsonrpc") != "2.0" or "method" not in data or not isinstance(data["method"], str):
        return None, make_error_response(
            data.get("id"), INVALID_REQUEST, "Invalid Request: missing jsonrpc 2.0 or method"
        )

    req = JsonRpcRequest(
        jsonrpc="2.0",
        id=data.get("id"),
        method=data["method"],
        params=data.get("params"),
    )
    return req, None


def serialize_response(response: JsonRpcResponse) -> str:
    """Serialize response to JSON string."""
    return json.dumps(response.model_dump(exclude_none=True))
