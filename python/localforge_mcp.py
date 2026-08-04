"""Dependency-free MCP stdio client and persisted server registry."""

from __future__ import annotations

import json
import hashlib
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2025-11-25"
MAX_MCP_SERVERS = 12
MAX_MCP_TOOLS_PER_PROMPT = 6


class MCPError(RuntimeError):
    pass


@dataclass
class MCPServerConfig:
    name: str
    command: list[str]
    enabled: bool = True
    permission: str = "ask"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MCPServerConfig":
        name = str(value.get("name", "")).strip()
        command = value.get("command", [])
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise MCPError("MCP command must be an argument list")
        return cls(
            name=name, command=command,
            enabled=bool(value.get("enabled", True)),
            permission=str(value.get("permission", "ask")),
        )

    def validate(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", self.name):
            raise MCPError("Server name may contain only letters, numbers, _ and -")
        if not self.command or len(self.command) > 32:
            raise MCPError("MCP command is empty or too long")
        if self.permission not in {"ask", "allow", "deny"}:
            raise MCPError("Permission must be ask, allow, or deny")
        executable = self.command[0]
        if not (Path(executable).is_file() or shutil.which(executable)):
            raise MCPError(f"Executable not found: {executable}")
        if Path(executable).name.lower() in {
            "sh", "bash", "dash", "zsh", "fish", "cmd", "cmd.exe",
            "powershell", "powershell.exe", "pwsh",
        }:
            raise MCPError("Shell commands are not allowed; configure the MCP executable directly")


class MCPStdioClient:
    def __init__(self, config: MCPServerConfig, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace.resolve()
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def connect(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.config.validate()
        environment = {
            key: value for key, value in os.environ.items()
            if key in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "XDG_RUNTIME_DIR"}
        }
        self.process = subprocess.Popen(
            self.config.command, cwd=self.workspace, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        threading.Thread(target=self._read_messages, daemon=True).start()
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "LocalForge AI", "version": "1.0"},
        })
        self._notify("notifications/initialized", {})

    def _read_messages(self) -> None:
        process = self.process
        if not process or not process.stdout:
            self._responses.put(None)
            return
        for line in process.stdout:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                self._responses.put(value)
        self._responses.put(None)

    def close(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process:
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin or self.process.poll() is not None:
            raise MCPError(f"MCP server {self.config.name} is not running")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(
        self, method: str, params: dict[str, Any], timeout: float = 15.0
    ) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            while True:
                try:
                    response = self._responses.get(timeout=timeout)
                except queue.Empty:
                    self.close()
                    raise MCPError(f"MCP server {self.config.name} timed out during {method}")
                if response is None:
                    raise MCPError(f"MCP server {self.config.name} closed unexpectedly")
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    raise MCPError(str(response["error"]))
                result = response.get("result", {})
                return result if isinstance(result, dict) else {"value": result}

    def list_tools(self) -> list[dict[str, Any]]:
        self.connect()
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.connect()
        result = self._request("tools/call", {"name": name, "arguments": arguments}, timeout=60.0)
        if result.get("isError"):
            raise MCPError(_content_text(result.get("content", [])) or "MCP tool failed")
        return _content_text(result.get("content", [])) or json.dumps(result, ensure_ascii=False)


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif item.get("type") in {"resource", "resource_link", "image", "audio"}:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


class MCPManager:
    def __init__(self, config_path: Path, workspace: Path) -> None:
        self.config_path = config_path
        self.workspace = workspace.resolve()
        self.configs: list[MCPServerConfig] = []
        self.clients: dict[str, MCPStdioClient] = {}
        self.tool_aliases: dict[str, tuple[str, str]] = {}
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            values = data.get("servers", []) if isinstance(data, dict) else []
            self.configs = [MCPServerConfig.from_dict(item) for item in values if isinstance(item, dict)]
        except (OSError, ValueError, MCPError):
            self.configs = []

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps({"version": 1, "servers": [asdict(item) for item in self.configs]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, name: str, command_text: str, permission: str = "ask") -> MCPServerConfig:
        if len(self.configs) >= MAX_MCP_SERVERS:
            raise MCPError(f"At most {MAX_MCP_SERVERS} MCP servers are allowed")
        config = MCPServerConfig(name=name.strip(), command=shlex.split(command_text), permission=permission)
        config.validate()
        if any(item.name == config.name for item in self.configs):
            raise MCPError(f"MCP server already exists: {config.name}")
        self.configs.append(config)
        self.save()
        return config

    def remove(self, name: str) -> None:
        client = self.clients.pop(name, None)
        if client:
            client.close()
        self.configs = [item for item in self.configs if item.name != name]
        self.save()

    def update(self, name: str, *, enabled: bool | None = None, permission: str | None = None) -> None:
        config = self.get_config(name)
        if enabled is not None:
            config.enabled = enabled
        if permission is not None:
            config.permission = permission
            config.validate()
        if not config.enabled:
            client = self.clients.pop(name, None)
            if client:
                client.close()
        self.save()

    def get_config(self, name: str) -> MCPServerConfig:
        found = next((item for item in self.configs if item.name == name), None)
        if not found:
            raise MCPError(f"Unknown MCP server: {name}")
        return found

    def _client(self, name: str) -> MCPStdioClient:
        config = self.get_config(name)
        if not config.enabled:
            raise MCPError(f"MCP server is disabled: {name}")
        client = self.clients.get(name)
        if not client:
            client = MCPStdioClient(config, self.workspace)
            self.clients[name] = client
        return client

    def discover(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        self.tool_aliases = {}
        for config in self.configs:
            if not config.enabled or config.permission == "deny":
                continue
            try:
                for tool in self._client(config.name).list_tools():
                    if not isinstance(tool, dict) or not tool.get("name"):
                        continue
                    original_name = str(tool["name"])
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name)
                    qualified_name = f"mcp__{config.name}__{safe_name}"
                    if len(qualified_name) > 64:
                        digest = hashlib.sha256(qualified_name.encode()).hexdigest()[:8]
                        qualified_name = qualified_name[:55] + "_" + digest
                    self.tool_aliases[qualified_name] = (config.name, original_name)
                    discovered.append({
                        "server": config.name,
                        "name": original_name,
                        "qualified_name": qualified_name,
                        "description": str(tool.get("description", ""))[:500],
                        "inputSchema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                        "permission": config.permission,
                    })
            except Exception:
                continue
        return discovered

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        alias = self.tool_aliases.get(qualified_name)
        if alias:
            return self._client(alias[0]).call_tool(alias[1], arguments)
        match = re.fullmatch(r"mcp__([a-zA-Z0-9_-]+)__(.+)", qualified_name)
        if not match:
            raise MCPError(f"Invalid MCP tool name: {qualified_name}")
        return self._client(match.group(1)).call_tool(match.group(2), arguments)

    def test(self, name: str) -> int:
        return len(self._client(name).list_tools())

    def close(self) -> None:
        for client in list(self.clients.values()):
            client.close()
        self.clients.clear()


def select_tools(tools: list[dict[str, Any]], request: str, limit: int = MAX_MCP_TOOLS_PER_PROMPT) -> list[dict[str, Any]]:
    """Rank compact MCP tool schemas so small models are not flooded with tools."""
    words = set(re.findall(r"[\w.-]{3,}", request.lower()))
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for tool in tools:
        haystack = f"{tool.get('qualified_name', '')} {tool.get('description', '')}".lower()
        score = sum(1 for word in words if word in haystack)
        if "mcp" in request.lower():
            score += 1
        ranked.append((-score, str(tool.get("qualified_name", "")), tool))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max(0, limit)] if -item[0] > 0]


def openai_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["qualified_name"],
            "description": tool.get("description", "MCP tool"),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }
