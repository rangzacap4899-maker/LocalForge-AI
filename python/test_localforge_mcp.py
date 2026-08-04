import json
import sys
import tempfile
import unittest
from pathlib import Path

from localforge_hooks import HookEngine
from localforge_mcp import (
    MCPError,
    MCPManager,
    MCPServerConfig,
    MCPStdioClient,
    openai_tool_schema,
    select_tools,
)


SERVER_SOURCE = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "test", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "echo", "description": "Echo supplied text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
            {"name": "repo.issue.create", "description": "Create repository issue", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}
        ]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


class MCPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = self.root / "server.py"
        self.server.write_text(SERVER_SOURCE, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_stdio_discovery_and_call(self):
        config = MCPServerConfig("test", [sys.executable, "-u", str(self.server)])
        client = MCPStdioClient(config, self.root)
        try:
            self.assertEqual(client.list_tools()[0]["name"], "echo")
            self.assertEqual(client.call_tool("echo", {"text": "สวัสดี"}), "สวัสดี")
        finally:
            client.close()

    def test_manager_persists_and_qualifies_tools(self):
        manager = MCPManager(self.root / "mcp.json", self.root)
        manager.add("demo", f'{sys.executable} -u "{self.server}"')
        try:
            tools = manager.discover()
            self.assertEqual(tools[0]["qualified_name"], "mcp__demo__echo")
            self.assertEqual(tools[1]["qualified_name"], "mcp__demo__repo_issue_create")
            self.assertEqual(manager.call("mcp__demo__echo", {"text": "ok"}), "ok")
            self.assertEqual(manager.call(tools[1]["qualified_name"], {"text": "issue"}), "issue")
            schema = openai_tool_schema(tools[0])
            self.assertEqual(schema["function"]["name"], "mcp__demo__echo")
        finally:
            manager.close()
        reloaded = MCPManager(self.root / "mcp.json", self.root)
        self.assertEqual(reloaded.configs[0].permission, "ask")

    def test_tool_selection_is_bounded_and_relevant(self):
        tools = [
            {"qualified_name": "mcp__git__create_issue", "description": "Create GitHub issue"},
            {"qualified_name": "mcp__db__query", "description": "Query SQL database"},
        ]
        selected = select_tools(tools, "create a GitHub issue", limit=1)
        self.assertEqual(selected[0]["qualified_name"], "mcp__git__create_issue")

    def test_shell_commands_are_rejected(self):
        with self.assertRaises(MCPError):
            MCPServerConfig("bad", ["sh", "-c", "anything"]).validate()


class HookTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audit = Path(self.temp.name) / "audit.jsonl"
        self.hooks = HookEngine(self.audit, max_result_chars=20)

    def tearDown(self):
        self.temp.cleanup()

    def test_external_tools_require_approval_by_default(self):
        decision = self.hooks.before_tool("mcp__demo__echo", {"token": "secret"}, "mcp", "ask")
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.require_approval)
        self.assertIn("[REDACTED]", self.audit.read_text())

    def test_after_tool_redacts_and_truncates(self):
        result = self.hooks.after_tool("demo", "api_key=secret " + "x" * 100, "mcp", 0.1)
        self.assertIn("[REDACTED]", result)
        self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
