import tempfile
import unittest
from pathlib import Path
import sys
import types

# CI/build machines often omit the optional Tk system package.  The tests below
# exercise only the non-GUI tool backend, so tiny import stubs are sufficient.
try:
    import tkinter  # noqa: F401
except ImportError:
    tkinter_stub = types.ModuleType("tkinter")
    tkinter_stub.filedialog = types.SimpleNamespace()
    tkinter_stub.messagebox = types.SimpleNamespace()
    sys.modules["tkinter"] = tkinter_stub

try:
    import customtkinter  # noqa: F401
except ImportError:
    customtkinter_stub = types.ModuleType("customtkinter")
    customtkinter_stub.CTk = type("CTk", (), {})
    sys.modules["customtkinter"] = customtkinter_stub

from chatbot_app import (
    ChatApp,
    MODEL_CATALOG,
    MODEL_DOWNLOAD_META,
    ToolError,
    Tools,
    looks_like_broken_tool_call,
    extract_generated_file,
    extract_generated_files,
    apply_requested_base_dir,
    parse_file_block,
    parse_plan,
    parse_review,
    parse_tool_call,
    requests_action,
    needs_tools,
    discover_models,
    estimate_tokens,
)


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tools = Tools(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_write_and_read(self):
        self.tools.write_file("notes/hello.txt", "สวัสดี")
        self.assertEqual(self.tools.read_file("notes/hello.txt"), "สวัสดี")

    def test_cannot_escape_workspace(self):
        with self.assertRaises(ToolError):
            self.tools.read_file("../secret.txt")

    def test_parse_tool_call(self):
        self.assertEqual(
            parse_tool_call('```json\n{"tool":"list_files","args":{"path":"."}}\n```'),
            ("list_files", {"path": "."}),
        )

    def test_parse_tool_call_surrounded_by_prose(self):
        self.assertEqual(
            parse_tool_call('กำลังทำให้ครับ {"tool":"read_file","args":{"path":"a.txt"}} เรียบร้อย'),
            ("read_file", {"path": "a.txt"}),
        )

    def test_broken_tool_call_is_detected(self):
        broken = '{"tool":"write_file","args":{"path":"index.html","content":"<p class="x">"}'
        self.assertIsNone(parse_tool_call(broken))
        self.assertTrue(looks_like_broken_tool_call(broken))

    def test_thai_action_request(self):
        self.assertTrue(requests_action("สร้างเว็บเกมให้หน่อย"))
        self.assertTrue(requests_action("เริ่มเลย"))

    def test_tool_intent_accepts_thai_file_spelling_variants(self):
        self.assertTrue(needs_tools("อ่านไฟล index.html"))
        self.assertTrue(needs_tools("อ่านไฟล์ index.html"))
        self.assertTrue(needs_tools("เปิด index.html แล้วสรุป"))
        self.assertFalse(needs_tools("วันนี้เป็นอย่างไรบ้าง"))

    def test_discovers_gguf_models_recursively(self):
        (Path(self.temp.name) / "qwen").mkdir()
        (Path(self.temp.name) / "qwen" / "model.gguf").write_bytes(b"GGUF")
        (Path(self.temp.name) / "qwen" / "model-mmproj.gguf").write_bytes(b"GGUF")
        (Path(self.temp.name) / "ignore.txt").write_text("x")
        self.assertEqual(
            discover_models(Path(self.temp.name)),
            [Path(self.temp.name) / "qwen" / "model.gguf"],
        )

    def test_token_estimate_handles_thai_and_empty_text(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("สวัสดีครับ"), 1)

    def test_every_catalog_model_has_integrity_metadata(self):
        self.assertEqual(set(MODEL_CATALOG), set(MODEL_DOWNLOAD_META))
        for size, digest in MODEL_DOWNLOAD_META.values():
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)

    def test_parse_raw_file_block(self):
        self.assertEqual(
            parse_file_block('<file path="space-arena/index.html">\n<h1>Hi</h1>\n</file>'),
            ("space-arena/index.html", "<h1>Hi</h1>"),
        )

    def test_extracts_raw_html_using_requested_path(self):
        self.assertEqual(
            extract_generated_file(
                "นี่คือผลงาน <!DOCTYPE html><html><body>OK</body></html>",
                "เขียน space-arena/index.html",
            ),
            ("space-arena/index.html", "<!DOCTYPE html><html><body>OK</body></html>"),
        )

    def test_extracts_plain_js_and_json(self):
        self.assertEqual(
            extract_generated_file("const running = true;", "space-arena/game.js"),
            ("space-arena/game.js", "const running = true;"),
        )
        self.assertEqual(
            extract_generated_file('```json\n{"speed": 4}\n```', "space-arena/data.json"),
            ("space-arena/data.json", '{"speed": 4}'),
        )

    def test_extracts_coder_write_file_json(self):
        self.assertEqual(
            extract_generated_files(
                '{"tool":"write_file","args":{"path":"app/game.js","content":"let x = 1;"}}',
                "app/game.js",
            ),
            [("app/game.js", "let x = 1;")],
        )

    def test_extracts_multiple_files(self):
        text = '<file path="index.html">HTML</file><file path="game.js">JS</file>'
        self.assertEqual(
            extract_generated_files(text, "แยก space-arena/index.html"),
            [("index.html", "HTML"), ("game.js", "JS")],
        )

    def test_applies_project_directory_to_siblings(self):
        self.assertEqual(
            apply_requested_base_dir(
                [("index.html", "H"), ("style.css", "C"), ("game.js", "J")],
                "แก้ space-arena/index.html แล้วแยกไฟล์",
            ),
            [
                ("space-arena/index.html", "H"),
                ("space-arena/style.css", "C"),
                ("space-arena/game.js", "J"),
            ],
        )

    def test_parses_agent_plan_and_review(self):
        self.assertEqual(
            parse_plan('<plan><file path="app/index.html">หน้าเว็บ</file><file path="app/app.js">เกม</file></plan>'),
            [("app/index.html", "หน้าเว็บ"), ("app/app.js", "เกม")],
        )
        self.assertEqual(parse_review('<review status="pass"/>'), [])
        self.assertEqual(
            parse_review('<review><issue file="app/app.js">syntax error</issue></review>'),
            [("app/app.js", "syntax error")],
        )

    def test_static_reviewer_finds_missing_asset(self):
        self.assertEqual(
            ChatApp._static_issues({"app/index.html": '<script src="game.js"></script>'}),
            [("app/index.html", "อ้างถึงไฟล์ game.js แต่ไม่มีใน manifest")],
        )

    def test_does_not_duplicate_workspace_directory(self):
        app = object.__new__(ChatApp)
        app.tools = Tools(Path("/tmp/space-arena"))
        self.assertEqual(
            app._workspace_relative_plan([
                ("space-arena/index.html", "หน้าเว็บ"),
                ("space-arena/js/game.js", "เกม"),
            ]),
            [("index.html", "หน้าเว็บ"), ("js/game.js", "เกม")],
        )


if __name__ == "__main__":
    unittest.main()
