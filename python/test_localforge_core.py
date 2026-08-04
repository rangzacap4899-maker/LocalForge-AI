import json
import tempfile
import unittest
from pathlib import Path

from localforge_core import (
    ConversationStore,
    FileTransaction,
    choose_model,
    context_report,
    inspect_model,
    select_recent_messages,
)


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_model_router_prefers_coder(self):
        gemma = self.root / "gemma-4b-q4.gguf"
        coder = self.root / "qwen2.5-coder-7b-q4_k_m.gguf"
        gemma.write_bytes(b"x")
        coder.write_bytes(b"xx")
        self.assertEqual(choose_model("สร้างเว็บด้วย JavaScript", [gemma, coder]), coder)
        self.assertEqual(choose_model("สวัสดี", [gemma, coder]), gemma)
        self.assertEqual(choose_model("สวัสดี", [gemma, coder], coder), coder)

    def test_model_metadata(self):
        model = self.root / "Qwen3-8B-Q4_K_M.gguf"
        model.write_bytes(b"x" * 1024)
        info = inspect_model(model)
        self.assertEqual(info.parameters, "8B")
        self.assertEqual(info.quantization, "Q4_K_M")

    def test_conversations_create_switch_search_export(self):
        store = ConversationStore(self.root / "conversations.json")
        first = store.active()["id"]
        store.set_messages([{"role": "user", "content": "สร้างเกม"}])
        second = store.create("งานใหม่")
        self.assertNotEqual(first, second)
        self.assertEqual(store.switch(first)[0]["content"], "สร้างเกม")
        self.assertEqual(len(store.search("สร้างเกม")), 1)
        target = self.root / "chat.md"
        store.export_markdown(target)
        self.assertIn("สร้างเกม", target.read_text())

    def test_file_transaction_preview_validate_apply_undo(self):
        workspace = self.root / "work"
        workspace.mkdir()
        target = workspace / "data.json"
        target.write_text('{"old": true}')
        transaction = FileTransaction(workspace, self.root / "backups")
        self.assertIn("-\u007b\"old\": true\u007d", transaction.preview("data.json", '{"new": true}'))
        transaction.apply([("data.json", json.dumps({"new": True}))])
        self.assertTrue(json.loads(target.read_text())["new"])
        transaction.undo()
        self.assertTrue(json.loads(target.read_text())["old"])
        with self.assertRaises(ValueError):
            transaction.apply([("broken.json", "{")])

    def test_context_report(self):
        report = context_report([{"role": "user", "content": "สวัสดี"}], 100)
        self.assertGreater(report["used"], 0)
        self.assertLessEqual(report["percent"], 100)

    def test_recent_context_uses_token_budget_for_thai(self):
        messages = [
            {"role": "user", "content": "ก" * 1000},
            {"role": "assistant", "content": "ข" * 1000},
            {"role": "user", "content": "ล่าสุด"},
        ]
        selected = select_recent_messages(messages, 500)
        self.assertEqual(selected, [messages[-1]])


if __name__ == "__main__":
    unittest.main()
