"""Testable core services for LocalForge AI."""

from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CODING_RE = re.compile(
    r"(?:โค้ด|โปรแกรม|เว็บ|เกม|ไฟล์|บั๊ก|แก้|สร้าง|เขียน|code|program|html|css|javascript|python|fix|build)",
    re.I,
)
PLANNING_RE = re.compile(r"(?:วางแผน|ออกแบบ|สถาปัตยกรรม|เปรียบเทียบ|เหตุผล|plan|design|architect|reason)", re.I)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    thai = len(re.findall(r"[\u0E00-\u0E7F]", text))
    other = len(text) - thai
    return max(1, round(thai * 0.8 + other / 4))


def choose_model(request: str, models: list[Path], current: Path | None = None) -> Path | None:
    """Choose an installed model by task, preferring specialized/fast local models."""
    if not models:
        return None
    lowered = request.lower()
    if CODING_RE.search(request):
        coder = next((path for path in models if "coder" in path.name.lower()), None)
        if coder:
            return coder
    if PLANNING_RE.search(request):
        larger = sorted(models, key=lambda path: path.stat().st_size if path.exists() else 0, reverse=True)
        if larger:
            return larger[0]
    gemma = next((path for path in models if "gemma" in path.name.lower() and "mmproj" not in path.name.lower()), None)
    return gemma or current or models[0]


@dataclass
class ModelInfo:
    path: str
    name: str
    size_bytes: int
    quantization: str
    parameters: str
    estimated_vram_gib: float


def inspect_model(path: Path) -> ModelInfo:
    size = path.stat().st_size
    name = path.name
    quant = (re.search(r"(?:^|[-_])(Q\d(?:_[A-Z0-9]+)+)(?:[-_.]|$)", name, re.I) or [None, "ไม่ทราบ"])[1]
    params_match = re.search(r"(?:^|[-_])E?(\d+(?:\.\d+)?B)(?:[-_.]|$)", name, re.I)
    params = params_match.group(1) if params_match else "ไม่ทราบ"
    return ModelInfo(
        path=str(path), name=name, size_bytes=size, quantization=str(quant).upper(),
        parameters=str(params).upper(), estimated_vram_gib=round(size / (1024**3) + 0.8, 1),
    )


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": 2, "active": "", "conversations": []}
        self.load()

    def load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("conversations"), list):
                self.data = loaded
        except (OSError, ValueError):
            pass
        if not self.data["conversations"]:
            self.create("บทสนทนาใหม่")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, title: str = "บทสนทนาใหม่") -> str:
        conversation_id = uuid.uuid4().hex[:12]
        now = time.time()
        self.data["conversations"].append({
            "id": conversation_id, "title": title, "created": now,
            "updated": now, "pinned": False, "messages": [],
        })
        self.data["active"] = conversation_id
        self.save()
        return conversation_id

    def active(self) -> dict[str, Any]:
        active_id = self.data.get("active")
        found = next((item for item in self.data["conversations"] if item["id"] == active_id), None)
        if found:
            return found
        self.data["active"] = self.data["conversations"][0]["id"]
        return self.data["conversations"][0]

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        conversation = self.active()
        conversation["messages"] = messages
        conversation["updated"] = time.time()
        if conversation["title"] == "บทสนทนาใหม่":
            first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if first:
                conversation["title"] = first.replace("\n", " ")[:42]
        self.save()

    def switch(self, conversation_id: str) -> list[dict[str, Any]]:
        if any(item["id"] == conversation_id for item in self.data["conversations"]):
            self.data["active"] = conversation_id
            self.save()
        return list(self.active()["messages"])

    def delete(self, conversation_id: str) -> None:
        self.data["conversations"] = [item for item in self.data["conversations"] if item["id"] != conversation_id]
        if not self.data["conversations"]:
            self.create()
        else:
            self.data["active"] = self.data["conversations"][0]["id"]
            self.save()

    def search(self, query: str) -> list[dict[str, Any]]:
        query = query.lower().strip()
        if not query:
            return list(self.data["conversations"])
        return [item for item in self.data["conversations"] if query in (item["title"] + " " + json.dumps(item["messages"], ensure_ascii=False)).lower()]

    def export_markdown(self, target: Path) -> None:
        conversation = self.active()
        lines = [f"# {conversation['title']}", ""]
        for message in conversation["messages"]:
            lines.extend([f"## {'คุณ' if message.get('role') == 'user' else 'LocalForge'}", "", str(message.get("content", "")), ""])
        target.write_text("\n".join(lines), encoding="utf-8")


class FileTransaction:
    def __init__(self, workspace: Path, backup_root: Path) -> None:
        self.workspace = workspace.resolve()
        self.backup_root = backup_root
        self.last_manifest: Path | None = None

    def _target(self, relative: str) -> Path:
        target = (self.workspace / relative).resolve()
        target.relative_to(self.workspace)
        return target

    @staticmethod
    def validate(path: str, content: str) -> list[str]:
        errors: list[str] = []
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".json":
                json.loads(content)
            elif suffix == ".py":
                ast.parse(content)
            elif suffix in {".html", ".htm"}:
                if "<html" in content.lower() and "</html>" not in content.lower():
                    errors.append("ไม่มีแท็ก </html>")
            elif suffix in {".js", ".css"}:
                pairs = {"{": "}", "(": ")", "[": "]"}
                stack: list[str] = []
                for char in re.sub(r"(?s)/\*.*?\*/|//.*?$|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"", "", content):
                    if char in pairs:
                        stack.append(pairs[char])
                    elif char in pairs.values() and (not stack or stack.pop() != char):
                        errors.append(f"วงเล็บ {char} ไม่สมดุล")
                        break
                if stack:
                    errors.append("วงเล็บปิดไม่ครบ")
        except (ValueError, SyntaxError) as exc:
            errors.append(str(exc))
        return errors

    def preview(self, path: str, content: str) -> str:
        target = self._target(path)
        old = target.read_text(encoding="utf-8") if target.is_file() else ""
        return "".join(difflib.unified_diff(
            old.splitlines(True), content.splitlines(True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=3,
        ))

    def apply(self, files: list[tuple[str, str]]) -> list[str]:
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time_ns() % 1_000_000):06d}"
        backup_dir = self.backup_root / stamp
        manifest: dict[str, Any] = {"workspace": str(self.workspace), "files": []}
        results: list[str] = []
        for relative, content in files:
            errors = self.validate(relative, content)
            if errors:
                raise ValueError(f"{relative}: " + "; ".join(errors))
            target = self._target(relative)
            existed = target.is_file()
            backup = backup_dir / relative
            if existed:
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            manifest["files"].append({"path": relative, "existed": existed})
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            results.append(f"เขียน {relative} ({len(content):,} ตัวอักษร)")
        backup_dir.mkdir(parents=True, exist_ok=True)
        self.last_manifest = backup_dir / "manifest.json"
        self.last_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return results

    def undo(self) -> list[str]:
        if not self.last_manifest or not self.last_manifest.is_file():
            raise RuntimeError("ยังไม่มีรายการแก้ไขให้ย้อนคืน")
        manifest = json.loads(self.last_manifest.read_text(encoding="utf-8"))
        restored: list[str] = []
        backup_dir = self.last_manifest.parent
        for item in manifest["files"]:
            target = self._target(item["path"])
            backup = backup_dir / item["path"]
            if item["existed"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
            elif target.exists():
                target.unlink()
            restored.append(item["path"])
        return restored


def context_report(messages: list[dict[str, Any]], max_tokens: int = 8192) -> dict[str, Any]:
    entries = [{"role": item.get("role", ""), "tokens": estimate_tokens(str(item.get("content", "")))} for item in messages]
    used = sum(item["tokens"] for item in entries)
    return {"used": used, "maximum": max_tokens, "percent": min(100, round(used / max_tokens * 100, 1)), "entries": entries}
