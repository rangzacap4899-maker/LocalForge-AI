#!/usr/bin/env python3
"""LocalForge AI desktop workspace for local OpenAI-compatible models."""

from __future__ import annotations

import html
import hashlib
import base64
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import webbrowser
import urllib.error
import urllib.parse
import urllib.request
import tkinter as tk
from localforge_algorithms import VectorDB, get_embedding
from pathlib import Path
from tkinter import Menu, filedialog, messagebox, simpledialog, ttk
from typing import Any

import customtkinter as ctk

from localforge_core import (
    ConversationStore,
    FileTransaction,
    choose_model,
    context_report,
    inspect_model,
    select_recent_messages,
)
from localforge_i18n import (
    LANGUAGE_CODES,
    LANGUAGE_FONTS,
    LANGUAGE_NAMES,
    translate,
)
from localforge_hooks import HookEngine
from localforge_mcp import MCPError, MCPManager, openai_tool_schema, select_tools


USER_AGENT = "LocalForgeAI/1.0"
MAX_FILE_BYTES = 1_000_000
MAX_WEB_BYTES = 1_500_000
MAX_TOOL_ROUNDS = 12
API_TIMEOUT_SECONDS = 900
# Image/audio preprocessing can keep a streaming connection silent for several
# seconds before the first token. Five seconds was enough for text but caused
# false timeouts on real photos; retain a finite timeout so cancellation and
# dead-server detection still return control to the UI.
STREAM_IDLE_TIMEOUT_SECONDS = 60
MAX_TOOL_RESULT_CHARS = 12_000
MODEL_CONTEXT_TOKENS = 8192
PROMPT_TOKEN_BUDGET = 5000
CHAT_MAX_TOKENS = 1024
TOOL_MAX_TOKENS = 2048
FILE_MAX_TOKENS = 4096
MAX_SAVED_MESSAGES = 100
TEXT_PROJECT_EXTENSIONS = {".html", ".htm", ".css", ".js", ".json", ".md", ".txt", ".py", ".svg"}
THAI_FONT = "Noto Sans Thai"
APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_BIN = str(APP_ROOT / "runtime/llama.cpp/build-vulkan/bin/llama-server")
DEFAULT_MODEL_ROOT = str(APP_ROOT / "models")

MODEL_CATALOG = {
    "Gemma 4 E4B IT — Q4_0": (
        "gemma-4-e4b/gemma-4-E4B_q4_0-it.gguf",
        "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
    ),
    "Qwen2.5 Coder 7B — Q4_K_M": (
        "qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    ),
    "Qwen3 8B — Q4_K_M": (
        "qwen3-8b/Qwen3-8B-Q4_K_M.gguf",
        "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    ),
    "Qwen3.5 4B — Q4_K_M": (
        "qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
    ),
    "Llama 3.1 8B Instruct — Q4_K_M": (
        "llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "Gemma 2 9B IT — Q4_K_M": (
        "gemma-2-9b/gemma-2-9b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
    ),
    "Mistral Nemo 12B Instruct — Q4_K_M": (
        "mistral-nemo-12b/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    ),
    "Phi-3 Mini 4K Instruct — Q4": (
        "phi-3-mini/Phi-3-mini-4k-instruct-q4.gguf",
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    ),
    "DeepSeek R1 Distill Qwen 7B — Q4_K_M": (
        "deepseek-r1/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    ),
    "GLM 4 9B Chat — Q4_K_M": (
        "glm-4/glm-4-9b-chat-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/glm-4-9b-chat-GGUF/resolve/main/glm-4-9b-chat-Q4_K_M.gguf",
    ),
}
MODEL_DOWNLOAD_META = {
    "Gemma 4 E4B IT — Q4_0": (5154941280, "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"),
    "Qwen2.5 Coder 7B — Q4_K_M": (4683073536, "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"),
    "Qwen3 8B — Q4_K_M": (5027783488, "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"),
    "Qwen3.5 4B — Q4_K_M": (2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    "Llama 3.1 8B Instruct — Q4_K_M": (4920739232, "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"),
    "Gemma 2 9B IT — Q4_K_M": (5761057728, "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe"),
    "Mistral Nemo 12B Instruct — Q4_K_M": (7405202976, "f3d538676f289659b8b7edcaaaad72fb3eb8fca02dcf579621379bd47ec6518a"),
    "Phi-3 Mini 4K Instruct — Q4": (2393275904, "30dc2cb2cf0bfce17bcadad93a743aa2f8650dfc31ad66318e8d8de958bf638b"),
    "DeepSeek R1 Distill Qwen 7B — Q4_K_M": (4683073504, "731ece8d06dc7eda6f6572997feb9ee1258db0784827e642909d9b565641937b"),
    "GLM 4 9B Chat — Q4_K_M": (6250926848, "aa6cb8f5ef0a70399bdbf92eef566c26c6017ee0b131a424b65b5c757f1c81a2"),
}
MODEL_COMPANIONS = {
    "Gemma 4 E4B IT — Q4_0": (
        "gemma-4-e4b/gemma-4-E4B-it-mmproj.gguf",
        "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B-it-mmproj.gguf",
        991552256,
        "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",
    ),
}


def media_content(path: Path) -> dict[str, Any]:
    """Create an OpenAI-compatible multimodal content part from a local file."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    if mime.startswith("audio/") or path.suffix.lower() == ".wav":
        audio_format = path.suffix.lower().lstrip(".") or "wav"
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}
    raise ValueError(f"ไม่รองรับไฟล์สื่อชนิดนี้: {path.name}")


def with_media(messages: list[dict[str, Any]], media: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach media to the latest user turn without mutating saved history."""
    result = [dict(message) for message in messages]
    if not media:
        return result
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            text = str(result[index].get("content", ""))
            images = [part for part in media if part.get("type") == "image_url"]
            audio = [part for part in media if part.get("type") == "input_audio"]
            result[index]["content"] = [*images, {"type": "text", "text": text}, *audio]
            break
    return result


def inference_profile(model_path: Path, cpu_count: int | None = None) -> dict[str, Any]:
    """Return conservative llama.cpp settings tuned for the selected model."""
    name = model_path.name.lower()
    threads = max(1, min(12, cpu_count or os.cpu_count() or 4))
    profile: dict[str, Any] = {
        "name": "Balanced",
        "gpu_layers": "auto",
        "threads": threads,
        "context": MODEL_CONTEXT_TOKENS,
        "batch": 1024,
        "ubatch": 256,
        "cache_type": "q8_0",
    }
    if "gemma-4-e4b" in name or "gemma_4_e4b" in name:
        # Q4_0 weights plus an 8K Q8 KV cache fit comfortably in this
        # workstation's 8 GiB GPU. Explicit all-layer offload avoids a future
        # llama.cpp heuristic change silently moving work back to the CPU.
        # Vision/audio encoders use non-causal attention. Their complete media
        # token block must fit in one ubatch or llama.cpp aborts while decoding
        # larger pasted images (GGML_ASSERT in llama-context.cpp).
        profile.update(
            name="Gemma 4 E4B · RX 8GB", gpu_layers="all",
            batch=2048, ubatch=2048,
        )
    elif "7b" in name or "8b" in name or "9b" in name or "12b" in name:
        # Larger models (7B - 12B) consume more VRAM for weights alone.
        # Reduce the context window to 4096 and use a smaller ubatch
        # to ensure it fits in an 8GB VRAM card without OOM crashing.
        profile.update(
            name="Large Model · 8GB Tuned", context=4096,
            batch=512, ubatch=128,
        )
    return profile


def estimate_tokens(text: str) -> int:
    """Small dependency-free fallback for messages restored from history."""
    if not text:
        return 0
    thai = len(re.findall(r"[\u0E00-\u0E7F]", text))
    other = len(text) - thai
    return max(1, round(thai * 0.8 + other / 4))


def discover_models(root: Path) -> list[Path]:
    try:
        return sorted(
            (
                path for path in root.rglob("*.gguf")
                if not re.search(r"(?:mmproj|projector)", path.name, re.I)
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return []

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders inside the current workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file inside the workspace. Use this when the user asks you to build or modify something.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Download and extract readable text from an HTTP or HTTPS page.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


class ToolError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


class Tools:
    """Constrained file and web tools exposed to the model."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def set_workspace(self, path: Path) -> None:
        self.workspace = path.resolve()

    def _path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError("ไม่อนุญาตให้เข้าถึงไฟล์นอก workspace") from exc
        return candidate

    def list_files(self, path: str = ".") -> str:
        folder = self._path(path)
        if not folder.is_dir():
            raise ToolError(f"ไม่พบโฟลเดอร์: {path}")
        items = []
        for item in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:300]:
            rel = item.relative_to(self.workspace)
            items.append(f"{'[DIR]' if item.is_dir() else '[FILE]'} {rel}")
        return "\n".join(items) or "(โฟลเดอร์ว่าง)"

    def read_file(self, path: str) -> str:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"ไม่พบไฟล์: {path}")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("ไฟล์ใหญ่เกิน 1 MB")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("รองรับเฉพาะไฟล์ข้อความ UTF-8") from exc

    def write_file(self, path: str, content: str) -> str:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"เขียนไฟล์ {target.relative_to(self.workspace)} แล้ว ({len(content)} ตัวอักษร)"

    @staticmethod
    def _download(url: str, limit: int) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError("รองรับเฉพาะ URL แบบ http/https")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raw = raw[:limit]
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")

    def web_search(self, query: str) -> str:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        page = self._download(url, MAX_WEB_BYTES)
        matches = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.I | re.S
        )
        results = []
        for link, title in matches[:8]:
            link = html.unescape(link)
            redirect = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("uddg")
            if redirect:
                link = redirect[0]
            clean_title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            results.append(f"- {clean_title}\n  {link}")
        return "\n".join(results) or "ไม่พบผลการค้นหา"

    def fetch_url(self, url: str) -> str:
        page = self._download(url, MAX_WEB_BYTES)
        page = re.sub(r"(?is)<(script|style|svg).*?>.*?</\1>", " ", page)
        page = re.sub(r"(?s)<[^>]+>", " ", page)
        text = html.unescape(page)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return text[:15_000]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        methods = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "web_search": self.web_search,
            "fetch_url": self.fetch_url,
        }
        if name not in methods:
            raise ToolError(f"ไม่รู้จักเครื่องมือ: {name}")
        try:
            return str(methods[name](**args))
        except TypeError as exc:
            raise ToolError(f"พารามิเตอร์ไม่ถูกต้อง: {exc}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise ToolError(str(exc)) from exc


SYSTEM_PROMPT = """คุณคือผู้ช่วยเดสก์ท็อปที่ตอบภาษาเดียวกับผู้ใช้ คุณมีความสามารถในการเข้าถึงอินเทอร์เน็ตและระบบไฟล์ผ่านเครื่องมือต่อไปนี้ (ห้ามปฏิเสธว่าทำไม่ได้):
- list_files: {"path":"."}
- read_file: {"path":"relative/path.txt"}
- write_file: {"path":"relative/path.txt","content":"..."}
- web_search: {"query":"คำค้นหาสั้นๆ กระชับ"}
- fetch_url: {"url":"https://..."}

เมื่อต้องใช้เครื่องมือ ให้ครอบ JSON ไว้ใน Code Block เสมอ ตัวอย่าง:
```json
{"tool":"ชื่อเครื่องมือ","args":{"พารามิเตอร์":"..."}}
```
คุณสามารถเขียนข้อความอธิบายก่อนเรียกเครื่องมือได้
เมื่อได้รับผลเครื่องมือแล้ว ให้ตัดสินใจว่าจะใช้เครื่องมือต่อหรือสรุปคำตอบ ห้ามอ้างว่าอ่านไฟล์หรือค้นเว็บถ้ายังไม่ได้ใช้เครื่องมือ
หากผู้ใช้สั่งให้อ่านเว็บ ค้นหาข้อมูล หรือเข้าไปอ่านลิงก์ ให้ใช้ web_search หรือ fetch_url ทันที ห้ามปฏิเสธ
พาธไฟล์ทั้งหมดเป็นพาธสัมพัทธ์ภายใน workspace เท่านั้น เมื่อได้ข้อมูลจากเว็บให้แนบ URL ที่เกี่ยวข้องในคำตอบสุดท้าย
เมื่อไม่ต้องใช้เครื่องมือ ให้ตอบผู้ใช้ตามปกติและห้ามครอบคำตอบด้วย JSON
ถ้าผู้ใช้สั่งให้สร้าง แก้ไข หรือทำโปรเจกต์ ให้ลงมือใช้ write_file ทันที เลือกรายละเอียดที่สมเหตุผลเอง
และสร้างผลงานที่เล่นหรือใช้งานได้จริง ห้ามปฏิเสธด้วยเหตุผลว่างานใหญ่ ห้ามตอบเพียงแผนหรือโค้ดตัวอย่าง
ถามกลับเฉพาะเมื่อขาดข้อมูลสำคัญจนไม่สามารถลงมือได้จริงเท่านั้น
สำหรับเว็บเกมขนาดเล็ก ให้สร้างเป็นไฟล์ index.html ไฟล์เดียวที่รวม CSS และ JavaScript ไว้ภายใน
เพื่อให้เปิดเล่นได้ทันทีและลดการเรียกเครื่องมือ เมื่อเครื่องมือรายงาน SUCCESS ห้ามเขียนไฟล์เดิมซ้ำ
งานเขียนโปรแกรมทั่วไปให้ใช้ความรู้ที่มีและเรียก write_file โดยตรง ห้ามค้นเว็บเว้นแต่ผู้ใช้สั่งค้นข้อมูลล่าสุด
และเมื่อสร้างหรือแก้ไข UI ต้องยึดหลัก Modern UI เสมอ: ห้ามใช้สีพื้นฐานดื้อๆ (ให้ใช้ Pastel/Dark Mode), เพิ่มมิติด้วย Gradient/Glassmorphism, ใช้ฟอนต์ Modern Sans-serif, มีเอฟเฟกต์ Hover/Transition นุ่มนวล, เว้นช่องไฟ (Whitespace) ให้โปร่ง, และทำให้ดูพรีเมียมที่สุดเสมอ
"""

CHAT_PROMPT = """คุณคือผู้ช่วย AI บนเครื่อง ตอบภาษาเดียวกับผู้ใช้
ตอบตรงคำถาม กระชับ และไม่แสดงขั้นตอนการคิดภายใน
และเมื่อต้องเขียนโค้ด UI ต้องยึดหลัก Modern UI เสมอ: ห้ามใช้สีพื้นฐานดื้อๆ (ให้ใช้ Pastel/Dark Mode), เพิ่มมิติด้วย Gradient/Glassmorphism, ใช้ฟอนต์ Modern Sans-serif, มีเอฟเฟกต์ Hover/Transition นุ่มนวล, เว้นช่องไฟให้โปร่ง และทำให้ออกมาพรีเมียมที่สุด"""


def needs_tools(text: str) -> bool:
    """Only attach the costly tool schema when the request actually needs it."""
    lowered = text.lower()
    explicit_tool_intent = bool(re.search(
        r"(?:ค้น(?:หา)?|เสิร์ช|อินเทอร์เน็ต|เว็บ|เว็บไซต์|url|https?://|"
        r"wiki|วิกิ|google|"
        r"อ่านไฟล[์]?|เปิดไฟล[์]?|ดูไฟล[์]?|รายชื่อไฟล[์]?|list files?|read files?|"
        r"search(?: the)? web|browse|internet)",
        lowered,
    ))
    # Natural requests often omit the word "ไฟล์", for example
    # "อ่าน index.html" or contain a minor Thai spelling omission.
    file_reference = bool(re.search(
        r"(?:^|[\s/])[^\s/]+\.(?:html?|css|js|json|md|txt|py|svg)(?:\s|$)",
        lowered,
    ))
    file_verb = bool(re.search(r"(?:อ่าน|เปิด|ดู|ตรวจ|สรุป)", lowered))
    return explicit_tool_intent or (file_reference and file_verb)


class GemmaClient:
    def __init__(
        self, base_url: str, model: str,
        tool_schemas: list[dict[str, Any]] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tool_schemas = tool_schemas if tool_schemas is not None else OPENAI_TOOLS
        self.last_native_tool_call: dict[str, Any] | None = None
        self.last_native_message: dict[str, Any] | None = None
        self.last_completion_tokens = 0
        self.last_prompt_tokens = 0

    def generate(
        self, messages: list[dict[str, Any]], enable_tools: bool = True,
        on_token: Any | None = None, cancel_event: threading.Event | None = None,
    ) -> str:
        self.last_native_tool_call = None
        self.last_native_message = None
        # llama.cpp exposes an OpenAI-compatible endpoint. Try it first so the
        # desktop app can also use local GGUF models, then fall back to the
        # native gemma.cpp Google-compatible endpoint.
        try:
            return self._generate_openai(messages, enable_tools, on_token, cancel_event)
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 405}:
                detail = exc.read().decode(errors="replace")
                raise RuntimeError(f"API ตอบกลับ {exc.code}: {detail[:500]}") from exc
        return self._generate_google(messages)

    def _read_stream(
        self, request: urllib.request.Request, on_token: Any,
        cancel_event: threading.Event | None,
    ) -> str:
        parts: list[str] = []
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            try:
                response.fp.raw._sock.settimeout(STREAM_IDLE_TIMEOUT_SECONDS)  # type: ignore[attr-defined]
            except AttributeError:
                pass
            while True:
                try:
                    raw_line = response.readline()
                except (TimeoutError, OSError) as exc:
                    if parts:
                        break
                    raise RuntimeError(
                        f"โมเดลยังไม่เริ่มตอบภายใน {STREAM_IDLE_TIMEOUT_SECONDS} วินาที"
                    ) from exc
                if not raw_line:
                    break
                if cancel_event and cancel_event.is_set():
                    raise GenerationCancelled("หยุดสร้างคำตอบแล้ว")
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                try:
                    chunk = json.loads(value)
                except ValueError:
                    continue
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    self.last_completion_tokens = int(usage["completion_tokens"] or 0)
                    self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    break
                piece = choice.get("delta", {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    on_token(piece)
        if not self.last_completion_tokens:
            self.last_completion_tokens = estimate_tokens("".join(parts))
        return "".join(parts).strip()

    def _generate_openai(
        self, messages: list[dict[str, Any]], enable_tools: bool,
        on_token: Any | None, cancel_event: threading.Event | None,
    ) -> str:
        self.last_prompt_tokens = sum(
            estimate_tokens(str(message.get("content", ""))) for message in messages
        ) + estimate_tokens(SYSTEM_PROMPT if enable_tools else CHAT_PROMPT)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT if enable_tools else CHAT_PROMPT},
                *messages,
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget_tokens": 0,
            "temperature": 0.6 if ("r1" in self.model.lower() or "reasoning" in self.model.lower()) else (0.2 if enable_tools else 0.35),
            "top_p": 0.9,
            # Ordinary chat should stop promptly; tool calls retain more room.
            # Full-file generation uses generate_file() and its larger budget.
            "max_tokens": TOOL_MAX_TOKENS if enable_tools else CHAT_MAX_TOKENS,
        }
        if enable_tools:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = "auto"
        if on_token:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        if on_token:
            return self._read_stream(request, on_token, cancel_event)
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            data = json.load(response)
        self.last_completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
        self.last_prompt_tokens = int(data.get("usage", {}).get("prompt_tokens", 0) or 0)
        try:
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                self.last_native_message = message
                self.last_native_tool_call = tool_calls[0]
                function = self.last_native_tool_call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                return json.dumps(
                    {"tool": function["name"], "args": arguments}, ensure_ascii=False
                )
            return (message.get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"รูปแบบคำตอบจาก llama.cpp ไม่ถูกต้อง: {data}") from exc

    def generate_file(
        self, request_text: str, on_token: Any | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Generate a raw file block without JSON escaping or tool parsing."""
        instruction = """คุณเป็นโปรแกรมสร้างไฟล์ ลงมือทำตามคำสั่งผู้ใช้ทันที
ตอบด้วยบล็อกตามรูปแบบนี้เท่านั้น ห้ามมีคำอธิบายหรือ Markdown:
<file path="relative/path.ext">
เนื้อหาไฟล์ฉบับสมบูรณ์
</file>
หากผู้ใช้ขอหลายไฟล์ ให้ส่งหลายบล็อก <file> ต่อกันจนครบในคำตอบเดียว
พาธต้องเป็นพาธสัมพัทธ์ สำหรับเว็บเกมให้รวม HTML, CSS และ JavaScript ใน index.html ไฟล์เดียว
เว้นแต่ผู้ใช้ระบุชัดว่าต้องการแยกไฟล์ ผลงานต้องเปิดใช้งานได้จริง กระชับ และห้ามใช้ JSON
เพราะเนื้อหาโค้ดไม่ต้อง escape
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": request_text},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget_tokens": 0,
            "temperature": 0.25,
            "top_p": 0.9,
            "max_tokens": FILE_MAX_TOKENS,
        }
        if on_token:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        if on_token:
            return self._read_stream(request, on_token, cancel_event)
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            data = json.load(response)
        self.last_completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
        self.last_prompt_tokens = int(data.get("usage", {}).get("prompt_tokens", 0) or 0)
        try:
            return (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"รูปแบบคำตอบสร้างไฟล์ไม่ถูกต้อง: {data}") from exc

    def _role_call(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget_tokens": 0,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            data = json.load(response)
        self.last_completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
        self.last_prompt_tokens = int(data.get("usage", {}).get("prompt_tokens", 0) or 0)
        try:
            return (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"รูปแบบคำตอบจาก agent ไม่ถูกต้อง: {data}") from exc

    def summarize(self, transcript: str) -> str:
        """Compress an older conversation transcript into a short LLM summary."""
        return self._role_call(
            "คุณคือผู้ช่วยที่สรุปประวัติการสนทนาก่อนหน้าให้กระชับ "
            "ตอบเฉพาะบทสรุปสั้นๆ 2-3 ประโยคในภาษาที่ใช้สนทนา "
            "ห้ามทักทาย ห้ามถามกลับ ห้ามอธิบายว่าคุณคือใคร",
            f"สรุปบทสนทนาก่อนหน้านี้:\n{transcript}",
            512,
        )

    def plan_project(self, request_text: str) -> str:
        return self._role_call(
            """คุณคือ Planner วางแผนไฟล์ขั้นต่ำที่จำเป็นสำหรับงานผู้ใช้
ตอบ XML เท่านั้น: <plan><file path="relative/path">หน้าที่สั้นๆ</file></plan>
ห้ามเขียนโค้ด ห้ามอธิบาย เพิ่มไม่เกิน 8 ไฟล์ และรักษาโฟลเดอร์ที่ผู้ใช้ระบุ
ใช้เฉพาะไฟล์ข้อความ .html .css .js .json .svg .md .txt .py ห้ามสร้าง .png .jpg หรือไฟล์ binary
ภาพในเว็บเกมให้วาดด้วย Canvas, CSS หรือ SVG เท่านั้น""",
            request_text,
            700,
        )

    def code_project_file(
        self, request_text: str, path: str, purpose: str, manifest: list[str],
        current: str = "", issue: str = "",
    ) -> str:
        context = (
            f"คำสั่งโครงการ:\n{request_text}\n\nManifest: {', '.join(manifest)}\n"
            f"สร้างเฉพาะไฟล์: {path}\nหน้าที่: {purpose}\n"
        )
        if current:
            context += f"\nเนื้อหาปัจจุบันที่ต้องแก้:\n{current[:MAX_TOOL_RESULT_CHARS]}\n"
        if issue:
            context += f"\nปัญหาที่ Reviewer พบ: {issue}\n"
        return self._role_call(
            """คุณคือ Coder สร้างหรือแก้ไฟล์เดียวให้สมบูรณ์และเชื่อมกับ manifest ถูกต้อง
ตอบเฉพาะ <file path="พาธที่กำหนด">เนื้อหาเต็ม</file> ห้าม Markdown ห้าม JSON ห้ามสร้างไฟล์อื่น
ข้อควรระวัง: ห้ามเขียนโค้ดย่อ ห้ามใส่ //... หรือ TODO คุณต้องเขียนโค้ดทั้งหมดให้ทำงานได้จริง 100% เท่านั้น""",
            context,
            4096,
        )

    def review_project(self, request_text: str, files_text: str) -> str:
        return self._role_call(
            """คุณคือ Reviewer ตรวจโค้ดว่าทำตามคำสั่งและไฟล์เชื่อมกันถูกต้อง
ถ้าผ่านตอบ <review status="pass"/> ถ้ามีปัญหาให้ตอบ XML เท่านั้น เช่น
<review><issue file="path">ปัญหาและวิธีแก้แบบสั้น</issue></review>
รายงานบั๊กที่ทำให้ใช้งานไม่ได้ และต้องแจ้งเตือนหากโค้ดไม่สมบูรณ์ (เช่นมี //... หรือ TODO) ไม่เขียนโค้ดเอง""",
            f"คำสั่ง:\n{request_text}\n\nไฟล์:\n{files_text[:MAX_TOOL_RESULT_CHARS]}",
            900,
        )

    def _generate_google(self, messages: list[dict[str, Any]]) -> str:
        contents = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3, "topK": 40, "maxOutputTokens": 2048},
        }
        endpoint = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"API ตอบกลับ {exc.code}: {detail[:500]}") from exc
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"รูปแบบคำตอบจาก API ไม่ถูกต้อง: {data}") from exc


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.I | re.S)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        # Some local models add a sentence before an otherwise valid call.
        # Try decoding a JSON object starting at each opening brace.
        value = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                decoded, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict) and "tool" in decoded:
                value = decoded
                break
        if value is None:
            return None
    if isinstance(value, dict) and isinstance(value.get("tool"), str) and isinstance(value.get("args"), dict):
        return value["tool"], value["args"]
    return None


def looks_like_broken_tool_call(text: str) -> bool:
    """Detect intended calls that could not be parsed, without false matching prose."""
    return bool(
        re.search(r'["\']tool["\']\s*:', text, re.I)
        and re.search(r"\b(list_files|read_file|write_file|web_search|fetch_url|mcp__[\w-]+__[\w.-]+)\b", text)
    )


def parse_file_block(text: str) -> tuple[str, str] | None:
    blocks = parse_file_blocks(text)
    return blocks[0] if blocks else None


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    matches = re.finditer(
        r"<file\s+path\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))\s*>\s*(.*?)\s*</file>",
        text,
        re.I | re.S,
    )
    blocks = []
    for match in matches:
        path = next(value for value in match.group(1, 2, 3) if value is not None)
        blocks.append((path.strip(), match.group(4)))
    return blocks


def parse_plan(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1).strip(), re.sub(r"<[^>]+>", "", match.group(2)).strip())
        for match in re.finditer(
            r"<file\s+path\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</file>", text, re.I | re.S
        )
    ]


def parse_review(text: str) -> list[tuple[str, str]]:
    if re.search(r"<review\s+status\s*=\s*[\"']pass[\"']\s*/?>", text, re.I):
        return []
    return [
        (match.group(1).strip(), re.sub(r"<[^>]+>", "", match.group(2)).strip())
        for match in re.finditer(
            r"<issue\s+file\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</issue>", text, re.I | re.S
        )
    ]


def extract_generated_file(text: str, request_text: str) -> tuple[str, str] | None:
    """Accept the XML contract plus common local-model formatting deviations."""
    parsed = parse_file_block(text)
    if parsed:
        return parsed
    path_match = re.search(
        r"(?<![\w.-])([\w.-]+(?:/[\w.-]+)*\.(?:html?|css|js|json|md|txt|py|svg))(?![\w.-])",
        request_text,
        re.I,
    )
    path = path_match.group(1) if path_match else "generated/index.html"
    fence = re.search(r"```(?:html?|css|javascript|js|json|svg|python|py|text)?\s*\n(.*?)```", text, re.I | re.S)
    if fence:
        return path, fence.group(1).strip()
    html_start = re.search(r"<!doctype\s+html|<html\b", text, re.I)
    if html_start:
        html_end = list(re.finditer(r"</html\s*>", text, re.I))
        end = html_end[-1].end() if html_end else len(text)
        return path, text[html_start.start():end].strip()
    # A file-specific Coder is allowed to return plain source without a
    # wrapper. This is safe because the orchestrator already owns the target
    # path and still applies the workspace sandbox before writing.
    if path_match and text.strip() and not re.search(r'\{["\']tool["\']\s*:', text):
        return path, text.strip()
    return None


def extract_generated_files(text: str, request_text: str) -> list[tuple[str, str]]:
    blocks = parse_file_blocks(text)
    if blocks:
        return blocks
    tool_call = parse_tool_call(text)
    if tool_call and tool_call[0] == "write_file":
        args = tool_call[1]
        if isinstance(args.get("path"), str) and isinstance(args.get("content"), str):
            return [(args["path"], args["content"])]
    single = extract_generated_file(text, request_text)
    return [single] if single else []


def apply_requested_base_dir(
    files: list[tuple[str, str]], request_text: str
) -> list[tuple[str, str]]:
    """Keep sibling files in a project directory mentioned in recent context."""
    path_match = re.search(
        r"(?<![\w.-])([\w.-]+(?:/[\w.-]+)+\.(?:html?|css|js|json|md|txt|py|svg))(?![\w.-])",
        request_text,
        re.I,
    )
    if not path_match:
        return files
    base = str(Path(path_match.group(1)).parent)
    if base == ".":
        return files
    return [
        (str(Path(base) / path) if Path(path).parent == Path(".") else path, content)
        for path, content in files
    ]


def requests_action(text: str) -> bool:
    return bool(re.search(r"(?:สร้าง|เขียน|แก้(?:ไข)?|ทำ(?:เว็บ|เกม|ไฟล์|โปรเจกต์)|เริ่มเลย|ไม่มี|ไม่ทำงาน|บั๊ก|ผิด|create|build|write|edit|fix)", text, re.I))


class LocalModelManager:
    """Own one llama-server process and downloaded GGUF models."""

    def __init__(self, root: Path, server_bin: Path, state_dir: Path) -> None:
        self.root = root.expanduser().resolve()
        self.server_bin = server_bin.expanduser().resolve()
        self.state_dir = state_dir
        self.process: subprocess.Popen[str] | None = None
        self.active_model: Path | None = None
        self.active_profile: dict[str, Any] | None = None
        self._log_handle: Any = None

    def models(self) -> list[Path]:
        return discover_models(self.root)

    @staticmethod
    def _health(api_url: str, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(api_url.rstrip("/") + "/health", timeout=timeout) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.active_model = None
        self.active_profile = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def load(self, model_path: Path, api_url: str, gpu_layers: int | str = "auto") -> None:
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise RuntimeError(f"ไม่พบไฟล์โมเดล: {model_path}")
        if not os.access(self.server_bin, os.X_OK):
            raise RuntimeError(f"ไม่พบ llama-server: {self.server_bin}")
        parsed = urllib.parse.urlparse(api_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("ตัวจัดการโมเดลรองรับเฉพาะ API ภายในเครื่อง")
        port = parsed.port or 8080
        self.stop()
        if self._health(api_url):
            raise RuntimeError(f"พอร์ต {port} ถูกใช้งานโดยเซิร์ฟเวอร์อื่น กรุณาปิดก่อน")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (self.state_dir / "server.log").open("w", encoding="utf-8")
        profile = inference_profile(model_path)
        if gpu_layers != "auto":
            profile["gpu_layers"] = gpu_layers
        command = [
            str(self.server_bin), "--model", str(model_path),
            "--gpu-layers", str(profile["gpu_layers"]), "--flash-attn", "auto",
            "--fit", "on", "--threads", str(profile["threads"]),
            "--threads-batch", str(profile["threads"]),
            "--batch-size", str(profile["batch"]), "--ubatch-size", str(profile["ubatch"]),
            "--cache-type-k", str(profile["cache_type"]),
            "--cache-type-v", str(profile["cache_type"]),
            "--load-mode", "mmap", "--embeddings",
            "--reasoning", "off", "--reasoning-budget", "0",
            "--ctx-size", str(profile["context"]),
            "--parallel", "1", "--no-cont-batching", "--port", str(port),
        ]
        mmproj = model_path.with_name("gemma-4-E4B-it-mmproj.gguf")
        if "gemma-4-e4b" in model_path.name.lower() and mmproj.is_file():
            command[3:3] = ["--mmproj", str(mmproj)]
        environment = os.environ.copy()
        library_dir = str(self.server_bin.parent)
        current_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            library_dir if not current_library_path
            else library_dir + os.pathsep + current_library_path
        )
        self.process = subprocess.Popen(
            command, stdout=self._log_handle, stderr=subprocess.STDOUT,
            text=True, env=environment,
        )
        for _ in range(120):
            if self._health(api_url):
                self.active_model = model_path
                self.active_profile = profile
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"โหลดโมเดลไม่สำเร็จ ดู log ที่ {self.state_dir / 'server.log'}")
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("โหลดโมเดลไม่สำเร็จภายใน 60 วินาที")

    def download(
        self, catalog_name: str, progress: Any | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        relative, url = MODEL_CATALOG[catalog_name]
        expected_size, expected_hash = MODEL_DOWNLOAD_META.get(catalog_name, (0, ""))
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and expected_size > 0 and target.stat().st_size == expected_size:
            self._download_companion(catalog_name, progress, cancel_event)
            return target
        distrobox = shutil.which("distrobox")
        if distrobox:
            command = [
                distrobox, "enter", "llama-build", "--", "aria2c",
                "--continue=true", "--max-connection-per-server=16", "--split=16",
                "--min-split-size=16M", "--file-allocation=none",
                f"--dir={target.parent}", f"--out={target.name}", url,
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if cancel_event and cancel_event.is_set():
                    process.terminate()
                    process.wait(timeout=5)
                    raise GenerationCancelled("ยกเลิกการดาวน์โหลดแล้ว")
                match = re.search(r"\((\d+)%\)", line)
                if match and progress:
                    progress(int(match.group(1)))
            result_code = process.wait()
            if result_code != 0:
                raise RuntimeError("aria2c ดาวน์โหลดไม่สำเร็จ")
        else:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
                downloaded = 0
                while True:
                    if cancel_event and cancel_event.is_set():
                        raise GenerationCancelled("ยกเลิกการดาวน์โหลดแล้ว")
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    if progress:
                        if expected_size > 0:
                            progress(min(100, int(downloaded / expected_size * 100)))
                        else:
                            # Fake progress if size is unknown
                            progress(min(99, int(downloaded / (1024 * 1024 * 100))))
        if expected_size > 0 and target.stat().st_size != expected_size:
            raise RuntimeError("ขนาดไฟล์ที่ดาวน์โหลดไม่ถูกต้อง")
        if expected_hash:
            digest = hashlib.sha256()
            with target.open("rb") as model_file:
                for block in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError("SHA-256 ของโมเดลไม่ถูกต้อง")
        if progress:
            progress(100)
        self._download_companion(catalog_name, progress, cancel_event)
        return target

    def _download_companion(
        self, catalog_name: str, progress: Any | None,
        cancel_event: threading.Event | None,
    ) -> Path | None:
        companion = MODEL_COMPANIONS.get(catalog_name)
        if not companion:
            return None
        relative, url, expected_size, expected_hash = companion
        target = self.root / relative
        if target.is_file() and target.stat().st_size == expected_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response, partial.open("wb") as output:
            received = 0
            while True:
                if cancel_event and cancel_event.is_set():
                    raise GenerationCancelled("ยกเลิกการดาวน์โหลดแล้ว")
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if progress:
                    progress(min(99, round(received * 100 / expected_size)))
        if partial.stat().st_size != expected_size:
            raise RuntimeError("ขนาดไฟล์ multimodal projector ไม่ถูกต้อง")
        digest = hashlib.sha256()
        with partial.open("rb") as model_file:
            for block in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError("SHA-256 ของ multimodal projector ไม่ถูกต้อง")
        partial.replace(target)
        return target

    def delete(self, model_path: Path) -> None:
        model_path = model_path.resolve()
        model_path.relative_to(self.root)
        if self.active_model and model_path == self.active_model.resolve():
            raise RuntimeError("ปิดโมเดลก่อนลบ")
        gio = shutil.which("gio")
        if gio:
            result = subprocess.run([gio, "trash", str(model_path)], capture_output=True, text=True)
            if result.returncode == 0:
                return
        model_path.unlink()


class ToolTip:
    def __init__(self, widget: ctk.CTkBaseClass, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window: tk.Toplevel | None = None
        self.id = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event: Any = None) -> None:
        self.schedule()

    def leave(self, event: Any = None) -> None:
        self.unschedule()
        self.hidetip()

    def schedule(self) -> None:
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self) -> None:
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def showtip(self, event: Any = None) -> None:
        x, y, cx, cy = self.widget.bbox("insert") or (0,0,0,0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        is_light = ctk.get_appearance_mode().lower() == "light"
        bg = "#FFFFFF" if is_light else "#1B263B"
        fg = "#0F172A" if is_light else "#F3F6FC"

        label = tk.Label(tw, text=self.text, justify='left',
                         background=bg, foreground=fg, relief='solid', borderwidth=1,
                         font=(THAI_FONT, 11, "normal"))
        label.pack(ipadx=6, ipady=3)

    def hidetip(self) -> None:
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()


class ChatApp(ctk.CTk):
    BG = ("#F8FAFC", "#0B0F19")
    SIDEBAR = ("#E2E8F0", "#111827")
    PANEL = ("#FFFFFF", "#151D2E")
    PANEL_HOVER = ("#F1F5F9", "#1B263B")
    BORDER = ("#94A3B8", "#22304A")
    TEXT = ("#0F172A", "#F3F6FC")
    MUTED = ("#475569", "#8CA0BE")
    ACCENT = ("#6D28D9", "#7C5CFC")
    ACCENT_HOVER = ("#5B21B6", "#6246CC")
    USER_BUBBLE = ("#4E3FBE", "#4E3FBE")
    BOT_BUBBLE = ("#F1F5F9", "#172238")

    def _resolve_color(self, color: tuple[str, str] | str) -> str:
        if isinstance(color, tuple):
            return color[0] if ctk.get_appearance_mode().lower() == "light" else color[1]
        return color

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title("LocalForge AI — Local AI Workspace")
        self.geometry("1120x760")
        self.minsize(880, 620)
        self.configure(fg_color=self.BG)
        ctk.set_appearance_mode("dark")

        workspace = Path(os.environ.get("LOCALFORGE_WORKSPACE", os.environ.get("GEMMA_WORKSPACE", Path.home())))
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        self.state_dir = state_home / "localforge-ai"
        legacy_state_dir = state_home / "gemma-assistant"
        if not self.state_dir.exists() and legacy_state_dir.exists():
            legacy_state_dir.rename(self.state_dir)
        self.history_file = self.state_dir / "conversation.json"
        self.settings_file = self.state_dir / "settings.json"
        self.model_manager = LocalModelManager(
            Path(os.environ.get("LOCALFORGE_MODEL_ROOT", os.environ.get("GEMMA_MODEL_ROOT", DEFAULT_MODEL_ROOT))),
            Path(os.environ.get("LLAMA_SERVER_BIN", DEFAULT_SERVER_BIN)),
            self.state_dir,
        )
        preferences = self._load_preferences()
        language = str(preferences.get("language", "th"))
        if language not in LANGUAGE_NAMES:
            language = "th"
        self.language_var = ctk.StringVar(value=language)
        self._lang_code = language
        self.language_name_var = ctk.StringVar(value=LANGUAGE_NAMES[language])
        global THAI_FONT
        THAI_FONT = LANGUAGE_FONTS[language]
        if not os.environ.get("LOCALFORGE_WORKSPACE") and not os.environ.get("GEMMA_WORKSPACE"):
            saved_workspace = str(preferences.get("workspace", "")).strip()
            if saved_workspace:
                try:
                    resolved = Path(saved_workspace).expanduser().resolve()
                    if resolved.is_dir():
                        workspace = resolved
                except OSError:
                    pass
        self.tools = Tools(workspace)
        self.file_transaction = FileTransaction(self.tools.workspace, self.state_dir / "backups")
        self.hooks = HookEngine(self.state_dir / "audit.jsonl", MAX_TOOL_RESULT_CHARS)
        self.mcp_manager = MCPManager(self.state_dir / "mcp_servers.json", self.tools.workspace)
        self.changed_files: set[str] = set()
        self.messages: list[dict[str, Any]] = self._load_history()
        self.conversation_store = ConversationStore(self.state_dir / "conversations.json")
        self.vector_db = VectorDB(self.state_dir / "vectordb.sqlite")
        if not self.conversation_store.active().get("messages") and self.messages:
            self.conversation_store.set_messages(self.messages)
        else:
            self.messages = list(self.conversation_store.active().get("messages", []))
        self.api_url_var = ctk.StringVar(
            value=os.environ.get("LOCALFORGE_API_URL", os.environ.get("GEMMA_API_URL", preferences.get("api_url", "http://localhost:8080")))
        )
        self.model_var = ctk.StringVar(value=os.environ.get("GEMMA_MODEL", "gemma3-4b"))
        self.selected_model_var = ctk.StringVar(value=preferences.get("selected_model", ""))
        self.download_model_var = ctk.StringVar(value=next(iter(MODEL_CATALOG)))
        self.model_status_var = ctk.StringVar(value=self._t("model_off"))
        self.auto_router_var = ctk.BooleanVar(value=bool(preferences.get("auto_router", True)))
        self.appearance_var = ctk.StringVar(value=str(preferences.get("appearance", "Dark")))
        saved_scale = max(0.85, min(1.25, float(preferences.get("font_scale", 1.0))))
        self.font_scale_var = ctk.DoubleVar(value=saved_scale)
        self.ui_scale_label_var = ctk.StringVar(value=f"{round(saved_scale * 100):.0f}%")
        self.download_cancel_event = threading.Event()
        # Multi-agent performs several sequential model calls and is much slower
        # on small local models. Keep the fast, single-agent path as the default;
        # users can enable multi-agent in Settings for larger coding tasks.
        self.multi_agent_var = ctk.BooleanVar(value=bool(preferences.get("multi_agent", False)))
        self.planner_model_var = ctk.StringVar(value=preferences.get("planner_model", "auto"))
        self.coder_model_var = ctk.StringVar(value=preferences.get("coder_model", "auto"))
        self.reviewer_model_var = ctk.StringVar(value=preferences.get("reviewer_model", "auto"))
        self.settings_window: ctk.CTkToplevel | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.cancel_event = threading.Event()
        self.request_started = 0.0
        self._pending_send: bool | None = None
        self.stream_buffer = ""
        self.stream_widgets: dict[str, Any] | None = None
        self.chat_images: list[tk.PhotoImage] = []
        self.pending_media: list[tuple[Path, dict[str, Any]]] = []
        self.recording_process: subprocess.Popen[Any] | None = None
        self.recording_path: Path | None = None
        self.speech_process: subprocess.Popen[Any] | None = None
        self.diff_request: dict[str, Any] | None = None
        self._compact_layout: bool | None = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_display_settings()
        self._build_ui()
        self.after(80, self._poll_events)
        self.after(1000, self._update_system_monitor)
        self.after(0, self._present_window)

    def _present_window(self) -> None:
        self.update_idletasks()
        self.deiconify()
        self.lift()

    def _t(self, key: str, **values: Any) -> str:
        return translate(self._lang_code, key, **values)

    def _load_preferences(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            selected = str(data.get("selected_model", ""))
            if selected:
                selected = Path(selected).name
            if not selected or not (self.model_manager.root / selected).is_file():
                try:
                    data["selected_model"] = discover_models(self.model_manager.root)[0].name
                except (IndexError, AttributeError):
                    pass
            return data
        except (OSError, ValueError):
            return {}

    def _save_preferences(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps({
            "api_url": self.api_url_var.get().strip(),
            "selected_model": self.selected_model_var.get(),
            "multi_agent": bool(self.multi_agent_var.get()),
            "planner_model": self.planner_model_var.get(),
            "coder_model": self.coder_model_var.get(),
            "reviewer_model": self.reviewer_model_var.get(),
            "auto_router": bool(self.auto_router_var.get()),
            "appearance": self.appearance_var.get(),
            "font_scale": float(self.font_scale_var.get()),
            "language": self.language_var.get(),
            "workspace": str(self.tools.workspace),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_history(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.history_file.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            if isinstance(messages, list):
                return [
                    {
                        "role": m["role"], "content": m["content"],
                        **({"media_paths": m["media_paths"]} if isinstance(m.get("media_paths"), list) else {}),
                    }
                    for m in messages
                    if isinstance(m, dict)
                    and m.get("role") in {"user", "assistant"}
                    and isinstance(m.get("content"), str)
                ][-MAX_SAVED_MESSAGES:]
        except (OSError, ValueError, TypeError):
            pass
        return []

    def _save_history(self) -> None:
        saved = [
            {
                "role": m["role"], "content": m["content"],
                **({"media_paths": m["media_paths"]} if isinstance(m.get("media_paths"), list) else {}),
            }
            for m in self.messages
            if m.get("role") in {"user", "assistant"}
            and isinstance(m.get("content"), str)
            and not m.get("tool_calls")
        ][-MAX_SAVED_MESSAGES:]
        self.conversation_store.set_messages(saved)
        if threading.current_thread() is threading.main_thread():
            if hasattr(self, "conversation_list"):
                self._refresh_conversations()
            self._update_context_meter()
        elif hasattr(self, "events"):
            self.events.put(("history_saved", ""))

    def _recent_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        profile = self.model_manager.active_profile or {}
        context_size = profile.get("context", MODEL_CONTEXT_TOKENS)
        # Leave room for system prompt and max_tokens generation (1200)
        budget = max(1000, context_size - 2000)
        return select_recent_messages(messages, budget)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, width=270, corner_radius=0, fg_color=self.SIDEBAR)
        self.sidebar = side
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        side.grid_propagate(False)
        self.bind("<Configure>", self._responsive_layout, add="+")
        brand = ctk.CTkFrame(side, fg_color="transparent")
        brand.pack(fill="x", padx=14, pady=(16, 16))
        icon_box = ctk.CTkFrame(brand, width=30, height=30, corner_radius=8, fg_color=self.ACCENT)
        icon_box.pack(side="left")
        icon_box.pack_propagate(False)
        ctk.CTkLabel(icon_box, text="⚙", text_color="#FFFFFF", font=ctk.CTkFont(size=16)).pack(expand=True)
        brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(side="left", padx=9)
        ctk.CTkLabel(
            brand_text, text="LocalForge AI", anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(0, 0))
        ctk.CTkLabel(
            brand_text, text=self._t("local_model"), anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(0, 0))

        ctk.CTkButton(
            side, text=self._t("new_chat"), height=32, corner_radius=8,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color="#B9A8FF",
            border_width=1, border_color=self.ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"), command=self._clear
        ).pack(fill="x", padx=14, pady=(0, 8))

        ctk.CTkButton(
            side, text=self._t("index_rag"), height=32, corner_radius=8,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color="#A7F3D0",
            border_width=1, border_color="#34D399",
            font=ctk.CTkFont(size=12, weight="bold"), command=self._index_document
        ).pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkButton(
            side, text=self._t("rag_manage"), height=28, corner_radius=8,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            border_width=1, border_color=self.BORDER,
            font=ctk.CTkFont(size=11), command=self._open_rag_manager
        ).pack(fill="x", padx=14, pady=(0, 18))

        # Reserve the sidebar footer before adding expandable content.  Tk's
        # packer allocates space in packing order; placing this after the
        # conversation list allowed that list to consume the whole sidebar at
        # small window sizes or higher UI scales.
        sidebar_footer = ctk.CTkFrame(side, fg_color=self.SIDEBAR)
        sidebar_footer.pack(side="bottom", fill="x")
        self.settings_button = ctk.CTkButton(
            sidebar_footer, text=self._t("settings"), height=40, corner_radius=20,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            font=ctk.CTkFont(size=13, weight="bold"), command=self._open_settings
        )
        self.settings_button.pack(fill="x", padx=18, pady=(4, 2))
        self.system_monitor = ctk.CTkFrame(sidebar_footer, fg_color="transparent")
        self.system_monitor.pack(fill="x", padx=14, pady=(8, 14))
        self.sys_cpu = ctk.CTkLabel(self.system_monitor, text="CPU —", fg_color=self.PANEL, corner_radius=6, font=ctk.CTkFont(size=11), text_color=self.MUTED)
        self.sys_cpu.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 4))
        self.sys_ram = ctk.CTkLabel(self.system_monitor, text="RAM —", fg_color=self.PANEL, corner_radius=6, font=ctk.CTkFont(size=11), text_color=self.MUTED)
        self.sys_ram.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 4))
        self.sys_gpu = ctk.CTkLabel(self.system_monitor, text="GPU —", fg_color=self.PANEL, corner_radius=6, font=ctk.CTkFont(size=11), text_color=self.MUTED)
        self.sys_gpu.grid(row=1, column=0, sticky="ew", padx=(0, 2))
        self.sys_vram = ctk.CTkLabel(self.system_monitor, text="VRAM —", fg_color=self.PANEL, corner_radius=6, font=ctk.CTkFont(size=11), text_color=self.MUTED)
        self.sys_vram.grid(row=1, column=1, sticky="ew", padx=(2, 0))
        self.system_monitor.grid_columnconfigure((0,1), weight=1)

        workspace_card = ctk.CTkFrame(side, fg_color="transparent")
        workspace_card.pack(fill="x", padx=14, pady=(0, 0))
        ctk.CTkLabel(
            workspace_card, text=self._t("workspace"), anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(fill="x", pady=(0, 8))
        self.workspace_button = ctk.CTkButton(
            workspace_card, text="📁 " + str(self.tools.workspace), height=28, corner_radius=7,
            fg_color=self.PANEL, hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            border_width=1, border_color=self.BORDER, anchor="w",
            font=ctk.CTkFont(size=11), command=self._choose_workspace
        )
        self.workspace_button.pack(fill="x", pady=(0, 6))
        ToolTip(self.workspace_button, self._t("change_folder"))

        ws_actions = ctk.CTkFrame(workspace_card, fg_color="transparent")
        ws_actions.pack(fill="x", pady=(0, 18))
        explorer_button = ctk.CTkButton(
            ws_actions, text=self._t("explorer"), height=26, corner_radius=6,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            border_width=1, border_color=self.BORDER,
            font=ctk.CTkFont(size=11), command=self._open_project_explorer,
        )
        explorer_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        ToolTip(explorer_button, self._t("project_explorer"))
        undo_button = ctk.CTkButton(
            ws_actions, text=self._t("undo"), height=26, corner_radius=6,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            border_width=1, border_color=self.BORDER,
            font=ctk.CTkFont(size=11), command=self._undo_files,
        )
        undo_button.pack(side="right", fill="x", expand=True, padx=(3, 0))
        ToolTip(undo_button, self._t("undo_short"))

        conversations = ctk.CTkFrame(side, fg_color="transparent")
        conversations.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        ctk.CTkLabel(
            conversations, text=self._t("conversations"), anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=11)
        ).pack(fill="x", pady=(0, 8))
        self.conversation_search_var = ctk.StringVar()
        search = ctk.CTkEntry(
            conversations, textvariable=self.conversation_search_var,
            placeholder_text=self._t("search_conversations"), height=28, corner_radius=7,
            fg_color="transparent", border_width=1, border_color=self.BORDER, text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=11),
        )
        search.pack(fill="x", pady=(0, 8))
        search.bind("<KeyRelease>", lambda _event: self._refresh_conversations())
        self.conversation_list = ctk.CTkScrollableFrame(
            conversations, fg_color="transparent", height=140,
            scrollbar_button_color=self.BORDER,
        )
        self.conversation_list.pack(fill="both", expand=True)
        convo_actions = ctk.CTkFrame(conversations, fg_color="transparent")
        convo_actions.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(convo_actions, text=self._t("export"), width=75, height=28, corner_radius=14, fg_color=self.PANEL, hover_color=self.PANEL_HOVER, text_color=self.TEXT, command=self._export_conversation).pack(side="left", padx=(0, 3))
        ctk.CTkButton(convo_actions, text=self._t("pin"), width=75, height=28, corner_radius=14, fg_color=self.PANEL, hover_color=self.PANEL_HOVER, text_color=self.TEXT, command=self._pin_conversation).pack(side="left", padx=3)
        ctk.CTkButton(convo_actions, text=self._t("delete"), width=55, height=28, corner_radius=14, fg_color="transparent", hover_color=("#FEE2E2", "#713747"), text_color=self.TEXT, command=self._delete_conversation).pack(side="right")
        self._refresh_conversations()

        main = ctk.CTkFrame(self, corner_radius=0, fg_color=self.BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(main, height=78, corner_radius=0, fg_color=self.BG)
        header.grid(row=0, column=0, sticky="ew", padx=28)
        header.grid_propagate(False)
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", pady=14)
        ctk.CTkLabel(
            title_box, text=self._t("your_assistant"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w")
        tagline_frame = ctk.CTkFrame(title_box, fg_color="transparent")
        tagline_frame.pack(anchor="w", pady=(2, 0))

        icon = ctk.CTkLabel(tagline_frame, text="✨", font=ctk.CTkFont(size=14), text_color=self.MUTED)
        icon.pack(side="left", padx=(0, 10))
        ToolTip(icon, self._t("tagline"))

        status_container = ctk.CTkFrame(header, fg_color="transparent")
        status_container.pack(side="right", pady=24)

        self.theme_btn = ctk.CTkButton(
            status_container, text="🌓", width=28, height=28, corner_radius=14,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            font=ctk.CTkFont(size=14), command=self._toggle_theme
        )
        self.theme_btn.pack(side="left", padx=(0, 12))

        choices = self._model_choices()
        self.header_model_menu = ctk.CTkOptionMenu(
            status_container, variable=self.selected_model_var, values=choices,
            font=ctk.CTkFont(size=11, weight="bold"), fg_color=self.BG,
            text_color=self.MUTED, button_color=self.BG,
            button_hover_color=self.PANEL_HOVER, dropdown_fg_color=self.PANEL,
            dropdown_text_color=self.TEXT, width=140, anchor="e",
            command=self._header_model_selected
        )
        self.header_model_menu.pack(side="left", padx=(0, 12))

        status_box = ctk.CTkFrame(status_container, fg_color=self.PANEL, corner_radius=13, border_width=1, border_color=self.BORDER)
        status_box.pack(side="left", padx=(0, 8))
        self.status_dot = ctk.CTkLabel(status_box, text="●", text_color=self.MUTED, font=ctk.CTkFont(size=11))
        self.status_dot.pack(side="left", padx=(10, 5))
        self.status = ctk.CTkLabel(
            status_box, text=self._t("model_off"), text_color=self.MUTED,
            font=ctk.CTkFont(size=11)
        )
        self.status.pack(side="left", padx=(0, 4), pady=4)

        self.model_toggle_btn = ctk.CTkButton(
            status_box, text="⏻", width=22, height=22, corner_radius=11,
            fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
            font=ctk.CTkFont(size=12), command=self._toggle_model_server
        )
        self.model_toggle_btn.pack(side="left", padx=(0, 6), pady=2)
        self.context_button = ctk.CTkButton(
            status_container, text="Context 0%", width=80, height=26, corner_radius=13,
            fg_color="transparent", hover_color=self.PANEL_HOVER,
            border_width=1, border_color=self.BORDER, text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=11),
            command=self._open_context_inspector,
        )
        self.context_button.pack(side="left")

        self.chat = ctk.CTkScrollableFrame(
            main, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=self.BORDER, scrollbar_button_hover_color=self.ACCENT
        )
        self.chat.grid(row=1, column=0, sticky="nsew", padx=(26, 18), pady=(0, 8))
        self.chat.grid_columnconfigure(0, weight=1)

        composer = ctk.CTkFrame(
            main, fg_color=self.PANEL, corner_radius=12,
            border_width=1, border_color=self.BORDER
        )
        composer.grid(row=2, column=0, sticky="ew", padx=28, pady=(8, 24))
        composer.grid_columnconfigure(0, weight=1)
        self.input = ctk.CTkTextbox(
            composer, height=82, wrap="word", font=ctk.CTkFont(family=THAI_FONT, size=14),
            fg_color="transparent", border_width=0, text_color=self.TEXT
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=8)
        self.input.bind("<Control-Return>", self._send_event)
        self.input.bind("<Control-Shift-V>", self._paste_image_event)
        self._bind_edit_menu(self.input, readonly=False)
        media_bar = ctk.CTkFrame(composer, fg_color="transparent")
        media_bar.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
        self.attach_button = ctk.CTkButton(
            media_bar, text=self._t("attach_image"), width=104, height=28,
            fg_color="transparent", hover_color=self.PANEL_HOVER,
            border_width=1, border_color=self.BORDER, text_color=self.TEXT, command=self._choose_image,
        )
        self.attach_button.pack(side="left", padx=(0, 6))
        self.voice_button = ctk.CTkButton(
            media_bar, text=self._t("voice"), width=92, height=28,
            fg_color="transparent", hover_color=self.PANEL_HOVER,
            border_width=1, border_color=self.BORDER, text_color=self.TEXT, command=self._toggle_recording,
        )
        self.voice_button.pack(side="left", padx=(0, 8))
        self.router_switch = ctk.CTkSwitch(
            media_bar, text=self._t("auto_router"), variable=self.auto_router_var,
            font=ctk.CTkFont(size=11), text_color=self.MUTED, switch_width=32, switch_height=16,
            command=self._save_preferences
        )
        self.router_switch.pack(side="left", padx=(12, 8))

        self.agent_switch = ctk.CTkSwitch(
            media_bar, text=self._t("multi_agent_switch"), variable=self.multi_agent_var,
            font=ctk.CTkFont(size=11), text_color=self.MUTED, switch_width=32, switch_height=16,
            command=self._save_preferences
        )
        self.agent_switch.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            media_bar, text=self._t("multi_agent_flow"), text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=9),
        ).pack(side="left", padx=(0, 8))
        self.media_status = ctk.CTkLabel(
            media_bar, text="", text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=10),
        )
        self.media_status.pack(side="left")
        self.send_button = ctk.CTkButton(
            composer, text=self._t("send"), width=94, height=48, corner_radius=8,
            fg_color=self.ACCENT, hover_color=self.ACCENT_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"), command=self.send
        )
        self.send_button.grid(row=0, column=1, padx=(0, 12), pady=12)
        self.stop_button = ctk.CTkButton(
            composer, text=self._t("stop"), width=94, height=48, corner_radius=8,
            fg_color="#8D3D52", hover_color="#A34860",
            font=ctk.CTkFont(family=THAI_FONT, size=14, weight="bold"),
            command=self._cancel_generation,
        )
        if self.messages:
            for index, message in enumerate(self.messages):
                self._append(
                    self._t("you") if message["role"] == "user" else "LocalForge",
                    message["content"], message_index=index,
                    media_paths=message.get("media_paths"),
                )
        else:
            self._append(self._t("system"), self._t("ready"))
        self._update_context_meter()

    def _bind_edit_menu(self, widget: Any, readonly: bool) -> None:
        target = getattr(widget, "_textbox", widget)
        menu = Menu(self, tearoff=False)

        def copy_selection() -> None:
            try:
                selected = target.get("sel.first", "sel.last")
            except Exception:
                return
            self.clipboard_clear()
            self.clipboard_append(selected)
            self.update_idletasks()

        def paste_clipboard() -> None:
            try:
                value = self.clipboard_get()
            except Exception:
                return
            try:
                target.delete("sel.first", "sel.last")
            except Exception:
                pass
            target.insert("insert", value)
            target.see("insert")

        def cut_selection() -> None:
            copy_selection()
            try:
                target.delete("sel.first", "sel.last")
            except Exception:
                pass

        def select_all() -> None:
            target.tag_add("sel", "1.0", "end-1c")
            target.mark_set("insert", "1.0")
            target.see("1.0")

        menu.add_command(label=self._t("copy"), command=copy_selection)
        if not readonly:
            menu.add_command(label=self._t("paste"), command=paste_clipboard)
            menu.add_command(label=self._t("cut"), command=cut_selection)
        menu.add_separator()
        menu.add_command(label=self._t("select_all"), command=select_all)

        def popup(event: Any) -> str:
            try:
                target.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
            return "break"

        target.bind("<Button-3>", popup, add="+")
        target.bind("<Control-c>", lambda _event: (copy_selection(), "break")[1], add="+")
        if not readonly:
            target.bind("<Control-v>", lambda _event: (paste_clipboard(), "break")[1], add="+")
            target.bind("<Control-x>", lambda _event: (cut_selection(), "break")[1], add="+")

    def _copy_text(self, text: str, button: Any | None = None) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
        if button is not None:
            button.configure(text=self._t("copied"))
            self.after(1200, lambda: button.winfo_exists() and button.configure(text=self._t("copy")))

    def _chat_thumbnail(self, path: Path) -> tk.PhotoImage | None:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"} or not path.is_file():
            return None
        try:
            identity = f"{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            directory = self.state_dir / "thumbnails"
            directory.mkdir(parents=True, exist_ok=True)
            thumbnail = directory / f"{digest}.png"
            if not thumbnail.is_file():
                converter = shutil.which("magick") or shutil.which("convert")
                if not converter:
                    return None
                command = [
                    converter, str(path), "-auto-orient", "-thumbnail", "520x300>",
                    "-strip", str(thumbnail),
                ]
                result = subprocess.run(command, capture_output=True, timeout=20)
                if result.returncode != 0:
                    return None
            return tk.PhotoImage(file=str(thumbnail))
        except (OSError, subprocess.SubprocessError, tk.TclError):
            return None

    def _append(
        self, who: str, text: str, token_count: int | None = None,
        metrics: dict[str, Any] | None = None, message_index: int | None = None,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        user_labels = {translate(code, "you") for code in LANGUAGE_NAMES}
        system_labels = {translate(code, "system") for code in LANGUAGE_NAMES}
        error_labels = {translate(code, "error") for code in LANGUAGE_NAMES}
        is_user = who in user_labels
        is_error = who in error_labels
        is_system = who in system_labels or is_error
        row = ctk.CTkFrame(self.chat, fg_color="transparent")
        row.grid(sticky="ew", padx=8, pady=7)
        row.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkFrame(
            row,
            fg_color="transparent",
            corner_radius=0,
            border_width=0,
        )
        bubble.grid(row=0, column=0, sticky="e" if is_user else "w", padx=(60, 0) if is_user else (0, 60))
        label_color = ("#EF4444", "#FF9EAE") if is_error else ("#7C5CFC", "#B9A8FF") if not is_user else self.MUTED
        ctk.CTkLabel(
            bubble, text=who.upper(), anchor="e" if is_user else "w", text_color=label_color,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(fill="x", padx=15, pady=(5, 2))
        for media_path in media_paths or []:
            thumbnail = self._chat_thumbnail(Path(media_path))
            if thumbnail is not None:
                self.chat_images.append(thumbnail)
                ctk.CTkLabel(
                    bubble, text="", image=thumbnail, corner_radius=10,
                    fg_color="transparent",
                ).pack(padx=15, pady=(6, 4))
        line_count = max(1, text.count("\n") + (len(text) // 78) + 1)
        bubble_width = max(360, min(800, self.winfo_width() - 300))
        body = ctk.CTkTextbox(
            bubble, width=bubble_width, height=max(34, line_count * 25), wrap="word",
            activate_scrollbars=False, fg_color="transparent",
            border_width=0, text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=14),
        )
        body.pack(fill="x", padx=9, pady=(2, 2))
        body.insert("1.0", text)
        self._highlight_markdown(body, text)
        body.configure(state="disabled")
        self._bind_edit_menu(body, readonly=True)
        if is_system:
            self.after_idle(lambda: self.chat._parent_canvas.yview_moveto(1.0))
            return {"bubble": bubble, "body": body, "token_label": None}

        footer = ctk.CTkFrame(bubble, fg_color="transparent")
        footer.pack(fill="x", padx=15, pady=(0, 9))
        token_label = None
        if not is_user and not is_system:
            shown_tokens = token_count if token_count is not None else estimate_tokens(text)
            suffix = "tokens" if token_count is not None else self._t("tokens_approx")
            detail = f"{shown_tokens:,} {suffix}"
            if metrics:
                elapsed = float(metrics.get("elapsed", 0))
                speed = shown_tokens / elapsed if elapsed > 0 else 0
                model_name = metrics.get("model", "")
                detail += f"  •  {elapsed:.1f}s  •  {speed:.1f} tok/s"
                if metrics.get("prompt_tokens"):
                    detail = self._t("in_out", prompt_tokens=int(metrics["prompt_tokens"]), detail=detail)
                if model_name:
                    detail += f"  •  {model_name}"
            token_label = ctk.CTkLabel(
                footer, text=detail, text_color=self.MUTED,
                font=ctk.CTkFont(family=THAI_FONT, size=10)
            )
            token_label.pack(side="left")
            copy_button = ctk.CTkButton(
                footer, text=self._t("copy"), width=74, height=26, corner_radius=8,
                fg_color="transparent", hover_color=self.PANEL_HOVER,
                border_width=1, border_color=self.BORDER, text_color=self.MUTED,
                font=ctk.CTkFont(family=THAI_FONT, size=10)
            )
            copy_button.configure(command=lambda value=text, btn=copy_button: self._copy_text(value, btn))
            copy_button.pack(side="right")
            ctk.CTkButton(
                footer, text=self._t("speak"), width=74, height=26, corner_radius=8,
                fg_color="transparent", hover_color=self.PANEL_HOVER,
                border_width=1, border_color=self.BORDER, text_color=self.MUTED,
                font=ctk.CTkFont(family=THAI_FONT, size=10),
                command=lambda value=text: self._speak_text(value),
            ).pack(side="right", padx=(0, 6))
            code_blocks = re.findall(r"```(?:[\w.+-]+)?\n(.*?)```", text, re.S)
            for block_index, code in enumerate(code_blocks[:2], 1):
                ctk.CTkButton(
                    footer, text=self._t("copy_code", index=block_index), width=86, height=26,
                    fg_color="transparent", hover_color=self.PANEL_HOVER,
                    border_width=1, border_color=self.BORDER, text_color=self.MUTED,
                    command=lambda value=code.strip(): self._copy_text(value),
                ).pack(side="right", padx=3)
            if "สร้างไฟล์" in text or "เขียนไฟล์" in text:
                ctk.CTkButton(
                    footer, text=self._t("open_project"), width=82, height=26,
                    fg_color="transparent", hover_color=self.PANEL_HOVER,
                    border_width=1, border_color=self.BORDER, text_color=self.MUTED,
                    command=self._open_project_explorer,
                ).pack(side="right", padx=3)
        if message_index is not None and not is_system:
            ctk.CTkButton(
                footer, text=self._t("delete"), width=42, height=26, corner_radius=8,
                fg_color="transparent", hover_color=("#FEE2E2", "#713747"), text_color=self.MUTED,
                command=lambda index=message_index: self._delete_message(index),
            ).pack(side="right", padx=4)
            if is_user:
                ctk.CTkButton(
                    footer, text=self._t("edit"), width=52, height=26, corner_radius=8,
                    fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
                    command=lambda index=message_index: self._edit_message(index),
                ).pack(side="right", padx=4)
            else:
                ctk.CTkButton(
                    footer, text=self._t("regenerate"), width=66, height=26, corner_radius=8,
                    fg_color="transparent", hover_color=self.PANEL_HOVER, text_color=self.MUTED,
                    command=lambda index=message_index: self._regenerate_message(index),
                ).pack(side="right", padx=4)
        self.after_idle(lambda: self.chat._parent_canvas.yview_moveto(1.0))
        return {"row": row, "body": body, "token_label": token_label}

    def _responsive_layout(self, _event: Any = None) -> None:
        if hasattr(self, "sidebar"):
            compact = self.winfo_width() < 1000
            if compact != self._compact_layout:
                self._compact_layout = compact
                self.sidebar.configure(width=220 if compact else 270)

    def _highlight_markdown(self, body: Any, text: str) -> None:
        target = getattr(body, "_textbox", body)
        target.tag_configure("code", foreground=self._resolve_color(("#1E293B", "#9BE7C4")), background=self._resolve_color(("#F1F5F9", "#0E1625")))
        target.tag_configure("heading", foreground=self._resolve_color(("#6246CC", "#B9A8FF")), font=(THAI_FONT, 15, "bold"))
        target.tag_configure("bold", foreground=self._resolve_color(self.TEXT), font=(THAI_FONT, 14, "bold"))
        target.tag_configure("think", foreground=self._resolve_color(self.MUTED), font=(THAI_FONT, 14, "italic"))
        for match in re.finditer(r"```(?:[\w.+-]+)?\n.*?```", text, re.S):
            target.tag_add("code", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
        for match in re.finditer(r"(?m)^#{1,3}\s+.*$", text):
            target.tag_add("heading", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
        for match in re.finditer(r"\*\*(.+?)\*\*", text):
            target.tag_add("bold", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
        for match in re.finditer(r"<think>.*?(?:</think>|$)", text, re.S):
            target.tag_add("think", f"1.0+{match.start()}c", f"1.0+{match.end()}c")

    def _update_system_monitor(self) -> None:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
            used = (values["MemTotal"] - values["MemAvailable"]) / values["MemTotal"] * 100
            cpu = min(100, os.getloadavg()[0] / max(1, os.cpu_count() or 1) * 100)
            gpu_busy, temperature, vram = "—", "—", "—"
            for card in Path("/sys/class/drm").glob("card*/device"):
                busy = card / "gpu_busy_percent"
                temps = list((card / "hwmon").glob("hwmon*/temp1_input"))
                if busy.exists():
                    gpu_busy = busy.read_text().strip() + "%"
                    vram_used, vram_total = card / "mem_info_vram_used", card / "mem_info_vram_total"
                    if vram_used.exists() and vram_total.exists():
                        vram = f"{int(vram_used.read_text()) / 1024**3:.1f}/{int(vram_total.read_text()) / 1024**3:.0f}G"
                    if temps:
                        temperature = f"{int(temps[0].read_text()) / 1000:.0f}°C"
                    break
            self.sys_cpu.configure(text=f"CPU {cpu:.0f}%")
            self.sys_ram.configure(text=f"RAM {used:.0f}%")
            self.sys_gpu.configure(text=f"GPU {gpu_busy}")
            vram_display = vram.split('/')[0] + "G" if '/' in vram else vram
            self.sys_vram.configure(text=f"VRAM {vram_display}")
        except Exception:
            pass
        if self.winfo_exists():
            self.after(2000, self._update_system_monitor)

    def _apply_display_settings(self, value: Any = None) -> None:
        if isinstance(value, str):
            scale = float(self.font_scale_var.get())
        else:
            scale = float(self.font_scale_var.get() if value is None else value)
        scale = max(0.85, min(1.25, scale))
        self.font_scale_var.set(scale)
        self.ui_scale_label_var.set(f"{round(scale * 100):.0f}%")
        ctk.set_appearance_mode(self.appearance_var.get())
        ctk.set_widget_scaling(scale)
        ctk.set_window_scaling(scale)

    def _notify_finished(self) -> None:
        notifier = shutil.which("notify-send")
        if notifier and self.focus_displayof() is None:
            subprocess.Popen(
                [notifier, "LocalForge AI", self._t("notify_done")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def _render_messages(self) -> None:
        for child in self.chat.winfo_children():
            child.destroy()
        self.chat_images.clear()
        for index, message in enumerate(self.messages):
            self._append(
                self._t("you") if message.get("role") == "user" else "LocalForge",
                str(message.get("content", "")), message_index=index,
                media_paths=message.get("media_paths"),
            )
        self._update_context_meter()

    def _update_context_meter(self) -> None:
        if not hasattr(self, "context_button"):
            return
        report = context_report(self.messages, MODEL_CONTEXT_TOKENS)
        color = ("#EF4444", "#FF9EAE") if report["percent"] >= 85 else ("#F59E0B", "#e5ad45") if report["percent"] >= 65 else self.MUTED
        self.context_button.configure(
            text=self._t("context_meter", percent=f"{report['percent']:.0f}"), text_color=color,
        )

    def _trim_context(self) -> None:
        if len(self.messages) <= 8:
            return
        self.messages = self.messages[-8:]
        self._save_history()
        self._render_messages()

    def _summarize_context(self) -> bool:
        """Summarize older messages; returns True when done synchronously
        (local fallback), False while an async LLM summary is running."""
        if len(self.messages) <= 8:
            return True
        old, recent = self.messages[:-6], self.messages[-6:]
        api_url = self.api_url_var.get().strip()
        if not self.model_manager._health(api_url):
            self._apply_summary(self._fallback_summary(old), recent)
            return True
        self.status.configure(text=self._t("summarizing"), text_color=("#F59E0B", "#e5ad45"))
        client = GemmaClient(api_url, self.model_var.get().strip())
        threading.Thread(
            target=self._summarize_worker, args=(client, old, recent), daemon=True
        ).start()
        return False

    def _summarize_worker(
        self, client: GemmaClient, old: list[dict[str, Any]], recent: list[dict[str, Any]]
    ) -> None:
        try:
            transcript = "\n".join(
                ("ผู้ใช้" if message.get("role") == "user" else "AI")
                + ": "
                + re.sub(r"\s+", " ", str(message.get("content", ""))).strip()[:240]
                for message in old[-20:]
            ) or "(ไม่มีข้อความ)"
            summary = client.summarize(transcript)
        except Exception:
            summary = self._fallback_summary(old)
        self.after(0, lambda: self._apply_summary(summary, recent))

    def _fallback_summary(self, old: list[dict[str, Any]]) -> str:
        facts = []
        for message in old:
            content = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
            if content:
                facts.append(("ผู้ใช้: " if message.get("role") == "user" else "AI: ") + content[:180])
        return self._t("summary_prefix") + "\n" + "\n".join(f"- {item}" for item in facts[-12:])

    def _apply_summary(self, summary: str, recent: list[dict[str, Any]]) -> None:
        if not summary:
            return
        self.messages = [{"role": "assistant", "content": summary}, *recent]
        self._save_history()
        self._render_messages()
        self.status.configure(text=self._t("idle"), text_color=("#10B981", "#63c174"))
        pending = self._pending_send
        self._pending_send = None
        if pending is not None:
            self.send(pending)

    def _open_context_inspector(self) -> None:
        report = context_report(self.messages, MODEL_CONTEXT_TOKENS)
        window = ctk.CTkToplevel(self)
        window.title(self._t("context_inspector_title"))
        window.geometry("620x560")
        ctk.CTkLabel(
            window, text=self._t("context_usage",
                                 used=report["used"], maximum=report["maximum"], percent=f"{report['percent']}"),
            font=ctk.CTkFont(family=THAI_FONT, size=20, weight="bold")
        ).pack(fill="x", padx=22, pady=(22, 8))
        progress = ctk.CTkProgressBar(window, progress_color=self.ACCENT)
        progress.set(report["percent"] / 100)
        progress.pack(fill="x", padx=22, pady=(0, 14))
        details = ctk.CTkTextbox(window, font=ctk.CTkFont(family=THAI_FONT, size=12))
        details.pack(fill="both", expand=True, padx=22, pady=8)
        lines = [f"{index + 1}. {entry['role']} — {entry['tokens']:,} tokens" for index, entry in enumerate(report["entries"])]
        details.insert("1.0", "\n".join(lines) or self._t("context_empty"))
        details.configure(state="disabled")
        buttons = ctk.CTkFrame(window, fg_color="transparent")
        buttons.pack(fill="x", padx=22, pady=(6, 20))
        ctk.CTkButton(buttons, text=self._t("trim_recent"), command=lambda: (self._trim_context(), window.destroy())).pack(side="left")
        ctk.CTkButton(buttons, text=self._t("summarize_old"), command=lambda: (self._summarize_context(), window.destroy())).pack(side="right")

    def _delete_message(self, index: int) -> None:
        if 0 <= index < len(self.messages):
            del self.messages[index]
            self._save_history()
            self._render_messages()

    def _edit_message(self, index: int) -> None:
        if not (0 <= index < len(self.messages)):
            return
        value = simpledialog.askstring(
            self._t("edit_message_title"), self._t("edit_message_prompt"),
            initialvalue=str(self.messages[index].get("content", "")), parent=self,
        )
        if value:
            self.messages = self.messages[:index]
            self._save_history()
            self._render_messages()
            self.input.delete("1.0", "end")
            self.input.insert("1.0", value)
            self.input.focus_set()

    def _regenerate_message(self, index: int) -> None:
        previous = next(
            (self.messages[pos].get("content", "") for pos in range(index - 1, -1, -1) if self.messages[pos].get("role") == "user"),
            "",
        )
        if previous:
            self.messages = self.messages[:index]
            self._save_history()
            self._render_messages()
            self.input.delete("1.0", "end")
            self.input.insert("1.0", str(previous))
            self.send(ignore_cache=True)

    def _begin_stream(self) -> None:
        if self.stream_widgets is None:
            self.stream_buffer = ""
            self.stream_widgets = self._append("LocalForge", "▌", 0)

    def _update_stream(self, piece: str) -> None:
        self._begin_stream()
        self.stream_buffer += piece
        body = self.stream_widgets["body"]
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert("1.0", self.stream_buffer + "▌")

        text = self.stream_buffer
        line_count = max(1, text.count("\n") + (len(text) // 78) + 1)
        body.configure(height=max(34, line_count * 25))
        self._highlight_markdown(body, text)

        body.configure(state="disabled")
        label = self.stream_widgets.get("token_label")
        if label:
            elapsed = max(0.1, time.monotonic() - self.request_started)
            tokens = estimate_tokens(self.stream_buffer)
            label.configure(text=self._t(
                "stream_speed", tokens=tokens, elapsed=elapsed, speed=tokens / elapsed
            ))
        self.chat._parent_canvas.yview_moveto(1.0)

    def _finish_stream(
        self, text: str, tokens: int, metrics: dict[str, Any],
        message_index: int | None = None,
    ) -> None:
        if not self.stream_widgets:
            self._append("LocalForge", text, tokens or None, metrics, message_index)
            return
        row = self.stream_widgets.get("row")
        if row and row.winfo_exists():
            row.destroy()
        self.stream_widgets = None
        self.stream_buffer = ""
        self._append("LocalForge", text, tokens or None, metrics, message_index)

    def _cancel_generation(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.status.configure(text=self._t("stopping_model"), text_color=("#EF4444", "#FF9EAE"))

    def _on_close(self) -> None:
        self._save_preferences()
        self.cancel_event.set()
        self.download_cancel_event.set()
        if self.recording_process and self.recording_process.poll() is None:
            self.recording_process.terminate()
        if self.speech_process and self.speech_process.poll() is None:
            self.speech_process.terminate()
        self.mcp_manager.close()
        self.model_manager.stop()
        self.destroy()

    def _choose_workspace(self) -> None:
        zenity = shutil.which("zenity")
        selected = None
        if zenity:
            result = subprocess.run(
                [zenity, "--file-selection", "--directory", f"--title={self._t('choose_workspace_title')}", f"--filename={self.tools.workspace}/"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                selected = result.stdout.strip()
        else:
            selected = filedialog.askdirectory(initialdir=self.tools.workspace)
        if selected:
            self.tools.set_workspace(Path(selected))
            self.mcp_manager.close()
            self.mcp_manager.workspace = self.tools.workspace
            self.file_transaction = FileTransaction(self.tools.workspace, self.state_dir / "backups")
            self.changed_files.clear()
            self.workspace_button.configure(text="📁 " + str(self.tools.workspace))

    def _refresh_conversations(self) -> None:
        if not hasattr(self, "conversation_list"):
            return
        for child in self.conversation_list.winfo_children():
            child.destroy()
        query = self.conversation_search_var.get() if hasattr(self, "conversation_search_var") else ""
        conversations = self.conversation_store.search(query)
        conversations.sort(key=lambda item: (not item.get("pinned", False), -float(item.get("updated", 0))))
        active_id = self.conversation_store.active()["id"]
        for item in conversations[:40]:
            prefix = "★ " if item.get("pinned") else ""
            ctk.CTkButton(
                self.conversation_list, text=prefix + item.get("title", self._t("conversations").rstrip("s"))[:28],
                height=30, corner_radius=15, anchor="w",
                fg_color=self.PANEL_HOVER if item["id"] == active_id else "transparent",
                hover_color=self.PANEL_HOVER, text_color=self.TEXT,
                font=ctk.CTkFont(family=THAI_FONT, size=10),
                command=lambda conversation_id=item["id"]: self._switch_conversation(conversation_id),
            ).pack(fill="x", pady=1)

    def _switch_conversation(self, conversation_id: str) -> None:
        if self.busy:
            return
        self.messages = self.conversation_store.switch(conversation_id)
        self._render_messages()
        self._refresh_conversations()

    def _pin_conversation(self) -> None:
        conversation = self.conversation_store.active()
        conversation["pinned"] = not conversation.get("pinned", False)
        self.conversation_store.save()
        self._refresh_conversations()

    def _delete_conversation(self) -> None:
        conversation = self.conversation_store.active()
        if messagebox.askyesno(self._t("delete"), f"{self._t('delete')} “{conversation.get('title', self._t('conversations').rstrip('s'))}”?"):
            self.conversation_store.delete(conversation["id"])
            self.messages = list(self.conversation_store.active()["messages"])
            self._render_messages()
            self._refresh_conversations()

    def _export_conversation(self) -> None:
        target = filedialog.asksaveasfilename(
            defaultextension=".md", filetypes=[("Markdown", "*.md"), ("JSON", "*.json")],
            initialfile="localforge-chat.md",
        )
        if not target:
            return
        path = Path(target)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.conversation_store.active(), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            self.conversation_store.export_markdown(path)

    def _request_file_approval(self, files: list[tuple[str, str]]) -> bool:
        diffs = [self.file_transaction.preview(path, content) or self._t("diff_new_file", path=path) + "\n" for path, content in files]
        request = {"files": files, "diff": "\n".join(diffs), "event": threading.Event(), "approved": False}
        self.events.put(("diff_request", request))
        while not request["event"].wait(0.1):
            if self.cancel_event.is_set():
                return False
        return bool(request["approved"])

    def _request_tool_approval(self, name: str, args: dict[str, Any]) -> bool:
        request = {
            "name": name, "args": args, "event": threading.Event(),
            "approved": False, "always": False,
        }
        self.events.put(("tool_approval", request))
        while not request["event"].wait(0.1):
            if self.cancel_event.is_set():
                return False
        if request["approved"] and request["always"]:
            match = re.match(r"mcp__([a-zA-Z0-9_-]+)__", name)
            if match:
                self.mcp_manager.update(match.group(1), permission="allow")
        return bool(request["approved"])

    def _show_tool_approval(self, request: dict[str, Any]) -> None:
        window = ctk.CTkToplevel(self)
        window.title(f"{self._t('tool_request')} — LocalForge AI")
        window.geometry("680x500")
        window.transient(self)
        ctk.CTkLabel(
            window, text=self._t("tool_request"),
            font=ctk.CTkFont(family=THAI_FONT, size=20, weight="bold"),
        ).pack(fill="x", padx=22, pady=(20, 5))
        ctk.CTkLabel(
            window, text=request["name"], anchor="w", text_color=self.ACCENT,
            font=ctk.CTkFont(family="Noto Sans Mono", size=12, weight="bold"),
        ).pack(fill="x", padx=22, pady=(0, 8))
        preview = ctk.CTkTextbox(
            window, wrap="word", font=ctk.CTkFont(family="Noto Sans Mono", size=12),
            fg_color=self.BG, border_width=1, border_color=self.BORDER,
        )
        preview.pack(fill="both", expand=True, padx=22, pady=8)
        preview.insert("1.0", json.dumps(request["args"], ensure_ascii=False, indent=2))
        preview.configure(state="disabled")
        buttons = ctk.CTkFrame(window, fg_color="transparent")
        buttons.pack(fill="x", padx=22, pady=(6, 20))

        def decide(approved: bool, always: bool = False) -> None:
            request["approved"], request["always"] = approved, always
            request["event"].set()
            window.destroy()

        ctk.CTkButton(buttons, text=self._t("deny"), command=lambda: decide(False), fg_color="#713747").pack(side="left")
        ctk.CTkButton(buttons, text=self._t("allow_once"), command=lambda: decide(True), fg_color=self.PANEL_HOVER).pack(side="right", padx=(8, 0))
        ctk.CTkButton(buttons, text=self._t("allow_always"), command=lambda: decide(True, True), fg_color=self.ACCENT).pack(side="right")
        window.protocol("WM_DELETE_WINDOW", lambda: decide(False))

    def _show_diff_review(self, request: dict[str, Any]) -> None:
        window = ctk.CTkToplevel(self)
        window.title(self._t("diff_title"))
        window.geometry("900x680")
        window.transient(self)
        ctk.CTkLabel(
            window, text=self._t("diff_count", count=len(request['files'])),
            font=ctk.CTkFont(family=THAI_FONT, size=20, weight="bold")
        ).pack(fill="x", padx=22, pady=(20, 8))
        preview = ctk.CTkTextbox(
            window, wrap="none", font=ctk.CTkFont(family="Noto Sans Mono", size=12),
            fg_color=self.BG, border_width=1, border_color=self.BORDER,
        )
        preview.pack(fill="both", expand=True, padx=22, pady=8)
        preview.insert("1.0", request["diff"] or self._t("diff_empty"))
        preview.configure(state="disabled")
        buttons = ctk.CTkFrame(window, fg_color="transparent")
        buttons.pack(fill="x", padx=22, pady=(6, 20))

        def decide(approved: bool) -> None:
            request["approved"] = approved
            request["event"].set()
            window.destroy()

        ctk.CTkButton(buttons, text=self._t("cancel"), command=lambda: decide(False), fg_color="#713747").pack(side="left")
        ctk.CTkButton(buttons, text=self._t("diff_approve"), command=lambda: decide(True), fg_color=self.ACCENT).pack(side="right")
        window.protocol("WM_DELETE_WINDOW", lambda: decide(False))
        window.grab_set()

    def _apply_generated_files(self, files: list[tuple[str, str]]) -> list[str]:
        if not self._request_file_approval(files):
            raise GenerationCancelled("ยกเลิกการเขียนไฟล์")
        results = self.file_transaction.apply(files)
        self.changed_files.update(path for path, _content in files)
        return results

    def _undo_files(self) -> None:
        try:
            restored = self.file_transaction.undo()
            self.changed_files.update(restored)
            self._append(self._t("system"), self._t("undo_done", paths=", ".join(restored)))
        except Exception as exc:
            messagebox.showwarning(self._t("undo_title"), str(exc))

    def _open_path(self, path: Path) -> None:
        if path.suffix.lower() in {".html", ".htm"}:
            webbrowser.open(path.resolve().as_uri())
            return
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _open_terminal(self) -> None:
        workspace = str(self.tools.workspace)
        candidates = [
            ("konsole", ["--workdir", workspace]),
            ("gnome-terminal", ["--working-directory", workspace]),
            ("xterm", ["-e", f"cd {workspace!r}; exec bash"]),
        ]
        for binary, arguments in candidates:
            executable = shutil.which(binary)
            if executable:
                subprocess.Popen([executable, *arguments])
                return
        messagebox.showwarning("Terminal", self._t("no_terminal"))

    def _open_project_explorer(self) -> None:
        window = ctk.CTkToplevel(self)
        window.title(self._t("pe_title"))
        window.geometry("720x650")
        window.transient(self)
        toolbar = ctk.CTkFrame(window, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(16, 8))
        tree_frame = ctk.CTkFrame(window, fg_color=self.PANEL, corner_radius=12, border_width=1, border_color=self.BORDER)
        tree_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        style = ttk.Style(window)
        style.theme_use("default")
        style.configure("Treeview",
                        background=self._resolve_color(self.PANEL), foreground=self._resolve_color(self.TEXT),
                        fieldbackground=self._resolve_color(self.PANEL), borderwidth=0,
                        font=(THAI_FONT, 12))
        style.map("Treeview", background=[("selected", self._resolve_color(self.PANEL_HOVER))], foreground=[("selected", self._resolve_color(self.TEXT))])

        tree = ttk.Treeview(tree_frame, columns=("kind",), show="tree", style="Treeview")
        scrollbar = ctk.CTkScrollbar(tree_frame, orientation="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=12)
        scrollbar.pack(side="right", fill="y", padx=(4, 12), pady=12)
        tree.tag_configure("changed", foreground="#C8BBFF")

        def refresh() -> None:
            tree.delete(*tree.get_children())
            nodes = {self.tools.workspace: ""}
            root_id = tree.insert("", "end", text=self.tools.workspace.name, open=True, values=("dir",))
            nodes[self.tools.workspace] = root_id
            items = sorted(self.tools.workspace.rglob("*"), key=lambda p: (len(p.parts), not p.is_dir(), p.name.lower()))
            for path in items[:1200]:
                try:
                    relative = str(path.relative_to(self.tools.workspace))
                except ValueError:
                    continue
                parent_id = nodes.get(path.parent, root_id)
                node = tree.insert(
                    parent_id, "end", text=("▸ " if path.is_dir() else "") + path.name,
                    values=("dir" if path.is_dir() else "file",),
                    tags=("changed",) if relative in self.changed_files else (),
                )
                nodes[path] = node
                tree.set(node, "#1", str(path))

        def selected_path() -> Path | None:
            selected = tree.selection()
            if not selected:
                return None
            raw = tree.set(selected[0], "#1")
            return Path(raw) if raw else self.tools.workspace

        def open_selected(_event: Any = None) -> None:
            path = selected_path()
            if path and path.is_file():
                self._open_path(path)

        def rename_selected() -> None:
            path = selected_path()
            if not path or path == self.tools.workspace:
                return
            name = simpledialog.askstring(self._t("pe_rename_title"), self._t("pe_rename_prompt"), initialvalue=path.name, parent=window)
            if name and Path(name).name == name:
                path.rename(path.with_name(name))
                refresh()

        def delete_selected() -> None:
            path = selected_path()
            if not path or path == self.tools.workspace:
                return
            if not messagebox.askyesno(self._t("pe_delete_title"), self._t("pe_delete_confirm", name=path.name), parent=window):
                return
            try:
                path.unlink() if path.is_file() else path.rmdir()
                refresh()
            except OSError as exc:
                messagebox.showerror(self._t("pe_delete_error_title"), self._t("pe_delete_error_text", error=exc), parent=window)

        for label, command in (
            (self._t("pe_refresh"), refresh), (self._t("pe_open"), open_selected),
            (self._t("pe_terminal"), self._open_terminal),
            (self._t("pe_open_index"), lambda: self._open_path(self.tools.workspace / "index.html")),
        ):
            ctk.CTkButton(
                toolbar, text=label, width=110, height=36, corner_radius=8,
                fg_color=self.PANEL, hover_color=self.PANEL_HOVER,
                border_width=1, border_color=self.BORDER, text_color=self.TEXT,
                font=ctk.CTkFont(size=12, weight="bold"), command=command
            ).pack(side="left", padx=4)
        context = Menu(window, tearoff=False)
        context.add_command(label=self._t("pe_open"), command=open_selected)
        context.add_command(label=self._t("pe_rename"), command=rename_selected)
        context.add_command(label=self._t("pe_delete"), command=delete_selected)
        tree.bind("<Double-1>", open_selected)
        tree.bind("<Button-3>", lambda event: (tree.selection_set(tree.identify_row(event.y)), context.tk_popup(event.x_root, event.y_root)))
        refresh()

    def _model_choices(self) -> list[str]:
        models = self.model_manager.models()
        return [path.name for path in models] or [self._t("no_models")]

    def _resolve_model_path(self, basename: str) -> Path | None:
        for p in self.model_manager.models():
            if p.name == basename:
                return p
        return None

    def _toggle_theme(self) -> None:
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.appearance_var.set(new_mode)
        self._save_preferences()

    def _header_model_selected(self, choice: str) -> None:
        self._model_selected(choice)
        self._save_preferences()

    def _rebuild_ui(self) -> None:
        self._compact_layout = None
        for child in self.winfo_children():
            child.destroy()
        try:
            self.unbind("<Configure>")
        except tk.TclError:
            pass
        self._build_ui()

    def _select_language(self, display_name: str) -> None:
        code = LANGUAGE_CODES.get(display_name, "en")
        if code == self.language_var.get():
            return
        if self.busy:
            messagebox.showwarning(self._t("language"), self._t("busy_blocked_language"))
            self.language_name_var.set(LANGUAGE_NAMES[self.language_var.get()])
            return
        self.language_var.set(code)
        self._lang_code = code
        global THAI_FONT
        THAI_FONT = LANGUAGE_FONTS[code]
        self._save_preferences()
        if self.settings_window is not None and self.settings_window.winfo_exists():
            try:
                self.settings_window.grab_release()
            except Exception:
                pass
            self.settings_window.destroy()
            self.settings_window = None
        self.after(120, self._rebuild_ui)

    def _model_selected(self, value: str | None = None) -> None:
        value = value or self.selected_model_var.get()
        try:
            model_path = self.model_manager.root / value if value else Path()
            info = inspect_model(model_path)
            text = self._t(
                "model_info_fmt",
                parameters=info.parameters, quantization=info.quantization,
                size=info.size_bytes / 1024**3, vram=info.estimated_vram_gib,
            )
        except OSError:
            text = self._t("model_info_missing")
        if hasattr(self, "model_info_label") and self.model_info_label.winfo_exists():
            self.model_info_label.configure(text=text)

    def _load_selected_model(self) -> None:
        value = self.selected_model_var.get()
        resolved = self._resolve_model_path(value) if value else None
        model_path = self.model_manager.root / resolved if resolved else Path()
        if not value or not resolved or not model_path.is_file():
            messagebox.showwarning(self._t("error"), self._t("select_model_first"))
            return
        self.model_status_var.set(self._t("loading_model"))
        self.status.configure(text=self._t("loading_model"), text_color=("#F59E0B", "#e5ad45"))

        def worker() -> None:
            try:
                self.model_manager.load(resolved, self.api_url_var.get().strip())
                profile = self.model_manager.active_profile or {}
                profile_text = self._t(
                    "profile_fmt",
                    name=profile.get('name', 'Balanced'),
                    gpu=profile.get('gpu_layers', 'auto'),
                    ctx=profile.get('context', MODEL_CONTEXT_TOKENS),
                )
                self.events.put(("model_loaded", (Path(value).name, profile_text)))
            except Exception as exc:
                self.events.put(("model_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_model(self) -> None:
        self.model_manager.stop()
        self.model_status_var.set(self._t("model_off"))
        self.status.configure(text=self._t("model_off"), text_color=self.MUTED)
        self.status_dot.configure(text_color=self.MUTED)

    def _toggle_model_server(self) -> None:
        if self.model_manager.process:
            self._stop_model()
        else:
            self._load_selected_model()

    def _download_selected_model(self) -> None:
        name = self.download_model_var.get()
        if name not in MODEL_CATALOG:
            return
        self.model_status_var.set(self._t("downloading_model", name=name))
        self.download_cancel_event = threading.Event()
        if hasattr(self, "download_progress"):
            self.download_progress.set(0)

        def worker() -> None:
            try:
                path = self.model_manager.download(
                    name,
                    progress=lambda value: self.events.put(("download_progress", value)),
                    cancel_event=self.download_cancel_event,
                )
                self.events.put(("model_downloaded", str(path)))
            except Exception as exc:
                self.events.put(("model_error", self._t("download_failed", error=exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_download(self) -> None:
        self.download_cancel_event.set()

    def _delete_selected_model(self) -> None:
        value = self.selected_model_var.get()
        resolved = self._resolve_model_path(value) if value else None
        model_path = self.model_manager.root / resolved if resolved else Path()
        if not value or not resolved or not model_path.is_file():
            return
        if not messagebox.askyesno(self._t("delete_model_title"), self._t("delete_model_confirm", name=model_path.name)):
            return
        try:
            self.model_manager.delete(model_path)
            choices = self._model_choices()
            self.selected_model_var.set(choices[0])
            if hasattr(self, "model_menu") and self.model_menu.winfo_exists():
                self.model_menu.configure(values=choices)
            if hasattr(self, "header_model_menu") and self.header_model_menu.winfo_exists():
                self.header_model_menu.configure(values=choices)
            self._model_selected(choices[0])
            self.model_status_var.set(self._t("model_deleted"))
        except Exception as exc:
            messagebox.showerror(self._t("delete_model_failed_title"), str(exc))

    def _benchmark_model(self) -> None:
        if not self.auto_router_var.get() and not self.model_manager._health(self.api_url_var.get().strip()):
            messagebox.showwarning(self._t("benchmark"), self._t("benchmark_warn"))
            return
        self.model_status_var.set(self._t("benchmark_running"))

        def worker() -> None:
            try:
                client = GemmaClient(self.api_url_var.get().strip(), "local")
                started = time.monotonic()
                answer = client.generate([{"role": "user", "content": "เขียนรายการเลข 1 ถึง 30 คั่นด้วยช่องว่างเท่านั้น"}], False)
                elapsed = time.monotonic() - started
                speed = client.last_completion_tokens / elapsed if elapsed else 0
                self.events.put(("benchmark", self._t(
                    "benchmark_done",
                    speed=f"{speed:.1f}",
                    tokens=client.last_completion_tokens,
                    elapsed=f"{elapsed:.1f}",
                )))
            except Exception as exc:
                self.events.put(("model_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_mcp_manager(self) -> None:
        window = ctk.CTkToplevel(self)
        window.title(f"{self._t('mcp_title')} — LocalForge AI")
        window.geometry("760x680")
        window.minsize(620, 540)
        window.transient(self)
        ctk.CTkLabel(
            window, text=self._t("mcp_title"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=22, weight="bold"),
        ).pack(fill="x", padx=24, pady=(22, 2))
        ctk.CTkLabel(
            window, text=self._t("audit_log", path=self.hooks.audit_path), anchor="w",
            text_color=self.MUTED, font=ctk.CTkFont(family=THAI_FONT, size=10),
        ).pack(fill="x", padx=24, pady=(0, 12))
        status_var = ctk.StringVar(value=self._t("mcp_ready"))
        status = ctk.CTkLabel(window, textvariable=status_var, anchor="w", text_color=self.MUTED)
        status.pack(fill="x", padx=24, pady=(0, 8))
        server_list = ctk.CTkScrollableFrame(window, fg_color="transparent", height=260)
        server_list.pack(fill="both", expand=True, padx=24, pady=(0, 10))

        def test_server(name: str) -> None:
            status_var.set(self._t("mcp_connecting", name=name))

            def worker() -> None:
                try:
                    count = self.mcp_manager.test(name)
                    self.events.put(("mcp_status", (status_var, self._t("mcp_test_ok", name=name, count=count))))
                except Exception as exc:
                    self.events.put(("mcp_status", (status_var, f"{name}: {exc}")))

            threading.Thread(target=worker, daemon=True).start()

        def refresh() -> None:
            for child in server_list.winfo_children():
                child.destroy()
            if not self.mcp_manager.configs:
                ctk.CTkLabel(server_list, text=self._t("no_mcp"), text_color=self.MUTED).pack(pady=20)
                return
            for config in self.mcp_manager.configs:
                row = ctk.CTkFrame(server_list, fg_color=self.PANEL, corner_radius=10)
                row.pack(fill="x", pady=4)
                info = ctk.CTkFrame(row, fg_color="transparent")
                info.pack(side="left", fill="x", expand=True, padx=12, pady=10)
                ctk.CTkLabel(info, text=config.name, anchor="w", text_color=self.TEXT,
                             font=ctk.CTkFont(size=12, weight="bold")).pack(fill="x")
                ctk.CTkLabel(info, text=" ".join(config.command), anchor="w", text_color=self.MUTED,
                             wraplength=360, font=ctk.CTkFont(family="Noto Sans Mono", size=9)).pack(fill="x")
                permission_var = ctk.StringVar(value=config.permission)
                ctk.CTkOptionMenu(
                    row, variable=permission_var, values=["ask", "allow", "deny"], width=90,
                    command=lambda value, name=config.name: self.mcp_manager.update(name, permission=value),
                ).pack(side="left", padx=4)
                enabled_var = ctk.BooleanVar(value=config.enabled)
                ctk.CTkSwitch(
                    row, text="", width=42, variable=enabled_var,
                    command=lambda name=config.name, var=enabled_var: self.mcp_manager.update(name, enabled=var.get()),
                ).pack(side="left", padx=4)
                ctk.CTkButton(row, text="Test", width=52, command=lambda name=config.name: test_server(name)).pack(side="left", padx=4)
                ctk.CTkButton(
                    row, text=self._t("delete"), width=46, fg_color="#713747",
                    command=lambda name=config.name: (self.mcp_manager.remove(name), refresh()),
                ).pack(side="left", padx=(4, 10))

        refresh()
        add_card = ctk.CTkFrame(window, fg_color=self.PANEL, corner_radius=12)
        add_card.pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(add_card, text=self._t("add_stdio"), anchor="w", text_color=self.TEXT,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(fill="x", padx=14, pady=(12, 6))
        inputs = ctk.CTkFrame(add_card, fg_color="transparent")
        inputs.pack(fill="x", padx=14)
        name_entry = ctk.CTkEntry(inputs, placeholder_text=self._t("server_name_hint"), width=170)
        name_entry.pack(side="left", padx=(0, 8))
        command_entry = ctk.CTkEntry(inputs, placeholder_text=self._t("command_hint"))
        command_entry.pack(side="left", fill="x", expand=True)

        def add_server() -> None:
            try:
                self.mcp_manager.add(name_entry.get(), command_entry.get(), "ask")
                name_entry.delete(0, "end")
                command_entry.delete(0, "end")
                status_var.set(self._t("mcp_added"))
                refresh()
            except Exception as exc:
                messagebox.showerror(self._t("mcp_add_failed"), str(exc), parent=window)

        ctk.CTkButton(add_card, text=self._t("add_server"), command=add_server).pack(fill="x", padx=14, pady=(8, 12))
        ctk.CTkLabel(
            window,
            text=self._t("mcp_warning"),
            text_color=("#B45309", "#F5C16C"), font=ctk.CTkFont(family=THAI_FONT, size=10),
        ).pack(fill="x", padx=24, pady=(0, 16))

    def _open_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.focus()
            return
        window = ctk.CTkToplevel(self)
        self.settings_window = window
        # On Wayland a Toplevel can be mapped before CustomTkinter finishes
        # laying out its children, leaving a permanently blank surface. Build
        # it while withdrawn and reveal it after idle layout has completed.
        window.withdraw()
        window.title(f"{self._t('settings_title')} — LocalForge AI")
        window.geometry("620x720")
        window.minsize(560, 620)
        window.resizable(True, True)
        window.configure(fg_color=self.BG)
        window.transient(self)

        def close_settings() -> None:
            self._save_preferences()
            try:
                window.grab_release()
            except Exception:
                pass
            self.settings_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_settings)

        content = ctk.CTkScrollableFrame(window, fg_color="transparent")
        content.pack(fill="both", expand=True)
        ctk.CTkLabel(
            content, text=self._t("settings_title"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(fill="x", padx=28, pady=(27, 4))
        ctk.CTkLabel(
            content, text=self._t("settings_subtitle"),
            anchor="w", text_color=self.MUTED, font=ctk.CTkFont(size=12)
        ).pack(fill="x", padx=28, pady=(0, 22))

        card = ctk.CTkFrame(
            content, fg_color=self.PANEL, corner_radius=14,
            border_width=1, border_color=self.BORDER
        )
        card.pack(fill="x", padx=28)
        ctk.CTkLabel(card, text=self._t("api_endpoint"), anchor="w", text_color=self.MUTED,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(fill="x", padx=16, pady=(16, 0))
        ctk.CTkEntry(
            card, textvariable=self.api_url_var, height=40, corner_radius=9,
            fg_color=self.BG, border_color=self.BORDER, text_color=self.TEXT
        ).pack(fill="x", padx=16, pady=(6, 14))
        ctk.CTkLabel(card, text=self._t("installed_models"), anchor="w", text_color=self.MUTED,
                     font=ctk.CTkFont(family=THAI_FONT, size=10, weight="bold")).pack(fill="x", padx=16)
        choices = self._model_choices()
        if self.selected_model_var.get() not in choices:
            self.selected_model_var.set(choices[0])
        self.model_menu = ctk.CTkOptionMenu(
            card, variable=self.selected_model_var, values=choices, height=40,
            fg_color=self.BG, button_color=self.ACCENT, button_hover_color=self.ACCENT_HOVER,
            text_color=self.TEXT, font=ctk.CTkFont(family=THAI_FONT, size=11),
            dropdown_font=ctk.CTkFont(family=THAI_FONT, size=11),
            command=self._model_selected,
        )
        self.model_menu.pack(fill="x", padx=16, pady=(6, 10))
        self.model_info_label = ctk.CTkLabel(
            card, text="", anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=10)
        )
        self.model_info_label.pack(fill="x", padx=16, pady=(0, 9))
        self._model_selected(choices[0])
        model_buttons = ctk.CTkFrame(card, fg_color="transparent")
        model_buttons.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(
            model_buttons, text=self._t("load_model"), height=38, command=self._load_selected_model,
            fg_color=self.ACCENT, hover_color=self.ACCENT_HOVER, text_color="#FFFFFF",
            font=ctk.CTkFont(family=THAI_FONT, size=12, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(
            model_buttons, text=self._t("stop_model"), height=38, command=self._stop_model,
            fg_color=("#EF4444", "#8D3D52"), hover_color=("#DC2626", "#A34860"), text_color="#FFFFFF",
            font=ctk.CTkFont(family=THAI_FONT, size=12, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))
        utility_buttons = ctk.CTkFrame(card, fg_color="transparent")
        utility_buttons.pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkButton(
            utility_buttons, text="Benchmark", height=34, command=self._benchmark_model,
            fg_color=self.PANEL_HOVER, hover_color=self.BORDER, text_color=self.TEXT,
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(
            utility_buttons, text=self._t("delete_model"), height=34, command=self._delete_selected_model,
            fg_color="transparent", hover_color=("#FEE2E2", "#713747"), text_color=self.TEXT,
            border_width=1, border_color=("#FECACA", "#713747"),
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ctk.CTkLabel(
            card, textvariable=self.model_status_var, anchor="w", text_color="#64D6A2",
            font=ctk.CTkFont(family=THAI_FONT, size=11)
        ).pack(fill="x", padx=16, pady=(0, 14))
        router_row = ctk.CTkFrame(card, fg_color="transparent")
        router_row.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(
            router_row, text=self._t("auto_router"), text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=11, weight="bold")
        ).pack(side="left")
        ctk.CTkSwitch(
            router_row, text="", variable=self.auto_router_var,
            progress_color=self.ACCENT, button_color="#FFFFFF"
        ).pack(side="right")
        agent_row = ctk.CTkFrame(card, fg_color="transparent")
        agent_row.pack(fill="x", padx=16, pady=(0, 15))
        agent_text = ctk.CTkFrame(agent_row, fg_color="transparent")
        agent_text.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            agent_text, text=self._t("multi_agent"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(fill="x")
        ctk.CTkLabel(
            agent_text, text="Planner → Coder → Reviewer → Fixer", anchor="w",
            text_color=self.MUTED, font=ctk.CTkFont(size=10)
        ).pack(fill="x")
        ctk.CTkSwitch(
            agent_row, text="", width=46, variable=self.multi_agent_var,
            progress_color=self.ACCENT, button_color="#FFFFFF"
        ).pack(side="right")

        # Multi-agent role model selections
        roles_frame = ctk.CTkFrame(card, fg_color="transparent")
        roles_frame.pack(fill="x", padx=16, pady=(0, 16))
        agent_choices = ["auto"] + choices

        for role, var_name, label in [
            ("planner", self.planner_model_var, self._t("planner_model")),
            ("coder", self.coder_model_var, self._t("coder_model")),
            ("reviewer", self.reviewer_model_var, self._t("reviewer_model")),
        ]:
            if var_name.get() not in agent_choices:
                var_name.set("auto")
            row = ctk.CTkFrame(roles_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(
                row, text=label, text_color=self.MUTED,
                font=ctk.CTkFont(family=THAI_FONT, size=11, weight="bold")
            ).pack(side="left")
            ctk.CTkOptionMenu(
                row, variable=var_name, values=agent_choices, height=26, width=150,
                fg_color=self.BG, button_color=self.ACCENT, button_hover_color=self.ACCENT_HOVER,
                text_color=self.TEXT, font=ctk.CTkFont(family=THAI_FONT, size=10),
                dropdown_font=ctk.CTkFont(family=THAI_FONT, size=10),
            ).pack(side="right")

        mcp_card = ctk.CTkFrame(
            content, fg_color=self.PANEL, corner_radius=14,
            border_width=1, border_color=self.BORDER,
        )
        mcp_card.pack(fill="x", padx=28, pady=(14, 0))
        ctk.CTkLabel(
            mcp_card, text=self._t("mcp_title"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=14, weight="bold"),
        ).pack(fill="x", padx=16, pady=(15, 2))
        enabled_servers = sum(config.enabled for config in self.mcp_manager.configs)
        ctk.CTkLabel(
            mcp_card,
            text=self._t("mcp_summary", enabled=enabled_servers, total=len(self.mcp_manager.configs)),
            anchor="w", text_color=self.MUTED, font=ctk.CTkFont(family=THAI_FONT, size=10),
        ).pack(fill="x", padx=16, pady=(0, 9))
        ctk.CTkButton(
            mcp_card, text=self._t("mcp_manage"), height=36,
            command=self._open_mcp_manager, fg_color=self.PANEL_HOVER,
            hover_color=self.BORDER, text_color=self.TEXT,
        ).pack(fill="x", padx=16, pady=(0, 14))
        download_card = ctk.CTkFrame(
            content, fg_color=self.PANEL, corner_radius=14,
            border_width=1, border_color=self.BORDER
        )
        download_card.pack(fill="x", padx=28, pady=(14, 0))
        ctk.CTkLabel(
            download_card, text=self._t("download_models"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=14, weight="bold")
        ).pack(fill="x", padx=16, pady=(15, 2))
        ctk.CTkLabel(
            download_card, text=self._t("models_location", path=self.model_manager.root),
            anchor="w", text_color=self.MUTED, font=ctk.CTkFont(family=THAI_FONT, size=10)
        ).pack(fill="x", padx=16, pady=(0, 9))
        ctk.CTkOptionMenu(
            download_card, variable=self.download_model_var, values=list(MODEL_CATALOG),
            height=40, fg_color=self.BG, button_color=self.ACCENT,
            font=ctk.CTkFont(family=THAI_FONT, size=11),
            dropdown_font=ctk.CTkFont(family=THAI_FONT, size=11)
        ).pack(fill="x", padx=16, pady=(0, 9))
        ctk.CTkButton(
            download_card, text=self._t("download"), height=38, command=self._download_selected_model,
            fg_color=self.ACCENT, hover_color=self.ACCENT_HOVER, text_color="#FFFFFF",
            font=ctk.CTkFont(family=THAI_FONT, size=12, weight="bold")
        ).pack(fill="x", padx=16, pady=(0, 16))
        self.download_progress = ctk.CTkProgressBar(
            download_card, progress_color=self.ACCENT, fg_color=self.BG
        )
        self.download_progress.set(0)
        self.download_progress.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(
            download_card, text=self._t("cancel_download"), height=32,
            command=self._cancel_download, fg_color="transparent", text_color=self.TEXT,
            border_width=1, border_color=self.BORDER,
        ).pack(fill="x", padx=16, pady=(0, 16))

        display_card = ctk.CTkFrame(
            content, fg_color=self.PANEL, corner_radius=14,
            border_width=1, border_color=self.BORDER,
        )
        display_card.pack(fill="x", padx=28, pady=(14, 0))
        ctk.CTkLabel(
            display_card, text=self._t("display"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=14, weight="bold"),
        ).pack(fill="x", padx=16, pady=(15, 8))
        language_row = ctk.CTkFrame(display_card, fg_color="transparent")
        language_row.pack(fill="x", padx=16, pady=(0, 12))
        language_text = ctk.CTkFrame(language_row, fg_color="transparent")
        language_text.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            language_text, text=self._t("language"), anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=11, weight="bold"),
        ).pack(fill="x")
        ctk.CTkLabel(
            language_text, text=self._t("restart_language"), anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=9),
        ).pack(fill="x")
        ctk.CTkOptionMenu(
            language_row, variable=self.language_name_var,
            values=list(LANGUAGE_CODES), command=self._select_language,
            width=135, font=ctk.CTkFont(family=THAI_FONT, size=11),
            dropdown_font=ctk.CTkFont(family=THAI_FONT, size=11),
        ).pack(side="right", padx=(10, 0))
        ctk.CTkSegmentedButton(
            display_card, values=["Dark", "Light", "System"],
            variable=self.appearance_var, command=self._apply_display_settings,
            fg_color=self.BG, selected_color=self.ACCENT, selected_hover_color=self.ACCENT_HOVER,
            unselected_color=self.BG, unselected_hover_color=self.PANEL_HOVER,
            text_color=self.TEXT,
        ).pack(fill="x", padx=16, pady=(0, 12))
        scale_header = ctk.CTkFrame(display_card, fg_color="transparent")
        scale_header.pack(fill="x", padx=16)
        ctk.CTkLabel(
            scale_header, text=self._t("ui_size"), anchor="w", text_color=self.MUTED,
            font=ctk.CTkFont(family=THAI_FONT, size=10),
        ).pack(side="left")
        ctk.CTkLabel(
            scale_header, textvariable=self.ui_scale_label_var, anchor="e",
            text_color=self.TEXT, font=ctk.CTkFont(family=THAI_FONT, size=10, weight="bold"),
        ).pack(side="right")
        ctk.CTkSlider(
            display_card, from_=0.85, to=1.25, number_of_steps=8,
            variable=self.font_scale_var, command=self._apply_display_settings,
        ).pack(fill="x", padx=16, pady=(6, 16))

        ctk.CTkButton(
            content, text=self._t("done"), height=40, corner_radius=11,
            fg_color=self.ACCENT, hover_color=self.ACCENT_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"), command=close_settings
        ).pack(fill="x", padx=28, pady=20)

        def show_settings() -> None:
            if not window.winfo_exists():
                return
            window.update_idletasks()
            width, height = 620, min(720, max(620, self.winfo_height() - 30))
            x = self.winfo_rootx() + max(0, (self.winfo_width() - width) // 2)
            y = self.winfo_rooty() + max(0, (self.winfo_height() - height) // 2)
            window.geometry(f"{width}x{height}+{x}+{y}")
            window.deiconify()
            window.lift()
            window.focus_force()
            window.grab_set()

        window.after(80, show_settings)

    def _clear(self) -> None:
        self.conversation_store.create()
        self.messages = []
        self._save_history()
        self._render_messages()
        self._refresh_conversations()
        self._append(self._t("system"), self._t("ready"))

    def _send_event(self, _event: Any) -> str:
        self.send()
        return "break"

    def _update_media_status(self) -> None:
        names = [path.name for path, _part in self.pending_media]
        self.media_status.configure(text=" • ".join(names[-2:]))

    def _attach_media_path(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(self._t("file_not_found", path=path))
        if path.stat().st_size > 25 * 1024 * 1024:
            raise RuntimeError(self._t("media_too_big"))
        part = media_content(path)
        self.pending_media.append((path, part))
        self._update_media_status()

    def _index_document(self) -> None:
        if not self.model_manager.active_model:
            messagebox.showwarning(self._t("rag_manage"), self._t("rag_warn"))
            return

        value = filedialog.askopenfilename(
            parent=self, title=self._t("rag_pick"),
            filetypes=[("Text files", "*.txt *.md *.csv *.json"), ("All files", "*")],
        )
        if not value:
            return

        try:
            content = Path(value).read_text(encoding="utf-8")
            # Simple chunking
            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
            self.status.configure(text=self._t("rag_indexing", count=len(chunks)), text_color="#34D399")

            def worker():
                for i, chunk in enumerate(chunks):
                    emb = get_embedding(self.api_url_var.get().strip(), chunk)
                    if emb:
                        self.vector_db.add_rag_chunk(Path(value).name, chunk, emb)
                self.events.put(("tool", self._t("rag_indexed", name=Path(value).name)))

            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            messagebox.showerror(self._t("error"), str(exc))

    def _open_rag_manager(self) -> None:
        window = ctk.CTkToplevel(self)
        window.title(self._t("rag_title"))
        window.geometry("560x360")
        window.transient(self)
        rag, cache = self.vector_db.counts()
        stats_var = ctk.StringVar(value=self._t("rag_stats", chunks=rag, cache=cache))
        ctk.CTkLabel(
            window, textvariable=stats_var, anchor="w", text_color=self.TEXT,
            font=ctk.CTkFont(family=THAI_FONT, size=18, weight="bold"),
        ).pack(fill="x", padx=28, pady=(28, 18))

        def refresh() -> None:
            rag, cache = self.vector_db.counts()
            stats_var.set(self._t("rag_stats", chunks=rag, cache=cache))

        def clear_rag() -> None:
            if not messagebox.askyesno(self._t("rag_clear_chunks"), self._t("rag_confirm"), parent=window):
                return
            self.vector_db.clear_rag()
            self.status.configure(text=self._t("rag_cleared"), text_color="#10B981")
            refresh()

        def clear_cache() -> None:
            if not messagebox.askyesno(self._t("rag_clear_cache"), self._t("rag_confirm"), parent=window):
                return
            self.vector_db.clear_cache()
            self.status.configure(text=self._t("rag_cleared"), text_color="#10B981")
            refresh()

        ctk.CTkButton(
            window, text=self._t("rag_clear_chunks"), height=40, corner_radius=10,
            fg_color="#713747", hover_color="#8a4458",
            font=ctk.CTkFont(size=13, weight="bold"), command=clear_rag,
        ).pack(fill="x", padx=28, pady=(0, 10))
        ctk.CTkButton(
            window, text=self._t("rag_clear_cache"), height=40, corner_radius=10,
            fg_color="#713747", hover_color="#8a4458",
            font=ctk.CTkFont(size=13, weight="bold"), command=clear_cache,
        ).pack(fill="x", padx=28, pady=(0, 10))
        ctk.CTkButton(
            window, text=self._t("done"), height=40, corner_radius=10,
            fg_color=self.ACCENT, hover_color=self.ACCENT_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"), command=window.destroy,
        ).pack(fill="x", padx=28, pady=(4, 22))

    def _choose_image(self) -> None:
        zenity = shutil.which("zenity")
        value = None
        if zenity:
            result = subprocess.run(
                [zenity, "--file-selection", "--title", self._t("attach_image"),
                 "--file-filter", "Images | *.png *.jpg *.jpeg *.webp *.gif",
                 "--file-filter", "All files | *"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                value = result.stdout.strip()
        else:
            value = filedialog.askopenfilename(
                parent=self, title=self._t("attach_image"),
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.gif"), ("All files", "*")],
            )
        if value:
            try:
                self._attach_media_path(Path(value))
            except Exception as exc:
                messagebox.showerror(self._t("error"), str(exc))

    def _paste_image_event(self, _event: Any = None) -> str:
        wl_paste = shutil.which("wl-paste")
        if not wl_paste:
            messagebox.showwarning(self._t("paste"), self._t("paste_missing_tool"))
            return "break"
        try:
            types = subprocess.run(
                [wl_paste, "--list-types"], capture_output=True, text=True, timeout=3, check=True,
            ).stdout.splitlines()
            mime = next((value for value in types if value in {"image/png", "image/jpeg", "image/webp"}), None)
            if not mime:
                raise RuntimeError(self._t("clipboard_no_image"))
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime]
            self.state_dir.mkdir(parents=True, exist_ok=True)
            path = self.state_dir / f"clipboard-{int(time.time() * 1000)}{suffix}"
            image_data = subprocess.run(
                [wl_paste, "--no-newline", "--type", mime],
                capture_output=True, timeout=8, check=True,
            ).stdout
            path.write_bytes(image_data)
            self._attach_media_path(path)
        except Exception as exc:
            messagebox.showerror(self._t("paste"), str(exc))
        return "break"

    def _toggle_recording(self) -> None:
        if self.recording_process and self.recording_process.poll() is None:
            self.recording_process.terminate()
            try:
                self.recording_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.recording_process.kill()
            self.recording_process = None
            self.voice_button.configure(text=self._t("voice"), fg_color="transparent")
            if self.recording_path and self.recording_path.is_file() and self.recording_path.stat().st_size > 44:
                try:
                    self._attach_media_path(self.recording_path)
                except Exception as exc:
                    messagebox.showerror(self._t("error"), str(exc))
            return
        recorder = shutil.which("pw-record")
        if not recorder:
            messagebox.showerror(self._t("voice"), self._t("voice_recorder_missing"))
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.recording_path = self.state_dir / f"voice-{int(time.time() * 1000)}.wav"
        self.recording_process = subprocess.Popen(
            [recorder, "--format=s16", "--rate=16000", "--channels=1", str(self.recording_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.voice_button.configure(text=self._t("stop_recording"), fg_color=("#EF4444", "#8D3D52"))
        self.media_status.configure(text=self._t("recording"))

    def _speak_text(self, text: str) -> None:
        if self.speech_process and self.speech_process.poll() is None:
            self.speech_process.terminate()
            self.speech_process = None
            return
        speaker = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
        if not speaker:
            messagebox.showerror(self._t("speak"), self._t("speak_missing"))
            return
        clean = re.sub(r"```.*?```", " ", text, flags=re.S)
        clean = re.sub(r"[*_#`]+", "", clean).strip()[:4000]
        args = [speaker, "-l", self.language_var.get(), clean] if Path(speaker).name == "spd-say" else [speaker, "-v", self.language_var.get(), clean]
        self.speech_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def send(self, ignore_cache: bool = False) -> None:
        text = self.input.get("1.0", "end").strip()
        if self.busy or (not text and not self.pending_media):
            return
        media = [part for _path, part in self.pending_media]
        if not text and media:
            text = self._t("transcribe_audio") if any(part["type"] == "input_audio" for part in media) else self._t("describe_image")
        if context_report(self.messages, MODEL_CONTEXT_TOKENS)["percent"] >= 85:
            if not self._summarize_context():
                self._pending_send = ignore_cache
                return
        if not self.auto_router_var.get() and not self.model_manager._health(self.api_url_var.get().strip()):
            messagebox.showwarning(self._t("model_not_loaded_title"), self._t("model_not_loaded_text"))
            return
        self.input.delete("1.0", "end")
        media_note = "" if not media else self._t("media_attached", count=len(media))
        media_paths = [str(path) for path, _part in self.pending_media]
        self._append(self._t("you"), text + media_note, media_paths=media_paths)
        self.messages.append({"role": "user", "content": text, "media_paths": media_paths})
        self.pending_media = []
        self._update_media_status()
        self._save_history()
        self.busy = True
        self.cancel_event = threading.Event()
        self.request_started = time.monotonic()
        self.stream_widgets = None
        self.stream_buffer = ""
        self.send_button.configure(state="disabled")
        self.send_button.grid_remove()
        self.stop_button.grid(row=0, column=1, padx=(0, 12), pady=12)
        self.status.configure(text=self._t("thinking_large"), text_color=("#F59E0B", "#e5ad45"))
        api_url = self.api_url_var.get().strip()
        model = self.model_var.get().strip()
        multi_agent = bool(self.multi_agent_var.get())
        auto_router = bool(self.auto_router_var.get())
        threading.Thread(
            target=self._agent_loop, args=(api_url, model, multi_agent, auto_router, media, ignore_cache), daemon=True
        ).start()

    def _save_project_state(
        self, request_text: str, manifest: list[str], status: dict[str, str]
    ) -> None:
        base = Path(manifest[0]).parent if manifest else Path(".")
        state_path = str(base / "project-state.json")
        self.tools.write_file(
            state_path,
            json.dumps(
                {"request": request_text, "files": manifest, "status": status},
                ensure_ascii=False, indent=2,
            ),
        )

    def _workspace_relative_plan(
        self, plan: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Avoid workspace/project/project when the workspace is already the project."""
        workspace_name = self.tools.workspace.name
        normalized = []
        for path, purpose in plan:
            parts = Path(path).parts
            if len(parts) > 1 and parts[0].lower() == workspace_name.lower():
                path = str(Path(*parts[1:]))
            normalized.append((path, purpose))
        return normalized

    @staticmethod
    def _static_issues(files: dict[str, str]) -> list[tuple[str, str]]:
        issues = []
        known = {str(Path(path)) for path in files}
        for path, content in files.items():
            if not path.lower().endswith((".html", ".htm")):
                continue
            for ref in re.findall(r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", content, re.I):
                if ref.startswith(("http://", "https://", "#", "data:")):
                    continue
                target = str(Path(path).parent / ref)
                if target not in known:
                    issues.append((path, f"อ้างถึงไฟล์ {ref} แต่ไม่มีใน manifest"))
        return issues

    def _get_role_client(self, role: str, api_url: str, default_client: GemmaClient) -> GemmaClient:
        role_var = getattr(self, f"{role}_model_var", None)
        model_name = role_var.get() if role_var else "auto"
        if model_name == "auto" or not model_name:
            return default_client
        model_info = next((m for m in self.model_manager.models() if m.name == model_name), None)
        if not model_info:
            return default_client
        if self.model_manager.active_model != model_info:
            self.events.put(("tool", self._t("switching_role_model", role=role.capitalize(), name=model_info.name)))
            self.model_manager.load(model_info, api_url)
            self.events.put(("router_model", str(model_info)))
        return GemmaClient(api_url, model_info.name, default_client.tool_schemas)

    def _run_multi_agent(self, api_url: str, default_client: GemmaClient, request_context: str) -> str:
        planner_client = self._get_role_client("planner", api_url, default_client)
        self.events.put(("tool", self._t("planner_working")))
        plan_text = planner_client.plan_project(request_context)
        plan = parse_plan(plan_text)
        explicit = list(dict.fromkeys(re.findall(
            r"(?<![\w.-])([\w.-]+(?:/[\w.-]+)*\.(?:html?|css|js|json|md|txt|py|svg))(?![\w.-])",
            request_context, re.I,
        )))
        if not plan:
            plan = [(path, "ไฟล์ที่ผู้ใช้ระบุ") for path in explicit]
        if not plan:
            plan = [("generated/index.html", "ผลงานหลัก")]
        # If the planner drops a project directory mentioned by the user,
        # apply it consistently to all sibling paths.
        base_context = request_context
        plan = apply_requested_base_dir(plan, base_context)
        plan = [item for item in plan if Path(item[0]).suffix.lower() in TEXT_PROJECT_EXTENSIONS][:8]
        plan = self._workspace_relative_plan(plan)
        if not plan:
            raise RuntimeError("Planner ไม่ได้เสนอไฟล์ข้อความที่รองรับ")
        manifest = [path for path, _ in plan]
        status = {path: "pending" for path in manifest}
        self._save_project_state(request_context, manifest, status)

        coder_client = self._get_role_client("coder", api_url, default_client)
        contents: dict[str, str] = {}
        for index, (path, purpose) in enumerate(plan, 1):
            self.events.put(("tool", self._t("coder_working", index=index, total=len(plan), path=path)))
            current = ""
            try:
                current = self.tools.read_file(path)
            except ToolError:
                pass
            blocks: list[tuple[str, str]] = []
            for attempt in range(2):
                output = coder_client.code_project_file(
                    request_context, path, purpose, manifest, current=current,
                    issue=(
                        "ครั้งก่อนรูปแบบคำตอบไม่ถูกต้อง ส่งเนื้อหาไฟล์เต็มใน <file> เพียงบล็อกเดียว"
                        if attempt else ""
                    ),
                )
                blocks = extract_generated_files(output, path)
                if blocks:
                    break
            if not blocks:
                status[path] = "failed: invalid file block after 2 attempts"
                self._save_project_state(request_context, manifest, status)
                continue
            content = blocks[0][1]
            contents[path] = content
            status[path] = "created"
            self._save_project_state(request_context, manifest, status)

        if not contents:
            raise RuntimeError("Coder ไม่สามารถสร้างไฟล์ที่สมบูรณ์ได้")

        for review_round in range(2):
            self.events.put(("tool", self._t("reviewer_working", round=review_round + 1)))
            reviewer_client = self._get_role_client("reviewer", api_url, default_client)
            files_text = "\n\n".join(
                f"--- {path} ---\n{content[:5000]}" for path, content in contents.items()
            )
            review = reviewer_client.review_project(request_context, files_text)
            issues = self._static_issues(contents) + parse_review(review)
            # Deduplicate identical reviewer/static findings.
            issues = list(dict.fromkeys(issues))
            if not issues:
                for path in contents:
                    status[path] = "verified"
                break
            # Load coder again if fixes are needed
            coder_client = self._get_role_client("coder", api_url, default_client)
            for issue_index, (path, issue) in enumerate(issues, 1):
                if path not in contents:
                    continue
                self.events.put(("tool", self._t("fixer_working", index=issue_index, total=len(issues), path=path)))
                output = coder_client.code_project_file(
                    request_context, path, "แก้ตาม Reviewer", manifest,
                    current=contents[path], issue=issue,
                )
                blocks = extract_generated_files(output, path)
                if blocks:
                    contents[path] = blocks[0][1]
                    status[path] = f"fixed round {review_round + 1}"
            self._save_project_state(request_context, manifest, status)

        self._apply_generated_files(list(contents.items()))
        self._save_project_state(request_context, manifest, status)
        lines = [
            f"{'✗' if status[path].startswith('failed') else '✓'} {path} — {status[path]}"
            for path in manifest
        ]

        # Restore the original model if it was changed
        if self.model_manager.active_model and self.model_manager.active_model.name != default_client.model:
            original_model_info = next((m for m in self.model_manager.models() if m.name == default_client.model), None)
            if original_model_info:
                self.events.put(("tool", self._t("restoring_main_model", name=original_model_info.name)))
                self.model_manager.load(original_model_info, api_url)
                self.events.put(("router_model", str(original_model_info)))

        return self._t("multi_agent_done") + "\n".join(lines)

    def _agent_loop(
        self, api_url: str, model: str, multi_agent: bool, auto_router: bool = False,
        media: list[dict[str, Any]] | None = None, ignore_cache: bool = False,
    ) -> None:
        try:
            original_request = next(
                (m["content"] for m in reversed(self.messages) if m["role"] == "user"), ""
            )
            if auto_router:
                desired = choose_model(
                    original_request, self.model_manager.models(), self.model_manager.active_model
                )
                if desired and desired != self.model_manager.active_model:
                    self.events.put(("tool", self._t("router_loaded", name=desired.name)))
                    self.model_manager.load(desired, api_url)
                    self.events.put(("router_model", str(desired)))

            # --- SEMANTIC CACHING (Check) ---
            try:
                if self.vector_db and not media and not ignore_cache:
                    query_embedding = get_embedding(api_url, original_request)
                    if query_embedding:
                        cached_answer = self.vector_db.search_cache(query_embedding)
                        if cached_answer:
                            self.events.put(("tool", self._t("cache_hit")))
                            self.events.put(("stream_begin", ""))
                            # Stream simulated chunks for UI UX
                            chunk_size = 50
                            for i in range(0, len(cached_answer), chunk_size):
                                self.events.put(("stream_delta", cached_answer[i:i+chunk_size]))
                                time.sleep(0.01)
                            self.messages.append({"role": "assistant", "content": cached_answer})
                            self._save_history()
                            self.events.put(("answer", (cached_answer, 0, 0)))
                            return
            except Exception as e:
                print(f"Cache check error: {e}")

            # --- RAG: Vector Search ---
            try:
                if self.vector_db and not media and 'query_embedding' in locals() and query_embedding:
                    rag_results = self.vector_db.search_rag(query_embedding)
                    if rag_results:
                        self.events.put(("tool", self._t("rag_found", count=len(rag_results))))
                        context_str = "\n\n".join([f"[{r['source']}]\n{r['content']}" for r in rag_results])
                        augmented_prompt = self._t("rag_prompt", context=context_str, question=original_request)
                        # Replace the last user message in the session with the augmented prompt
                        for i in range(len(self.messages)-1, -1, -1):
                            if self.messages[i]["role"] == "user":
                                self.messages[i]["content"] = augmented_prompt
                                break
            except Exception as e:
                print(f"RAG search error: {e}")

            discovered_mcp = self.mcp_manager.discover()
            selected_mcp = select_tools(discovered_mcp, original_request)
            builtin_needed = needs_tools(original_request)
            tool_schemas = (OPENAI_TOOLS if builtin_needed else []) + [
                openai_tool_schema(tool) for tool in selected_mcp
            ]
            client = GemmaClient(api_url, model, tool_schemas)
            if requests_action(original_request) and not selected_mcp:
                recent_user_messages = select_recent_messages(
                    [m for m in self.messages if m["role"] == "user"], 3500
                )
                request_context = "\n".join(
                    str(message.get("content", "")) for message in recent_user_messages
                )
                if multi_agent:
                    media_paths = next(
                        (m.get("media_paths") for m in reversed(self.messages) if m["role"] == "user"),
                        None,
                    )
                    if media and media_paths:
                        request_context += "\n\n" + self._t("media_note_in_multiagent", paths=", ".join(media_paths))
                    final = self._run_multi_agent(api_url, client, request_context)
                    self.messages.append({"role": "assistant", "content": final})
                    self._save_history()
                    self.events.put(("answer", (final, client.last_completion_tokens, client.last_prompt_tokens)))
                    return
                self.events.put(("tool", self._t("generating_files")))
                referenced = re.search(
                    r"(?<![\w.-])([\w.-]+(?:/[\w.-]+)*\.(?:html?|css|js|json|md|txt|py|svg))(?![\w.-])",
                    request_context,
                    re.I,
                )
                if referenced:
                    try:
                        current = self.tools.read_file(referenced.group(1))
                        if len(current) <= MAX_TOOL_RESULT_CHARS:
                            request_context += (
                                f"\n\nเนื้อหาไฟล์ {referenced.group(1)} ปัจจุบันสำหรับใช้อ้างอิง:\n{current}"
                            )
                    except ToolError:
                        pass
                self.events.put(("stream_begin", ""))
                generated = client.generate_file(
                    request_context,
                    on_token=lambda piece: self.events.put(("stream_delta", piece)),
                    cancel_event=self.cancel_event,
                )
                files = extract_generated_files(generated, request_context)
                files = apply_requested_base_dir(files, request_context)
                if not files:
                    raise RuntimeError("โมเดลไม่ได้ส่งบล็อกไฟล์ที่สมบูรณ์ กรุณาลองสั่งใหม่ให้สั้นลง")
                results = self._apply_generated_files(files)
                paths = ", ".join(path for path, _ in files)
                final = self._t("created_files", paths=paths) + "\n".join(results)
                self.messages.append({"role": "assistant", "content": final})
                self._save_history()
                self.events.put(("answer", (final, client.last_completion_tokens, client.last_prompt_tokens)))
                return
            action_nudge_used = False
            enable_tools = bool(tool_schemas)
            for _ in range(MAX_TOOL_ROUNDS):
                recent_context = with_media(self._recent_context(self.messages), media or [])
                self.hooks.before_model(
                    len(recent_context),
                    sum(estimate_tokens(str(item.get("content", ""))) for item in recent_context),
                )
                self.events.put(("stream_begin", ""))
                answer = client.generate(
                    recent_context, enable_tools,
                    on_token=lambda piece: self.events.put(("stream_delta", piece)),
                    cancel_event=self.cancel_event,
                )
                call = parse_tool_call(answer)
                if not call:
                    if looks_like_broken_tool_call(answer):
                        self.events.put(("tool", self._t("fixing_tool_json")))
                        # Never retain the large, truncated JSON blob. Keeping
                        # it caused the retry prompt itself to exceed 8K.
                        self.messages.append({
                            "role": "user",
                            "content": (
                                "SYSTEM TOOL ERROR: JSON ของ tool-call ไม่ถูกต้อง ห้ามอธิบาย "
                                "ให้เรียกเครื่องมือเดิมใหม่ทันทีด้วย JSON ที่ valid และกระชับเท่านั้น "
                                "ตรวจ escape เครื่องหมายคำพูดและปิดวงเล็บให้ครบ"
                            ),
                        })
                        continue
                    if requests_action(original_request) and not action_nudge_used:
                        action_nudge_used = True
                        self.events.put(("tool", self._t("nudging_action")))
                        self.messages.append({"role": "assistant", "content": answer})
                        self.messages.append({
                            "role": "user",
                            "content": (
                                "ผู้ใช้อนุญาตให้คุณเลือกรายละเอียดทั้งหมดเองแล้ว ห้ามถามกลับหรือเสนอแผน "
                                "ให้ลงมือเรียก write_file เพื่อสร้างผลงานที่ใช้งานได้จริงทันที"
                            ),
                        })
                        continue
                    self.messages.append({"role": "assistant", "content": answer})
                    self._save_history()
                    # --- SEMANTIC CACHING (Save) ---
                    try:
                        if self.vector_db and not media:
                            # Re-compute embedding or store it during Check phase. Re-computing is easier here.
                            emb = get_embedding(api_url, original_request)
                            if emb:
                                self.vector_db.add_to_cache(original_request, emb, answer)
                    except Exception as e:
                        print(f"Cache save error: {e}")

                    self.events.put(("answer", (answer, client.last_completion_tokens, client.last_prompt_tokens)))
                    return
                name, args = call
                self.events.put(("tool", self._t("using_tool", name=name, args=json.dumps(args, ensure_ascii=False))))
                source = "mcp" if name.startswith("mcp__") else "builtin"
                started = time.monotonic()
                try:
                    if source == "mcp":
                        server_name = name.split("__", 2)[1]
                        permission = self.mcp_manager.get_config(server_name).permission
                        decision = self.hooks.before_tool(name, args, source, permission)
                        if not decision.allowed:
                            result = f"ERROR: {decision.reason}"
                        elif decision.require_approval and not self._request_tool_approval(name, args):
                            result = self._t("tool_denied_mcp")
                        else:
                            result = self.mcp_manager.call(name, args)
                    elif name == "write_file":
                        self.hooks.before_tool(name, args, source, "allow")
                        path = str(args.get("path", ""))
                        content = str(args.get("content", ""))
                        result = "\n".join(self._apply_generated_files([(path, content)]))
                    else:
                        self.hooks.before_tool(name, args, source, "allow")
                        result = self.tools.execute(name, args)
                    result = self.hooks.after_tool(
                        name, str(result), source, time.monotonic() - started
                    )
                except (ToolError, MCPError) as exc:
                    self.hooks.on_error(f"tool:{name}", exc)
                    result = f"ERROR: {exc}"
                if name == "write_file" and not result.startswith("ERROR:"):
                    path = str(args.get("path", "ไฟล์"))
                    final = self._t("created_file", path=path, result=result)
                    # The tool call can contain thousands of tokens of source
                    # code. Do not send it back merely to ask for a summary;
                    # that can overflow the model context after a successful
                    # write. The application can report success deterministically.
                    self.messages.append({"role": "assistant", "content": final})
                    self._save_history()
                    self.events.put(("answer", (final, client.last_completion_tokens, client.last_prompt_tokens)))
                    return
                model_result = result
                if len(model_result) > MAX_TOOL_RESULT_CHARS:
                    model_result = model_result[:MAX_TOOL_RESULT_CHARS] + "\n...[ตัดผลลัพธ์เพื่อไม่ให้เกิน context]"
                if client.last_native_tool_call and client.last_native_message:
                    self.messages.append(client.last_native_message)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": client.last_native_tool_call.get("id", "call_local"),
                        "name": name,
                        "content": f"SUCCESS\n{model_result}" if not result.startswith("ERROR:") else model_result,
                    })
                else:
                    self.messages.append({"role": "assistant", "content": answer})
                    self.messages.append({
                        "role": "user",
                        "content": self._t("tool_result", name=name, result=model_result),
                    })
            raise RuntimeError(self._t("tool_limit_reached"))
        except GenerationCancelled:
            self.events.put(("cancelled", self._t("stopped_by_user")))
        except Exception as exc:  # surfaced in the GUI, including connectivity errors
            process = self.model_manager.process
            if isinstance(exc, urllib.error.URLError) and process and process.poll() is not None:
                self.model_manager.stop()
                self.events.put((
                    "model_crashed",
                    self._t("model_crashed_hint"),
                ))
            else:
                self.events.put(("error", str(exc)))
        finally:
            self.events.put(("done", ""))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, text = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "answer":
                if isinstance(text, tuple) and len(text) == 3:
                    answer, tokens, prompt_tokens = text
                elif isinstance(text, tuple):
                    answer, tokens = text
                    prompt_tokens = 0
                else:
                    answer, tokens, prompt_tokens = str(text), None, 0
                metrics = {
                    "elapsed": time.monotonic() - self.request_started,
                    "model": Path(self.model_manager.active_model).name if self.model_manager.active_model else "local",
                    "prompt_tokens": prompt_tokens,
                }
                self._finish_stream(answer, tokens or 0, metrics, len(self.messages) - 1)
            elif kind == "stream_begin":
                self._begin_stream()
            elif kind == "stream_delta":
                self._update_stream(str(text))
            elif kind == "history_saved":
                self._refresh_conversations()
                self._update_context_meter()
            elif kind == "diff_request":
                self._show_diff_review(text)
            elif kind == "tool_approval":
                self._show_tool_approval(text)
            elif kind == "mcp_status":
                status_var, value = text
                try:
                    status_var.set(value)
                except Exception:
                    pass
            elif kind == "tool":
                self.status.configure(text=text[:38] + ("…" if len(text) > 38 else ""))
            elif kind == "error":
                self._append(self._t("error"), text)
                messagebox.showerror(self._t("error"), text)
            elif kind == "model_crashed":
                self.model_status_var.set(self._t("model_crashed_status"))
                self.status.configure(text=self._t("model_crashed_short"), text_color=("#EF4444", "#FF9EAE"))
                self.status_dot.configure(text_color="#EF4444")
                self._append(self._t("error"), text)
                messagebox.showerror(self._t("error"), text)
            elif kind == "model_loaded":
                model_name, profile_text = text
                self.model_status_var.set(self._t("using_model", name=model_name, profile=profile_text))
                self.status.configure(text=self._t("model_ready"), text_color=("#10B981", "#63c174"))
                self.status_dot.configure(text_color="#43D19E")
            elif kind == "model_downloaded":
                self.model_status_var.set(self._t("downloaded_done", name=Path(text).name))
                choices = self._model_choices()
                self.selected_model_var.set(Path(text).name)
                if hasattr(self, "model_menu") and self.model_menu.winfo_exists():
                    self.model_menu.configure(values=choices)
                if hasattr(self, "header_model_menu") and self.header_model_menu.winfo_exists():
                    self.header_model_menu.configure(values=choices)
                if hasattr(self, "download_progress"):
                    self.download_progress.set(1)
            elif kind == "download_progress":
                if hasattr(self, "download_progress") and self.download_progress.winfo_exists():
                    self.download_progress.set(float(text) / 100)
                self.model_status_var.set(self._t("downloading_progress", percent=int(text)))
            elif kind == "benchmark":
                self.model_status_var.set(f"Benchmark: {text}")
            elif kind == "router_model":
                self.selected_model_var.set(Path(text).name)
                self.model_status_var.set(self._t("router_chose", name=Path(text).name))
            elif kind == "model_error":
                self.model_status_var.set(str(text))
                self.status.configure(text=self._t("model_error_short"), text_color=("#EF4444", "#FF9EAE"))
                messagebox.showerror(self._t("model_manage_failed"), str(text))
            elif kind == "cancelled":
                if self.stream_widgets:
                    self._finish_stream(self.stream_buffer + self._t("stopped_by_user"), estimate_tokens(self.stream_buffer), {
                        "elapsed": time.monotonic() - self.request_started,
                        "model": Path(self.model_manager.active_model).name if self.model_manager.active_model else "local",
                    })
                self.status.configure(text=self._t("stopped"), text_color=self.MUTED)
            elif kind == "done":
                self.busy = False
                self.send_button.configure(state="normal")
                self.stop_button.grid_remove()
                self.send_button.grid(row=0, column=1, padx=(0, 12), pady=12)
                self.status.configure(text=self._t("idle"), text_color=("#10B981", "#63c174"))
                self._notify_finished()
        self.after(80, self._poll_events)


if __name__ == "__main__":
    ChatApp().mainloop()
