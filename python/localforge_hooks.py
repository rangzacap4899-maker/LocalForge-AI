"""Policy hooks and audit logging around LocalForge model/tool activity."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)([\"']?(?:api[_-]?key|token|secret|password)[\"']?\s*[=:]\s*[\"']?)"
        r"[^\"'\s,;}]+"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
)


@dataclass
class HookDecision:
    allowed: bool = True
    require_approval: bool = False
    reason: str = ""


class HookEngine:
    """Built-in lifecycle hooks with conservative defaults for external tools."""

    def __init__(self, audit_path: Path, max_result_chars: int = 12_000) -> None:
        self.audit_path = audit_path
        self.max_result_chars = max_result_chars
        self._lock = threading.Lock()

    def _audit(self, event: str, **details: Any) -> None:
        record = {"time": time.time(), "event": event, **details}
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        try:
            self.audit_path.chmod(0o600)
        except OSError:
            pass

    def before_model(self, message_count: int, prompt_tokens: int) -> None:
        self._audit("before_model", message_count=message_count, prompt_tokens=prompt_tokens)

    def before_tool(
        self, name: str, args: dict[str, Any], source: str, permission: str = "ask"
    ) -> HookDecision:
        permission = permission if permission in {"ask", "allow", "deny"} else "ask"
        if permission == "deny":
            decision = HookDecision(False, False, "Tool permission is set to Deny")
        elif source == "mcp" and permission == "ask":
            decision = HookDecision(True, True, "External MCP tool requires approval")
        else:
            decision = HookDecision()
        self._audit(
            "before_tool", name=name, source=source, permission=permission,
            args=self.redact(json.dumps(args, ensure_ascii=False)), decision=asdict(decision),
        )
        return decision

    def after_tool(self, name: str, result: str, source: str, elapsed: float) -> str:
        cleaned = self.redact(result)
        if len(cleaned) > self.max_result_chars:
            cleaned = cleaned[:self.max_result_chars] + "\n...[tool result truncated]"
        self._audit(
            "after_tool", name=name, source=source, elapsed=round(elapsed, 3),
            result_chars=len(cleaned),
        )
        return cleaned

    def on_error(self, stage: str, error: Exception | str) -> None:
        self._audit("error", stage=stage, error=self.redact(str(error))[:1000])

    @staticmethod
    def redact(text: str) -> str:
        cleaned = text
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                cleaned = pattern.sub(r"\1[REDACTED]", cleaned)
            else:
                cleaned = pattern.sub("[REDACTED PRIVATE KEY]", cleaned)
        return cleaned
