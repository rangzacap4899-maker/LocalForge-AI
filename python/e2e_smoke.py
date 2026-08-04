#!/usr/bin/env python3
"""Headless end-to-end smoke test; run with xvfb-run."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

state = tempfile.TemporaryDirectory(prefix="localforge-e2e-")
os.environ["XDG_STATE_HOME"] = state.name
os.environ["LOCALFORGE_WORKSPACE"] = "/tmp"
os.environ["LOCALFORGE_MODEL_ROOT"] = str(Path(__file__).resolve().parents[1] / "models")

import chatbot_app  # noqa: E402

chatbot_app.messagebox.showerror = lambda title, message, **_kwargs: print("UI_ERROR", title, message, flush=True)
chatbot_app.messagebox.showwarning = lambda title, message, **_kwargs: print("UI_WARNING", title, message, flush=True)
ChatApp = chatbot_app.ChatApp

app = ChatApp()
app.update()
app.input.insert("1.0", "ตอบสั้นๆ ว่า E2E พร้อมใช้งาน")
app.send()
deadline = time.time() + 45
while app.busy and time.time() < deadline:
    app.update()
    time.sleep(0.03)
app.update()
assert not app.busy, "request did not finish"
assert app.model_manager.active_model is not None, "router did not load a model"
assert app.messages[-1]["role"] == "assistant", "assistant answer missing"
print("E2E_OK", app.model_manager.active_model.name, app.messages[-1]["content"])
app._on_close()
state.cleanup()
