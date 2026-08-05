# LocalForge AI — handoff context

Last updated: 2026-08-05 (Asia/Bangkok)

This file is the durable context for another AI assistant taking over after the
current assistant reaches a usage limit. Read the current source and `git diff`
before acting; this document explains intent and constraints but source code is
authoritative.

## User goal and communication

The user is building a polished, click-to-run local AI desktop workspace on
Bazzite/Fedora. The UI is primarily Thai but supports English, Chinese, and
Japanese. The user prefers direct implementation and real testing over plans or
repeated clarification. Reply in Thai unless asked otherwise.

Main repository:
`https://github.com/rangzacap4899-maker/LocalForge-AI.git`

## Machine and runtime

- CPU: Intel Xeon E5-2680 v3, 12 cores / 24 threads
- RAM: 15 GiB
- GPU: Radeon RX 470/480 (Ellesmere), 8 GiB VRAM, Vulkan/amdgpu
- OS: Bazzite/Fedora; do not suggest `apt`
- Build parallelism: use at most `-j4` to avoid exhausting RAM
- Python virtual environment: `.venv`
- llama.cpp server:
  `runtime/llama.cpp/build-vulkan/bin/llama-server`
- Application launcher: `./launch_localforge_ai.sh`
- Runtime state/logs: `~/.local/state/localforge-ai/`

Installed primary model files (ignored by Git):

- `models/gemma-4-e4b/gemma-4-E4B_q4_0-it.gguf`
  - 5,154,941,280 bytes
  - SHA-256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`
- `models/gemma-4-e4b/gemma-4-E4B-it-mmproj.gguf`
  - 991,552,256 bytes
  - SHA-256 `7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578`

## Current product capabilities

- CustomTkinter chat UI with responsive layout, themes, UI scaling, Thai fonts
- Conversation storage/search/pin/export/delete and context meter
- Streaming output, cancellation, token/time/tokens-per-second display
- Copy, paste, selection, answer-copy and code-copy controls
- Workspace-scoped file tools, diff approval, backups and Undo
- Project Explorer and direct multi-file generation
- Web search/read tools
- Local model download/load/stop/delete/benchmark and automatic model routing
- Optional multi-agent coding workflow
- MCP stdio clients, permission hooks, audit log and secret redaction
- Languages: Thai, English, Simplified Chinese, Japanese
- Image selection and Wayland clipboard image paste (`Ctrl+Shift+V`)
- Gemma 4 image/audio requests through the official multimodal projector
- PipeWire microphone recording (`pw-record`, PCM WAV 16 kHz mono)
- Local answer speech using `spd-say`, with eSpeak fallback
- Image thumbnails inside user chat bubbles; media paths persist in history

## Important source files

- `python/chatbot_app.py`: UI, model manager, API client, tools and multimodal UI
- `python/localforge_core.py`: conversation store, routing, model info, transactions
- `python/localforge_i18n.py`: all four language catalogs
- `python/localforge_mcp.py`: MCP stdio lifecycle and tool selection
- `python/localforge_hooks.py`: permissions, audit and redaction hooks
- `python/test_*.py`: unit tests
- `python/CHATBOT_README.md` and `README.md`: user documentation
- `packaging/localforge-ai.desktop`: desktop launcher entry

## Gemma 4 E4B profile and recent bug history

`inference_profile()` in `python/chatbot_app.py` must keep these settings for
Gemma 4 E4B on this machine:

- `gpu_layers = all`
- context 8,192 (do not use the model's theoretical 128K on this hardware)
- Q8 K/V cache
- 12 inference and batch threads
- batch 2,048 and ubatch 2,048
- one slot, no continuous batching, reasoning disabled
- load sibling `gemma-4-E4B-it-mmproj.gguf` with `--mmproj` when present

Do not reduce multimodal ubatch to 256. A real pasted image caused llama.cpp to
abort with:
`non-causal attention requires n_ubatch >= n_tokens`.

Do not restore the streaming socket timeout to 5 seconds. Vision prefill can be
silent for 10–15 seconds, which caused false `timed out` errors and canceled
server tasks. `STREAM_IDLE_TIMEOUT_SECONDS` is currently 60 seconds.

Verified results after fixes:

- 3840x2160 JPEG, non-streaming: complete response in 10.42 s
- Same large JPEG through the UI streaming path: 33 chunks in 11.68 s
- Microphone capture produced valid PCM WAV, mono 16 kHz
- Silent microphone sample correctly returned `ไม่มีเสียงพูด`
- Unit suite: 40 tests passing at the time of this handoff

Image/audio notes:

- OpenAI-compatible content types are `image_url` data URLs and `input_audio`.
- Put images before prompt text and audio after prompt text (`with_media`).
- Media is limited to 25 MB.
- Base64 data is transient and is not persisted; only local `media_paths` are.
- Thumbnails are PNG files cached in `~/.local/state/localforge-ai/thumbnails/`.
- Old conversations created before thumbnail support lack media paths and cannot
  reconstruct their images; the user must attach those images again.
- llama.cpp currently logs that Gemma 4 audio is experimental.

## Testing and diagnostics

Fast required checks after Python changes:

```bash
python3 -m py_compile python/chatbot_app.py python/localforge_i18n.py
.venv/bin/python -m unittest discover -s python -p 'test_*.py'
git diff --check
```

Before starting an end-to-end server test, check whether the user's app already
owns port 8080:

```bash
pgrep -a llama-server
ss -ltnp | rg ':8080'
tail -120 ~/.local/state/localforge-ai/server.log
```

Do not kill a server belonging to the currently open user app merely to run a
test. Use another port only if VRAM permits; loading two Gemma 4 instances at
once normally does not fit. If a test starts its own manager, always stop it in
`finally` and confirm no test-owned server remains.

The screenshot/thumbnail path depends on ImageMagick (`magick`) and Tk
`PhotoImage`; Wayland clipboard paste depends on `wl-paste`; recording depends
on `pw-record`; speech output uses `spd-say` or eSpeak. These commands exist on
the development machine.

## Git and publishing workflow — critical

The local working tree originated from `google/gemma.cpp`, so remotes are:

- `origin` -> `https://github.com/google/gemma.cpp.git` (do not push LocalForge)
- `localforge` -> the user's GitHub repository

There is an unrelated user modification in root `CMakeLists.txt` changing
Highway and SentencePiece tags. Preserve it and do not stage, revert, or commit
it unless explicitly requested. Normal commits must name only intended files.

Local commits contain the app history but cannot be pushed directly as the clean
GitHub repository has unrelated ancestry. Established safe publishing flow:

1. Run tests and `git diff --check`.
2. Stage explicit LocalForge files only; never `git add -A`.
3. Commit locally with repo-local author
   `rangzacap4899-maker <rangzacap4899-maker@users.noreply.github.com>`.
4. Clone `https://github.com/rangzacap4899-maker/LocalForge-AI.git` into a fresh
   `mktemp -d` directory.
5. Copy only the changed files into that clean clone, commit, and push `main`.

Latest commits at handoff:

- Local working history: `2119ddc Redesign UI to Deep Space theme and add model toggle`
- Clean GitHub history: `daf4921 Redesign UI to Deep Space theme and add model toggle`

The hashes differ because of the separate histories; the file contents should
match. Never force-push or rewrite the user's GitHub history.

## Known cleanup and next checks

- The app must be restarted after source changes; an already open GUI continues
  running old Python code.
- Verify image thumbnails with a newly attached image after restart. Old image
  messages cannot display because their paths were never saved.
- The UI currently mixes some hard-coded Thai status/error strings with the
  translation catalog. Future polishing can move them into i18n.
- Speech output quality depends on installed system voices.
- Continue keeping prompts compact for 4B-class local models.
- After every material fix, update this file's date, verified test count, recent
  bug notes, and latest local/GitHub commits.

