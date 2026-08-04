"""Small dependency-free UI translation catalog for LocalForge AI."""

from __future__ import annotations

from typing import Any


LANGUAGE_NAMES = {"th": "ไทย", "en": "English", "zh": "中文", "ja": "日本語"}
LANGUAGE_CODES = {name: code for code, name in LANGUAGE_NAMES.items()}
LANGUAGE_FONTS = {
    "th": "Noto Sans Thai",
    "en": "Noto Sans",
    "zh": "Noto Sans CJK SC",
    "ja": "Noto Sans CJK JP",
}


TRANSLATIONS: dict[str, dict[str, str]] = {
    "th": {
        "new_chat": "＋  บทสนทนาใหม่", "settings": "⚙  ตั้งค่า",
        "change_folder": "▣  เปลี่ยนโฟลเดอร์", "project_explorer": "◫  Project Explorer",
        "conversations": "บทสนทนา", "search_conversations": "ค้นหาบทสนทนา…",
        "export": "ส่งออก", "pin": "ปักหมุด", "delete": "ลบ", "undo": "↶  ย้อนคืนล่าสุด",
        "your_assistant": "ผู้ช่วยของคุณ", "tagline": "สนทนา • สร้างไฟล์ • ค้นหาข้อมูล",
        "model_off": "โมเดลปิดอยู่", "send": "ส่ง  ➜", "stop": "หยุด ■",
        "ready": "พร้อมแล้ว — ถามคำถาม หรือสั่งให้อ่าน/เขียนไฟล์และค้นเว็บได้เลย",
        "system": "ระบบ", "you": "คุณ", "error": "ข้อผิดพลาด",
        "copy": "คัดลอก", "paste": "วาง", "cut": "ตัด", "select_all": "เลือกทั้งหมด",
        "copied": "คัดลอกแล้ว ✓", "copy_code": "คัดลอกโค้ด {index}",
        "open_project": "เปิดโปรเจกต์", "edit": "แก้ไข", "regenerate": "สร้างใหม่",
        "settings_title": "ตั้งค่า", "settings_subtitle": "จัดการโมเดล การเชื่อมต่อ และโหมดการทำงาน",
        "installed_models": "โมเดลที่ติดตั้ง", "load_model": "โหลดโมเดล", "stop_model": "ปิดโมเดล",
        "delete_model": "ลบโมเดล", "auto_router": "เลือกโมเดลอัตโนมัติตามงาน",
        "multi_agent": "โหมด Multi-agent", "download_models": "ดาวน์โหลดโมเดล",
        "models_location": "โมเดลจะเก็บไว้ใน {path}", "download": "ดาวน์โหลด",
        "cancel_download": "ยกเลิกดาวน์โหลด", "display": "การแสดงผล", "ui_size": "ขนาด UI",
        "language": "ภาษา", "restart_language": "ภาษาจะเปลี่ยนหลังเปิดโปรแกรมใหม่",
        "done": "เสร็จสิ้น", "no_models": "(ยังไม่มีโมเดล GGUF)",
    },
    "en": {
        "new_chat": "＋  New chat", "settings": "⚙  Settings",
        "change_folder": "▣  Change folder", "project_explorer": "◫  Project Explorer",
        "conversations": "Conversations", "search_conversations": "Search conversations…",
        "export": "Export", "pin": "Pin", "delete": "Delete", "undo": "↶  Undo latest",
        "your_assistant": "Your assistant", "tagline": "Chat • Create files • Search the web",
        "model_off": "Model is stopped", "send": "Send  ➜", "stop": "Stop ■",
        "ready": "Ready — ask a question or request file and web tasks.",
        "system": "System", "you": "You", "error": "Error",
        "copy": "Copy", "paste": "Paste", "cut": "Cut", "select_all": "Select all",
        "copied": "Copied ✓", "copy_code": "Copy code {index}",
        "open_project": "Open project", "edit": "Edit", "regenerate": "Regenerate",
        "settings_title": "Settings", "settings_subtitle": "Manage models, connections, and operating modes",
        "installed_models": "Installed models", "load_model": "Load model", "stop_model": "Stop model",
        "delete_model": "Delete model", "auto_router": "Automatically select a model for each task",
        "multi_agent": "Multi-agent mode", "download_models": "Download models",
        "models_location": "Models are stored in {path}", "download": "Download",
        "cancel_download": "Cancel download", "display": "Appearance", "ui_size": "UI size",
        "language": "Language", "restart_language": "The language changes after restarting the app",
        "done": "Done", "no_models": "(No GGUF models installed)",
    },
    "zh": {
        "new_chat": "＋  新对话", "settings": "⚙  设置",
        "change_folder": "▣  更改文件夹", "project_explorer": "◫  项目浏览器",
        "conversations": "对话", "search_conversations": "搜索对话…",
        "export": "导出", "pin": "置顶", "delete": "删除", "undo": "↶  撤销上次操作",
        "your_assistant": "您的助手", "tagline": "聊天 • 创建文件 • 搜索信息",
        "model_off": "模型已停止", "send": "发送  ➜", "stop": "停止 ■",
        "ready": "已就绪 — 可以提问，或要求读写文件和搜索网页。",
        "system": "系统", "you": "你", "error": "错误",
        "copy": "复制", "paste": "粘贴", "cut": "剪切", "select_all": "全选",
        "copied": "已复制 ✓", "copy_code": "复制代码 {index}",
        "open_project": "打开项目", "edit": "编辑", "regenerate": "重新生成",
        "settings_title": "设置", "settings_subtitle": "管理模型、连接和运行模式",
        "installed_models": "已安装模型", "load_model": "加载模型", "stop_model": "停止模型",
        "delete_model": "删除模型", "auto_router": "根据任务自动选择模型",
        "multi_agent": "多智能体模式", "download_models": "下载模型",
        "models_location": "模型将保存在 {path}", "download": "下载",
        "cancel_download": "取消下载", "display": "显示", "ui_size": "界面大小",
        "language": "语言", "restart_language": "重启应用后语言生效",
        "done": "完成", "no_models": "（尚未安装 GGUF 模型）",
    },
    "ja": {
        "new_chat": "＋  新しいチャット", "settings": "⚙  設定",
        "change_folder": "▣  フォルダーを変更", "project_explorer": "◫  プロジェクト",
        "conversations": "会話", "search_conversations": "会話を検索…",
        "export": "書き出す", "pin": "固定", "delete": "削除", "undo": "↶  直前を元に戻す",
        "your_assistant": "あなたのアシスタント", "tagline": "チャット • ファイル作成 • Web検索",
        "model_off": "モデル停止中", "send": "送信  ➜", "stop": "停止 ■",
        "ready": "準備完了 — 質問、ファイル操作、Web検索を依頼できます。",
        "system": "システム", "you": "あなた", "error": "エラー",
        "copy": "コピー", "paste": "貼り付け", "cut": "切り取り", "select_all": "すべて選択",
        "copied": "コピー済み ✓", "copy_code": "コードをコピー {index}",
        "open_project": "プロジェクトを開く", "edit": "編集", "regenerate": "再生成",
        "settings_title": "設定", "settings_subtitle": "モデル、接続、動作モードを管理",
        "installed_models": "インストール済みモデル", "load_model": "モデルをロード", "stop_model": "モデルを停止",
        "delete_model": "モデルを削除", "auto_router": "タスクに応じてモデルを自動選択",
        "multi_agent": "マルチエージェントモード", "download_models": "モデルをダウンロード",
        "models_location": "モデルの保存先: {path}", "download": "ダウンロード",
        "cancel_download": "ダウンロードを中止", "display": "表示", "ui_size": "UIサイズ",
        "language": "言語", "restart_language": "アプリの再起動後に言語が変わります",
        "done": "完了", "no_models": "（GGUFモデルは未インストールです）",
    },
}


def translate(language: str, key: str, **values: Any) -> str:
    catalog = TRANSLATIONS.get(language, TRANSLATIONS["en"])
    template = catalog.get(key, TRANSLATIONS["en"].get(key, key))
    return template.format(**values)
