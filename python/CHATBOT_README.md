# LocalForge AI

ผู้ช่วย AI ภายในเครื่องด้วย CustomTkinter ใช้ `llama.cpp`/Vulkan และโมเดล GGUF
สำหรับสนทนา เขียนโค้ด จัดการไฟล์ใน workspace และค้นเว็บ

## เปิดโปรแกรม

เปิด **LocalForge AI** จากเมนูแอปหรือ Desktop หรือรัน:

```bash
cd ~/LocalForge-AI
./launch_localforge_ai.sh
```

โปรแกรมไม่โหลดโมเดลจนกว่าจะกด **ตั้งค่า → โหลดโมเดล** และจะปิด
`llama-server` เมื่อปิดโปรแกรม Log อยู่ที่:

```text
~/.local/state/localforge-ai/server.log
```

## โครงสร้าง

```text
~/LocalForge-AI/
├── models/                 โมเดล GGUF
├── runtime/llama.cpp/      llama-server และ Vulkan libraries
├── python/                 แอปและชุดทดสอบ
└── launch_localforge_ai.sh
```

## ความสามารถหลัก

- Streaming พร้อมปุ่มหยุด เวลา และความเร็ว tokens/s
- Model Router เลือกโมเดลสนทนา/เขียนโค้ดอัตโนมัติ
- ดาวน์โหลด ตรวจ SHA-256, benchmark, โหลด ปิด และย้ายโมเดลลงถังขยะ
- Diff approval, syntax validation, backup และ Undo ก่อน AI เขียนไฟล์
- Project Explorer พร้อมเปิด/rename/delete/Terminal/Browser
- หลายบทสนทนา ค้นหา ปักหมุด แก้ไข สร้างใหม่ ลบ และ export
- Context Inspector พร้อม token meter, trim และ summary
- Markdown highlighting, copy/code-copy, UI ไทย/English/中文/日本語, theme และ UI scaling
- CPU/RAM/GPU/VRAM/temperature monitor และ desktop notification

## คีย์ลัด

- `Ctrl+Enter` — ส่งข้อความ
- `Ctrl+C`, `Ctrl+V`, `Ctrl+X` — คัดลอก วาง และตัด
- คลิกขวาในช่องข้อความ — เมนูแก้ไข

## ความปลอดภัย

- เครื่องมือไฟล์เข้าถึงได้เฉพาะ workspace
- ทุกไฟล์ที่ AI สร้างหรือแก้ต้องได้รับอนุมัติจากหน้าต่าง diff
- ไฟล์เดิมถูกสำรองก่อนเขียนและย้อนคืนได้
- การลบโมเดลใช้ถังขยะของระบบเมื่อรองรับ
- URL รองรับเฉพาะ HTTP/HTTPS

## ทดสอบ

```bash
cd ~/LocalForge-AI
.venv/bin/python -m unittest discover -s python -p 'test_*.py'
xvfb-run -a .venv/bin/python python/e2e_smoke.py
```
