# LocalForge AI

LocalForge AI คือแอปเดสก์ท็อปสำหรับใช้งานโมเดลภาษาแบบ local ผ่าน
`llama.cpp` โดยมีหน้าต่างภาษาไทยที่สร้างด้วย CustomTkinter ข้อมูลการสนทนา
โมเดล และไฟล์งานยังอยู่ภายในเครื่องของผู้ใช้

> **English:** A modern desktop workspace for running local GGUF models with
> chat, file tools, web search, model management, and coding assistance.

## ความสามารถ

- สนทนาแบบ streaming พร้อมปุ่มหยุด จำนวน token เวลา และความเร็ว `tokens/s`
- โหลด ปิด ดาวน์โหลด ตรวจสอบ SHA-256 และ benchmark โมเดล GGUF
- Model Router เลือกโมเดลสนทนาหรือเขียนโค้ดตามงานโดยอัตโนมัติ
- อ่านไฟล์และเสนอการแก้ไขภายใน workspace พร้อม diff ก่อนอนุมัติ
- สำรองไฟล์และ Undo การเปลี่ยนแปลงล่าสุด
- Project Explorer สำหรับเปิด เปลี่ยนชื่อ และจัดการไฟล์โปรเจกต์
- ค้นเว็บและอ่านหน้าเว็บสาธารณะผ่าน HTTP/HTTPS
- จัดเก็บ ค้นหา ปักหมุด ส่งออก และลบบทสนทนา
- Context Inspector, token meter, markdown highlighting และปุ่มคัดลอก
- UI รองรับไทย, English, 中文 และ日本語 พร้อมฟอนต์ตามภาษา
- รองรับธีมมืด/สว่างและการปรับขนาด UI
- แสดงสถานะ CPU, RAM, GPU, VRAM และอุณหภูมิ
- เชื่อม local MCP servers ผ่าน stdio พร้อม Permission Center และ audit hooks

## ความต้องการของระบบ

- Linux (พัฒนาบน Bazzite/Fedora)
- Python 3.10 ขึ้นไป พร้อม Tkinter
- `git`, `cmake` และ compiler สำหรับสร้าง `llama.cpp`
- การ์ดจอที่รองรับ Vulkan หรือใช้ `llama.cpp` backend อื่นที่เหมาะกับเครื่อง
- RAM/VRAM และพื้นที่จัดเก็บตามขนาดโมเดลที่เลือก

โมเดล Q4 ขนาด 4B มักต้องใช้พื้นที่ประมาณ 3 GB ส่วน 7B–8B ประมาณ 5 GB
ทั้งนี้หน่วยความจำจริงขึ้นกับ context size และ backend

เมนูดาวน์โหลดรองรับ Gemma 4 E4B IT Q4_0 รุ่นทางการ (ไฟล์ประมาณ 5.15 GB)
เมื่อเลือก Gemma 4 E4B แอปจะใช้โปรไฟล์สำหรับการ์ดจอ VRAM 8 GB โดยอัตโนมัติ:
offload ทุก layer ไปยัง GPU, context 8,192 tokens, KV cache Q8, multimodal batch 2,048
และจำกัด CPU ที่ 12 threads
เพื่อให้ตอบสนองไวและไม่ใช้หน่วยความจำเกินจำเป็น

Gemma 4 E4B รองรับภาพและเสียงใน LocalForge เมื่อมีไฟล์ `gemma-4-E4B-it-mmproj.gguf`
อยู่ข้างโมเดลหลัก เมนูดาวน์โหลดจะติดตั้งไฟล์นี้ให้อัตโนมัติ สามารถกด `＋ รูปภาพ`,
กด `Ctrl+Shift+V` เพื่อวางภาพจากคลิปบอร์ด Wayland หรือกด `🎙 เสียง` เพื่อเริ่ม/หยุดอัดเสียง
ผ่าน PipeWire ได้ ปุ่ม `🔊 อ่าน` ในคำตอบใช้ Speech Dispatcher อ่านคำตอบออกเสียง
ไฟล์สื่อจำกัดขนาด 25 MB และไม่ถูกบันทึกเป็น base64 ลงประวัติสนทนา
ภาพที่แนบจะแสดงเป็น thumbnail ในบอลลูนและจำพาธไว้เพื่อแสดงอีกครั้งเมื่อเปิดบทสนทนา
การประมวลผลภาพอาจเงียบก่อน token แรกได้นานกว่าแชตข้อความ แอปจึงรอ prefill
ได้นานสูงสุด 60 วินาทีก่อนแจ้ง timeout
ในโหมดข้อความ ส่วน multimodal projector และการรับภาพ/เสียงยังไม่เปิดใช้ในรุ่นนี้

## ติดตั้ง

### 1. ดาวน์โหลด LocalForge AI

```bash
git clone https://github.com/rangzacap4899-maker/LocalForge-AI.git
cd LocalForge-AI
```

### 2. เตรียม Python

ตรวจสอบ Tkinter ก่อน:

```bash
python3 -c "import tkinter; print('Tkinter OK')"
```

หาก Bazzite/Fedora ไม่มี Tkinter ให้ติดตั้งแพ็กเกจ `python3-tkinter`
ด้วยวิธีจัดการแพ็กเกจที่เหมาะกับระบบของคุณ แล้วสร้าง virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/chatbot_requirements.txt
```

### 3. สร้าง llama.cpp พร้อม Vulkan

```bash
git clone https://github.com/ggml-org/llama.cpp.git runtime/llama.cpp
cmake -S runtime/llama.cpp -B runtime/llama.cpp/build-vulkan \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/llama.cpp/build-vulkan --target llama-server -j4
```

ใช้ `-j4` เพื่อจำกัดงาน build พร้อมกันไม่ให้ใช้ RAM มากเกินไป หากต้องการใช้
CPU หรือ backend อื่น โปรดดูเอกสารของ
[llama.cpp](https://github.com/ggml-org/llama.cpp)

### 4. เปิดโปรแกรม

```bash
./launch_localforge_ai.sh
```

จากนั้นเปิด **ตั้งค่า → โมเดล** เพื่อดาวน์โหลดหรือเลือกไฟล์ `.gguf` แล้วกด
**โหลดโมเดล** โปรแกรมจะเปิด `llama-server` ให้เองและปิดเมื่อออกจากโปรแกรม

โมเดลที่ดาวน์โหลดผ่านโปรแกรมจะอยู่ใน `models/` และไม่ถูกเก็บใน Git

## เพิ่มไอคอนในเมนูแอป

ไฟล์ desktop entry ที่ให้มาใช้ตำแหน่ง `/home/addrang/LocalForge-AI` หากติดตั้ง
ไว้ตำแหน่งอื่น ให้แก้ค่า `Exec` ก่อนคัดลอก:

```bash
mkdir -p ~/.local/share/applications
cp packaging/localforge-ai.desktop ~/.local/share/applications/
```

## การใช้งาน

1. เลือก workspace จากแถบด้านซ้าย
2. โหลดโมเดลในเมนูตั้งค่า
3. พิมพ์คำถามหรือสั่งให้อ่าน/สร้าง/แก้ไขไฟล์
4. ตรวจ diff และกดยืนยันก่อนให้โปรแกรมเขียนไฟล์

คีย์ลัด:

- `Ctrl+Enter` — ส่งข้อความ
- `Ctrl+C`, `Ctrl+V`, `Ctrl+X` — คัดลอก วาง และตัด
- คลิกขวาในช่องข้อความ — เปิดเมนูแก้ไข

## ความปลอดภัยและความเป็นส่วนตัว

- การประมวลผลโมเดลทำภายในเครื่อง
- เครื่องมือไฟล์ถูกจำกัดให้อยู่ภายใน workspace ที่เลือก
- โปรแกรมแสดง diff และรออนุมัติก่อนเขียนไฟล์
- ไฟล์เดิมจะถูกสำรองเพื่อให้ย้อนคืนได้
- ฟีเจอร์ค้นเว็บจะส่งคำค้นหรือ URL ไปยังบริการอินเทอร์เน็ตที่เกี่ยวข้อง
- MCP tool ภายนอกใช้สิทธิ์ `Ask` เป็นค่าเริ่มต้นและต้องได้รับอนุมัติก่อนทำงาน
- LocalForge ไม่รัน MCP command ผ่าน shell และจำกัดจำนวน tool schema ที่ส่งให้โมเดล

ข้อมูลและ log ของโปรแกรมอยู่ที่:

```text
~/.local/state/localforge-ai/
```

## MCP Servers และ Hooks

เปิด **ตั้งค่า → MCP Servers & Hooks** เพื่อเพิ่ม local MCP server ที่สื่อสาร
ผ่าน stdio โดยกรอกชื่อและคำสั่ง executable พร้อม arguments โดยตรง ห้ามใส่
shell operator เช่น `|`, `&&` หรือ `>`

สิทธิ์แต่ละ server:

- `ask` — แสดงชื่อ tool และ arguments เพื่อขออนุมัติทุกครั้ง (ค่าเริ่มต้น)
- `allow` — อนุญาตโดยไม่ถาม เหมาะเฉพาะ server ที่เชื่อถือได้
- `deny` — ไม่เปิดเผย tools ของ server ให้โมเดล

Hook engine ทำงานก่อนและหลัง model/tool calls เพื่อบันทึก audit, ปิดบัง token
หรือ private key และจำกัดขนาดผลลัพธ์ เครื่องมือ MCP ถูกใส่ namespace เช่น
`mcp__github__create_issue` และโปรแกรมจะเลือกส่งให้โมเดลสูงสุด 6 tools ที่เกี่ยวข้อง
กับคำถาม เพื่อลด context และความสับสนของโมเดลขนาดเล็ก

ไฟล์การตั้งค่าและ audit:

```text
~/.local/state/localforge-ai/mcp_servers.json
~/.local/state/localforge-ai/audit.jsonl
```

> MCP server เป็นโปรแกรมที่ทำงานด้วยสิทธิ์เดียวกับผู้ใช้ โปรดติดตั้งและเปิดใช้
> เฉพาะ server ที่ตรวจสอบแหล่งที่มาแล้ว

## ทดสอบ

```bash
.venv/bin/python -m unittest discover -s python -p 'test_*.py'
```

เมื่อมีโมเดลและ `llama-server` พร้อมแล้ว สามารถทดสอบ end-to-end ได้ด้วย:

```bash
xvfb-run -a .venv/bin/python python/e2e_smoke.py
```

## โครงสร้างโปรเจกต์

```text
LocalForge-AI/
├── launch_localforge_ai.sh
├── packaging/
│   └── localforge-ai.desktop
├── python/
│   ├── chatbot_app.py
│   ├── localforge_core.py
│   ├── localforge_hooks.py
│   ├── localforge_mcp.py
│   └── test_*.py
├── models/                 # สร้างภายในเครื่องและไม่เก็บใน Git
└── runtime/llama.cpp/      # ติดตั้งภายในเครื่องและไม่เก็บใน Git
```

## สถานะโครงการ

โครงการนี้อยู่ระหว่างพัฒนา โปรดสำรองไฟล์สำคัญก่อนให้โมเดลแก้ไขโปรเจกต์จริง
และตรวจสอบ diff ทุกครั้ง โมเดลขนาดเล็กอาจเรียกเครื่องมือผิดรูปแบบหรือให้คำตอบ
ที่ไม่ถูกต้องได้

## เครดิต

- [llama.cpp](https://github.com/ggml-org/llama.cpp) สำหรับ local inference server
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) สำหรับส่วนติดต่อผู้ใช้
- เจ้าของโมเดลแต่ละรายสำหรับไฟล์ GGUF ที่เลือกใช้งาน

Repository นี้ยังไม่ได้ประกาศ license สำหรับซอร์สของ LocalForge AI
