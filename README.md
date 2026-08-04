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

## ความต้องการของระบบ

- Linux (พัฒนาบน Bazzite/Fedora)
- Python 3.10 ขึ้นไป พร้อม Tkinter
- `git`, `cmake` และ compiler สำหรับสร้าง `llama.cpp`
- การ์ดจอที่รองรับ Vulkan หรือใช้ `llama.cpp` backend อื่นที่เหมาะกับเครื่อง
- RAM/VRAM และพื้นที่จัดเก็บตามขนาดโมเดลที่เลือก

โมเดล Q4 ขนาด 4B มักต้องใช้พื้นที่ประมาณ 3 GB ส่วน 7B–8B ประมาณ 5 GB
ทั้งนี้หน่วยความจำจริงขึ้นกับ context size และ backend

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

ข้อมูลและ log ของโปรแกรมอยู่ที่:

```text
~/.local/state/localforge-ai/
```

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
