---
name: Modern UI Design
description: ทักษะพื้นฐานและแนวทางในการออกแบบ UI ให้มีความทันสมัย สวยงาม และพรีเมียม (Modern, Premium, Dynamic)
---

# 🎨 Modern UI Design Skill

สกิลนี้กำหนดมาตรฐานการออกแบบ (Design Standards) ที่ AI ต้องปฏิบัติตามทุกครั้งที่มีการสร้างหรือแก้ไข User Interface (UI) ไม่ว่าจะเป็น Web App หรือ Desktop App เพื่อให้ผลลัพธ์ออกมาดู **"พรีเมียม"** และ **"ทันสมัย"** ที่สุด

## 1. 🌈 ระบบสี (Color Palette & Aesthetics)
- **ห้ามใช้สีพื้นฐานแบบดื้อๆ:** (เช่น สีแดงล้วน `red`, สีน้ำเงินล้วน `blue`, สีเขียวล้วน `green`)
- **ใช้สีที่ผ่านการจับคู่มาอย่างดี:** เน้นการใช้สีแนว Pastel, HSL-tailored colors หรือโทนสีแบบ Modern Dark Mode (เช่น `Slate`, `Zinc`, `Midnight Blue`)
- **สร้างความมีมิติ:** ใช้ Gradients แบบนุ่มนวล (Smooth gradients) หรือเทคนิค Glassmorphism (พื้นหลังกึ่งโปร่งใสพร้อม Blur effect) เพื่อให้ UI ดูมีมิติ ไม่แบนเรียบ

## 2. 🔤 ตัวอักษร (Typography)
- ห้ามใช้ฟอนต์เริ่มต้นของเบราว์เซอร์
- ให้ใช้ฟอนต์ตระกูล Modern Sans-serif เสมอ เช่น `Inter`, `Roboto`, `Outfit`, หรือ `Noto Sans Thai` สำหรับภาษาไทย
- จัดลำดับความสำคัญของข้อความ (Visual Hierarchy) ด้วยขนาด (Size), น้ำหนัก (Weight), และความสว่างของสี (Contrast) อย่างชัดเจน

## 3. ✨ การโต้ตอบและภาพเคลื่อนไหว (Interaction & Micro-animations)
- **UI ต้องดูมีชีวิตชีวา (Dynamic & Alive):**
- ต้องมีเอฟเฟกต์ **Hover** หรือ **Active** เสมอสำหรับปุ่มกดและองค์ประกอบที่คลิกได้
- ใช้ **Micro-animations** หรือ Transitions แบบนุ่มนวล (เช่น `transition: all 0.2s ease-in-out;`) เมื่อมีการเปลี่ยนสถานะ
- ห้ามสร้าง UI ที่เมื่อเอาเมาส์ไปชี้แล้วไม่มีการตอบสนองใดๆ

## 4. 📐 โครงสร้างและการจัดวาง (Layout & Spacing)
- **Whitespace is King:** ใช้ช่องไฟ (Padding & Margin) ให้เพียงพอเพื่อให้ UI ดูสะอาดตาและไม่อึดอัด
- ใช้มุมโค้งมน (Border Radius) ที่เหมาะสมกับสไตล์ (เช่น `8px` ถึง `12px` สำหรับการ์ด, และวงรี `pill shape` สำหรับปุ่มกดบางประเภท)
- ต้องรองรับการแสดงผลแบบ Responsive เสมอ

## 5. 🚫 ข้อควรระวัง (Strict Rules)
- ห้ามสร้างหน้าตาแอปแบบ "Minimum Viable Product (MVP)" ที่ดูแข็งกระด้าง
- ถ้าต้องใช้รูปภาพ ให้ใช้เครื่องมือสร้างรูปภาพ (`generate_image`) เพื่อจำลองของจริงแทนการใช้ Placeholder โล่งๆ
- ผลลัพธ์สุดท้ายต้องทำให้ผู้ใช้รู้สึก "WOW" ตั้งแต่แรกเห็น
