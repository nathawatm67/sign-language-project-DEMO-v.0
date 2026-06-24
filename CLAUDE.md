# Smart Thai Sign Language Translator
## Project Context for Claude Code

### ภาพรวมโครงงาน
- **ชื่อ**: Smart Thai Sign Language Translator (SCI-INNO 001)
- **นิสิต**: นายณัฐวัตร มายูร — IT ปี 2 มหาวิทยาลัยนเรศวร
- **วันส่ง**: 9 สิงหาคม 2569 (เหลือ ~51 วัน)
- **งบประมาณ**: 5,000 บาท
- **เป้าหมาย**: Prototype ระบบแปลภาษามือไทย Real-time สำหรับส่งคณะวิทยาศาสตร์

---

## สถานะปัจจุบัน (Current State)

### ✅ เสร็จแล้ว
- `thai_sign_3words.py` — Rule-based gesture classifier ทำงานได้
  - รู้จัก 3 ท่า: สวัสดี (HELLO), ขอบคุณ (THANK YOU), ขอโทษ (SORRY)
  - ใช้ MediaPipe Hands + OpenCV
  - มี Smoothing logic (window=14, min_count=8, hold=2s)
  - มี Debug mode (กด D), Screenshot (กด S), Toggle landmark (กด H)
  - ผ่าน unit test ทั้ง 3 gesture แล้ว
- `run_windows.bat` — One-click installer + runner สำหรับ Windows
- Business Model Canvas + Value Proposition Canvas (HTML file)

### 🔄 กำลังทำ / ขั้นตอนถัดไป
| สัปดาห์ | วันที่ | งาน |
|---------|--------|-----|
| Week 2 | 26 มิ.ย – 2 ก.ค. | เก็บ Dataset .npy (30 ซีน/คำ/คน) |
| Week 3 | 3–9 ก.ค. | เทรน LSTM model ครั้งแรก |
| Week 4 | 10–16 ก.ค. | ปรับ accuracy > 85% |
| Week 5 | 17–23 ก.ค. | เชื่อม LSTM → GUI Real-time |
| Week 6 | 24–30 ก.ค. | ทดสอบจริง + เพิ่ม TTS |
| Week 7 | 31 ก.ค – 9 ส.ค. | รายงาน + นำเสนอ |

---

## Tech Stack

```
Python 3.x
opencv-python
mediapipe==0.10.13   ← version นี้เท่านั้น (ต้องมี .solutions API)
numpy
tensorflow / keras   ← ยังไม่ได้ติดตั้ง (ต้องทำ Week 3)
tkinter / PyQt5      ← GUI (ยังไม่ทำ)
gTTS                 ← Text-to-Speech (ยังไม่ทำ)
```

---

## โครงสร้างไฟล์ปัจจุบัน

```
project/
├── thai_sign_3words.py      ✅ rule-based classifier (DONE)
├── run_windows.bat          ✅ installer (DONE)
├── CLAUDE.md                ← ไฟล์นี้
│
├── data/                    ← ยังไม่มี (Week 2)
│   ├── sawatdi/             .npy files
│   ├── khob_khun/
│   └── kho_thot/
│
├── model/                   ← ยังไม่มี (Week 3)
│   ├── lstm_model.h5
│   └── label_map.json
│
├── gui/                     ← ยังไม่มี (Week 5)
│   └── app.py
│
└── report/                  ← ยังไม่มี (Week 7)
    └── accuracy_report.md
```

---

## Gesture Logic (Rule-based — ใน thai_sign_3words.py)

```python
# ใช้ lm = numpy array (21, 3) จาก MediaPipe

# สวัสดี: นิ้ว >= 4 ขึ้น + spread > 0.14
if total >= 4 and spread > 0.14:
    → "sawatdi"

# ขอโทษ: กำมือ (total==0) + โป้งพับ (up[0]==0)
if total == 0 and up[0] == 0:
    → "kho_thot"

# ขอบคุณ: โป้งขึ้น + มือยกสูง (wrist_y < 0.62)
if up[0] == 1 and total <= 2 and wrist_y < 0.62:
    → "khob_khun"
```

---

## งานถัดไปที่ Claude Code ควรช่วย

### Priority 1 — Dataset Collection Script (Week 2)
สร้าง `collect_data.py`:
- เปิดกล้อง กด SPACE เพื่อบันทึก 1 sequence
- บันทึก 30 frames/sequence เป็น numpy array shape (30, 21, 3)
- บันทึกลงใน `data/{gesture_name}/{seq_number}.npy`
- แสดง progress บนหน้าจอ (เช่น "12/30 sequences recorded")

### Priority 2 — LSTM Training Script (Week 3)
สร้าง `train_model.py`:
- โหลด dataset จาก `data/` ทั้งหมด
- สร้าง model: Input(30,63) → LSTM(64) → LSTM(32) → Dense(32) → Softmax(3)
- เทรน 100 epochs, บันทึก best model
- แสดง accuracy/loss graph และ confusion matrix

### Priority 3 — Real-time Inference (Week 4–5)
อัปเดต `thai_sign_3words.py` ให้:
- โหลด LSTM model แทน rule-based
- ใช้ sliding window 30 frames
- Fallback กลับ rule-based ถ้าโหลด model ไม่ได้

### Priority 4 — GUI + TTS (Week 5–6)
สร้าง `gui/app.py`:
- Tkinter window แสดงผล Webcam
- แสดงคำแปลไทยด้วย font ที่ถูกต้อง
- เพิ่ม gTTS พูดออกเสียงเมื่อตรวจพบ gesture ใหม่

---

## ข้อจำกัดที่ต้องรู้

1. **mediapipe version**: ต้องใช้ `==0.10.13` เท่านั้น
   - version ใหม่กว่าไม่มี `mp.solutions` API
   - ถ้าเจอ `AttributeError: module 'mediapipe' has no attribute 'solutions'` → downgrade ทันที

2. **Camera index**: Default = 0, ถ้าไม่ขึ้นลอง 1 หรือ 2

3. **Thai font**: OpenCV ไม่รองรับ Unicode โดยตรง
   - ถ้าต้องแสดงภาษาไทยใน OpenCV ต้องใช้ PIL/Pillow แปลงก่อน
   - หรือใช้ Tkinter/PyQt5 ที่รองรับ Unicode

4. **Dataset size**: ต้องการ >= 30 sequences/gesture, แนะนำ 60–80
   - บันทึกจากหลายคน (ไม่ใช่คนเดียว) เพื่อให้ model generalize

5. **Accuracy target**: >= 85% บน test set
   - ถ้าต่ำกว่า → เพิ่ม data ก่อน ไม่ใช่แก้ architecture

---

## ทรัพยากรอ้างอิง

- YouTube: Nicholas Renotte "Sign Language Detection LSTM" → https://youtube.com/watch?v=doDUihpj6ro
- MediaPipe Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- งานวิจัยอ้างอิง: https://link.springer.com/article/10.1007/s44163-024-00113-8

---

## คำสั่งรัน

```bash
# ติดตั้ง
pip install opencv-python mediapipe==0.10.13 numpy

# รันโปรแกรมปัจจุบัน
python thai_sign_3words.py

# Windows one-click
run_windows.bat
```

---

*ไฟล์นี้สร้างโดย Claude (claude.ai) เพื่อใช้เป็น context ใน Claude Code*
*อัปเดตล่าสุด: 19 มิถุนายน 2569*
