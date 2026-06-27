# Accuracy Report — Smart Thai Sign Language Translator

> โครงงาน SCI-INNO 001 · นายณัฐวัตร มายูร · IT ปี 2 มหาวิทยาลัยนเรศวร
> อัปเดตล่าสุด: [PLACEHOLDER — วันที่จัดทำรายงาน]

---

## 1. Project Overview

Smart Thai Sign Language Translator เป็นระบบต้นแบบ (prototype) สำหรับแปลภาษามือไทยแบบเรียลไทม์
รองรับคำพื้นฐาน 3 คำ ได้แก่ **สวัสดี (HELLO)**, **ขอบคุณ (THANK YOU)**, และ **ขอโทษ (SORRY)**
ระบบใช้กล้องเว็บแคมจับภาพมือ สกัดจุด landmark ด้วย MediaPipe Hands แล้วจำแนกท่าทางด้วยโมเดล LSTM
ที่เทรนจากชุดข้อมูลที่เก็บเอง พร้อมระบบแปลงข้อความเป็นเสียงพูดภาษาไทย (Text-to-Speech)
เป้าหมายของโครงงานคือสร้างต้นแบบที่ทำงานได้จริงเพื่อนำเสนอต่อคณะวิทยาศาสตร์ ภายในงบประมาณ 5,000 บาท

---

## 2. Dataset Summary

| Gesture | คำแปลไทย | จำนวน sequences |
|---------|----------|----------------|
| `sawatdi`   | สวัสดี  | 91 |
| `khob_khun` | ขอบคุณ | 90 |
| `kho_thot`  | ขอโทษ  | 90 |
| **รวม (Total)** | | **271** |

- **รูปแบบข้อมูลต่อ sequence:** shape `(30, 126)` — 30 เฟรม × 126 ฟีเจอร์ (2 มือ × 21 landmark × พิกัด x,y,z)
- **จำนวนผู้ร่วมเก็บข้อมูล (contributors):** [PLACEHOLDER — จำนวนคนที่ช่วยบันทึกท่า เช่น 1–3 คน; ถ้าเก็บคนเดียวให้ระบุว่า 1 คน]
- **เครื่องมือเก็บข้อมูล:** `collect_data.py` (บันทึก 30 เฟรม/sequence เมื่อกด SPACE)

---

## 3. Model Architecture

โมเดลเป็น Sequential LSTM สร้างด้วย Keras (จาก `train_model.py`):

| ลำดับ | Layer | พารามิเตอร์ | หมายเหตุ |
|------|-------|------------|----------|
| Input | — | shape `(30, 126)` | 30 เฟรม × 126 ฟีเจอร์ |
| 1 | `LSTM(64)` | `return_sequences=True` | ส่ง sequence ต่อให้ LSTM ชั้นถัดไป |
| 2 | `LSTM(32)` | `return_sequences=False` | สรุปเป็นเวกเตอร์เดียว |
| 3 | `Dense(32)` | `activation="relu"` | ชั้น fully-connected |
| 4 | `Dropout(0.3)` | rate = 0.3 | ลด overfitting |
| 5 | `Dense(3)` | `activation="softmax"` | output 3 คลาส |

- **Optimizer:** Adam
- **Loss:** `sparse_categorical_crossentropy`
- **Metric:** accuracy

---

## 4. Training Results

| รายการ | ค่า |
|--------|-----|
| Epochs ที่ตั้งไว้ | 200 |
| Epochs ที่ใช้จริง (หลัง EarlyStopping) | [PLACEHOLDER — ดูจาก log ตอนเทรน; EarlyStopping patience=20] |
| Batch size | 32 |
| Train/Validation split | 80 / 20 |
| Final train accuracy | [PLACEHOLDER — ค่า accuracy บรรทัดสุดท้ายตอนเทรน] |
| Final validation accuracy | [PLACEHOLDER — ค่า val_accuracy บรรทัดสุดท้าย] |
| Best validation accuracy | [PLACEHOLDER — ค่า best val_accuracy ที่ระบบ print ออกมา] |

> หมายเหตุ: โมเดลที่บันทึก (`model/lstm_model.h5`) เป็น checkpoint ที่ให้ val_accuracy สูงสุด

---

## 5. Per-Gesture Accuracy

อ้างอิงจาก `model/confusion_matrix.png` (validation set)

| Gesture จริง \ ทำนาย | sawatdi | khob_khun | kho_thot | Recall (%) |
|----------------------|---------|-----------|----------|-----------|
| **sawatdi**   | [PH] | [PH] | [PH] | [PLACEHOLDER — % ที่ทายถูกของท่าสวัสดี] |
| **khob_khun** | [PH] | [PH] | [PH] | [PLACEHOLDER — % ที่ทายถูกของท่าขอบคุณ] |
| **kho_thot**  | [PH] | [PH] | [PH] | [PLACEHOLDER — % ที่ทายถูกของท่าขอโทษ] |

> `[PH]` = จำนวนตัวอย่างในแต่ละช่องของ confusion matrix (อ่านจากภาพ)
> ท่าที่มี accuracy ต่ำสุด: [PLACEHOLDER — ระบุท่าที่สับสนบ่อยที่สุด เช่น kho_thot มักสับสนกับ khob_khun]

---

## 6. Known Limitations

- **ชุดข้อมูลมาจากผู้ร่วมเก็บจำนวนจำกัด** — หากเก็บจากคนเดียวหรือไม่กี่คน โมเดลอาจ generalize
  กับผู้ใช้คนอื่น/รูปร่างมือต่างกันได้ไม่ดี และอ่อนไหวต่อมุมกล้อง/แสงที่เปลี่ยนไป
- **รองรับเพียง 3 คำ** — ยังไม่ครอบคลุมภาษามือไทยจริงที่มีคำและไวยากรณ์ซับซ้อนกว่ามาก
  จึงเป็นต้นแบบเชิงพิสูจน์แนวคิด (proof of concept) ไม่ใช่ระบบใช้งานจริง
- **ประสิทธิภาพขึ้นกับ MediaPipe** — หากตรวจจับมือหลุด (แสงน้อย, มือบังกัน, มือออกนอกเฟรม)
  ฟีเจอร์จะกลายเป็นศูนย์และทำให้การทำนายผิดพลาด

---

## 7. Next Steps (หากพัฒนาต่อ)

1. **เพิ่มขนาดและความหลากหลายของชุดข้อมูล** — เก็บจากผู้ร่วมหลายคน หลายมุมกล้อง หลายสภาพแสง
   เพื่อยกระดับความแม่นยำและความทนทาน
2. **ขยายจำนวนคำศัพท์** — เพิ่มคำพื้นฐานที่ใช้บ่อย (เช่น ใช่/ไม่, ชื่อ, ตัวเลข) ทีละชุด
3. **ปรับปรุงประสบการณ์ผู้ใช้** — เพิ่ม GUI ที่รองรับฟอนต์ไทยเต็มรูปแบบ และปรับ TTS ให้ลื่นไหลขึ้น
4. **ประเมินบนชุดทดสอบอิสระ** — แยก test set จากผู้ใช้ที่ไม่เคยอยู่ใน training data เพื่อวัดผลที่สมจริง
5. **ทดลองสถาปัตยกรรมอื่น** — เช่น GRU, 1D-CNN + LSTM, หรือ Transformer เมื่อมีข้อมูลมากพอ
