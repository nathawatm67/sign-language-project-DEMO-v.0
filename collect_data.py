"""
=======================================================
  Dataset Collection — Thai Sign Language
  เก็บ dataset สำหรับ LSTM model
=======================================================
วิธีใช้:  python collect_data.py
ปุ่ม:    1/2/3 = เลือก gesture   SPACE = บันทึก   Q = ออก
เป้าหมาย: 30 sequences/gesture  →  data/{gesture}/{0000}.npy
=======================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

CAM_INDEX       = 1
SEQUENCE_LENGTH = 30
TARGET_SEQ      = 90
GESTURES        = ["sawatdi", "khob_khun", "kho_thot"]

FONT  = cv2.FONT_HERSHEY_DUPLEX
WHITE = (235, 235, 235)
GREEN = (40, 200, 110)
YELLOW= (20, 210, 230)
RED   = (60, 60, 255)
GRAY  = (110, 110, 110)
BLACK = (0, 0, 0)
BG    = (18, 18, 24)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.60, min_tracking_confidence=0.50)


def extract_both_hands(res):
    """Return flat array of shape (126,): Left hand (63) + Right hand (63).
    Missing hand is filled with zeros so shape is always consistent."""
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    if res.multi_hand_landmarks:
        for hlm, hinfo in zip(res.multi_hand_landmarks, res.multi_handedness):
            arr = np.array([[l.x, l.y, l.z] for l in hlm.landmark],
                           dtype=np.float32).flatten()
            if hinfo.classification[0].label == "Left":
                left = arr
            else:
                right = arr
    return np.concatenate([left, right])   # (126,)


def count_saved(gesture):
    folder = os.path.join("data", gesture)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def next_seq_num(gesture):
    folder = os.path.join("data", gesture)
    if not os.path.exists(folder):
        return 0
    nums = []
    for f in os.listdir(folder):
        name = os.path.splitext(f)[0]
        if name.isdigit():
            nums.append(int(name))
    return max(nums) + 1 if nums else 0


def tx(fr, text, pos, scale, color, thick=1):
    cv2.putText(fr, text, pos, FONT, scale, BLACK,  thick + 3, cv2.LINE_AA)
    cv2.putText(fr, text, pos, FONT, scale, color,  thick,     cv2.LINE_AA)


def ov(fr, x1, y1, x2, y2, a=0.78):
    ol = fr.copy()
    cv2.rectangle(ol, (x1, y1), (x2, y2), BG, -1)
    cv2.addWeighted(ol, a, fr, 1 - a, 0, fr)


def draw_ui(frame, gesture, count, recording, seq_len, hand_ok, saved_flash):
    H, W = frame.shape[:2]

    # ── Top bar ──
    ov(frame, 0, 0, W, 82)
    tx(frame, f"Gesture: {gesture}", (20, 36), 0.85, GREEN, 2)
    pct = int(count / TARGET_SEQ * 100)
    tx(frame, f"{count} / {TARGET_SEQ} sequences  ({pct}%)", (20, 66), 0.58, WHITE, 1)

    # ── Progress bar (top) ──
    bw = int((W - 40) * min(count / TARGET_SEQ, 1.0))
    cv2.rectangle(frame, (20, 74), (W - 20, 80), (45, 45, 55), -1)
    cv2.rectangle(frame, (20, 74), (20 + bw, 80),
                  GREEN if count >= TARGET_SEQ else YELLOW, -1)

    # ── Bottom bar ──
    ov(frame, 0, H - 88, W, H)
    cv2.line(frame, (0, H - 88), (W, H - 88), GRAY, 1)

    if recording:
        prog = int((W - 40) * seq_len / SEQUENCE_LENGTH)
        cv2.rectangle(frame, (20, H - 84), (W - 20, H - 68), (45, 45, 55), -1)
        cv2.rectangle(frame, (20, H - 84), (20 + prog, H - 68), RED, -1)
        tx(frame, f"RECORDING  {seq_len} / {SEQUENCE_LENGTH} frames",
           (20, H - 50), 0.70, RED, 2)
    else:
        status = "Hand detected  — press SPACE to record" if hand_ok \
                 else "No hand visible"
        col = YELLOW if hand_ok else GRAY
        tx(frame, status, (20, H - 54), 0.58, col, 1)

    tx(frame, "1=sawatdi   2=khob_khun   3=kho_thot   |   SPACE=record   Q=quit",
       (20, H - 18), 0.44, GRAY, 1)

    # ── SAVED flash (1 s green overlay) ──
    if time.time() - saved_flash < 1.0:
        ol = frame.copy()
        cv2.rectangle(ol, (0, 0), (W, H), (20, 160, 60), -1)
        cv2.addWeighted(ol, 0.22, frame, 0.78, 0, frame)
        tx(frame, "SAVED!", (W // 2 - 100, H // 2 + 20), 2.2, GREEN, 3)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open camera (index={CAM_INDEX})")
        print("  Change CAM_INDEX = 1 or 2 and retry\n")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gesture_idx = 0
    recording   = False
    sequence    = []
    saved_flash = 0.0

    print("=" * 56)
    print("  Dataset Collection — Thai Sign Language")
    print("=" * 56)
    print("  1=sawatdi   2=khob_khun   3=kho_thot")
    print("  SPACE=record  Q=quit")
    print("=" * 56)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Cannot read camera frame")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = detector.process(rgb)

        gesture   = GESTURES[gesture_idx]
        hand_ok   = False
        landmarks = None

        if res.multi_hand_landmarks:
            for hlm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())
            hand_ok = True

        landmarks = extract_both_hands(res)   # (126,) — zeros if hand absent

        # ── Recording logic ──
        if recording:
            sequence.append(landmarks)

            if len(sequence) >= SEQUENCE_LENGTH:
                recording = False
                arr    = np.array(sequence, dtype=np.float32)   # (30, 126)
                folder = os.path.join("data", gesture)
                os.makedirs(folder, exist_ok=True)
                path   = os.path.join(folder, f"{next_seq_num(gesture):04d}.npy")
                np.save(path, arr)
                saved = count_saved(gesture)
                print(f"  [SAVED] {path}  shape={arr.shape}  total={saved}/{TARGET_SEQ}")
                sequence    = []
                saved_flash = time.time()

        draw_ui(frame, gesture, count_saved(gesture),
                recording, len(sequence), hand_ok, saved_flash)

        cv2.imshow("Collect Data — Thai Sign Language", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord('1'):
            gesture_idx = 0; sequence = []; recording = False
            print(f"  >> sawatdi  ({count_saved('sawatdi')}/{TARGET_SEQ})")
        elif k == ord('2'):
            gesture_idx = 1; sequence = []; recording = False
            print(f"  >> khob_khun  ({count_saved('khob_khun')}/{TARGET_SEQ})")
        elif k == ord('3'):
            gesture_idx = 2; sequence = []; recording = False
            print(f"  >> kho_thot  ({count_saved('kho_thot')}/{TARGET_SEQ})")
        elif k == ord(' ') and not recording:
            recording = True
            sequence  = []
            print(f"  Recording #{count_saved(gesture) + 1} for [{gesture}]...")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Done. Saved files:")
    for g in GESTURES:
        print(f"    {g}: {count_saved(g)}/{TARGET_SEQ} sequences")


if __name__ == "__main__":
    main()
