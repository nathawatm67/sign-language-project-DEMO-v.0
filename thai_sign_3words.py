"""
=======================================================
  Smart Thai Sign Language Translator
  ระบบแปลภาษามือไทย 3 คำพื้นฐาน
  สวัสดี | ขอบคุณ | ขอโทษ
=======================================================
ต้องการ:
  pip install opencv-python mediapipe==0.10.13 numpy

วิธีใช้:  python thai_sign_3words.py
ปุ่มลัด: Q=ออก  H=landmark  S=ภาพ  D=debug
=======================================================
"""

import cv2, mediapipe as mp, numpy as np, time, sys, json
from collections import deque

CAM_INDEX   = 1
SHOW_LM     = True
DEBUG       = False
SMOOTH_WIN  = 6
SMOOTH_MIN  = 3
HOLD_SECS   = 2.0
FONT        = cv2.FONT_HERSHEY_DUPLEX

BG=(18,18,24); WHITE=(235,235,235); GRAY=(110,110,110); BLACK=(0,0,0)
GREEN=(40,200,110); YELLOW=(20,210,230); ORANGE=(30,140,255)

SEQUENCE_LENGTH = 30
FEATURES        = 126   # 2 hands × 21 × 3
GESTURES        = ["sawatdi", "khob_khun", "kho_thot"]
CONF_THRESHOLD  = 0.55
PREDICT_EVERY   = 2     # predict ทุก N frames เพื่อเพิ่ม FPS
MODEL_PATH      = "model/lstm_model.h5"

try:
    import tensorflow as tf
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  [LSTM] Model loaded: {MODEL_PATH}")
    USE_LSTM = True
except Exception as e:
    print(f"  [FALLBACK] Rule-based mode ({e})")
    lstm_model = None
    USE_LSTM   = False

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.60, min_tracking_confidence=0.50)

TIPS=[4,8,12,16,20]; DIPS=[3,7,11,15,19]; MCPS=[2,5,9,13,17]

def lm_np(lm): return np.array([[l.x,l.y,l.z] for l in lm.landmark])

def extract_both_hands(res):
    """Left(63) + Right(63) = 126 floats. Missing hand → zeros."""
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    if res.multi_hand_landmarks:
        for hlm, hinfo in zip(res.multi_hand_landmarks, res.multi_handedness):
            arr = np.array([[l.x,l.y,l.z] for l in hlm.landmark],
                           dtype=np.float32).flatten()
            if hinfo.classification[0].label == "Left":
                left = arr
            else:
                right = arr
    return np.concatenate([left, right])

def fingers_up(lm, side):
    up=[]
    up.append(1 if (lm[4][0]<lm[3][0] if side=="Right" else lm[4][0]>lm[3][0]) else 0)
    for i in range(1,5): up.append(1 if lm[TIPS[i]][1]<lm[DIPS[i]][1] else 0)
    return up

def finger_spread(lm): return float(np.linalg.norm(lm[8][:2]-lm[20][:2]))

def classify(lm, side):
    up=fingers_up(lm,side); total=sum(up)
    spread=finger_spread(lm); wy=float(lm[0][1])
    # สวัสดี: นิ้ว>=4 ขึ้น + กางนิ้ว
    if total>=4 and spread>0.14:
        conf=round(min(total/5,1.0)*0.35+min(spread/0.22,1.0)*0.65,2)
        return ("sawatdi",min(conf,0.97),"แบมือ กางนิ้ว → สวัสดี")
    # ขอโทษ: กำมือ+โป้งพับ
    if total==0 and up[0]==0:
        return ("kho_thot",0.90,"กำมือแน่น โป้งพับ → ขอโทษ")
    # ขอบคุณ: โป้งขึ้น+ยกสูง (wrist_y < 0.62)
    if up[0]==1 and total<=2 and wy<0.62:
        conf=round(0.40+min((0.62-wy)/0.25,1.0)*0.35+min((2-total)/2,1.0)*0.25,2)
        return ("khob_khun",min(conf,0.95),"โป้งชู ยกมือสูง → ขอบคุณ")
    return (None,0.0,"ยังไม่พบท่า")

INFO={
    "sawatdi":   {"th":"สวัสดี",  "en":"HELLO / SAWASDEE",      "color":GREEN,
                  "steps":["1. Open palm, spread all 5 fingers","2. Raise hand to forehead level","3. Face palm outward"]},
    "khob_khun": {"th":"ขอบคุณ", "en":"THANK YOU / KHOB KHUN",  "color":YELLOW,
                  "steps":["1. Loose fist","2. Extend thumb upward","3. Raise hand to mouth/chin level"]},
    "kho_thot":  {"th":"ขอโทษ",  "en":"SORRY / KHO THOT",       "color":ORANGE,
                  "steps":["1. Make a tight fist","2. Fold thumb inside fist","3. Rotate fist at chest"]},
}

class Smoother:
    def __init__(self):
        self.buf=[];self.stable=(None,0.0,"");self._ts=time.time()
    def push(self,g,c,h):
        self.buf.append((g,c,h))
        if len(self.buf)>SMOOTH_WIN: self.buf.pop(0)
        counts={}
        for gi,ci,hi in self.buf:
            if gi: counts[gi]=counts.get(gi,0)+1
        if counts:
            best=max(counts,key=counts.get)
            if counts[best]>=SMOOTH_MIN:
                avg_c=float(np.mean([ci for gi,ci,hi in self.buf if gi==best]))
                hint=next((hi for gi,ci,hi in self.buf if gi==best),"")
                if self.stable[0]!=best:
                    self.stable=(best,round(avg_c,2),hint);self._ts=time.time()
                return self.stable
        if self.stable[0] and (time.time()-self._ts)<HOLD_SECS: return self.stable
        self.stable=(None,0.0,h);return self.stable

def tx(fr,text,pos,scale,color,thick=1):
    cv2.putText(fr,text,pos,FONT,scale,BLACK,thick+3,cv2.LINE_AA)
    cv2.putText(fr,text,pos,FONT,scale,color,thick,cv2.LINE_AA)

def ov(fr,x1,y1,x2,y2,a=0.78):
    ol=fr.copy();cv2.rectangle(ol,(x1,y1),(x2,y2),BG,-1);cv2.addWeighted(ol,a,fr,1-a,0,fr)

def draw_result(fr,gesture,conf,hint):
    H,W=fr.shape[:2];ph=140
    ov(fr,0,H-ph,W,H)
    cv2.line(fr,(0,H-ph),(W,H-ph),INFO[gesture]["color"] if gesture else GRAY,2)
    if gesture:
        info=INFO[gesture];col=info["color"]
        tx(fr,info["en"],(24,H-ph+40),1.15,col,2)
        for i,s in enumerate(info["steps"]): tx(fr,s,(24,H-ph+72+i*24),0.50,WHITE,1)
        BX1,BX2,BY1,BY2=24,W-24,H-28,H-10
        cv2.rectangle(fr,(BX1,BY1),(BX2,BY2),(45,45,55),-1)
        bw=int((BX2-BX1)*conf)
        cv2.rectangle(fr,(BX1,BY1),(BX1+bw,BY2),GREEN if conf>0.80 else YELLOW,-1)
        cv2.rectangle(fr,(BX1,BY1),(BX2,BY2),(80,80,90),1)
        tx(fr,f"Confidence  {conf:.0%}",(BX1+4,BY2-2),0.42,GRAY,1)
    else:
        tx(fr,"-- No gesture detected --",(24,H-ph+50),0.85,GRAY,1)
        tx(fr,hint,(24,H-ph+84),0.52,GRAY,1)
        tx(fr,"Try:  HELLO  |  THANK YOU  |  SORRY",(24,H-ph+116),0.50,GRAY,1)

def draw_guide(fr,active):
    H,W=fr.shape[:2];PW=278;PX=W-PW-10;PY=10
    rows=[
        ("THAI SIGN GUIDE",WHITE,0.52,True),("",WHITE,0.4,False),
        ("[HELLO]  Sawasdee",GREEN,0.50,False),("  Open palm 5 fingers",WHITE,0.42,False),("",WHITE,0.4,False),
        ("[THANK YOU]  Khob Khun",YELLOW,0.50,False),("  Thumbs-up, raise high",WHITE,0.42,False),("",WHITE,0.4,False),
        ("[SORRY]  Kho Thot",ORANGE,0.50,False),("  Tight fist, thumb inside",WHITE,0.42,False),("",WHITE,0.4,False),
        ("Q=quit  H=landmarks",GRAY,0.42,False),("S=screenshot  D=debug",GRAY,0.42,False),
    ]
    LH=22;PH=len(rows)*LH+20
    ov(fr,PX-8,PY,PX+PW,PY+PH,a=0.72);cv2.rectangle(fr,(PX-8,PY),(PX+PW,PY+PH),GRAY,1)
    for i,(text,col,scale,bold) in enumerate(rows):
        if not text: continue
        if active:
            en=INFO.get(active,{}).get("en","").split("/")[0].strip()
            if en and en in text: col=INFO[active]["color"]
        tx(fr,text,(PX,PY+20+i*LH),scale,col,2 if bold else 1)

def draw_steps(fr,gesture):
    if not gesture: return
    info=INFO[gesture];col=info["color"]
    ov(fr,8,44,340,44+len(info["steps"])*28+36,a=0.70)
    tx(fr,info["en"],(16,68),0.60,col,1)
    for i,s in enumerate(info["steps"]): tx(fr,s,(16,100+i*28),0.50,WHITE,1)

def draw_debug(fr,lm,side):
    H=fr.shape[0];up=fingers_up(lm,side);sp=finger_spread(lm);wy=float(lm[0][1])
    lines=[f"fingers:{up} sum={sum(up)}",f"spread:{sp:.3f}",f"wrist_y:{wy:.3f}",f"side:{side}"]
    ov(fr,8,H-220,340,H-140,a=0.75)
    for i,line in enumerate(lines): tx(fr,line,(16,H-204+i*20),0.46,YELLOW,1)
    print(f"[DEBUG] fingers:{up} sum={sum(up)}  spread:{sp:.3f}  wrist_y:{wy:.3f}  side:{side}")

def main():
    global SHOW_LM,DEBUG
    cap=cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"\n[ERROR] เปิดกล้องไม่ได้ (index={CAM_INDEX})")
        print("  แก้ CAM_INDEX = 1 หรือ 2 ในไฟล์แล้วลองใหม่\n"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720); cap.set(cv2.CAP_PROP_FPS,30)
    smoother=Smoother(); prev_t=time.time(); shot_n=0
    lstm_buf=deque(maxlen=SEQUENCE_LENGTH); frame_count=0
    print("="*56)
    print("  Smart Thai Sign Language  —  3 Basic Words")
    print("="*56)
    print("  สวัสดี  /  ขอบคุณ  /  ขอโทษ")
    print("  Q=ออก  H=landmark  S=ภาพ  D=debug")
    print("="*56)
    while True:
        ok,frame=cap.read()
        if not ok: print("[ERROR] อ่านกล้องไม่ได้"); break
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=detector.process(rgb)
        now=time.time(); fps=1.0/max(now-prev_t,1e-9); prev_t=now
        rg,rc,rh=None,0.0,"Show hand to camera"

        # วาด landmarks ทุกมือที่เห็น
        if res.multi_hand_landmarks:
            for hlm,hinfo in zip(res.multi_hand_landmarks,res.multi_handedness):
                if SHOW_LM:
                    mp_draw.draw_landmarks(frame,hlm,mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style())
                if DEBUG:
                    side=hinfo.classification[0].label; lm=lm_np(hlm)
                    draw_debug(frame,lm,side)

        if USE_LSTM:
            # LSTM inference — sliding window 30 frames
            lstm_buf.append(extract_both_hands(res))
            frame_count+=1
            if len(lstm_buf)==SEQUENCE_LENGTH and frame_count%PREDICT_EVERY==0:
                try:
                    seq=np.array(lstm_buf,dtype=np.float32)[np.newaxis]
                    probs=lstm_model.predict(seq,verbose=0)[0]
                    idx=int(np.argmax(probs)); conf=float(probs[idx])
                    rh=f"Model: {GESTURES[idx]} ({conf:.0%})"
                    if conf>=CONF_THRESHOLD:
                        rg=GESTURES[idx]; rc=conf; rh=""
                except Exception as e:
                    print(f"[LSTM ERROR] {e}")
                    rh=f"LSTM error: {e}"
        else:
            # Fallback: rule-based (ใช้มือแรกที่เห็น)
            if res.multi_hand_landmarks:
                for hlm,hinfo in zip(res.multi_hand_landmarks,res.multi_handedness):
                    side=hinfo.classification[0].label; lm=lm_np(hlm)
                    rg,rc,rh=classify(lm,side); break
        gesture,conf,hint=smoother.push(rg,rc,rh)
        draw_result(frame,gesture,conf,hint)
        draw_guide(frame,gesture)
        mode_text = f"LSTM  FPS {fps:.0f}" if USE_LSTM else f"RULE  FPS {fps:.0f}"
        mode_col  = GREEN if USE_LSTM else YELLOW
        tx(frame,mode_text,(frame.shape[1]-180,28),0.52,mode_col,1)
        cv2.imshow("Thai Sign Language",frame)
        k=cv2.waitKey(1)&0xFF
        if k in (ord('q'),27): break
        elif k==ord('h'): SHOW_LM=not SHOW_LM; print(f"  Landmarks: {'ON' if SHOW_LM else 'OFF'}")
        elif k==ord('d'): DEBUG=not DEBUG; print(f"  Debug: {'ON' if DEBUG else 'OFF'}")
        elif k==ord('s'):
            shot_n+=1; fn=f"shot_{shot_n:03d}.png"; cv2.imwrite(fn,frame); print(f"  [saved] {fn}")
    cap.release(); cv2.destroyAllWindows(); print("\nปิดโปรแกรมแล้ว")

if __name__=="__main__":
    main()
