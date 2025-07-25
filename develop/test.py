import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import time
from datetime import datetime
import mediapipe as mp

# 전역 상태 변수
recording = False
rows = []
frame_idx = 0
label = "user_recording"

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# CSV에 저장할 열 구조
columns = ['frame', 'joint', 'x', 'y', 'angle_type', 'angle', 'label']

# 각도 계산 함수
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# 웹캠 연결
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 프레임 처리 함수
def update_frame():
    global frame_idx, rows, recording

    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    image = frame.copy()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 좌표 추출
        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height]
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width,
               lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_height]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame_width,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame_height]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame_width,
                 lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame_height]

        # 각도 계산
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)

        # 텍스트 표시
        cv2.putText(image, f'Hip: {int(hip_angle)}', tuple(np.int32(hip)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f'Knee: {int(knee_angle)}', tuple(np.int32(knee)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # CSV 데이터 기록
        if recording:
            rows.append([frame_idx, 'RIGHT_SHOULDER', shoulder[0], shoulder[1], 'none', None, label])
            rows.append([frame_idx, 'RIGHT_HIP', hip[0], hip[1], 'hip', round(hip_angle, 2), label])
            rows.append([frame_idx, 'RIGHT_KNEE', knee[0], knee[1], 'knee', round(knee_angle, 2), label])

    # Tkinter 이미지 출력
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    frame_idx += 1
    root.after(10, update_frame)

# 버튼 기능
def start_recording():
    global recording, rows, frame_idx
    if recording:
        return
    rows.clear()
    frame_idx = 0
    recording = True
    messagebox.showinfo("시작", "자세 분석 시작됨")

def stop_and_save():
    global recording, rows
    if not recording:
        return
    recording = False
    if not rows:
        messagebox.showinfo("알림", "저장할 데이터가 없습니다.")
        return

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"angle_data_{now}.csv"
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filename, index=False)
    messagebox.showinfo("완료", f"CSV 저장됨: {filename}")

def on_closing():
    global cap
    if messagebox.askokcancel("종료", "프로그램을 종료할까요?"):
        cap.release()
        root.destroy()

# Tkinter UI 구성
root = tk.Tk()
root.title("트링커 자세 분석기")
root.protocol("WM_DELETE_WINDOW", on_closing)

video_label = tk.Label(root)
video_label.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="시작", width=15, command=start_recording).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="종료 및 저장", width=15, command=stop_and_save).pack(side=tk.LEFT, padx=10)

# ✅ 프레임 업데이트 시작
update_frame()

# ✅ Tkinter 루프 실행
root.mainloop()
