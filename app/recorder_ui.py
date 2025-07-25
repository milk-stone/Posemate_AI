import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import mediapipe as mp
import threading
import queue


class SquatRecorder:
    """
    A class that uses a Tkinter GUI to record squat posture from a webcam
    and returns the result as a pandas DataFrame. (Thread-based)
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("1. Record Posture (Posemate)")

        # --- Initialize state variables ---
        self.is_recording = False
        self.recorded_rows = []
        self.frame_idx = 0
        self.cap = None
        self.camera_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)

        # --- Initialize MediaPipe ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.setup_gui()

    def setup_gui(self):
        """Sets up the GUI widgets and adjusts the window size."""
        # Set window size to half the monitor's dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = screen_width // 2
        win_height = screen_height // 2
        x_coord = (screen_width // 2) - (win_width // 2)
        y_coord = (screen_height // 2) - (win_height // 2)
        self.root.geometry(f"{win_width}x{win_height}+{x_coord}+{y_coord}")

        # Place the button frame at the bottom of the window first
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=10)

        # The video label will fill the remaining space
        self.video_label = tk.Label(self.root, text="Connecting to camera...", font=("Arial", 16))
        self.video_label.pack(padx=10, pady=10, expand=True, fill="both")

        tk.Button(btn_frame, text="Start Recording", width=15, command=self._start_recording).pack(side=tk.LEFT,
                                                                                                   padx=10)
        tk.Button(btn_frame, text="Stop & Analyze", width=15, command=self._stop_and_analyze).pack(side=tk.LEFT,
                                                                                                   padx=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def _camera_worker(self):
        """(Background thread) Finds a camera and puts frames into the queue."""
        camera_index = next((i for i in range(5) if cv2.VideoCapture(i).isOpened()), -1)

        if camera_index == -1:
            self.frame_queue.put("ERROR_NO_CAM")
            return

        self.cap = cv2.VideoCapture(camera_index)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and self.frame_queue.empty():
                self.frame_queue.put(frame)
        self.cap.release()

    def _update_gui(self):
        """Updates the GUI."""
        try:
            frame = self.frame_queue.get_nowait()
            if isinstance(frame, str) and frame == "ERROR_NO_CAM":
                messagebox.showerror("Camera Error", "Webcam could not be found.")
                self._on_closing()
                return

            image = self.process_frame(frame)
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if w > 1 and h > 1:
                img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")
        except queue.Empty:
            pass
        self.root.after(15, self._update_gui)

    def process_frame(self, frame):
        """Processes a frame to draw landmarks and angles."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            shoulder = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * self.frame_width,
                        lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * self.frame_height]
            hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * self.frame_width,
                   lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * self.frame_height]
            knee = [lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * self.frame_width,
                    lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * self.frame_height]
            ankle = [lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * self.frame_width,
                     lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * self.frame_height]

            hip_angle = self._calculate_angle(shoulder, hip, knee)
            knee_angle = self._calculate_angle(hip, knee, ankle)

            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f'Hip: {int(hip_angle)}', tuple(np.int32(hip)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Knee: {int(knee_angle)}', tuple(np.int32(knee)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            if self.is_recording:
                self.recorded_rows.append(
                    ['user_video', self.frame_idx, 'RIGHT_SHOULDER', shoulder[0], shoulder[1], 'shoulder', None,
                     'user_recording'])
                self.recorded_rows.append(
                    ['user_video', self.frame_idx, 'RIGHT_HIP', hip[0], hip[1], 'hip', round(hip_angle, 2),
                     'user_recording'])
                self.recorded_rows.append(
                    ['user_video', self.frame_idx, 'RIGHT_KNEE', knee[0], knee[1], 'knee', round(knee_angle, 2),
                     'user_recording'])
                self.recorded_rows.append(
                    ['user_video', self.frame_idx, 'RIGHT_ANKLE', ankle[0], ankle[1], 'ankle', None, 'user_recording'])
                self.frame_idx += 1
        return frame

    def _start_recording(self):
        if self.is_recording: return
        self.recorded_rows.clear()
        self.frame_idx = 0
        self.is_recording = True
        messagebox.showinfo("Start", "Recording started. Please perform 2-3 reps and then press 'Stop & Analyze'.")

    def _stop_and_analyze(self):
        if not self.is_recording:
            messagebox.showwarning("Notice", "Recording has not started.")
            return
        self.is_recording = False
        messagebox.showinfo("Finished", "Recording has finished. Closing the window to start analysis.")
        self._on_closing()

    def _on_closing(self):
        self.stop_event.set()
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
        self.root.destroy()

    def start(self):
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()
        self._update_gui()
        self.root.mainloop()

        if not self.recorded_rows:
            return None

        columns = ['idx', 'frame', 'joint', 'x', 'y', 'angle_type', 'angle', 'label']
        return pd.DataFrame(self.recorded_rows, columns=columns)
