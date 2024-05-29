import tkinter as tk
from tkinter import Label, LabelFrame, Button
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pyautogui
import threading
import speech_recognition as sr
import subprocess
import pyttsx3
import time
import os

# Constants for eye aspect ratio to indicate 

EYE_AR_THRESH = 0.25  # Adjust this threshold based on calibration
MOUTH_AR_THRESH = 0.7  # Adjust this threshold based on calibration

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Feature and Voice Command Tracking")
        self.root.configure(background="white")

        # Set the window size and position
        self.root.geometry("400x700+0+0")
        self.root.resizable(False, False)

        # Prevent window from being minimized or closed
        self.root.protocol("WM_DELETE_WINDOW", self.prevent_minimize)

        # Initialize the video capture and models
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.stop_event = threading.Event()  # Event to handle thread termination

        # Setup GUI components
        self.lmain = Label(self.root)
        self.lmain.pack(padx=10, pady=10)
        self.frame_controls = LabelFrame(self.root, text="Controls", bd=5, font=('times', 14, 'bold'), bg="light grey")
        self.frame_controls.pack(fill="both", expand="yes", padx=20, pady=20)
        self.start_button = Button(self.frame_controls, text="Start", command=self.start_all, width=15, height=1, font=('times', 15, 'bold'), bg="white")
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.exit_button = Button(self.frame_controls, text="Exit", command=self.stop_all, width=15, height=1, font=('times', 15, 'bold'), bg="red")
        self.exit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Variables for click detection
        self.left_eye_start_time = None
        self.right_eye_start_time = None
        self.mouth_start_time = None
        self.blink_duration = 0.4  # Duration in seconds
        self.mouth_open_duration = 2.0  # Duration in seconds
        self.scrolling_active = False
        self.typing_active = False
        self.talkback_active = False

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Start the camera feed
        self.update_frame()

        # Start voice control for start and exit commands
        self.voice_control_thread = threading.Thread(target=self.voice_control_for_gui)
        self.voice_control_thread.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
        if not self.stop_event.is_set():
            self.root.after(5, self.update_frame)
        else:
            self.root.after(5, self.update_frame)  # Keep updating the frame even if stop_event is set

    def prevent_minimize(self):
        pass  # Do nothing to prevent the window from being minimized or closed

    def start_all(self):
        self.stop_event.clear()
        self.thread_head_tracking = threading.Thread(target=self.head_tracking)
        self.thread_speech_recognition = threading.Thread(target=self.continuous_audio_to_text)
        self.thread_head_tracking.start()
        self.thread_speech_recognition.start()

    def head_tracking(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if not faces:
                continue
            landmarks = self.predictor(gray, faces[0])
            shape = face_utils.shape_to_np(landmarks)
            left_eye_points = shape[36:42]
            right_eye_points = shape[42:48]
            mouth_points = shape[48:60]
            left_eye_center = self.get_eye_center(left_eye_points)
            right_eye_center = self.get_eye_center(right_eye_points)
            self.move_mouse_based_on_eyes(left_eye_center, right_eye_center, frame)
            left_ear = self.eye_aspect_ratio(left_eye_points)
            right_ear = self.eye_aspect_ratio(right_eye_points)
            mouth_ear = self.mouth_aspect_ratio(mouth_points)

            current_time = time.time()

            # Right eye blink triggers left mouse button click
            if right_ear < EYE_AR_THRESH:
                if self.right_eye_start_time is None:
                    self.right_eye_start_time = current_time
                elif current_time - self.right_eye_start_time >= self.blink_duration:
                    pyautogui.click(button='left')
                    print("Left click triggered by right eye blink")
                    self.right_eye_start_time = None
                    self.left_eye_start_time = None  # Reset left eye timer to avoid double click
            else:
                self.right_eye_start_time = None

            # Left eye blink triggers right mouse button click
            if left_ear < EYE_AR_THRESH:
                if self.left_eye_start_time is None:
                    self.left_eye_start_time = current_time
                elif current_time - self.left_eye_start_time >= self.blink_duration:
                    pyautogui.click(button='right')
                    print("Right click triggered by left eye blink")
                    self.left_eye_start_time = None
                    self.right_eye_start_time = None  # Reset right eye timer to avoid double click
            else:
                self.left_eye_start_time = None

            # Mouth open triggers scrolling
            if mouth_ear > MOUTH_AR_THRESH:
                if self.mouth_start_time is None:
                    self.mouth_start_time = current_time
                elif current_time - self.mouth_start_time >= self.mouth_open_duration:
                    if not self.scrolling_active:
                        self.scrolling_active = True
                        print("Scrolling activated")
            else:
                if self.scrolling_active:
                    self.scrolling_active = False
                    print("Scrolling deactivated")
                self.mouth_start_time = None

            # Simulate scrolling
            if self.scrolling_active:
                scroll_speed = int((mouth_ear - MOUTH_AR_THRESH) * 50)  # Adjust multiplier as needed
                pyautogui.scroll(-scroll_speed)  # Scroll down with speed based on mouth opening

    def get_eye_center(self, eye_points):
        x = [p[0] for p in eye_points]
        y = [p[1] for p in eye_points]
        return (int(np.mean(x)), int(np.mean(y)))

    def move_mouse_based_on_eyes(self, left_eye_center, right_eye_center, frame):
        screen_width, screen_height = pyautogui.size()
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
        # Reduced duration for faster mouse movement
        pyautogui.moveTo((1 - eye_center_x / frame.shape[1]) * screen_width,
                         eye_center_y / frame.shape[0] * screen_height, duration=0.05)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[3], mouth[9])  # 51, 59
        B = dist.euclidean(mouth[2], mouth[10])  # 50, 58
        C = dist.euclidean(mouth[4], mouth[8])  # 52, 56
        D = dist.euclidean(mouth[0], mouth[6])  # 48, 54
        return (A + B + C) / (3.0 * D)

    def continuous_audio_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for 1 second to better capture ambient noise
            while not self.stop_event.is_set():
                try:
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=5)  # Set a timeout to avoid long listening periods
                    command = recognizer.recognize_google(audio)
                    print(f"Recognized: {command}")
                    self.process_command(command.lower())  # Convert to lowercase here
                except sr.UnknownValueError:
                    print("[unintelligible]")  # Only print once per error
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
                except sr.WaitTimeoutError:
                    print("Listening timed out while waiting for phrase to start")

    def process_command(self, command):
        if 'start tracker' in command or 'start' in command:
            self.speak("Starting tracker")
            self.start_all()
        elif 'stop tracker' in command or 'stop' in command:
            self.speak("Stopping tracker")
            self.stop_tracking()
        elif 'open' in command:
            if 'chrome' in command:
                self.speak("Opening Chrome")
                subprocess.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'])
            elif 'notepad' in command:
                self.speak("Opening Notepad")
                subprocess.Popen(['notepad.exe'])
            elif 'calculator' in command:
                self.speak("Opening Calculator")
                subprocess.Popen(['calc.exe'])
            else:
                self.speak("Unknown application")
                print("Unknown application")
        elif 'press' in command:
            key = command.replace('press', '').strip()
            pyautogui.press(key)
        elif 'type' in command:
            text = command.replace('type', '').strip()
            pyautogui.write(text)
        elif 'maximize window' in command or 'maximise window' in command:
            pyautogui.hotkey('win', 'up')
            print("Window maximized")
        elif 'minimize window' in command or 'minimise window' in command:
            pyautogui.hotkey('win', 'down')
            print("Window minimized")
        elif 'close window' in command:
            pyautogui.hotkey('alt', 'f4')
            print("Window closed")    
        elif 'exit' in command:
            self.stop_all()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def voice_control_for_gui(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for 1 second to better capture ambient noise
            while not self.stop_event.is_set():
                try:
                    print("Listening for start/exit command...")
                    audio = recognizer.listen(source, timeout=5)  # Set a timeout to avoid long listening periods
                    command = recognizer.recognize_google(audio)
                    print(f"Recognized: {command}")
                    self.process_command(command.lower())  # Process recognized command
                except sr.UnknownValueError:
                    print("[unintelligible]")  # Only print once per error
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
                except sr.WaitTimeoutError:
                    print("Listening timed out while waiting for phrase to start")

    def stop_tracking(self):
        self.stop_event.set()
        if hasattr(self, 'thread_head_tracking'):
            self.thread_head_tracking.join()
        if hasattr(self, 'thread_speech_recognition'):
            self.thread_speech_recognition.join()
        print("Stopped tracking")

    def stop_all(self):
        self.stop_event.set()
        if hasattr(self, 'thread_head_tracking'):
            self.thread_head_tracking.join()
        if hasattr(self, 'thread_speech_recognition'):
            self.thread_speech_recognition.join()
        if hasattr(self, 'voice_control_thread'):
            self.voice_control_thread.join()
        self.cap.release()
        self.root.destroy()
        print("Stopped all")

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
