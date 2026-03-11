import customtkinter as ctk
import cv2
import threading
import time
import random
import ollama
from PIL import Image, ImageTk
import stress_ai as dft

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class StressGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("StressGuard AI - Companion")
        self.geometry("500x700")

        self.running = True
        self.stress_manager = dft.SmartStressManager()

        self.current_stress = 0.0
        self.current_emotion = "Neutral"
        self.current_level = "NORMAL"

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.status_frame = ctk.CTkFrame(self, height=60, fg_color="#1a1a1a")
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Status: SCANNING", font=("Arial", 16, "bold"),
                                       text_color="gray")
        self.lbl_status.pack(side="left", padx=15, pady=10)

        self.lbl_stress = ctk.CTkLabel(self.status_frame, text="Stress: 0%", font=("Arial", 16, "bold"),
                                       text_color="green")
        self.lbl_stress.pack(side="right", padx=15, pady=10)

        self.chat_box = ctk.CTkTextbox(self, font=("Roboto", 14), wrap="word")
        self.chat_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_box.insert("end",
                             "StressGuard: System active. I'm monitoring your stress levels via the camera window.\n\n")
        self.chat_box.configure(state="disabled")

        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.input_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type message...", font=("Roboto", 14))
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self.send_message)

        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", width=60, command=self.send_message)
        self.send_btn.pack(side="right")

        threading.Thread(target=self.video_loop, daemon=True).start()

    def update_ui_status(self, emotion, stress, level):
        self.current_emotion = emotion
        self.current_stress = stress
        self.current_level = level

        self.lbl_status.configure(text=f"Status: {emotion.upper()}")
        self.lbl_stress.configure(text=f"Stress: {stress:.1f}%")

        # --- UPDATE 1: Match new WHO/PSS-10 Terminology ---
        if level == "SEVERE":
            self.lbl_stress.configure(text_color="#FF4444")
        elif level == "WARNING":
            self.lbl_stress.configure(text_color="#FFBB00")
        else:
            self.lbl_stress.configure(text_color="#00CC00")

    def ai_alert_worker_bg(self, emotion, stress_level):
        try:
            prompt = f"""
            Situation: User is at computer.
            Status: Stress is SEVERE ({stress_level:.0f}%) and face looks {emotion}.
            Task: Write a 10-word URGENT notification. 
            Tone: Caring but commanding. No quotes but also make it comforting.
            """
            response = ollama.chat(model='gpt-oss:120b-cloud', messages=[{'role': 'user', 'content': prompt}])

            self.stress_manager.send_system_alert(
                title=f"⚠️ Severe Stress ({stress_level:.0f}%)",
                message=response['message']['content']
            )
        except Exception:
            self.stress_manager.send_system_alert(title="High Stress", message=random.choice(dft.CALMING_TIPS))

    def video_loop(self):
        cap = cv2.VideoCapture(0)

        cached_color = (0, 255, 0)
        cached_level = "NORMAL"

        last_analysis_time = 0
        ai_interval = 7.0

        while self.running:
            ret, frame = cap.read()
            if not ret: break

            output_frame, faces_data = self.stress_manager.preprocessor.process_frame(frame)

            for face in faces_data:
                x, y, w, h = face['coords']

                if face['emotion_ready']:

                    # RUN AI EVERY 7 SECONDS
                    if (time.time() - last_analysis_time > ai_interval):
                        last_analysis_time = time.time()

                        emo, stress = self.stress_manager.analyze_face(face['image'])
                        self.stress_manager.stress_buffer.append(stress)
                        avg_stress = sum(self.stress_manager.stress_buffer) / len(self.stress_manager.stress_buffer)

                        self.current_stress = avg_stress
                        self.current_emotion = emo

                        # --- UPDATE 2: Match new WHO/PSS-10 Terminology ---
                        if avg_stress >= self.stress_manager.level_3_threshold:
                            cached_level = "SEVERE"
                            cached_color = (0, 0, 255)
                            if (
                                    time.time() - self.stress_manager.last_notification_time > self.stress_manager.notification_cooldown):
                                self.stress_manager.last_notification_time = time.time()
                                threading.Thread(target=self.ai_alert_worker_bg, args=(emo, avg_stress)).start()

                        elif avg_stress >= self.stress_manager.level_2_threshold:
                            cached_level = "WARNING"
                            cached_color = (0, 200, 255)
                        else:
                            cached_level = "NORMAL"
                            cached_color = (0, 255, 0)

                        self.update_ui_status(emo, avg_stress, cached_level)

                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), cached_color, 2)
                    label = f"{self.current_emotion.upper()} | {self.current_stress:.0f}%"
                    cv2.rectangle(output_frame, (x, y - 40), (x + w, y), cached_color, -1)
                    cv2.putText(output_frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("StressGuard Vision Feed", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on_closing()
                break

        cap.release()
        cv2.destroyAllWindows()

    def send_message(self, event=None):
        msg = self.input_entry.get()
        if not msg: return

        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"You: {msg}\n\n")
        self.chat_box.configure(state="disabled")
        self.input_entry.delete(0, "end")

        threading.Thread(target=self.generate_ai_reply, args=(msg,)).start()

    def generate_ai_reply(self, user_msg):
        try:
            # --- UPDATE 3: Fixed prompt to match the new 67% threshold ---
            prompt = f"""
            ROLE: Empathetic AI therapist.
            USER INPUT: "{user_msg}"
            USER VISUAL STATE: Emotion={self.current_emotion}, Stress={self.current_stress:.1f}% ({self.current_level})

            INSTRUCTIONS:
            1. If Stress > 66% (SEVERE): Ignore "I'm fine" text. Prioritize health.
            2. If Normal: Respond naturally.
            3. Keep it brief (max 2 sentences).
            """

            self.chat_box.configure(state="normal")
            current_status = f"[Emotion: {self.current_emotion.upper()}]"
            self.chat_box.insert("end", f"AI {current_status}: ")
            self.chat_box.see("end")

            # Streaming response for sub-1 second latency
            stream = ollama.chat(model='gpt-oss:120b-cloud', messages=[{'role': 'user', 'content': prompt}],
                                 stream=True)

            for chunk in stream:
                text_chunk = chunk['message']['content']
                self.chat_box.insert("end", text_chunk)
                self.chat_box.see("end")

            self.chat_box.insert("end", "\n" + "-" * 40 + "\n")
            self.chat_box.configure(state="disabled")

        except Exception as e:
            print(f"Chat Error: {e}")
            self.chat_box.configure(state="disabled")

    def on_closing(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    app = StressGuardApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()