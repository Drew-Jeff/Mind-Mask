import web_cam_stress as stress
from plyer import notification
import ollama
import random
import threading
import pywhatkit  # <-- The free automation library

CALMING_TIPS = [
    "Hey bud take a breather and chill out.",
    "Looks like you've been working for a long time, take a break, you need it.",
    "Go and have a small walk and get some air.",
    "Drink a glass of water to refresh yourself.",
    "It's time to stretch: Roll your neck gently from side to side.",
    "Close your eyes for 10 seconds and rest, you need it.",
    "Stop what you are doing and rest for 5 mins."
]


class SmartStressManager(stress.UnifiedEmotionStressSystem):
    def __init__(self):
        super().__init__()
        self.last_notification_time = 0

        # Increased cooldown to 45 seconds to prevent browser popup spam
        self.notification_cooldown = 45.0

        self.current_tip = random.choice(CALMING_TIPS)
        self.is_generating = False

        # --- FREE WHATSAPP CONFIGURATION ---
        # Just put the target phone number here (Must include country code)
        self.target_phone = '+91-'

    def generate_ai_tip(self):
        self.is_generating = True
        try:
            response = ollama.chat(model='gpt-oss:120b-cloud', messages=[
                {
                    'role': 'system',
                    'content': 'You are a caring friend. Write ONE short, casual sentence and make it charming and calming(under 15 words) telling someone to relax. No quotes. and do not repeat responses '
                },
            ])
            self.current_tip = response['message']['content']

        except Exception as e:
            print(f"AI/Ollama unavailable: {e}")
            self.current_tip = random.choice(CALMING_TIPS)
        finally:
            self.is_generating = False

    def send_whatsapp_bg(self, full_message):
        """Opens browser in the background to send WhatsApp message for free."""
        try:
            print(f"Opening browser to send free WhatsApp alert to {self.target_phone}...")

            # wait_time=15 gives WhatsApp Web enough time to load
            # tab_close=True automatically closes the browser tab after sending
            pywhatkit.sendwhatmsg_instantly(
                phone_no=self.target_phone,
                message=full_message,
                wait_time=15,
                tab_close=True,
                close_time=3
            )
            print("WhatsApp alert sent and tab closed.")
        except Exception as e:
            print(f"WhatsApp Web Automation Error: {e}")

    def send_system_alert(self, title, message):
        # 1. Send Native Desktop Notification
        try:
            notification.notify(
                title=title,
                message=message,
                app_name='StressGuard AI',
                timeout=5
            )
        except Exception as e:
            print(f"Notification Error: {e}")

        # 2. Send Free WhatsApp Notification via pywhatkit
        if self.target_phone:
            wa_message = f"*{title}*\n{message}"

            # Spawn a thread so opening the browser doesn't freeze the OpenCV video loop
            wa_thread = threading.Thread(target=self.send_whatsapp_bg, args=(wa_message,))
            wa_thread.start()
        else:

            print("Skipping WhatsApp: Target phone number not configured.")
