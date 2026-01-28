

sender_email = 'kingspan.miami.vision.system@gmail.com'
app_password = 'mujj keze islt kzol'  # App password, not your real one
recipient_email = 'mark.klimek@kingspan.com'
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg["Subject"] = "SMTP Test (Gmail)"
msg["From"] = sender_email
msg["To"] = recipient_email
msg.set_content("This is a plain test message sent via Gmail SMTP.")

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)
    print("✅ Email sent successfully.")
except Exception as e:
    print("❌ Failed to send email:", e)