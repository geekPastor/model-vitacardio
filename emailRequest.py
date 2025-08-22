import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email(to_email: str, subject: str, content: str):
    try:
        message = Mail(
            from_email="chrinovic.mm@gmail.com",
            to_emails=to_email,
            subject=subject,
            html_content=content
        )
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"[DEBUG] Email envoyé : {response.status_code}")
        return {"status": "sent", "code": response.status_code}
    except Exception as e:
        print("[ERROR] Échec d’envoi :", str(e))
        return {"status": "error", "message": str(e)}
