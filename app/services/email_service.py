import logging

import aiosmtplib
from email.message import EmailMessage

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmailService:
    """Send verification/reset emails via SMTP; fall back to mock logging when SMTP is unset."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def _smtp_enabled(self) -> bool:
        return bool(self.settings.smtp_host)

    async def _send(self, to: str, subject: str, body: str) -> None:
        if not self._smtp_enabled:
            logger.info("[mock-email] to=%s subject=%s body=%s", to, subject, body)
            return

        message = EmailMessage()
        message["From"] = self.settings.smtp_from
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

        await aiosmtplib.send(
            message,
            hostname=self.settings.smtp_host,
            port=self.settings.smtp_port,
            username=self.settings.smtp_user or None,
            password=self.settings.smtp_password or None,
            start_tls=self.settings.smtp_use_tls,
        )

    async def send_verification_email(self, email: str, token: str) -> None:
        base = self.settings.email_verify_web_base.rstrip("/")
        link = f"{base}/verify?token={token}"
        body = (
            "Welcome to LR-Agent!\n\n"
            f"Please verify your email by opening this link:\n{link}\n\n"
            "This link expires in 24 hours."
        )
        await self._send(email, "Verify your LR-Agent email", body)

    async def send_password_reset_email(self, email: str, token: str) -> None:
        link = f"{self.settings.email_deep_link_base}reset-password?token={token}"
        body = (
            "You requested a password reset for your LR-Agent account.\n\n"
            f"Reset your password using this link:\n{link}\n\n"
            "This link expires in 1 hour. If you did not request this, ignore this email."
        )
        await self._send(email, "Reset your LR-Agent password", body)
