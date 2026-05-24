import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import (
    create_token,
    generate_opaque_token,
    hash_password,
    hash_token,
    verify_password,
    verify_token,
)
from app.models.user_session import UserSession
from app.schemas.user import UserPublic
from app.services.email_service import EmailService
from app.services.user_service import UserService

logger = logging.getLogger(__name__)

REFRESH_KEY_PREFIX = "lr:auth:refresh:"
REFRESH_REVOKED_PREFIX = "lr:auth:refresh:revoked:"
USER_SESSIONS_PREFIX = "lr:auth:user_sessions:"
VERIFY_EMAIL_PREFIX = "lr:auth:verify:email:"
RESET_PWD_PREFIX = "lr:auth:reset:pwd:"


class AuthService:
    def __init__(self, db: AsyncSession, redis: Redis, settings: Settings) -> None:
        self.db = db
        self.redis = redis
        self.settings = settings
        self.users = UserService(db)
        self.email = EmailService(settings)

    async def register(self, email: str, password: str) -> None:
        existing = await self.users.get_by_email(email)
        if existing is not None:
            raise ValueError("email_already_registered")

        user = await self.users.create(
            email=email,
            password_hash=hash_password(password),
        )

        token = generate_opaque_token()
        await self.redis.setex(
            f"{VERIFY_EMAIL_PREFIX}{hash_token(token)}",
            timedelta(hours=24),
            str(user.id),
        )
        await self.email.send_verification_email(email, token)
        logger.info("auth.register user_id=%s email=%s", user.id, email)

    async def resend_verification_email(self, user_id: UUID) -> None:
        user = await self.users.get_by_id(user_id)
        if user is None or not user.is_active:
            raise ValueError("invalid_credentials")

        if user.email_verified:
            raise ValueError("email_already_verified")

        token = generate_opaque_token()
        await self.redis.setex(
            f"{VERIFY_EMAIL_PREFIX}{hash_token(token)}",
            timedelta(hours=24),
            str(user.id),
        )
        await self.email.send_verification_email(user.email, token)
        logger.info("auth.resend_verification user_id=%s", user.id)

    async def verify_email(self, token: str) -> None:
        user_id = await self.redis.get(f"{VERIFY_EMAIL_PREFIX}{hash_token(token)}")
        if not user_id:
            raise ValueError("invalid_or_expired_token")

        user = await self.users.get_by_id(UUID(user_id))
        if user is None or not user.is_active:
            raise ValueError("invalid_or_expired_token")

        user.email_verified = True
        await self.redis.delete(f"{VERIFY_EMAIL_PREFIX}{hash_token(token)}")
        logger.info("auth.verify_email user_id=%s", user.id)

    async def login(
        self,
        email: str,
        password: str,
        *,
        ip_address: str | None = None,
        device_info: str | None = None,
    ) -> dict:
        user = await self.users.get_by_email(email)
        if user is None or not verify_password(password, user.password_hash):
            raise ValueError("invalid_credentials")

        if not user.is_active:
            raise ValueError("invalid_credentials")

        access_token, _ = create_token(
            subject=str(user.id),
            token_type="access",
            expires_delta=timedelta(minutes=self.settings.access_token_expire_minutes),
            settings=self.settings,
        )
        refresh_token, jti = create_token(
            subject=str(user.id),
            token_type="refresh",
            expires_delta=timedelta(days=self.settings.refresh_token_expire_days),
            settings=self.settings,
        )
        ttl = timedelta(days=self.settings.refresh_token_expire_days)
        await self._store_refresh_session(
            user_id=str(user.id),
            jti=jti,
            ttl=ttl,
            ip_address=ip_address,
            device_info=device_info,
        )

        user.last_login_at = datetime.now(timezone.utc)
        logger.info("auth.login user_id=%s ip=%s", user.id, ip_address)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": self.settings.access_token_expire_minutes * 60,
            "user": UserPublic.model_validate(user),
        }

    async def refresh(self, refresh_token: str) -> dict:
        payload = verify_token(refresh_token, expected_type="refresh", settings=self.settings)
        jti = payload.get("jti")
        user_id = payload.get("sub")
        if not jti or not user_id:
            raise ValueError("invalid_token")

        revoked = await self.redis.get(f"{REFRESH_REVOKED_PREFIX}{jti}")
        if revoked:
            await self._revoke_all_sessions(user_id)
            raise ValueError("token_reuse_detected")

        stored_user_id = await self.redis.get(f"{REFRESH_KEY_PREFIX}{jti}")
        if stored_user_id != user_id:
            raise ValueError("invalid_token")

        await self._revoke_refresh_session(user_id, jti)

        user = await self.users.get_by_id(UUID(user_id))
        if user is None or not user.is_active:
            raise ValueError("invalid_credentials")

        access_token, _ = create_token(
            subject=str(user.id),
            token_type="access",
            expires_delta=timedelta(minutes=self.settings.access_token_expire_minutes),
            settings=self.settings,
        )
        new_refresh_token, new_jti = create_token(
            subject=str(user.id),
            token_type="refresh",
            expires_delta=timedelta(days=self.settings.refresh_token_expire_days),
            settings=self.settings,
        )
        ttl = timedelta(days=self.settings.refresh_token_expire_days)
        await self._store_refresh_session(user_id=str(user.id), jti=new_jti, ttl=ttl)

        logger.info("auth.refresh user_id=%s old_jti=%s new_jti=%s", user.id, jti, new_jti)

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "expires_in": self.settings.access_token_expire_minutes * 60,
            "user": UserPublic.model_validate(user),
        }

    async def logout(self, refresh_token: str) -> None:
        try:
            payload = verify_token(refresh_token, expected_type="refresh", settings=self.settings)
        except ValueError:
            return

        jti = payload.get("jti")
        user_id = payload.get("sub")
        if jti and user_id:
            await self._revoke_refresh_session(user_id, jti)
            logger.info("auth.logout user_id=%s jti=%s", user_id, jti)

    async def revoke_all_sessions(self, user_id: str) -> None:
        await self._revoke_all_sessions(user_id)
        logger.info("auth.revoke_all_sessions user_id=%s", user_id)

    async def delete_account(self, user_id: UUID, password: str) -> None:
        user = await self.users.get_by_id(user_id)
        if user is None or not user.is_active:
            raise ValueError("invalid_credentials")

        if not verify_password(password, user.password_hash):
            raise ValueError("invalid_credentials")

        await self._revoke_all_sessions(str(user.id))
        await self.users.deactivate_account(user)
        logger.info("auth.delete_account user_id=%s", user.id)

    async def forgot_password(self, email: str) -> None:
        user = await self.users.get_by_email(email)
        if user is None:
            return

        token = generate_opaque_token()
        await self.redis.setex(
            f"{RESET_PWD_PREFIX}{hash_token(token)}",
            timedelta(hours=1),
            str(user.id),
        )
        await self.email.send_password_reset_email(email, token)
        logger.info("auth.forgot_password user_id=%s", user.id)

    async def reset_password(self, token: str, password: str) -> None:
        user_id = await self.redis.get(f"{RESET_PWD_PREFIX}{hash_token(token)}")
        if not user_id:
            raise ValueError("invalid_or_expired_token")

        user = await self.users.get_by_id(UUID(user_id))
        if user is None:
            raise ValueError("invalid_or_expired_token")

        user.password_hash = hash_password(password)
        await self.redis.delete(f"{RESET_PWD_PREFIX}{hash_token(token)}")
        await self._revoke_all_sessions(str(user.id))
        logger.info("auth.reset_password user_id=%s", user.id)

    async def _store_refresh_session(
        self,
        *,
        user_id: str,
        jti: str,
        ttl: timedelta,
        ip_address: str | None = None,
        device_info: str | None = None,
    ) -> None:
        await self.redis.setex(f"{REFRESH_KEY_PREFIX}{jti}", ttl, user_id)
        sessions_key = f"{USER_SESSIONS_PREFIX}{user_id}"
        await self.redis.sadd(sessions_key, jti)
        await self.redis.expire(sessions_key, ttl)

        session = UserSession(
            user_id=UUID(user_id),
            jti=jti,
            ip_address=ip_address,
            device_info=device_info,
        )
        self.db.add(session)

    async def _revoke_refresh_session(self, user_id: str, jti: str) -> None:
        ttl = timedelta(days=self.settings.refresh_token_expire_days)
        await self.redis.delete(f"{REFRESH_KEY_PREFIX}{jti}")
        await self.redis.setex(f"{REFRESH_REVOKED_PREFIX}{jti}", ttl, "1")
        await self.redis.srem(f"{USER_SESSIONS_PREFIX}{user_id}", jti)

        await self.db.execute(
            update(UserSession)
            .where(UserSession.jti == jti, UserSession.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )

    async def _revoke_all_sessions(self, user_id: str) -> None:
        sessions_key = f"{USER_SESSIONS_PREFIX}{user_id}"
        jtis = await self.redis.smembers(sessions_key)
        ttl = timedelta(days=self.settings.refresh_token_expire_days)

        for jti in jtis:
            await self.redis.delete(f"{REFRESH_KEY_PREFIX}{jti}")
            await self.redis.setex(f"{REFRESH_REVOKED_PREFIX}{jti}", ttl, "1")

        await self.redis.delete(sessions_key)

        await self.db.execute(
            update(UserSession)
            .where(
                UserSession.user_id == UUID(user_id),
                UserSession.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )
