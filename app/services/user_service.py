from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, user_id: UUID) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def create(self, *, email: str, password_hash: str) -> User:
        user = User(email=email, password_hash=password_hash)
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def update_profile(
        self,
        user: User,
        *,
        username: str | None = None,
        display_name: str | None = None,
    ) -> User:
        if username is not None:
            user.username = username
        if display_name is not None:
            user.display_name = display_name
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def update_avatar(
        self,
        user: User,
        *,
        avatar_url: str,
        avatar_key: str,
    ) -> User:
        user.avatar_url = avatar_url
        user.avatar_key = avatar_key
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def clear_avatar(self, user: User) -> User:
        user.avatar_url = None
        user.avatar_key = None
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def deactivate_account(self, user: User) -> User:
        user.is_active = False
        user.email = f"deleted_{user.id}@deleted.local"
        user.username = None
        user.display_name = None
        user.avatar_url = None
        user.avatar_key = None
        user.email_verified = False
        await self.db.flush()
        await self.db.refresh(user)
        return user
