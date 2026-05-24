import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if not re.search(r"[a-z]", value):
            raise ValueError("password_must_contain_lowercase")
        if not re.search(r"[A-Z]", value):
            raise ValueError("password_must_contain_uppercase")
        if not re.search(r"\d", value):
            raise ValueError("password_must_contain_digit")
        return value


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class VerifyEmailRequest(BaseModel):
    token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if not re.search(r"[a-z]", value):
            raise ValueError("password_must_contain_lowercase")
        if not re.search(r"[A-Z]", value):
            raise ValueError("password_must_contain_uppercase")
        if not re.search(r"\d", value):
            raise ValueError("password_must_contain_digit")
        return value


class UserUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=32)
    display_name: str | None = Field(default=None, max_length=64)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is not None and not USERNAME_PATTERN.match(value):
            raise ValueError("invalid_username")
        return value


class DeleteAccountRequest(BaseModel):
    password: str


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    email_verified: bool
    username: str | None
    display_name: str | None
    avatar_url: str | None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


class MessageResponse(BaseModel):
    message: str
