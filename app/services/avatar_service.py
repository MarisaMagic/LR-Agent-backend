import asyncio
import io
import json
import logging
import uuid
from typing import Final

from minio import Minio
from minio.error import S3Error
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
MAX_AVATAR_BYTES: Final[int] = 2 * 1024 * 1024
AVATAR_SIZES: Final[tuple[int, ...]] = (128, 256, 512)


class AvatarService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Minio | None = None

    def _get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                secure=self.settings.minio_secure,
            )
        return self._client

    def _public_url(self, object_key: str) -> str:
        base = self.settings.minio_public_base_url
        if not base:
            scheme = "https" if self.settings.minio_secure else "http"
            base = f"{scheme}://{self.settings.minio_endpoint}/{self.settings.minio_bucket}"
        return f"{base.rstrip('/')}/{object_key}"

    @staticmethod
    def _validate_image(file_bytes: bytes, content_type: str) -> Image.Image:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError("invalid_image_type")

        if len(file_bytes) > MAX_AVATAR_BYTES:
            raise ValueError("file_too_large")

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()
            image = Image.open(io.BytesIO(file_bytes))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("invalid_image") from exc

        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("invalid_image")

        return image

    @staticmethod
    def _resize_to_webp(image: Image.Image, size: int) -> bytes:
        if image.mode in {"RGBA", "LA", "P"}:
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=92, method=6)
        return buffer.getvalue()

    async def ensure_bucket(self) -> None:
        client = self._get_client()
        bucket = self.settings.minio_bucket

        def _setup() -> None:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)

            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket}/avatars/*"],
                    }
                ],
            }
            try:
                client.set_bucket_policy(bucket, json.dumps(policy))
            except S3Error as exc:
                logger.warning("Could not set MinIO bucket policy: %s", exc)

        await asyncio.to_thread(_setup)

    async def upload(self, user_id: str, file_bytes: bytes, content_type: str) -> tuple[str, str]:
        image = self._validate_image(file_bytes, content_type)
        avatar_id = uuid.uuid4().hex
        base_key = f"avatars/{user_id}/{avatar_id}"
        client = self._get_client()
        bucket = self.settings.minio_bucket

        def _upload_all() -> None:
            for size in AVATAR_SIZES:
                webp_bytes = self._resize_to_webp(image, size)
                object_key = f"{base_key}_{size}.webp"
                client.put_object(
                    bucket,
                    object_key,
                    io.BytesIO(webp_bytes),
                    length=len(webp_bytes),
                    content_type="image/webp",
                )

        await asyncio.to_thread(_upload_all)
        primary_key = f"{base_key}_512.webp"
        return self._public_url(primary_key), base_key

    async def delete(self, avatar_key: str) -> None:
        client = self._get_client()
        bucket = self.settings.minio_bucket

        def _delete_all() -> None:
            for size in AVATAR_SIZES:
                object_key = f"{avatar_key}_{size}.webp"
                try:
                    client.remove_object(bucket, object_key)
                except S3Error as exc:
                    logger.warning("Failed to delete avatar object %s: %s", object_key, exc)

        await asyncio.to_thread(_delete_all)
