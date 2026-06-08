from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "LR-Agent API"
    app_env: Literal["development", "staging", "production", "testing"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    postgres_user: str = "lr_agent"
    postgres_password: str
    postgres_db: str = "lr_agent"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: PostgresDsn

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str
    redis_db: int = 0
    redis_url: RedisDsn

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""
    minio_bucket: str = "lr-agent-avatars"
    minio_secure: bool = False
    minio_public_base_url: str = ""

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@example.com"
    smtp_use_tls: bool = True
    email_deep_link_base: str = "lr-agent://"
    email_verify_web_base: str = "http://localhost:1212"

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:1212"]

    agent_max_tool_rounds: int = 4
    agent_job_cancel_ttl_seconds: int = 3600
    agent_default_max_context_tokens: int = 12_000
    agent_default_reserve_completion_tokens: int = 2_048
    agent_default_max_turns_in_window: int = 20
    agent_default_summarize_trigger_ratio: float = 0.85
    agent_default_min_turns_before_summarize: int = 6
    agent_chat_cache_ttl_seconds: int = 2_592_000
    agent_chat_sessions_index_limit: int = 500
    agent_session_list_default_limit: int = 50
    agent_session_list_max_limit: int = 100
    agent_message_page_default_limit: int = 50
    agent_message_page_max_limit: int = 100
    agent_stream_rate_limit_per_minute: int = 10
    agent_stream_rate_limit_per_day: int = 200
    agent_session_write_rate_limit_per_hour: int = 60

    llm_secrets_master_key: str | None = None
    llm_secrets_active_key_id: str = "v1"
    llm_base_url_allow_http: bool = False

    # When true, batch annotation follows fusion-reasoning tool chain (ReAct sub-agent).
    annotation_fusion_parity: bool = True
    annotation_sub_agent_max_iterations: int = 12
    annotation_sub_agent_max_rounds: int = 10
    annotation_vision_map_concurrency: int = 3
    annotation_llm_temperature: float = 0.0
    annotation_prepare_temperature: float = 0.1
    annotation_vision_map_validate: bool = True
    annotation_vision_map_max_retries: int = 2
    annotation_label_pool_preflight: Literal["off", "auto", "always"] = "auto"
    annotation_label_pool_preflight_min_extra: int = 2
    agent_chat_vision_max_edge: int = 1280
    agent_chat_vision_jpeg_quality: int = 85
    agent_read_file_max_bytes: int = 524_288
    agent_read_file_max_lines: int = 2000
    agent_read_document_max_pages: int = 30

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            import json

            return json.loads(value)
        return value

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
