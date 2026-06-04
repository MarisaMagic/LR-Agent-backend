from pydantic import BaseModel, Field


class LlmProviderCreateRequest(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=128)
    enabled: bool = True
    is_default: bool = False


class LlmProviderUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    api_key: str | None = Field(default=None, min_length=1, max_length=512)
    model: str | None = Field(default=None, min_length=1, max_length=128)
    enabled: bool | None = None
    is_default: bool | None = None


class LlmProviderPublic(BaseModel):
    id: str
    name: str
    base_url: str
    api_key: str
    model: str
    enabled: bool
    is_default: bool
    supports_vision: bool = False
    vision_probed_at: int | None = None
    vision_probe_detail: str = ""
    created_at: int
    updated_at: int
