from pydantic import BaseModel, Field

from app.schemas.annotation_agent import AnnotationLlmBaseRequest


class QualityReportComposeRequest(AnnotationLlmBaseRequest):
    compose_payload: dict = Field(default_factory=dict)
