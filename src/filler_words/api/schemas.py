from pydantic import BaseModel, Field, field_validator, model_validator

from filler_words.core.labels import FillerType


class FillerRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Current user query from ASR or upstream system.")
    persona_tag: str = Field(..., min_length=1, description="Assistant persona used only for template selection.")
    request_id: str | None = Field(default=None, description="Optional trace id from upstream.")
    locale: str = Field(default="zh-CN", description="Template locale.")

    @field_validator("query", "persona_tag", "locale")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class FillerResponse(BaseModel):
    trigger: int = Field(..., ge=0, le=1)
    filler_type: FillerType
    filler_text: str = ""
    template_id: str = ""
    model_version: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_trigger_contract(self) -> "FillerResponse":
        if self.trigger == 0 and self.filler_type is not FillerType.NONE:
            raise ValueError("trigger=0 requires filler_type=NONE")
        if self.trigger == 0 and (self.filler_text or self.template_id):
            raise ValueError("trigger=0 requires empty filler_text and template_id")
        if self.trigger == 1 and self.filler_type is FillerType.NONE:
            raise ValueError("trigger=1 requires a non-NONE filler_type")
        return self
