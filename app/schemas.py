from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    excessFee: float | None = None
    rrp: float | None = None
    productName: str | None = None
    productDesc: str | None = None
    policyStatus: str | None = None
    retailerName: str | None = None
    deviceType: str | None = None
    model: str | None = None
    channel: str | None = None
    claimType: str | None = None
    country: str | None = None
    turnOnOff: float | None = None
    touchScreen: float | None = None
    frontCamera: float | None = None
    backCamera: float | None = None
    audio: float | None = None
    mic: float | None = None
    buttons: float | None = None
    connection: float | None = None
    charging: float | None = None
    issueDesc: str | None = None


class ContributingFactor(BaseModel):
    feature: str
    contribution: float


class PersonaExplanation(BaseModel):
    persona: str
    explanation: str


class ExplanationResponse(BaseModel):
    decision: str = Field(..., description="'approved' or 'declined'")
    approval_probability: float
    contributing_factors: list[ContributingFactor]
    text_contribution: float
    explanations: list[PersonaExplanation]
