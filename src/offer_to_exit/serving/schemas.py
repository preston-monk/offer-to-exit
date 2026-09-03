"""Stable input and output contracts for a pricing recommendation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(min_length=1)
    estimated_market_value: float = Field(gt=25_000, le=10_000_000)
    repair_cost: float = Field(ge=0, le=1_000_000)
    weekly_holding_cost: float = Field(gt=0, le=25_000)
    risk_aversion: float = Field(ge=0, le=10)
    target_margin: float = Field(ge=0, le=0.30)


class WeeklyAction(BaseModel):
    week: int = Field(ge=0, le=52)
    action: str
    list_price: float = Field(gt=0)
    sale_probability: float = Field(ge=0, le=1)


class PricingResponse(BaseModel):
    property_id: str
    recommendation: str
    acquisition_offer: float | None = Field(default=None, gt=0)
    expected_profit: float = Field(
        description="Expected contribution profit per quoted lead, including seller rejection."
    )
    probability_of_loss: float = Field(
        ge=0,
        le=1,
        description=(
            "Approximate loss probability per quoted lead after outcome-grid compression, "
            "including seller rejection."
        ),
    )
    sale_by_120_days: float = Field(
        ge=0,
        le=1,
        description="Cumulative sale probability conditional on acquisition.",
    )
    policy: list[WeeklyAction]
    reasons: list[str]

    @model_validator(mode="after")
    def offer_required_for_price_decision(self) -> PricingResponse:
        if self.recommendation == "price" and self.acquisition_offer is None:
            raise ValueError("acquisition_offer is required for a price recommendation")
        return self
