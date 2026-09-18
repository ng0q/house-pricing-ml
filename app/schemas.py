from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    location: str = Field(
        ...,
        description='Location(City), type: "thane"',
        examples=['thane']
    )
    transaction: int = Field(
        ...,
        ge=0,
        le=1,
        description='0 - Resale, 1 - New Property'
    )
    furnishing: int = Field(
        ...,
        ge=0,
        le=2,
        description='0 - Unfurnished, 1 - Semi-Furnished, 2 - Furnished'
    )
    facing: str = Field(
        ...,
        description='"east", "west", "missing" etc'
    )
    bathroom: float = Field(
        ...,
        ge=0,
        description='Bathrooms count'
    )
    balcony: float = Field(
        ...,
        ge=0,
        description='Balcony count'
    )
    car_covered: int = Field(
        ...,
        ge=0,
        le=1,
        description='1 - Covered parking, 0 - open or without parking'
    )
    parking_count: float = Field(
        ...,
        ge=0,
        description='Parking count'
    )
    super_area_sqft: float | None = Field(
        None,
        description='Super area in Sqft (can be null)'
    )
    carpet_area_sqft: float | None = Field(
        None,
        description='Carpet area in Sqft (can be null)'
    )
    floor_current: float | None = Field(
        None,
        description='Current floor (Ground = 0)'
    )
    floor_total: float | None = Field(
        None,
        description='Total floors'
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(
        ...,
        description='Predicted price in rupees'
    )
    predicted_price_formatted: str = Field(
        ...,
        description='Convenient format, like "85.00 Lac" or "1.20 Cr"'
    )