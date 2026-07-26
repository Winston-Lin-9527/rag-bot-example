from pydantic import BaseModel


class CollectionSummary(BaseModel):
    name: str
    raw_name: str


class CollectionsResponse(BaseModel):
    collections: list[CollectionSummary]