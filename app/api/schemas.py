from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    question: str
    route: str
    answer: str
    sources: list[dict] = []
    web_results: list[dict] = []


class IngestResponse(BaseModel):
    message: str
    status: str