from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class QueryMode(str, Enum):
    guidance = "guidance"
    story = "story"
    children = "children"
    scholar = "scholar"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    mode: QueryMode = QueryMode.guidance
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)
    language: str = "en"  # "en", "hi", "te", "ta", etc.
    filter_kanda: Optional[str] = None  # e.g. "Sundara Kanda"


class SourceChunk(BaseModel):
    text: str
    citation: str
    kanda: Optional[str] = None
    sarga: Optional[int] = None
    score: float
    source_type: str = "text"  # "text" | "audio"
    audio_timestamp: Optional[str] = None
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    mode: QueryMode
    language: str
    query_id: str


class AudioTranscribeRequest(BaseModel):
    language_code: str = "hi-IN"
    kanda: Optional[str] = None
    description: Optional[str] = None


class AudioTranscribeResponse(BaseModel):
    transcript: str
    chunks_created: int
    file_name: str


class HealthResponse(BaseModel):
    status: str
    pinecone: bool
    anthropic: bool
    sarvam: bool
