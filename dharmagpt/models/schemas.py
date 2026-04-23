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


class ManualTranslationItem(BaseModel):
    chunk_index: int = Field(..., ge=0)
    text_en_manual: str = Field(..., min_length=1)
    review_status: str = "pending"


class ManualTranslationSingleRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    text_en_manual: str = Field(..., min_length=1)
    reviewer: Optional[str] = None
    review_status: str = "pending"
    review_note: Optional[str] = None


class ManualTranslationBulkRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    translations: list[ManualTranslationItem] = Field(default_factory=list)


class ManualTranslationReviewRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    review_status: str = Field(..., pattern="^(approved|rejected|needs_work)$")
    reviewer: Optional[str] = None
    review_note: Optional[str] = None


class ManualTranslationApplyResponse(BaseModel):
    status: str
    dataset_id: str
    file_path: str
    updated_chunks: int
    total_chunks: int


class ManualTranslationRecord(BaseModel):
    chunk_index: int
    text_te: Optional[str] = None
    text_en_model: Optional[str] = None
    text_en_manual: Optional[str] = None
    review_status: Optional[str] = None
    reviewer: Optional[str] = None
    review_note: Optional[str] = None
    reviewed_at: Optional[str] = None


class ManualTranslationPendingResponse(BaseModel):
    dataset_id: str
    total_chunks: int
    pending_chunks: list[ManualTranslationRecord]


class HealthResponse(BaseModel):
    status: str
    pinecone: bool
    anthropic: bool
    sarvam: bool
