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
    filter_section: Optional[str] = None  # e.g. "Sundara Kanda", "Adi Parva"


class SourceChunk(BaseModel):
    text: str
    citation: str
    section: Optional[str] = None   # Kanda / Parva / Adhyaya / Skandha etc.
    chapter: Optional[int] = None   # Sarga / Adhyaya / Chapter number
    verse: Optional[int] = None     # Shloka / verse number within the chapter
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
    section: Optional[str] = None
    description: Optional[str] = None


class AudioTranscribeResponse(BaseModel):
    transcript: str
    translated_transcript: Optional[str] = None
    chunks_created: int
    file_name: str
    transcript_file_name: Optional[str] = None
    transcription_mode: str = "sarvam_stt"
    transcription_version: str = "saaras:v3"
    translation_mode: Optional[str] = None
    translation_backend: Optional[str] = None
    translation_version: Optional[str] = None
    translation_fallback_reason: Optional[str] = None


class FeedbackRating(str, Enum):
    up = "up"
    down = "down"


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    mode: QueryMode = QueryMode.guidance
    sources: list[SourceChunk] = Field(default_factory=list)
    rating: FeedbackRating = FeedbackRating.up
    note: Optional[str] = None


class CorpusUploadResponse(BaseModel):
    status: str
    role: str
    dataset_id: Optional[str] = None
    file_path: str
    records: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    ingested: bool = False
    vectors_written: int = 0


class HealthResponse(BaseModel):
    status: str
    pinecone: bool
    vector_backend: str = "pinecone"
    vector_store: bool = False
    anthropic: bool
    sarvam: bool
    vector_name: str
    llm_backend: str = "anthropic"
    llm_local: bool = False
