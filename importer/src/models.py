from typing import List
from pydantic import BaseModel, Field, ConfigDict


class DocumentWork(BaseModel):
    """Data structure representing a document processing work unit."""

    model_config = ConfigDict(extra='forbid')

    doc_id: int = Field(..., description="Paperless document ID")
    source_url: str = Field("", description="Original source download URL, may be empty")
    chunks: List[str] = Field(default_factory=list, description="Chunks of extracted text")
    text_hash: str = Field(..., description="SHA256 hash of the full text for idempotency")
    doc: dict = Field(default_factory=dict, description="Original Paperless document payload")
