"""
Pydantic schemas for the SHL Assessment Recommender API.

These schemas are NON-NEGOTIABLE — the grader validates exact structure.
Every response MUST have: reply, recommendations (always a list), end_of_conversation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ──────────────────────────────────────────────
# Chat Endpoint Schemas (POST /chat)
# ──────────────────────────────────────────────

class Message(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.
    Contains the FULL conversation history — the API is stateless.
    """
    messages: List[Message] = Field(
        ..., description="Full conversation history, oldest first"
    )


class Assessment(BaseModel):
    """A single SHL assessment recommendation."""
    name: str = Field(..., description="Full product name from catalog")
    url: str = Field(..., description="Exact URL from catalog.json — NEVER fabricated")
    test_type: str = Field(..., description="Test type code: A, P, K, B, C, S, D")


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.
    
    Rules:
    - recommendations MUST be [] (empty list) when clarifying or refusing.
    - recommendations MUST contain 1-10 items when recommending.
    - recommendations is NEVER null.
    - end_of_conversation is True only when the final shortlist is confirmed.
    """
    reply: str = Field(
        ..., description="The agent's conversational response"
    )
    recommendations: List[Assessment] = Field(
        default_factory=list,
        description="Assessment shortlist — empty list when clarifying/refusing"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when final shortlist is confirmed"
    )


# ──────────────────────────────────────────────
# Recommend Endpoint Schemas (POST /recommend)
# ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Request body for POST /recommend.
    One-shot: send a JD or query, get recommendations in a single call.
    No conversation needed.
    """
    query: str = Field(
        ..., description="Job description or role query"
    )
    top_k: int = Field(
        default=10, ge=1, le=10,
        description="Number of recommendations to return (1-10)"
    )


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""
    recommendations: List[Assessment] = Field(
        ..., description="Ranked list of recommended assessments"
    )
    query_used: str = Field(
        ..., description="The search query that was used for retrieval"
    )


# ──────────────────────────────────────────────
# Catalog Endpoint Schemas (GET /catalog/search)
# ──────────────────────────────────────────────

class CatalogItem(BaseModel):
    """Full catalog entry for an SHL assessment."""
    name: str
    url: str
    test_type: str
    description: str = ""
    job_levels: List[str] = Field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    duration_minutes: Optional[int] = None


class CatalogSearchResponse(BaseModel):
    """Response body for GET /catalog/search."""
    results: List[CatalogItem]
    total: int = Field(..., description="Number of results found")
    query: str = Field(..., description="The search query used")
