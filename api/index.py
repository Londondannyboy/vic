"""
VIC CLM Server - Custom Language Model for Hume EVI

This FastAPI server implements the OpenAI-compatible /chat/completions endpoint
that Hume EVI requires for Custom Language Model integration.

All responses are validated through Pydantic AI to ensure factual accuracy
before being spoken by the voice assistant.
"""

import os
import time
import json
import random
import asyncio
import re
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

from .agent import generate_response
from .database import Database
from .tools import save_user_message

# Token for authenticating Hume requests
CLM_AUTH_TOKEN = os.environ.get("CLM_AUTH_TOKEN", "")

# Tokenizer for streaming response chunks
enc = tiktoken.encoding_for_model("gpt-4o")

# Vic-style filler phrases to stream immediately while searching
FILLER_PHRASES = [
    "Ah, let me think about that...",
    "Now that's an interesting question...",
    "Let me search through my collection of stories...",
    "Hmm, let me see what I have on that...",
    "That's a fascinating topic, give me a moment...",
]

# Topic-aware filler phrases (use {topic} placeholder)
TOPIC_FILLER_PHRASES = [
    "Ah, {topic}... let me see what I have on that...",
    "{topic}, you say? Let me search my archives...",
    "Now, {topic} is an interesting one... let me think...",
]


def extract_topic(user_message: str) -> str | None:
    """Extract a potential topic/subject from the user's message."""
    # Remove common question words
    cleaned = re.sub(
        r'\b(tell me about|what is|who was|where is|when did|how did|can you tell me about)\b',
        '',
        user_message.lower()
    ).strip()
    # Take first few significant words
    words = [w for w in cleaned.split() if len(w) > 3 and w not in ('the', 'and', 'was', 'were', 'have', 'been')]
    if words:
        # Capitalize and return first 2-3 words as topic
        topic = ' '.join(words[:3]).title()
        return topic
    return None

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection lifecycle."""
    # Startup
    yield
    # Shutdown - close database pool
    await Database.close()


app = FastAPI(
    title="VIC CLM",
    description="Custom Language Model for Lost London voice assistant",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: HTTPAuthorizationCredentials | None) -> bool:
    """Verify the Bearer token from Hume."""
    if not CLM_AUTH_TOKEN:
        # No token configured - allow all requests (dev mode)
        return True
    if not credentials:
        return False
    return credentials.credentials == CLM_AUTH_TOKEN


def extract_user_message(messages: list[dict]) -> str | None:
    """Extract the last user message from conversation history."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Handle both string and list content formats
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                ]
                return " ".join(text_parts)
            return content
    return None


def extract_session_id(request: Request, body: dict | None = None) -> str | None:
    """
    Extract custom_session_id from request.
    Hume may send it in query params, headers, or body.
    """
    import sys

    # Try query params first
    session_id = request.query_params.get("custom_session_id")
    if session_id:
        print(f"[VIC CLM] Session ID from query params: {session_id}", file=sys.stderr)
        return session_id

    # Try headers (X-Hume-Session-Id or similar)
    for header_name in ["x-hume-session-id", "x-session-id", "x-custom-session-id"]:
        session_id = request.headers.get(header_name)
        if session_id:
            print(f"[VIC CLM] Session ID from header {header_name}: {session_id}", file=sys.stderr)
            return session_id

    # Try body (Hume might include it in the request)
    if body:
        # Check various possible locations in body
        session_id = body.get("custom_session_id") or body.get("session_id")
        if session_id:
            print(f"[VIC CLM] Session ID from body: {session_id}", file=sys.stderr)
            return session_id

        # Check in metadata if present
        metadata = body.get("metadata", {})
        session_id = metadata.get("custom_session_id") or metadata.get("session_id")
        if session_id:
            print(f"[VIC CLM] Session ID from body.metadata: {session_id}", file=sys.stderr)
            return session_id

    print(f"[VIC CLM] No session ID found. Query: {dict(request.query_params)}, Headers: {dict(request.headers)}", file=sys.stderr)
    return None


def extract_user_name_from_session(session_id: str | None) -> str | None:
    """
    Extract user_name from session ID.
    Format: "Name|userId" or just "userId"
    """
    if not session_id:
        return None
    if '|' in session_id:
        name = session_id.split('|')[0]
        # Validate it's a reasonable name
        if name and len(name) >= 2 and len(name) <= 20 and name.isalpha():
            return name
    return None


def create_chunk(chunk_id: str, created: int, content: str, session_id: str | None, is_first: bool = False) -> str:
    """Create a single SSE chunk in OpenAI format."""
    chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=content,
                    role="assistant" if is_first else None,
                ),
                finish_reason=None,
                index=0,
            )
        ],
        created=created,
        model="vic-clm-2.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


async def stream_response(text: str, session_id: str | None = None):
    """
    Stream response as OpenAI-compatible ChatCompletionChunks.

    Hume EVI expects responses in the exact format of OpenAI's
    streaming chat completions API.
    """
    chunk_id = str(uuid4())
    created = int(time.time())

    # Stream token by token for natural speech pacing
    tokens = enc.encode(text)

    for i, token_id in enumerate(tokens):
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id, is_first=(i == 0))

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=created,
        model="vic-clm-2.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Signal end of stream
    yield "data: [DONE]\n\n"


async def stream_with_padding(
    user_message: str,
    session_id: str | None = None,
    user_name: str | None = None
):
    """
    Stream filler phrases immediately while generating the real response in background.

    This improves perceived responsiveness by giving the user immediate feedback
    while the search/validation takes place.
    """
    chunk_id = str(uuid4())
    created = int(time.time())

    # Start generating the real response in the background IMMEDIATELY
    response_task = asyncio.create_task(generate_response(user_message, session_id, user_name))

    # Choose a filler phrase (topic-aware if we can extract a topic)
    topic = extract_topic(user_message)
    if topic and random.random() > 0.3:  # 70% chance to use topic-aware filler
        filler = random.choice(TOPIC_FILLER_PHRASES).format(topic=topic)
    else:
        filler = random.choice(FILLER_PHRASES)

    # Stream the filler phrase token by token
    filler_tokens = enc.encode(filler)
    for i, token_id in enumerate(filler_tokens):
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id, is_first=(i == 0))
        # Small delay to make it sound natural
        await asyncio.sleep(0.02)

    # Add a natural pause (ellipsis already in filler, but add breathing room)
    yield create_chunk(chunk_id, created, " ", session_id)

    # Wait for the real response to complete
    response_text = await response_task

    # Stream the actual response
    response_tokens = enc.encode(response_text)
    for token_id in response_tokens:
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id)

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=created,
        model="vic-clm-2.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Signal end of stream
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security),
):
    """
    OpenAI-compatible chat completions endpoint for Hume CLM.

    Receives conversation history, generates a validated response using
    Pydantic AI, and streams it back in OpenAI's format.
    """
    # Verify authentication
    if not verify_token(credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    session_id = extract_session_id(request, body)
    user_name = extract_user_name_from_session(session_id)

    # Debug logging
    import sys
    print(f"[VIC CLM] ===== REQUEST DEBUG =====", file=sys.stderr)
    print(f"[VIC CLM] Session ID: {session_id}", file=sys.stderr)
    print(f"[VIC CLM] User Name: {user_name}", file=sys.stderr)
    print(f"[VIC CLM] Body keys: {list(body.keys())}", file=sys.stderr)
    print(f"[VIC CLM] ===========================", file=sys.stderr)

    # Extract the user's message
    user_message = extract_user_message(messages)

    if not user_message:
        # No user message - return a prompt
        fallback = "I didn't quite catch that. Could you say that again?"
        return StreamingResponse(
            stream_response(fallback, session_id),
            media_type="text/event-stream",
        )

    # Save user message to memory (fire and forget)
    if session_id:
        asyncio.create_task(save_user_message(session_id, user_message, "user"))

    # Generate and stream response
    response_text = await generate_response(user_message, session_id, user_name)

    return StreamingResponse(
        stream_response(response_text, session_id),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    """Health check and info endpoint."""
    return {
        "status": "ok",
        "service": "VIC CLM",
        "version": "2.0.0",
        "description": "Custom Language Model for Lost London voice assistant",
        "endpoint": "/chat/completions",
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
