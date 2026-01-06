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
from typing import Optional

from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

from .agent import (
    generate_response,
    generate_response_with_enrichment,
    get_suggestion_teaser,
    should_use_name,
    mark_name_used,
    increment_turn_counter,
    is_affirmation,
    get_last_suggestion,
    set_last_suggestion,
)
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


def extract_topic(user_message: str) -> Optional[str]:
    """Extract a potential topic/subject from the user's message."""
    # Remove conversational phrases and common question patterns
    cleaned = re.sub(
        r'\b(tell me about|what is|who was|where is|when did|how did|can you tell me about|'
        r"i'd like to know about|i would like to know about|"
        r"that is interesting|that's interesting|that sounds interesting|"
        r"yes please|no thanks|okay|sure|"
        r"like to know|want to know|curious about)\b",
        '',
        user_message.lower()
    ).strip()

    # Remove leading punctuation and filler words
    cleaned = re.sub(r'^[.,!?\s]+', '', cleaned)
    cleaned = re.sub(r'^\b(the|a|an|some|any)\b\s*', '', cleaned)

    # Take significant words (longer than 3 chars, not common words)
    stop_words = {'the', 'and', 'was', 'were', 'have', 'been', 'that', 'this', 'with', 'from', 'about', 'like', 'know'}
    words = [w for w in cleaned.split() if len(w) > 3 and w not in stop_words]

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


def verify_token(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Verify the Bearer token from Hume."""
    if not CLM_AUTH_TOKEN:
        # No token configured - allow all requests (dev mode)
        return True
    if not credentials:
        return False
    return credentials.credentials == CLM_AUTH_TOKEN


def extract_user_message(messages: list[dict]) -> Optional[str]:
    """Extract the last user message from conversation history.

    Returns None if the most recent user message is silence or a system instruction,
    to prevent responding to old messages.
    """
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
                content = " ".join(text_parts)

            # If the most recent user message is silence, return None
            # Don't fall back to earlier messages (that causes repetition)
            if content and content.lower().strip() == "[user silent]":
                return None

            # If the most recent user message is a greeting instruction, return None
            # The greeting handler will catch this case separately
            if content and content.lower().startswith("speak your greeting"):
                return None

            # Strip Hume emotion tags like {very interested, quite contemplative}
            if content and "{" in content:
                content = re.sub(r'\s*\{[^}]+\}\s*$', '', content).strip()

            return content
    return None


def extract_user_name_from_messages(messages: list[dict]) -> Optional[str]:
    """
    Extract user name from system message if present.
    Looks for patterns like "USER'S NAME: Dan" in the system prompt.
    """
    import sys

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                # Look for "USER'S NAME: X" pattern
                match = re.search(r"USER'S NAME:\s*(\w+)", content, re.IGNORECASE)
                if match:
                    name = match.group(1)
                    print(f"[VIC CLM] Found user name in system message: {name}", file=sys.stderr)
                    return name

                # Also try "Hello X" or "Welcome X" patterns
                match = re.search(r"(?:Hello|Welcome back),?\s+(\w+)", content)
                if match:
                    name = match.group(1)
                    print(f"[VIC CLM] Found user name in greeting pattern: {name}", file=sys.stderr)
                    return name

    print(f"[VIC CLM] No user name found in messages", file=sys.stderr)
    return None


def extract_session_id(request: Request, body: Optional[dict] = None) -> Optional[str]:
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


def extract_user_name_from_session(session_id: Optional[str]) -> Optional[str]:
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


def create_chunk(chunk_id: str, created: int, content: str, session_id: Optional[str], is_first: bool = False) -> str:
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


async def stream_response(text: str, session_id: Optional[str] = None):
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
    session_id: Optional[str] = None,
    user_name: Optional[str] = None
):
    """
    Stream filler phrases immediately while generating the real response in background.

    Uses dual-path architecture:
    - Fast path: Immediate response streamed to user
    - Enrichment path: Background context building for better follow-ups

    This improves perceived responsiveness by giving the user immediate feedback
    while the search/validation takes place.
    """
    chunk_id = str(uuid4())
    created = int(time.time())

    # Start generating response with enrichment in background IMMEDIATELY
    # This returns (response_text, enrichment_task) - enrichment runs in background
    response_task = asyncio.create_task(
        generate_response_with_enrichment(user_message, session_id, user_name)
    )

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

    # Wait for the fast response to complete (enrichment continues in background)
    response_text, enrichment_task = await response_task

    # Extract any suggested topic from the response and store it
    # Patterns: "Would you like to hear about X?" "Shall I tell you about X?"
    import re
    suggestion_patterns = [
        r"would you like to hear (?:more )?about ([^?]+)\?",
        r"shall i tell you (?:more )?about ([^?]+)\?",
        r"would you like to know (?:more )?about ([^?]+)\?",
        r"i could tell you about ([^?.,]+)",
        r"there's quite a story (?:about|there) ([^?.,]+)",
    ]
    for pattern in suggestion_patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            suggested_topic = match.group(1).strip()
            set_last_suggestion(session_id, suggested_topic)
            import sys
            print(f"[VIC CLM] Stored suggestion for next turn: '{suggested_topic}'", file=sys.stderr)
            break

    # Stream the actual response
    response_tokens = enc.encode(response_text)
    for token_id in response_tokens:
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id)

    # Check if enrichment has a suggestion ready (from previous turn)
    # This adds proactive follow-up suggestions when available
    suggestion_teaser = get_suggestion_teaser(session_id)
    if suggestion_teaser:
        teaser_tokens = enc.encode(suggestion_teaser)
        for token_id in teaser_tokens:
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

    # Note: enrichment_task continues running in background
    # It will populate session context for the next query


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """
    OpenAI-compatible chat completions endpoint for Hume CLM.

    Receives conversation history, generates a validated response using
    Pydantic AI, and streams it back in OpenAI's format.
    """
    global _last_request_debug

    # Verify authentication
    if not verify_token(credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])

    session_id = extract_session_id(request, body)
    user_message_extracted = extract_user_message(messages)
    topic_extracted = extract_topic(user_message_extracted) if user_message_extracted else None

    # Store for debugging
    _last_request_debug = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "messages_count": len(messages),
        "user_message_extracted": user_message_extracted,
        "topic_extracted": topic_extracted,
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content", "")[:500] if isinstance(m.get("content"), str) else str(m.get("content", ""))[:500]
            }
            for m in messages
        ],
        "body_keys": list(body.keys()),
    }
    user_name = extract_user_name_from_session(session_id)

    # If no user name from session, try extracting from system message
    if not user_name:
        user_name = extract_user_name_from_messages(messages)

    # Debug logging
    import sys
    print(f"[VIC CLM] ===== REQUEST DEBUG =====", file=sys.stderr)
    print(f"[VIC CLM] Session ID: {session_id}", file=sys.stderr)
    print(f"[VIC CLM] User Name: {user_name}", file=sys.stderr)
    print(f"[VIC CLM] User Message: {user_message_extracted}", file=sys.stderr)
    print(f"[VIC CLM] Topic Extracted: {topic_extracted}", file=sys.stderr)
    print(f"[VIC CLM] Body keys: {list(body.keys())}", file=sys.stderr)
    print(f"[VIC CLM] Number of messages: {len(messages)}", file=sys.stderr)
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            preview = content[:200] + "..." if len(content) > 200 else content
        else:
            preview = str(content)[:200]
        print(f"[VIC CLM] Message {i}: role={role}, content={preview}", file=sys.stderr)
    print(f"[VIC CLM] ===========================", file=sys.stderr)

    # Use the already extracted user message
    user_message = user_message_extracted

    # Check if this is a greeting request by looking at the MOST RECENT user message only
    # (not any message in history - that caused greeting loops)
    most_recent_user_content = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                most_recent_user_content = content.lower().strip()
            break

    is_greeting_request = (
        most_recent_user_content is not None and
        most_recent_user_content.startswith("speak your greeting")
    )

    if is_greeting_request:
        # Generate a proper greeting - warm and conversational, like Vic chatting over tea
        # Avoid exclamation marks - they make the TTS sound too excited/different
        if user_name:
            greeting = f"Ah, hello {user_name}. Good to have you here. I'm Vic, and I've collected over 370 stories about London's hidden history. What corner of the city shall we explore together?"
            mark_name_used(session_id, is_greeting=True)
        else:
            greeting = "Ah, hello there. I'm Vic, the voice of Vic Keegan. I've spent years uncovering London's hidden stories, and I'd love to share them with you. What should I call you, and where shall we begin?"
        return StreamingResponse(
            stream_response(greeting, session_id),
            media_type="text/event-stream",
        )

    if not user_message:
        # No user message (silence) - return 204 No Content
        # This tells Hume there's nothing to process, preventing restart loops
        import sys
        print(f"[VIC CLM] No user message found (silence), returning 204 No Content", file=sys.stderr)
        from fastapi.responses import Response
        return Response(status_code=204)

    # Check if this is an affirmation of a previous suggestion
    # e.g., user says "yes" after VIC asked "Would you like to hear about X?"
    actual_query = user_message
    is_affirm, topic_hint = is_affirmation(user_message)

    if is_affirm:
        import sys
        if topic_hint:
            # User said something like "yeah, the Thames" - use their topic hint
            print(f"[VIC CLM] Affirmation with topic hint: '{topic_hint}'", file=sys.stderr)
            actual_query = topic_hint
        else:
            # Pure affirmation like "yes" - use last suggestion
            last_suggestion = get_last_suggestion(session_id)
            if last_suggestion:
                print(f"[VIC CLM] Pure affirmation '{user_message}' -> using last suggestion: '{last_suggestion}'", file=sys.stderr)
                actual_query = last_suggestion
            else:
                print(f"[VIC CLM] Affirmation detected but no last suggestion stored", file=sys.stderr)

    # Save user message to memory (fire and forget)
    if session_id:
        asyncio.create_task(save_user_message(session_id, user_message, "user"))

    # Check if we should use the name in this response (spacing rule)
    use_name = should_use_name(session_id, is_greeting=False)
    effective_name = user_name if use_name else None

    if use_name and user_name:
        mark_name_used(session_id)

    # Always increment turn counter for tracking
    increment_turn_counter(session_id)

    # OPTIMIZATION: Stream filler phrases while generating response
    # This disguises the delay and makes VIC feel more responsive
    return StreamingResponse(
        stream_with_padding(actual_query, session_id, effective_name),
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


@app.get("/debug/last-request")
async def debug_last_request():
    """Return the last request received for debugging."""
    return _last_request_debug


# Store last request for debugging
_last_request_debug: dict = {"status": "no requests yet"}


@app.get("/debug/search")
async def debug_search():
    """Debug endpoint to test search functionality."""
    from .tools import get_voyage_embedding, normalize_query
    from .database import search_articles_hybrid
    import os

    try:
        # Test 1: Check DATABASE_URL
        db_url = os.environ.get("DATABASE_URL", "")
        db_status = "set" if db_url else "NOT SET"

        # Test 2: Get embedding
        query = "royal aquarium"
        normalized = normalize_query(query)
        embedding = await get_voyage_embedding(normalized)

        # Test 3: Search database
        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=normalized,
            limit=3,
            similarity_threshold=0.3,
        )

        return {
            "status": "ok",
            "database_url": db_status,
            "query": query,
            "normalized": normalized,
            "embedding_length": len(embedding) if embedding else 0,
            "results_count": len(results),
            "results": [
                {"title": r.get("title", "NO TITLE"), "score": r.get("score", 0)}
                for r in results[:3]
            ],
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
