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
    detect_topic_switch,
    set_current_topic,
    set_pending_topic_switch,
    get_pending_topic_switch,
    clear_pending_topic_switch,
    clean_query,
    check_returning_user,
    update_interaction_time,
    mark_greeted_this_session,
    set_user_emotion,
    get_emotion_adjustment,
    extract_emotion_from_message,
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

# Include validated interests router for human-in-the-loop Zep storage
from .validated_interests import router as validated_interests_router
app.include_router(validated_interests_router)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Verify the Bearer token from Hume."""
    if not CLM_AUTH_TOKEN:
        # No token configured - allow all requests (dev mode)
        return True
    if not credentials:
        return False
    return credentials.credentials == CLM_AUTH_TOKEN


def extract_user_message_and_emotion(messages: list[dict]) -> tuple[Optional[str], str]:
    """Extract the last user message and emotion from conversation history.

    Returns (message, emotion):
    - message: None if silence/instruction, otherwise the cleaned message
    - emotion: The emotion tags if present, otherwise empty string
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
                return (None, "")

            # If the most recent user message is a greeting instruction, return None
            # The greeting handler will catch this case separately
            if content and content.lower().startswith("speak your greeting"):
                return (None, "")

            # Extract Hume emotion tags like {very interested, quite contemplative}
            emotion = ""
            if content and "{" in content:
                cleaned, emotion = extract_emotion_from_message(content)
                content = cleaned

            return (content, emotion)
    return (None, "")


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

    Includes natural pacing with micro-delays at punctuation.
    """
    chunk_id = str(uuid4())
    created = int(time.time())

    # Stream token by token for natural speech pacing
    tokens = enc.encode(text)

    for i, token_id in enumerate(tokens):
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id, is_first=(i == 0))

        # Add micro-delays at punctuation for natural pacing
        if token_text.rstrip() in {'.', '!', '?'}:
            await asyncio.sleep(0.05)  # Longer pause at sentence end
        elif token_text.rstrip() in {',', ';', ':', '...'}:
            await asyncio.sleep(0.02)  # Shorter pause at clauses

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
    user_message_extracted, user_emotion = extract_user_message_and_emotion(messages)
    topic_extracted = extract_topic(user_message_extracted) if user_message_extracted else None

    # Store user emotion for response adjustment
    if user_emotion and session_id:
        set_user_emotion(session_id, user_emotion)
        import sys
        print(f"[VIC CLM] Detected emotion: {user_emotion}", file=sys.stderr)

    # Store for debugging
    _last_request_debug = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "messages_count": len(messages),
        "user_message_extracted": user_message_extracted,
        "user_emotion": user_emotion,
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
        # Check if this is a returning user (after a time gap)
        is_returning, last_topic = check_returning_user(session_id)

        # Extract user_id to check Zep for their interests
        user_id = None
        if session_id and '|' in session_id:
            user_id = session_id.split('|')[1].split('_')[0]

        # Try to get user's ACTUAL recent topics from user_queries table (ground truth)
        # This is more reliable than Zep's inferred facts
        zep_topics = []
        if user_id and not last_topic:
            try:
                from .database import get_connection
                async with get_connection() as conn:
                    # Get user's most recent queries with article titles
                    recent = await conn.fetch("""
                        SELECT DISTINCT article_title, created_at
                        FROM user_queries
                        WHERE user_id = $1 AND article_title IS NOT NULL
                        ORDER BY created_at DESC
                        LIMIT 5
                    """, user_id)

                    for row in recent:
                        title = row['article_title']
                        query = row.get('query', '').lower().strip()

                        # Skip affirmations - they're follow-ups, not real topic queries
                        if query in ['yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope']:
                            continue

                        if title:
                            # Clean up title - extract main topic
                            # "Vic Keegan's Lost London 103: Thorney Island" -> "Thorney Island"
                            if ':' in title:
                                topic = title.split(':')[-1].strip()
                            else:
                                topic = title
                            if topic and len(topic) < 50 and topic not in zep_topics:
                                zep_topics.append(topic)

                    print(f"[VIC Greeting] Recent topics from DB: {zep_topics}", file=sys.stderr)
            except Exception as e:
                print(f"[VIC Greeting] Error fetching DB topics: {e}", file=sys.stderr)

                # Fallback to Zep if DB fails
                try:
                    from .agent import get_zep_client, ZEP_API_KEY
                    if ZEP_API_KEY:
                        client = get_zep_client()
                        if client:
                            response = await client.post(
                                "/api/v2/graph/search",
                                json={
                                    "user_id": user_id,
                                    "query": "user interested in learning about topics",
                                    "limit": 5,
                                    "scope": "edges",
                                },
                            )
                            if response.status_code == 200:
                                edges = response.json().get("edges", [])
                                # Sort by created_at (most recent first)
                                edges.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                                for edge in edges:
                                    fact = edge.get("fact", "")
                                    if "interest" in fact.lower() and "learning about" in fact.lower():
                                        import re
                                        match = re.search(r"learning about ([^.]+)", fact, re.IGNORECASE)
                                        if match:
                                            topic = match.group(1).strip().rstrip('.')
                                            if topic and len(topic) < 50:
                                                zep_topics.append(topic)
                                print(f"[VIC Greeting] Fallback Zep topics: {zep_topics}", file=sys.stderr)
                except Exception as e2:
                    print(f"[VIC Greeting] Zep fallback also failed: {e2}", file=sys.stderr)

        # Varied greeting templates for returning users with topics
        import random
        RETURNING_WITH_TOPIC = [
            "Ah, {name}, lovely to have you back. Last time you were curious about {topic}. Shall we pick up where we left off, or venture somewhere new?",
            "Welcome back, {name}. I remember you were exploring {topic}. Would you like to continue that journey, or discover something different?",
            "{name}, good to see you again. We were discussing {topic} before. Fancy hearing more about that, or shall we wander elsewhere?",
            "Ah, {name}. I was hoping you'd return. You seemed quite taken with {topic}. More of that, or a fresh adventure?",
        ]

        # Varied greetings for returning users without specific topic
        RETURNING_NO_TOPIC = [
            "Ah, {name}, welcome back to Lost London. What hidden corner shall we explore today?",
            "{name}, good to see you again. I've been collecting more stories since we last spoke. What takes your fancy?",
            "Welcome back, {name}. The city has so many secrets yet to share. Where shall we begin?",
        ]

        # First-time user greetings
        FIRST_TIME_GREETINGS = [
            "Ah, hello {name}. I'm Vic, and I've spent years uncovering London's hidden stories. Over 370 of them, in fact. What corner of the city shall we explore together?",
            "Welcome, {name}. I'm Vic Keegan, and I'd love to share some of London's forgotten tales with you. What piques your curiosity?",
            "{name}, good to meet you. I'm Vic, collector of London's hidden history. From lost rivers to forgotten palaces, where shall we start?",
        ]

        # Anonymous user greetings
        ANONYMOUS_GREETINGS = [
            "Ah, hello there. I'm Vic, the voice of Vic Keegan. I've spent years uncovering London's hidden stories. What should I call you, and where shall we begin?",
            "Welcome to Lost London. I'm Vic, and I've collected over 370 stories about this city's secret past. Might I ask your name before we start exploring?",
        ]

        # Determine the best topic to offer
        topic_to_offer = last_topic
        if not topic_to_offer and zep_topics:
            topic_to_offer = zep_topics[0]

        if user_name and topic_to_offer:
            # Returning user with a topic to continue
            template = random.choice(RETURNING_WITH_TOPIC)
            greeting = template.format(name=user_name, topic=topic_to_offer)
            mark_name_used(session_id, is_greeting=True)
        elif user_name and (is_returning or zep_topics):
            # Returning user but no specific topic
            template = random.choice(RETURNING_NO_TOPIC)
            greeting = template.format(name=user_name)
            mark_name_used(session_id, is_greeting=True)
        elif user_name:
            # First-time user with name
            template = random.choice(FIRST_TIME_GREETINGS)
            greeting = template.format(name=user_name)
            mark_name_used(session_id, is_greeting=True)
        else:
            # Anonymous user
            greeting = random.choice(ANONYMOUS_GREETINGS)

        # Mark that we've greeted this session
        mark_greeted_this_session(session_id)
        update_interaction_time(session_id)

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

    # Check if there's a pending topic switch awaiting confirmation
    pending_switch = get_pending_topic_switch(session_id)
    actual_query = user_message
    import sys

    if pending_switch:
        is_affirm, _ = is_affirmation(user_message)
        if is_affirm:
            # User confirmed the topic switch
            print(f"[VIC CLM] User confirmed topic switch to: '{pending_switch}'", file=sys.stderr)
            actual_query = pending_switch
            clear_pending_topic_switch(session_id)
            set_current_topic(session_id, pending_switch)
        else:
            # User didn't confirm - they might be asking something else
            clear_pending_topic_switch(session_id)
            print(f"[VIC CLM] Topic switch not confirmed, processing: '{user_message}'", file=sys.stderr)
    else:
        # Check if this is an affirmation of a previous suggestion
        is_affirm, topic_hint = is_affirmation(user_message)

        if is_affirm:
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
        else:
            # Check for topic switch - if detected, clear old context and switch
            is_switch, new_topic, _ = detect_topic_switch(user_message, session_id)
            if is_switch:
                print(f"[VIC CLM] Topic switch detected to: '{new_topic}'", file=sys.stderr)
                # Clear old context for clean switch
                set_current_topic(session_id, new_topic)
                # Use the new topic as the query
                actual_query = new_topic

    # Clean the query - remove yes/no prefixes that are confirmations, not search terms
    # "Yes, the Royal Aquarium" -> "the Royal Aquarium"
    actual_query = clean_query(actual_query)
    print(f"[VIC CLM] Cleaned query: '{actual_query}'", file=sys.stderr)

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

    # Update interaction time for returning user detection
    update_interaction_time(session_id)

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


@app.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Return session context for debugging - shows what Zep/enrichment data is available."""
    from .agent import get_session_context

    context = get_session_context(session_id)

    return {
        "session_id": session_id,
        "enrichment_complete": context.enrichment_complete,
        "current_topic": context.current_topic,
        "last_suggested_topic": context.last_suggested_topic,
        "topics_discussed": context.topics_discussed,
        "entities_count": len(context.entities),
        "entities": [{"name": e.name, "type": e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)} for e in context.entities[:10]],
        "connections_count": len(context.connections),
        "connections": [
            {"from": c.from_entity, "relation": c.relation, "to": c.to_entity}
            for c in context.connections[:10]
        ],
        "suggestions_count": len(context.suggestions),
        "suggestions": [{"topic": s.topic, "teaser": s.teaser} for s in context.suggestions[:5]],
        "turns_since_name_used": context.turns_since_name_used,
    }


@app.get("/debug/popular-topics")
async def debug_popular_topics():
    """Return popular topics for analytics - shows what users are asking about."""
    from .agent import get_popular_topics

    topics = get_popular_topics(limit=20)

    return {
        "popular_topics": [
            {"topic": t[0], "weighted_count": t[1], "articles": t[2] if len(t) > 2 else []}
            for t in topics
        ],
        "total_tracked": len(topics),
    }


@app.get("/debug/cache-stats")
async def debug_cache_stats():
    """Return cache statistics for monitoring."""
    from .tools import _embedding_cache, EMBEDDING_CACHE_TTL_SECONDS, MAX_EMBEDDING_CACHE_SIZE
    import time

    current_time = time.time()

    # Calculate cache stats
    total_entries = len(_embedding_cache)
    valid_entries = 0
    expired_entries = 0

    for key, data in _embedding_cache.items():
        age = current_time - data["timestamp"]
        if age < EMBEDDING_CACHE_TTL_SECONDS:
            valid_entries += 1
        else:
            expired_entries += 1

    return {
        "embedding_cache": {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "max_size": MAX_EMBEDDING_CACHE_SIZE,
            "ttl_seconds": EMBEDDING_CACHE_TTL_SECONDS,
            "recent_keys": list(_embedding_cache.keys())[:10],
        }
    }


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


# =============================================================================
# User History & Zep Integration Endpoints
# =============================================================================

@app.get("/api/user/{user_id}/conversations")
async def get_user_conversations(user_id: str):
    """
    Get all conversation threads for a user from Zep.
    Returns list of sessions with messages.
    """
    from .agent import get_zep_client, ZEP_API_KEY
    import sys

    if not ZEP_API_KEY:
        return {"error": "Zep not configured", "conversations": []}

    try:
        client = get_zep_client()
        if not client:
            return {"error": "Zep client unavailable", "conversations": []}

        # Get all threads for this user
        response = await client.get(
            f"/api/v2/users/{user_id}/threads",
        )

        if response.status_code != 200:
            print(f"[Zep API] Failed to get threads: {response.status_code}", file=sys.stderr)
            return {"error": f"Zep error: {response.status_code}", "conversations": []}

        threads = response.json()
        conversations = []

        # Fetch messages for each thread (limit to recent 10 threads)
        for thread in threads[:10]:
            thread_id = thread.get("thread_id") or thread.get("id")
            if not thread_id:
                continue

            msg_response = await client.get(
                f"/api/v2/threads/{thread_id}/messages",
                params={"limit": 50},
            )

            if msg_response.status_code == 200:
                messages = msg_response.json()
                conversations.append({
                    "session_id": thread_id,
                    "created_at": thread.get("created_at"),
                    "messages": messages.get("messages", messages) if isinstance(messages, dict) else messages,
                })

        return {
            "user_id": user_id,
            "conversation_count": len(conversations),
            "conversations": conversations,
        }

    except Exception as e:
        print(f"[Zep API] Error: {e}", file=sys.stderr)
        return {"error": str(e), "conversations": []}


@app.get("/api/user/{user_id}/facts")
async def get_user_facts(user_id: str):
    """
    Get knowledge graph facts about the user from Zep.
    Returns user profile, preferences, and extracted facts.
    """
    from .agent import get_zep_client, ZEP_API_KEY
    import sys

    if not ZEP_API_KEY:
        return {"error": "Zep not configured", "facts": []}

    try:
        client = get_zep_client()
        if not client:
            return {"error": "Zep client unavailable", "facts": []}

        # Get user's knowledge graph edges (facts)
        response = await client.post(
            "/api/v2/graph/search",
            json={
                "user_id": user_id,
                "query": "user preferences interests topics history",
                "limit": 50,
                "scope": "edges",
            },
        )

        if response.status_code != 200:
            print(f"[Zep API] Failed to get facts: {response.status_code}", file=sys.stderr)
            return {"error": f"Zep error: {response.status_code}", "facts": []}

        data = response.json()
        edges = data.get("edges", [])

        facts = []
        for edge in edges:
            fact = edge.get("fact")
            if fact:
                facts.append({
                    "fact": fact,
                    "created_at": edge.get("created_at"),
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                })

        return {
            "user_id": user_id,
            "fact_count": len(facts),
            "facts": facts,
        }

    except Exception as e:
        print(f"[Zep API] Error: {e}", file=sys.stderr)
        return {"error": str(e), "facts": []}


@app.delete("/api/user/{user_id}/clear")
async def clear_user_history(user_id: str):
    """
    Clear all conversation history and facts for a user.
    Deletes from both Zep and local user_queries table.
    """
    from .agent import get_zep_client, ZEP_API_KEY
    from .database import get_connection
    import sys

    results = {
        "user_id": user_id,
        "zep_cleared": False,
        "local_cleared": False,
        "errors": [],
    }

    # Clear from Zep
    if ZEP_API_KEY:
        try:
            client = get_zep_client()
            if client:
                # Delete user from Zep (cascades to threads and facts)
                response = await client.delete(f"/api/v2/users/{user_id}")
                if response.status_code in [200, 204, 404]:
                    results["zep_cleared"] = True
                    print(f"[Zep API] Cleared user {user_id}", file=sys.stderr)
                else:
                    results["errors"].append(f"Zep delete failed: {response.status_code}")
        except Exception as e:
            results["errors"].append(f"Zep error: {str(e)}")

    # Clear from local database
    try:
        async with get_connection() as conn:
            deleted = await conn.execute(
                "DELETE FROM user_queries WHERE user_id = $1",
                user_id
            )
            results["local_cleared"] = True
            print(f"[DB] Cleared queries for user {user_id}", file=sys.stderr)
    except Exception as e:
        results["errors"].append(f"DB error: {str(e)}")

    results["success"] = results["zep_cleared"] or results["local_cleared"]
    return results


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
