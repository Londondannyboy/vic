"""Pydantic AI validation for VIC content moderation and entity validation.

Validates:
1. Content is not offensive/inappropriate before storing
2. Topics match actual articles in our database
3. Provides warnings for inappropriate content
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional, Literal
from enum import Enum
import os
import re


# =============================================================================
# Pydantic Models for Validation
# =============================================================================

class ContentCategory(str, Enum):
    """Categories of content for moderation."""
    SAFE = "safe"
    OFFENSIVE = "offensive"
    INAPPROPRIATE = "inappropriate"
    OFF_TOPIC = "off_topic"
    SPAM = "spam"


class ContentValidationResult(BaseModel):
    """Result of content validation check."""
    is_valid: bool = Field(description="Whether content is appropriate to store")
    category: ContentCategory = Field(description="Category of the content")
    reason: Optional[str] = Field(default=None, description="Reason if content is invalid")
    sanitized_text: Optional[str] = Field(default=None, description="Cleaned version if minor issues")
    requires_user_confirmation: bool = Field(default=False, description="If user should confirm appropriateness")
    vic_warning: Optional[str] = Field(default=None, description="Warning message VIC should say")


class TopicValidationResult(BaseModel):
    """Result of topic validation against article database."""
    is_valid_topic: bool = Field(description="Whether topic matches an article")
    matched_article_id: Optional[int] = Field(default=None, description="ID of matched article")
    matched_article_title: Optional[str] = Field(default=None, description="Title of matched article")
    matched_article_slug: Optional[str] = Field(default=None, description="Slug of matched article")
    confidence: float = Field(default=0.0, description="Confidence in the match (0-1)")


class ValidatedEntity(BaseModel):
    """An entity that has been validated against our database."""
    name: str = Field(description="Entity name")
    entity_type: Literal["article", "location", "person", "era", "unknown"] = Field(description="Type of entity")
    is_from_database: bool = Field(description="Whether this entity exists in our database")
    article_id: Optional[int] = Field(default=None, description="Linked article ID if applicable")
    article_title: Optional[str] = Field(default=None, description="Linked article title")


# =============================================================================
# Offensive Content Patterns (Fast Rule-Based Check)
# =============================================================================

# Words/phrases that are always inappropriate
BANNED_WORDS = {
    # Slurs and hate speech
    "nigger", "faggot", "retard", "spastic", "cunt",
    # Explicit sexual content
    "fuck me", "suck my", "dick pic", "nude pic",
    # Violence
    "kill yourself", "kys", "die in a fire",
    # Add more as needed
}

# Patterns that indicate potentially inappropriate content
SUSPICIOUS_PATTERNS = [
    r'\b(fuck|shit|ass|bitch|dick|cock|pussy)\b',  # Profanity
    r'\b(sex|porn|nude|naked)\b',  # Sexual content
    r'\b(kill|murder|rape|assault)\b.*\b(you|me|them|her|him)\b',  # Violent threats
    r'\b(racist|sexist)\s+joke',  # Discriminatory content requests
]

# Patterns that suggest off-topic (not London history)
OFF_TOPIC_PATTERNS = [
    r'\b(crypto|bitcoin|nft|forex|stock market)\b',  # Finance spam
    r'\b(diet pill|weight loss|viagra|cialis)\b',  # Spam products
    r'\b(my (car|dog|cat|pet))\b',  # Personal non-London topics
    r'\b(politics|election|trump|biden|brexit)\b',  # Political content
]


def fast_content_check(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Fast rule-based content check (no LLM needed).

    Returns: (is_clean, category, warning_message)
    """
    text_lower = text.lower()

    # Check banned words first (immediate rejection)
    for banned in BANNED_WORDS:
        if banned in text_lower:
            return (
                False,
                "offensive",
                "I'm afraid I can't engage with that kind of language. Let's keep our conversation respectful. Is there something about London's history I can help you with?"
            )

    # Check suspicious patterns
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text_lower):
            return (
                False,
                "inappropriate",
                "That's not quite the sort of topic I cover. I specialise in London's hidden history - fascinating tales of the city's past. What would you like to explore?"
            )

    # Check off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return (
                False,
                "off_topic",
                "That's a bit outside my area of expertise. I'm VIC, your guide to London's hidden history. Would you like to hear about something from London's past instead?"
            )

    return (True, "safe", None)


# =============================================================================
# Database Validation Functions
# =============================================================================

async def validate_topic_against_database(
    topic: str,
    similarity_threshold: float = 0.4
) -> TopicValidationResult:
    """
    Validate that a topic matches an article in our database.

    Uses semantic search to find matching articles.
    """
    from .database import search_articles_hybrid, get_connection
    from .tools import get_voyage_embedding, normalize_query

    try:
        # Normalize and embed the topic
        normalized = normalize_query(topic)
        embedding = await get_voyage_embedding(normalized)

        if not embedding:
            return TopicValidationResult(is_valid_topic=False, confidence=0.0)

        # Search for matching articles
        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=normalized,
            limit=1,
            similarity_threshold=similarity_threshold,
        )

        if results and len(results) > 0:
            top_result = results[0]
            score = top_result.get('score', 0)

            # Convert RRF score (typically 0.01-0.03) to confidence (0-1)
            # Good matches are >0.02, excellent matches >0.025
            confidence = min(score / 0.03, 1.0)

            return TopicValidationResult(
                is_valid_topic=confidence > 0.5,  # At least 50% confidence
                matched_article_id=top_result.get('article_id'),
                matched_article_title=top_result.get('title'),
                matched_article_slug=top_result.get('article_slug'),
                confidence=confidence,
            )

        return TopicValidationResult(is_valid_topic=False, confidence=0.0)

    except Exception as e:
        import sys
        print(f"[Validation] Topic validation error: {e}", file=sys.stderr)
        return TopicValidationResult(is_valid_topic=False, confidence=0.0)


async def validate_entity(entity_name: str) -> ValidatedEntity:
    """
    Validate an entity exists in our database.

    Checks if the entity matches an article title, location, or known person.
    """
    from .database import get_connection

    try:
        async with get_connection() as conn:
            # Check if it matches an article title
            article = await conn.fetchrow("""
                SELECT id, title, slug
                FROM articles
                WHERE LOWER(title) LIKE $1
                OR title ILIKE $2
                LIMIT 1
            """, f"%{entity_name.lower()}%", f"%{entity_name}%")

            if article:
                return ValidatedEntity(
                    name=entity_name,
                    entity_type="article",
                    is_from_database=True,
                    article_id=article['id'],
                    article_title=article['title'],
                )

            # Check knowledge chunks for locations/people mentions
            chunk = await conn.fetchrow("""
                SELECT source_id, title
                FROM knowledge_chunks
                WHERE content ILIKE $1
                LIMIT 1
            """, f"%{entity_name}%")

            if chunk:
                return ValidatedEntity(
                    name=entity_name,
                    entity_type="location" if any(loc in entity_name.lower() for loc in ['street', 'bridge', 'square', 'lane', 'road', 'palace', 'tower']) else "person",
                    is_from_database=True,
                    article_id=chunk['source_id'],
                    article_title=chunk['title'],
                )

            # Not found in database
            return ValidatedEntity(
                name=entity_name,
                entity_type="unknown",
                is_from_database=False,
            )

    except Exception as e:
        import sys
        print(f"[Validation] Entity validation error: {e}", file=sys.stderr)
        return ValidatedEntity(
            name=entity_name,
            entity_type="unknown",
            is_from_database=False,
        )


# =============================================================================
# Main Validation Function
# =============================================================================

async def validate_user_input(
    user_message: str,
    check_topic: bool = True,
) -> ContentValidationResult:
    """
    Full validation of user input before storing/processing.

    Args:
        user_message: The user's message to validate
        check_topic: Whether to also validate the topic matches our database

    Returns:
        ContentValidationResult with validation status and any warnings
    """
    import sys

    # Step 1: Fast rule-based check (no LLM needed)
    is_clean, category, warning = fast_content_check(user_message)

    if not is_clean:
        print(f"[Validation] Content blocked: {category}", file=sys.stderr)
        return ContentValidationResult(
            is_valid=False,
            category=ContentCategory(category),
            reason=f"Content flagged as {category}",
            vic_warning=warning,
        )

    # Step 2: Check if topic matches our database (optional)
    if check_topic and len(user_message) > 3:
        topic_result = await validate_topic_against_database(user_message)

        if not topic_result.is_valid_topic:
            print(f"[Validation] Topic not in database: {user_message[:50]}", file=sys.stderr)
            return ContentValidationResult(
                is_valid=False,
                category=ContentCategory.OFF_TOPIC,
                reason="Topic does not match any article in database",
                requires_user_confirmation=True,
                vic_warning="I'm not sure I have information about that specific topic. My collection focuses on London's hidden history. Would you like to try a different topic?",
            )

    # Content is valid
    return ContentValidationResult(
        is_valid=True,
        category=ContentCategory.SAFE,
    )


async def should_store_to_zep(
    user_message: str,
    article_matched: bool,
    article_title: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Decide whether to store this interaction in Zep.

    Only stores validated, on-topic conversations with matched articles.

    Returns: (should_store, reason_if_not)
    """
    # Don't store short messages (affirmations, etc.)
    if len(user_message.strip()) < 5:
        return (False, "Message too short")

    # Don't store if content check fails
    is_clean, category, _ = fast_content_check(user_message)
    if not is_clean:
        return (False, f"Content flagged: {category}")

    # Only store if we matched an article (validates topic is London-related)
    if not article_matched:
        return (False, "No article match - topic not validated")

    # Passed all checks - safe to store
    return (True, None)


# =============================================================================
# VIC Warning Responses
# =============================================================================

VIC_WARNINGS = {
    ContentCategory.OFFENSIVE: [
        "I'm afraid I can't engage with that kind of language. Let's keep our conversation respectful and focused on London's fascinating history.",
        "That's not the sort of thing I discuss. I'm here to share stories of London's past. Shall we explore something more uplifting?",
        "I must ask that we keep things civil. Now, is there something about London's hidden history I can help you discover?",
    ],
    ContentCategory.INAPPROPRIATE: [
        "That's rather outside my remit. I specialise in London's historical tales - the hidden stories of the city.",
        "I'm not sure that's quite appropriate for our discussion. Would you like to hear about something from London's past instead?",
        "Let's steer back to more suitable territory. I've got wonderful stories about London waiting to be told.",
    ],
    ContentCategory.OFF_TOPIC: [
        "That's not really my area of expertise. I'm VIC - your guide to London's hidden history. What aspect of London would you like to explore?",
        "I'm afraid that's outside my collection. I focus on London's fascinating past. Shall I suggest something?",
        "That topic is a bit beyond me. But I've got 370 articles about London's hidden history - where shall we start?",
    ],
    ContentCategory.SPAM: [
        "That doesn't seem like a genuine question about London. I'm here to share the city's hidden history.",
        "I think there might be some confusion. I'm VIC, a guide to London's past. How can I help you explore the city's history?",
    ],
}


def get_vic_warning(category: ContentCategory) -> str:
    """Get an appropriate warning response from VIC."""
    import random
    warnings = VIC_WARNINGS.get(category, VIC_WARNINGS[ContentCategory.OFF_TOPIC])
    return random.choice(warnings)
