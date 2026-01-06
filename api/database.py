"""Database connection and queries for Neon PostgreSQL with pgvector."""

import os
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

DATABASE_URL = os.environ.get("DATABASE_URL", "")


class Database:
    """Async database connection manager for Neon PostgreSQL."""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create connection pool."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        return cls._pool

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a database connection from the pool."""
    pool = await Database.get_pool()
    async with pool.acquire() as conn:
        yield conn


async def search_articles_hybrid(
    query_embedding: list[float],
    query_text: str,
    limit: int = 5,
    similarity_threshold: float = 0.3
) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).

    RRF combines vector and keyword search by rank position, not raw scores.
    Formula: RRF_score = 1/(k + vector_rank) + 1/(k + keyword_rank)
    where k=60 is the standard constant.

    This approach (used by Cole Meddin's MongoDB-RAG-Agent) provides better
    accuracy than weighted score combination.

    Args:
        query_embedding: Vector embedding of the query
        query_text: Original query text for keyword matching
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score (for filtering)

    Returns:
        List of matching articles with RRF scores
    """
    import json
    import sys

    async with get_connection() as conn:
        embedding_json = json.dumps(query_embedding)

        # RRF with k=60 (industry standard)
        # Get ranked results from both vector and keyword searches
        results = await conn.fetch("""
            WITH
            -- Vector search: rank by embedding similarity
            vector_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) as vector_rank,
                    1 - (embedding <=> $1::vector) as vector_score
                FROM knowledge_chunks
                WHERE 1 - (embedding <=> $1::vector) > 0.3  -- Basic threshold
                LIMIT 50
            ),
            -- Keyword search: rank by text match quality
            keyword_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY keyword_score DESC) as keyword_rank,
                    keyword_score
                FROM (
                    SELECT id,
                        CASE
                            WHEN LOWER(content) LIKE '%' || $2 || '%' THEN 3
                            WHEN LOWER(title) LIKE '%' || $2 || '%' THEN 2
                            ELSE 0
                        END as keyword_score
                    FROM knowledge_chunks
                    WHERE LOWER(content) LIKE '%' || $2 || '%'
                       OR LOWER(title) LIKE '%' || $2 || '%'
                ) keyword_matches
                WHERE keyword_score > 0
            ),
            -- RRF: Combine ranks using reciprocal rank fusion
            rrf_combined AS (
                SELECT
                    COALESCE(v.id, k.id) as id,
                    -- RRF formula: 1/(60 + rank) for each search method
                    COALESCE(1.0 / (60 + v.vector_rank), 0) +
                    COALESCE(1.0 / (60 + k.keyword_rank), 0) as rrf_score,
                    v.vector_score,
                    v.vector_rank,
                    k.keyword_rank
                FROM vector_ranked v
                FULL OUTER JOIN keyword_ranked k ON v.id = k.id
            )
            SELECT
                kc.id::text,
                kc.title,
                kc.content,
                kc.source_type,
                r.rrf_score as score,
                r.vector_score,
                r.vector_rank,
                r.keyword_rank
            FROM rrf_combined r
            JOIN knowledge_chunks kc ON kc.id = r.id
            ORDER BY r.rrf_score DESC
            LIMIT $3
        """, embedding_json, query_text.lower(), limit)

        print(f"[VIC RRF] Query: '{query_text[:30]}...' → {len(results)} results", file=sys.stderr)
        for r in results[:3]:
            print(f"[VIC RRF]   {r['title'][:40]}... (RRF={r['score']:.4f}, vec_rank={r['vector_rank']}, kw_rank={r['keyword_rank']})", file=sys.stderr)

        return [dict(r) for r in results]


async def get_cached_response(query: str) -> Optional[dict]:
    """
    Check if we have a cached response for this query or its variations.

    Returns cached response if found, None otherwise.
    """
    query_lower = query.lower().strip()

    async with get_connection() as conn:
        # Check if query matches any variation
        result = await conn.fetchrow("""
            SELECT normalized_query, response_text, article_titles
            FROM vic_response_cache
            WHERE response_text IS NOT NULL
              AND ($1 = ANY(variations) OR normalized_query = $1)
        """, query_lower)

        if result and result['response_text']:
            # Update hit count
            await conn.execute("""
                UPDATE vic_response_cache
                SET hit_count = hit_count + 1, last_hit_at = NOW()
                WHERE normalized_query = $1
            """, result['normalized_query'])

            return {
                "response": result['response_text'],
                "articles": result['article_titles'] or [],
                "cached": True
            }

    return None


async def cache_response(query: str, response: str, article_titles: list[str]) -> None:
    """
    Cache a response for a query.

    If the query matches an existing variation, updates that entry.
    Otherwise creates a new cache entry.
    """
    query_lower = query.lower().strip()

    async with get_connection() as conn:
        # Check if this query matches an existing cache entry's variations
        existing = await conn.fetchrow("""
            SELECT normalized_query FROM vic_response_cache
            WHERE $1 = ANY(variations) OR normalized_query = $1
        """, query_lower)

        if existing:
            # Update existing entry with response
            await conn.execute("""
                UPDATE vic_response_cache
                SET response_text = $1, article_titles = $2, last_hit_at = NOW()
                WHERE normalized_query = $3
            """, response, article_titles, existing['normalized_query'])
        else:
            # Create new cache entry
            await conn.execute("""
                INSERT INTO vic_response_cache (normalized_query, variations, response_text, article_titles)
                VALUES ($1, ARRAY[$1], $2, $3)
                ON CONFLICT (normalized_query) DO UPDATE
                SET response_text = $2, article_titles = $3, last_hit_at = NOW()
            """, query_lower, response, article_titles)


