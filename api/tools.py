"""Tools for the Pydantic AI agent - article search and user memory."""

import os
import httpx
from typing import Optional
from pydantic_ai import RunContext

from .models import SearchResults, ArticleResult
from .database import search_articles_hybrid

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL = "voyage-2"
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
LOST_LONDON_GRAPH_ID = "lost-london"

# OPTIMIZATION: Persistent HTTP clients for connection reuse
_voyage_client: Optional[httpx.AsyncClient] = None


def get_voyage_client() -> httpx.AsyncClient:
    """Get or create persistent Voyage HTTP client."""
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = httpx.AsyncClient(
            base_url="https://api.voyageai.com",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
    return _voyage_client


# Phonetic corrections for voice input - matching lost.london
PHONETIC_CORRECTIONS: dict[str, str] = {
    # Names
    "ignacio": "ignatius",
    "ignasio": "ignatius",
    "ignacius": "ignatius",
    # Places
    "thorny": "thorney",
    "fawny": "thorney",
    "fauny": "thorney",
    "forney": "thorney",
    "tie burn": "tyburn",
    "tieburn": "tyburn",
    "aquarim": "aquarium",
    "aquariam": "aquarium",
    "royale": "royal",
    "cristal": "crystal",
    "crystle": "crystal",
    "shakespear": "shakespeare",
    "shakespere": "shakespeare",
    "westmister": "westminster",
    "white hall": "whitehall",
    "parliment": "parliament",
    "tems": "thames",
    "devils acre": "devil's acre",
    "devil acre": "devil's acre",
    # Additional corrections
    "voxhall": "vauxhall",
    "vox hall": "vauxhall",
    "southwork": "southwark",
    "grenwich": "greenwich",
    "wolwich": "woolwich",
    "bermondsy": "bermondsey",
    "holbourn": "holborn",
    "aldwich": "aldwych",
    "chisick": "chiswick",
    "dulwitch": "dulwich",
}


def normalize_query(query: str) -> str:
    """Apply phonetic corrections to normalize the query."""
    import re
    normalized = query.lower().strip()

    for wrong, correct in PHONETIC_CORRECTIONS.items():
        # Use word boundary matching for accuracy
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        normalized = pattern.sub(correct, normalized)

    return normalized


async def get_voyage_embedding(text: str) -> list[float]:
    """Generate embedding using Voyage AI with persistent client."""
    client = get_voyage_client()
    response = await client.post(
        "/v1/embeddings",
        json={
            "model": VOYAGE_MODEL,
            "input": text,
            "input_type": "query",
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


async def search_zep_graph(query: str) -> dict:
    """
    Search Zep knowledge graph for entity connections.

    Returns facts and relationships that span multiple articles.
    This enriches responses with connections the user might not know about.
    """
    if not ZEP_API_KEY:
        return {"facts": [], "connections": []}

    try:
        async with httpx.AsyncClient() as client:
            # Search for edges (relationships between entities)
            edge_response = await client.post(
                f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "limit": 5,
                    "scope": "edges",
                    "reranker": "rrf",
                },
                timeout=5.0,  # Fast timeout - enrichment shouldn't slow us down
            )

            facts = []
            connections = []

            if edge_response.status_code == 200:
                data = edge_response.json()
                edges = data.get("edges", [])

                for edge in edges:
                    # Extract fact
                    if edge.get("fact"):
                        facts.append(edge["fact"])

                    # Extract connection
                    source = edge.get("source_node_name")
                    target = edge.get("target_node_name")
                    relation = edge.get("relation")
                    if source and target and relation:
                        connections.append({
                            "from": source,
                            "relation": relation,
                            "to": target
                        })

            return {"facts": facts[:3], "connections": connections[:3]}

    except Exception as e:
        import sys
        print(f"[Zep Graph] Error: {e}", file=sys.stderr)
        return {"facts": [], "connections": []}


async def search_articles(ctx: RunContext[None], query: str) -> SearchResults:
    """
    Search Lost London articles using hybrid vector + keyword search.

    This tool searches the knowledge base for articles relevant to the query.
    Results are ranked by semantic similarity and keyword matching.

    Args:
        query: The user's question or topic to search for

    Returns:
        SearchResults containing matching articles and the normalized query
    """
    # Normalize query with phonetic corrections
    normalized_query = normalize_query(query)

    # Get embedding
    embedding = await get_voyage_embedding(normalized_query)

    # Search database
    results = await search_articles_hybrid(
        query_embedding=embedding,
        query_text=normalized_query,
        limit=5,
        similarity_threshold=0.45,
    )

    articles = [
        ArticleResult(
            id=r["id"],
            title=r["title"],
            content=r["content"],
            score=r["score"],
        )
        for r in results
    ]

    return SearchResults(articles=articles, query=normalized_query)


async def get_user_memory(ctx: RunContext[None], user_id: str) -> dict:
    """
    Retrieve user facts and preferences from Zep Cloud knowledge graph.

    Zep automatically extracts facts and entities from conversations,
    so this retrieves what Zep knows about the user.

    Args:
        user_id: The unique identifier for the user

    Returns:
        Dictionary containing user facts and memories from the graph
    """
    if not ZEP_API_KEY or not user_id:
        return {"facts": [], "memories": []}

    async with httpx.AsyncClient() as client:
        try:
            # Search user's personal graph for facts about them
            response = await client.post(
                "https://api.getzep.com/api/v2/graph/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "user_id": user_id,
                    "query": "user preferences interests name",
                    "limit": 10,
                    "scope": "edges",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Extract facts from edges
            facts = []
            for edge in data.get("edges", []):
                if edge.get("fact"):
                    facts.append(edge["fact"])

            return {
                "facts": facts,
                "memories": data.get("edges", []),
            }
        except Exception:
            # Fail gracefully - memory is optional
            return {"facts": [], "memories": []}


async def save_user_message(user_id: str, message: str, role: str = "user") -> bool:
    """
    Add a message to the user's Zep memory graph.

    Zep automatically extracts facts and entities from messages,
    so no explicit "remember" logic is needed.

    Args:
        user_id: The unique identifier for the user
        message: The message content
        role: Either "user" or "assistant"

    Returns:
        True if saved successfully, False otherwise
    """
    if not ZEP_API_KEY or not user_id:
        return False

    async with httpx.AsyncClient() as client:
        try:
            # First ensure user exists
            await client.post(
                "https://api.getzep.com/api/v2/users",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"user_id": user_id},
                timeout=10.0,
            )
        except Exception:
            pass  # User may already exist

        try:
            # Add message to user's graph
            response = await client.post(
                "https://api.getzep.com/api/v2/graph",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "user_id": user_id,
                    "type": "message",
                    "data": f"{role}: {message}",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False


async def search_knowledge_graph(query: str, limit: int = 10) -> list[dict]:
    """
    Search the Lost London knowledge graph via Zep.

    This provides relationship-aware search across articles.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of relevant edges/relationships from the graph
    """
    if not ZEP_API_KEY:
        return []

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.getzep.com/api/v2/graph/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "graph_id": LOST_LONDON_GRAPH_ID,
                    "query": query,
                    "limit": limit,
                    "scope": "edges",
                    "reranker": "rrf",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("edges", [])
        except Exception:
            return []


# =============================================================================
# Agent Tools - For enrichment and multi-step reasoning
# =============================================================================

from .models import ExtractedEntity, EntityType, EntityConnection, SuggestedTopic, RelatedArticleResult
from .agent_deps import VICAgentDeps
import re


# Known London places for entity extraction
KNOWN_PLACES = {
    "westminster", "whitehall", "crystal palace", "hyde park", "tyburn",
    "southwark", "greenwich", "woolwich", "bermondsey", "holborn", "aldwych",
    "chiswick", "dulwich", "vauxhall", "lambeth", "islington", "hackney",
    "shoreditch", "spitalfields", "clerkenwell", "bloomsbury", "soho",
    "covent garden", "strand", "fleet street", "cheapside", "lombard street",
    "tower hill", "tower of london", "london bridge", "westminster bridge",
    "trafalgar square", "piccadilly", "mayfair", "chelsea", "kensington",
    "notting hill", "paddington", "marylebone", "regent's park", "st james",
    "thorney island", "bankside", "southbank", "embankment", "thames",
    "royal aquarium", "methodist central hall", "parliament square",
}

# Known eras for entity extraction
KNOWN_ERAS = {
    "victorian": "Victorian Era (1837-1901)",
    "georgian": "Georgian Era (1714-1830)",
    "elizabethan": "Elizabethan Era (1558-1603)",
    "medieval": "Medieval Period (500-1500)",
    "tudor": "Tudor Period (1485-1603)",
    "stuart": "Stuart Period (1603-1714)",
    "regency": "Regency Era (1811-1820)",
    "edwardian": "Edwardian Era (1901-1910)",
    "roman": "Roman Britain (43-410 AD)",
    "anglo-saxon": "Anglo-Saxon Period (450-1066)",
    "18th century": "18th Century",
    "19th century": "19th Century",
    "20th century": "20th Century",
}


async def extract_entities(
    ctx: RunContext[VICAgentDeps],
    article_content: str,
    article_title: str
) -> list[ExtractedEntity]:
    """
    Extract entities (people, places, eras, buildings) from article content.

    Uses fast regex-based extraction - no LLM call required.
    This runs in the enrichment path to build context for future queries.

    Args:
        article_content: The full text of the article
        article_title: Title of the article (for context)

    Returns:
        List of extracted entities with their types and context snippets
    """
    entities: list[ExtractedEntity] = []
    content_lower = article_content.lower()

    # Extract known places
    for place in KNOWN_PLACES:
        if place in content_lower:
            # Find context snippet
            idx = content_lower.find(place)
            start = max(0, idx - 30)
            end = min(len(article_content), idx + len(place) + 50)
            context = article_content[start:end].strip()

            entities.append(ExtractedEntity(
                name=place.title(),
                entity_type=EntityType.PLACE,
                context=f"...{context}...",
                article_title=article_title,
            ))

    # Extract eras
    for era_key, era_name in KNOWN_ERAS.items():
        if era_key in content_lower:
            idx = content_lower.find(era_key)
            start = max(0, idx - 30)
            end = min(len(article_content), idx + len(era_key) + 50)
            context = article_content[start:end].strip()

            entities.append(ExtractedEntity(
                name=era_name,
                entity_type=EntityType.ERA,
                context=f"...{context}...",
                article_title=article_title,
            ))

    # Extract people (capitalized names - heuristic)
    # Pattern: Two or more capitalized words together (likely a person's name)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    names = re.findall(name_pattern, article_content)

    # Filter out known places and common false positives
    false_positives = {"Crystal Palace", "Hyde Park", "Tower Of", "Tower Hill",
                       "London Bridge", "Royal Aquarium", "Central Hall"}
    for name in set(names):
        if name not in false_positives and len(name.split()) <= 4:
            # Find context
            idx = article_content.find(name)
            if idx >= 0:
                start = max(0, idx - 30)
                end = min(len(article_content), idx + len(name) + 50)
                context = article_content[start:end].strip()

                entities.append(ExtractedEntity(
                    name=name,
                    entity_type=EntityType.PERSON,
                    context=f"...{context}...",
                    article_title=article_title,
                ))

    # Extract buildings (pattern: "The X" followed by building-like words)
    building_pattern = r'(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Hall|Palace|Theatre|Theater|Church|Cathedral|Abbey|House|Hotel|Station|Market|Exchange|Hospital|Museum|Gallery)'
    buildings = re.findall(building_pattern, article_content)
    for building in set(buildings):
        full_name = building
        idx = article_content.find(building)
        if idx >= 0:
            # Get the full building name with suffix
            end_match = re.search(rf'{re.escape(building)}\s+\w+', article_content[idx:])
            if end_match:
                full_name = end_match.group()

            start = max(0, idx - 20)
            end_idx = min(len(article_content), idx + len(full_name) + 40)
            context = article_content[start:end_idx].strip()

            entities.append(ExtractedEntity(
                name=full_name,
                entity_type=EntityType.BUILDING,
                context=f"...{context}...",
                article_title=article_title,
            ))

    # Deduplicate by name
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.name.lower() not in seen:
            seen.add(entity.name.lower())
            unique_entities.append(entity)

    return unique_entities[:20]  # Limit to top 20 entities


async def traverse_graph_connections(
    ctx: RunContext[VICAgentDeps],
    start_entity: str,
    max_depth: int = 2
) -> list[EntityConnection]:
    """
    Traverse the Zep knowledge graph to find connections from an entity.

    Performs multi-hop traversal to discover hidden relationships.
    E.g., "Royal Aquarium" -> "Zazel" -> "Human Cannonball performances"

    Args:
        start_entity: The entity name to start from
        max_depth: Maximum hops to traverse (default 2)

    Returns:
        List of entity connections discovered through traversal
    """
    if not ZEP_API_KEY:
        return []

    connections: list[EntityConnection] = []

    try:
        async with httpx.AsyncClient() as client:
            # Search for edges involving this entity
            response = await client.post(
                f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                headers={
                    "Authorization": f"Api-Key {ZEP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": start_entity,
                    "limit": 10,
                    "scope": "edges",
                    "reranker": "rrf",
                },
                timeout=5.0,
            )

            if response.status_code != 200:
                return []

            data = response.json()
            edges = data.get("edges", [])

            # First hop - direct connections
            for edge in edges:
                source = edge.get("source_node_name")
                target = edge.get("target_node_name")
                relation = edge.get("relation")
                fact = edge.get("fact")

                if source and target and relation:
                    connections.append(EntityConnection(
                        from_entity=source,
                        relation=relation,
                        to_entity=target,
                        fact=fact,
                    ))

            # Second hop - follow connections if max_depth > 1
            if max_depth > 1 and connections:
                # Get unique target entities from first hop
                second_hop_entities = set(
                    c.to_entity for c in connections
                    if c.to_entity.lower() != start_entity.lower()
                )

                for entity in list(second_hop_entities)[:3]:  # Limit second hop
                    hop_response = await client.post(
                        f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                        headers={
                            "Authorization": f"Api-Key {ZEP_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": entity,
                            "limit": 5,
                            "scope": "edges",
                            "reranker": "rrf",
                        },
                        timeout=3.0,
                    )

                    if hop_response.status_code == 200:
                        hop_data = hop_response.json()
                        for edge in hop_data.get("edges", []):
                            source = edge.get("source_node_name")
                            target = edge.get("target_node_name")
                            relation = edge.get("relation")
                            fact = edge.get("fact")

                            if source and target and relation:
                                connections.append(EntityConnection(
                                    from_entity=source,
                                    relation=relation,
                                    to_entity=target,
                                    fact=fact,
                                ))

    except Exception as e:
        import sys
        print(f"[Graph Traversal] Error: {e}", file=sys.stderr)

    # Deduplicate connections
    seen = set()
    unique_connections = []
    for conn in connections:
        key = (conn.from_entity.lower(), conn.relation.lower(), conn.to_entity.lower())
        if key not in seen:
            seen.add(key)
            unique_connections.append(conn)

    return unique_connections[:15]


async def find_related_articles(
    ctx: RunContext[VICAgentDeps],
    entity_names: list[str],
    relation_type: str = "any"
) -> list[RelatedArticleResult]:
    """
    Find articles related to the given entities through the knowledge graph.

    Queries Zep for connected entities, then fetches matching articles from Neon.

    Args:
        entity_names: List of entity names to find related articles for
        relation_type: Filter by relation type ("same_era", "same_location", "same_person", "any")

    Returns:
        List of related articles with connection explanations
    """
    related_articles: list[RelatedArticleResult] = []

    if not entity_names:
        return related_articles

    # First, find connected entities through the graph
    all_connected_entities: set[str] = set()

    if ZEP_API_KEY:
        async with httpx.AsyncClient() as client:
            for entity in entity_names[:3]:  # Limit to top 3 entities
                try:
                    response = await client.post(
                        f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                        headers={
                            "Authorization": f"Api-Key {ZEP_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": entity,
                            "limit": 5,
                            "scope": "edges",
                        },
                        timeout=3.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for edge in data.get("edges", []):
                            target = edge.get("target_node_name")
                            source = edge.get("source_node_name")
                            if target:
                                all_connected_entities.add(target)
                            if source and source.lower() != entity.lower():
                                all_connected_entities.add(source)
                except Exception:
                    pass

    # Now search for articles mentioning these connected entities
    # Combine into a search query
    if all_connected_entities:
        search_query = " ".join(list(all_connected_entities)[:5])
    else:
        search_query = " ".join(entity_names)

    # Get embedding and search
    try:
        embedding = await get_voyage_embedding(search_query)
        results = await search_articles_hybrid(
            query_embedding=embedding,
            query_text=search_query,
            limit=5,
            similarity_threshold=0.3,
        )

        for r in results:
            # Determine relation type
            content_lower = r["content"].lower()
            detected_relation = "related_topic"
            relation_detail = "Related to your query"

            # Check for specific relation types
            for entity in entity_names:
                if entity.lower() in content_lower:
                    detected_relation = "same_entity"
                    relation_detail = f"Also mentions {entity}"
                    break

            for era in KNOWN_ERAS:
                if era in content_lower and era in search_query.lower():
                    detected_relation = "same_era"
                    relation_detail = f"From the same era: {KNOWN_ERAS[era]}"
                    break

            for place in KNOWN_PLACES:
                if place in content_lower and place in search_query.lower():
                    detected_relation = "same_location"
                    relation_detail = f"Same location: {place.title()}"
                    break

            # Apply relation_type filter
            if relation_type != "any" and detected_relation != relation_type:
                continue

            related_articles.append(RelatedArticleResult(
                id=r["id"],
                title=r["title"],
                content=r["content"][:500],  # Truncate for efficiency
                score=r["score"],
                relation_type=detected_relation,
                relation_detail=relation_detail,
            ))

    except Exception as e:
        import sys
        print(f"[Related Articles] Error: {e}", file=sys.stderr)

    return related_articles[:5]


async def suggest_followup_topics(
    ctx: RunContext[VICAgentDeps],
    current_topic: str,
    entities: list[str],
) -> list[SuggestedTopic]:
    """
    Suggest compelling follow-up topics based on current discussion.

    Uses graph connections, entities mentioned, and user interests to
    generate engaging suggestions with teasers.

    Args:
        current_topic: The main topic currently being discussed
        entities: List of entities mentioned in the current conversation

    Returns:
        List of suggested topics with reasons and teasers
    """
    suggestions: list[SuggestedTopic] = []
    deps = ctx.deps

    # Get user's prior interests if available
    prior_topics = deps.prior_topics if deps else []
    prior_entities = deps.prior_entities if deps else []

    # Find graph connections for suggestions
    connected_topics: list[tuple[str, str]] = []  # (topic, reason)

    if ZEP_API_KEY and entities:
        async with httpx.AsyncClient() as client:
            for entity in entities[:2]:  # Check top 2 entities
                try:
                    response = await client.post(
                        f"https://api.getzep.com/api/v2/graph/{LOST_LONDON_GRAPH_ID}/search",
                        headers={
                            "Authorization": f"Api-Key {ZEP_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": entity,
                            "limit": 5,
                            "scope": "edges",
                        },
                        timeout=3.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for edge in data.get("edges", []):
                            target = edge.get("target_node_name")
                            relation = edge.get("relation")
                            fact = edge.get("fact")
                            if target and target.lower() != entity.lower():
                                reason = f"Connected to {entity} via {relation}" if relation else f"Related to {entity}"
                                connected_topics.append((target, reason))
                except Exception:
                    pass

    # Generate suggestions from connected topics
    for topic, reason in connected_topics[:3]:
        # Skip if already discussed
        if topic.lower() in [t.lower() for t in prior_topics]:
            continue

        # Create engaging teaser
        teaser = f"Speaking of {current_topic}, did you know about {topic}? There's a fascinating connection..."

        suggestions.append(SuggestedTopic(
            topic=topic,
            reason=reason,
            teaser=teaser,
        ))

    # If we don't have enough from graph, suggest based on current entities
    if len(suggestions) < 2:
        for entity in entities:
            if entity.lower() not in [s.topic.lower() for s in suggestions]:
                suggestions.append(SuggestedTopic(
                    topic=entity,
                    reason=f"Mentioned in the {current_topic} discussion",
                    teaser=f"Would you like to hear more about {entity}? There's quite a story there...",
                ))
                if len(suggestions) >= 3:
                    break

    return suggestions[:3]
