# VIC CLM - Claude Code Notes

## Project Overview
Custom Language Model (CLM) backend for Lost London voice assistant (VIC).
- **Frontend**: `/Users/dankeegan/lost.london-app` (Next.js, deployed to lost.london)
- **CLM Backend**: This repo, deployed to `https://vic-clm.vercel.app`
- **Voice**: Hume EVI integration

## Current Status (Jan 6, 2026)

### What Happened - The Outage
1. **Jan 5, 2026**: Made changes to switch from Groq to Google Gemini to fix rate limiting
2. **The Problem**: Added `google-generativeai` package (200MB+) which exceeded Vercel's 250MB serverless limit
3. **Result**: Deployment failed silently, app broke with "UnboundLocalError" messages
4. **Fix**: Rolled back to Dec 27 stable version (`fa549cf`)

### What Was Rolled Back
~2300 lines of code from Jan 5th including:
- Pydantic AI content validation/moderation
- Validation encouragement prompts
- Human-in-the-loop interest validation
- Smart Zep-powered returning user greetings
- Query tracking improvements
- Content filtering for affirmations

### What Still Works (Dec 27 version)
- **Zep integration** - Full conversation memory, user graphs, entity search
- **Groq LLM** - Using llama-3.3-70b-versatile (fast, reliable)
- **Pydantic AI agent** - Structured responses with validation
- **Article search** - pgvector + hybrid search
- **Name spacing** - Doesn't say user's name every turn
- **Affirmation handling** - "yes/sure" uses last suggestion
- **Safe topic fallbacks** - When search fails

### Frontend Features (Unaffected)
The frontend (lost.london-app) was NOT rolled back:
- Dashboard with transcripts
- Recent topics/conversations
- User query history
- Interest validation UI
- All UI improvements intact

## Architecture

```
User Voice → Hume EVI → vic-clm.vercel.app/chat/completions → Response
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              Groq LLM          pgvector search
           (llama-3.3-70b)      (Neon database)
                    │                   │
                    └─────────┬─────────┘
                              ↓
                         Zep Memory
                    (user conversation history)
```

## Key Files
| File | Purpose |
|------|---------|
| `api/index.py` | FastAPI + `/chat/completions` endpoint |
| `api/agent.py` | Pydantic AI agent with Vic persona |
| `api/tools.py` | Article search + Zep memory integration |
| `api/database.py` | Neon pgvector queries |
| `api/agent_config.py` | Agent configuration, system prompts |

## Environment Variables (Vercel)
- `DATABASE_URL` - Neon PostgreSQL
- `VOYAGE_API_KEY` - Embeddings
- `GROQ_API_KEY` - LLM (llama-3.3-70b)
- `ZEP_API_KEY` - Conversation memory
- `CLM_AUTH_TOKEN` - Hume authentication

## Lessons Learned

### 1. Package Size Matters on Vercel
- Vercel serverless has 250MB limit
- `google-generativeai` SDK is huge (~200MB with dependencies)
- Use REST APIs via `httpx` instead of full SDKs when possible

### 2. Test Deployments
- Vercel can fail silently in production
- Always check `vercel ls` for deployment status
- Test the actual production endpoint after deploying

### 3. Keep Rollback Points
- The Dec 27 commit was a clean, stable state
- Tag important stable versions for easy rollback

## TODO - Reintroduce Lost Features
The Jan 5th features were good but need to be reintroduced without the heavy dependencies:

1. [ ] Content validation - use Groq instead of Gemini
2. [ ] Returning user greetings - check if this was in Dec 27 version
3. [ ] Query tracking to database
4. [ ] Human-in-the-loop validation

## Commands

```bash
# Local development
source .venv/bin/activate
source .env.local && uvicorn api.index:app --port 8000

# Deploy
vercel --prod

# Check deployments
vercel ls

# Test endpoint
curl -s -X POST https://vic-clm.vercel.app/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer vic-clm-hume-secret-2024' \
  -d '{"messages":[{"role":"user","content":"thorney island"}]}'
```

---
*Last updated: Jan 6, 2026*
