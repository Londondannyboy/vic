# VIC CLM v2.0 - Restart Plan

## Current Status: 95% Complete

### What's Done ✅

1. **FastAPI/Pydantic AI Server Built**
   - `/Users/dankeegan/vic-clm/` - Complete project
   - GitHub: https://github.com/Londondannyboy/vic

2. **Deployed to Vercel**
   - URL: `https://vic-clm.vercel.app/chat/completions`
   - All env vars configured (DATABASE_URL, VOYAGE_API_KEY, GOOGLE_API_KEY, ZEP_API_KEY, CLM_AUTH_TOKEN)

3. **Tested Working**
   - Database connection: ✅
   - Voyage AI embeddings: ✅
   - Google Gemini LLM: ✅
   - SSE streaming: ✅
   - Pydantic AI validation: ✅ (built into models.py)

### What's Left ⏳

1. **Hume Integration Issue**
   - Auth token: `vic-clm-hume-secret-2024`
   - Hume config ID: `caa1fc82-46ed-4f29-9b9b-733c3d5bfad9`
   - Issue: Hume says "API key not valid" but curl with same token works
   - Possible cause: Hume may send key differently than Bearer token

2. **Debug Hume Auth**
   - Check how Hume sends the `language_model_api_key`
   - May need to adjust `api/index.py` to handle Hume's format
   - Or try with auth disabled (remove CLM_AUTH_TOKEN env var)

---

## Quick Restart Commands

```bash
# Go to project
cd ~/vic-clm

# Activate venv
source .venv/bin/activate

# Test locally
source .env.local && uvicorn api.index:app --port 8000

# Test endpoint
curl -s -X POST http://127.0.0.1:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer vic-clm-hume-secret-2024' \
  -d '{"messages":[{"role":"user","content":"Tell me about Ignatius Sancho"}]}'

# Deploy to Vercel
vercel --prod

# Check Vercel env vars
vercel env ls
```

---

## Key Files

| File | Purpose |
|------|---------|
| `api/index.py` | FastAPI + `/chat/completions` endpoint |
| `api/agent.py` | Pydantic AI agent with Vic persona |
| `api/models.py` | ValidatedVICResponse + fact validators |
| `api/tools.py` | search_articles + Zep memory |
| `api/database.py` | Neon pgvector queries |

---

## Pydantic AI Validation (Already Built)

The validation is in `api/models.py`:

```python
class ValidatedVICResponse(BaseModel):
    response_text: str
    facts_stated: list[str]
    source_content: str
    source_titles: list[str]

    # Validators:
    # 1. facts_must_be_in_source - ensures facts exist in articles
    # 2. no_architect_unless_mentioned - prevents architect hallucinations
    # 3. no_specific_dates_unless_in_source - prevents date hallucinations
```

Tests in `tests/test_validation.py` - all 28 pass.

---

## Next Steps

1. **Fix Hume auth** - either debug the token format or disable auth temporarily
2. **Test live with Hume** - voice interaction working
3. **Fine-tune validation** - adjust strictness based on real usage
4. **Add more phonetic corrections** - based on voice recognition errors

---

## Environment Variables (Vercel)

```
GOOGLE_API_KEY=AIzaSyAfcWEAGd2s-8bmJQlj8xUZDwIbSDNqCqw
DATABASE_URL=postgresql://neondb_owner:...@...neon.tech/neondb
VOYAGE_API_KEY=pa-NBONsUWEcXy-DPKM4J_Jtq82H6BPCDauNBq-zl0ViB0
ZEP_API_KEY=z_1dWlkIjoiMmNkYWVjZjkt...
CLM_AUTH_TOKEN=vic-clm-hume-secret-2024
```

---

*Last updated: December 23, 2024*
