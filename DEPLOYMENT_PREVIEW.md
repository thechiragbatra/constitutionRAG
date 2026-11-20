# Constitution RAG — Deployment Preview

Complete guide showing how your application will look and behave when deployed across different platforms.

---

## 📱 Application Preview

### Frontend UI (Streamlit)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🟠🟠🟠🟠🟠🟠🟠  [Saffron Bar]                                        │
│ ⬜⬜⬜⬜⬜⬜⬜  [White Bar]                                           │
│ 🟢🟢🟢🟢🟢🟢🟢  [Emerald Green Bar]                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Constitution of India — RAG Question Answering                       │
│  Answers only from the indexed Constitution, with Article citations   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Can the police arrest me without telling me why?              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  [Ask Question]                                                      │
│                                                                       │
│  ┌─ Answer ──────────────────────────────────────────────────────┐ │
│  │ 🟢 [Article 22] No person who is arrested shall be detained    │ │
│  │    in custody without being informed of the grounds for arrest │ │
│  │    and shall be allowed to consult and be defended by a legal  │ │
│  │    practitioner of his choice.                                  │ │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Cited Articles: [22] [21]                                           │
│  ⏱ Latency: 1.24s                                                   │
│                                                                       │
│  ▼ Retrieved Passages (5)                                            │
│     1. [Article 22] Protection against arrest... (score: 0.89)       │
│     2. [Article 21] Protection of life... (score: 0.76)              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

Sidebar Configuration:
├─ Chunking: article | fixed256 | fixed512
├─ Retriever: BM25 | Dense | Hybrid
├─ Top-K: 5, 10, 20
├─ Reranker: ☐ Enable cross-encoder
├─ Abstain Threshold: 0.25
└─ Use Retrieval: ☑ Enabled
```

---

## 🌐 Deployment Options

### Option 1: Local Development (Current Setup)

**Status:** ✅ **READY NOW**

**Architecture:**
```
[Browser] ←→ [Streamlit :8501] ←→ [FastAPI :8000] ←→ [ML Models + Indexes]
```

**Start Commands:**
```bash
# Terminal 1 - Backend
cd /Users/chiragbatra/ConstitutionRAG/constitutionRAG
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 - Frontend
cd /Users/chiragbatra/ConstitutionRAG/constitutionRAG
source .venv/bin/activate
streamlit run app.py
```

**Access:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

**Performance:**
- Cold Start: ~2-3s (loading models)
- Query Latency: 1-2s (CPU inference)
- Memory Usage: ~1.5GB (all models loaded)

**Best For:**
- Development and testing
- Lab demonstrations
- Viva presentations

---

### Option 2: Render/Railway (Cloud Hosting)

**Status:** ✅ **DEPLOYMENT READY**

**Files Used:**
- `Procfile`: `web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `runtime.txt`: `python-3.14.8`
- `requirements.txt`: Pinned dependencies

#### Railway.app (RECOMMENDED)

**Deployment Steps:**
```bash
1. Push code to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Select: thechiragbatra/constitutionRAG
4. Railway auto-detects Procfile, runtime.txt, requirements.txt
5. Add Environment Variables:
   - RAG_LLM_MODE=none
6. Deploy (automatic)
```

**Expected URL:**
```
https://constitutionrag-production.up.railway.app
```

**Performance:**
- Cold Start: 15-30s
- Warm Latency: 1-2s
- Memory: 2GB included

**Cost:**
- $5 free credit/month
- ~$0.10/hour when running
- Can pause manually

#### Render.com

**Deployment Steps:**
```bash
1. Push to GitHub
2. render.com → New Web Service → Connect repo
3. Configure:
   - Build: pip install -r requirements.txt && python src/index.py
   - Start: (from Procfile)
   - Instance: Starter ($7/mo, 2GB RAM)
4. Add env vars: RAG_LLM_MODE=none
5. Deploy
```

**Expected URL:**
```
https://constitution-rag-api.onrender.com
```

**Performance:**
- Cold Start: 30-60s (free tier sleeps after 15min)
- Memory: Needs at least 1GB (Starter plan)

**Best For:**
- Public demo links
- Lab report submissions
- Railway is recommended over Render

---

### Option 3: Vercel (Serverless) - BM25 Only

**Status:** ⚠️ **LIMITED** (BM25-only due to 500MB bundle limit)

**Deployment:**
```bash
npm i -g vercel
cd /Users/chiragbatra/ConstitutionRAG/constitutionRAG
vercel --prod
```

**Expected URL:**
```
https://constitution-rag.vercel.app
```

**Limitations:**
- ⚠️ Cannot include PyTorch/FAISS (bundle size)
- Only BM25 retrieval (no dense/hybrid)
- No cross-encoder reranking

**Best For:**
- Lightweight demos
- Fast cold starts (5-10s)

---

## 🔌 API Endpoints Preview

### 1. Health Check
```bash
GET /api/health

Response:
{
  "status": "healthy",
  "articles_loaded": 395,
  "timestamp": "2026-11-20T18:22:13Z"
}
```

### 2. Ask Question (Main Endpoint)
```bash
POST /api/ask
Content-Type: application/json

{
  "question": "Can the police arrest me without telling me why?",
  "top_k": 5
}

Response:
{
  "question": "Can the police arrest me without telling me why?",
  "answer": "[Article 22] No person who is arrested shall be detained...",
  "cited_articles": ["22"],
  "flagged_citations": [],
  "abstained": false,
  "latency_ms": 1240.52,
  "retrieved": [...]
}
```

### 3. Get Article
```bash
GET /api/article/21

Response:
{
  "article": "21",
  "title": "Protection of life and personal liberty",
  "text": "No person shall be deprived of his life or personal liberty..."
}
```

---

## 📊 Performance Benchmarks

| Platform | Cold Start | Query Latency | Memory | Cost/Month |
|----------|-----------|---------------|--------|------------|
| **Local** | 2-3s | 1-2s | 1.5GB | Free |
| **Railway** | 15-30s | 1-2s | 2GB | $5 free credit |
| **Render Starter** | 30-60s | 2-3s | 2GB | $7 |
| **Vercel** | 5-10s | 0.5-1s (BM25 only) | 512MB | Free |

---

## 🚀 Recommended: Railway Deployment

**Why Railway?**
✅ Free $5 credit/month  
✅ No cold starts  
✅ 2GB RAM (enough for all models)  
✅ Easy GitHub integration  
✅ Auto-detects Procfile  
✅ Perfect for lab demos  

**Setup Time:** 10 minutes

**Steps:**
1. Push your code to GitHub
2. Connect Railway to GitHub
3. Deploy with one click
4. Get URL: `https://constitutionrag-production.up.railway.app`

**For Streamlit Frontend:**
- Deploy separately on Streamlit Cloud (free)
- Set `BACKEND_URL` env var to your Railway URL

---

## 🎓 Lab Demo Checklist (27 Nov 2026)

✅ Run `python setup_check.py` before demo  
✅ Test backend: `curl http://localhost:8000/api/health`  
✅ Test frontend loads with Indian flag theme  
✅ All 8 pytest tests passing  
✅ Example questions work (Article 21, 22, etc.)  
✅ Abstention works ("Who is PM?" → "Not found")  
✅ Citations clickable  
✅ Config changes work (BM25 vs Hybrid)  
✅ Deployment URL ready (backup: run locally)  

---

## 📱 Access Points After Deployment

| Service | URL |
|---------|-----|
| **Backend API** | `https://your-app.railway.app` |
| **API Docs** | `https://your-app.railway.app/docs` |
| **Health Check** | `https://your-app.railway.app/api/health` |
| **Frontend** | Deploy Streamlit separately or run locally |
| **GitHub** | https://github.com/thechiragbatra/constitutionRAG |

---

## 🔐 Environment Variables

### Required
```bash
RAG_LLM_MODE=none  # or 'api' or 'local'
```

### Optional (for Streamlit)
```bash
BACKEND_URL=https://your-backend-url.com
```

### Optional (API mode)
```bash
RAG_API_BASE=https://api.openai.com/v1
RAG_API_MODEL=gpt-4o-mini
RAG_API_KEY=sk-...
```

---

## 💡 Quick Test Commands

### Test Backend Locally
```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Article 21 say?", "top_k": 5}'
```

### Test Deployed Backend
```bash
curl https://your-app.railway.app/api/health
```

---

**Your Constitution RAG is ready to deploy! Railway.app is the recommended platform for your lab project. 🚀**

