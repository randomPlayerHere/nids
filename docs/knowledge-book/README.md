# NIDS Project Knowledge Book

> A teach-don't-describe handbook for the **Network Intrusion Detection System (NIDS)**.
> Goal: read this and be able to explain, defend, modify, debug, and interview on every
> major backend/ML decision in this repo **as if you built it**.

This book is split by module so it reads well on an iPad. Read top-to-bottom the first time;
after that, jump straight to the cheat sheets and interview prep.

---

## How to read this book

| # | File | What it covers | Read if you want to… |
|---|------|----------------|----------------------|
| 00 | [00-Overview.md](00-Overview.md) | Business problem, users, architecture, the **big honest truths** about this system | Get the 10-minute mental model |
| 01 | [01-Request-Lifecycle.md](01-Request-Lifecycle.md) | End-to-end path of every request (REST + WebSocket) | Explain "what happens when I click X" |
| 02 | [02-Backend-Architecture.md](02-Backend-Architecture.md) | Folder structure, layered design, patterns, lazy loading | Defend the code organization |
| 03 | [03-API-Deep-Dive.md](03-API-Deep-Dive.md) | Every endpoint: request/response/validation/failure modes + sequence diagrams | Answer "walk me through this endpoint" |
| 04 | [04-Data-and-Model-Store.md](04-Data-and-Model-Store.md) | The "database" of this system: model files, scaler, label maps, demo data, in-memory state | Talk about persistence & data design |
| 05 | [05-AI-ML-Pipeline.md](05-AI-ML-Pipeline.md) | Preprocessing → 1D-CNN → SHAP → severity. The heart of the project | Explain the ML deeply |
| 06 | [06-Services-and-Functions.md](06-Services-and-Functions.md) | Every service + function: purpose, complexity, edge cases, bugs | Refactor/debug confidently |
| 07 | [07-Security.md](07-Security.md) | What's protected, what isn't, threat model, what you'd add | Defend security posture honestly |
| 08 | [08-Performance-and-Scalability.md](08-Performance-and-Scalability.md) | Latency, caching, 100→1M users scaling story | Pass the system-design round |
| 09 | [09-Design-Decisions.md](09-Design-Decisions.md) | Why FastAPI, why CNN, why SHAP, why no DB, why Docker… with alternatives | Win every "why did you…" question |
| 10 | [10-Deployment-and-Debugging.md](10-Deployment-and-Debugging.md) | Docker, HF Spaces + Vercel, env vars, debugging playbook | Ship it & fix it |
| 11 | [11-Improvements.md](11-Improvements.md) | What a 30-day roadmap looks like + tradeoffs | Show senior-level thinking |
| 12 | [12-Interview-Prep.md](12-Interview-Prep.md) | **150+ Q&A** with reasoning, mistakes, follow-ups | Drill before the interview |
| 13 | [13-Cheat-Sheets.md](13-Cheat-Sheets.md) | One-glance summaries of everything | Review the night before |

---

## The 60-second pitch (memorize this)

> "It's a **real-time network intrusion detection system**. A **1D Convolutional Neural Network**
> classifies each network *flow* (78 CICFlowMeter features from the CICIDS2017 dataset) into
> **11 classes** — BENIGN plus 10 attack families — at **99.48% accuracy**. The backend is
> **FastAPI**: it exposes fast prediction, **SHAP-explained** prediction, batch CSV analysis,
> and a **WebSocket** live-alert stream. A **React/Vite** dashboard consumes those. It's a
> **stateless ML inference service** — no database, no auth — packaged as a multi-stage
> **Docker** image and deployed to Hugging Face Spaces (backend) + Vercel (frontend)."

That sentence answers 80% of opening interview questions. The rest of this book is the depth behind it.

---

## ⚠️ Honesty notes (these make you credible, not weaker)

This project intentionally does **not** have several things a generic "full-stack app" template
assumes. Knowing *why they're absent* is itself a strong interview signal:

- **No SQL/NoSQL database.** It's a stateless inference service; its "data" is model weights +
  a scaler + label JSON. See [04-Data-and-Model-Store.md](04-Data-and-Model-Store.md).
- **No authentication / authorization / JWT / RBAC.** It's a demo/research API. See
  [07-Security.md](07-Security.md) for the honest threat model and what you'd add for production.
- **No LLM / RAG / vector database / embeddings.** The "AI" is a supervised CNN classifier
  with SHAP explanations — not generative AI. Don't conflate the two in an interview.
- **No Redis / message queue / Celery.** Inference is synchronous and fast (single-digit ms for
  fast predictions). The scaling chapter explains when you'd add these.

Throughout the book, wherever the original documentation template asked about one of these,
you'll find a **"Not in this project — here's why, and here's what would play that role"** box.
