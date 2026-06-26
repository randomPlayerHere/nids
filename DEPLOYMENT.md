# Deploying the NIDS demo (free, shareable link)

This is the **Tier 2** setup: a public URL you can put in your report or share,
using only free tiers.

| Piece | Host | Why |
|-------|------|-----|
| **Backend** (FastAPI + TensorFlow + WebSocket) | **Hugging Face Spaces (Docker)** | Free tier gives **16 GB RAM** — enough for TensorFlow, unlike most free hosts — and supports Docker + WebSockets. |
| **Frontend** (React/Vite static bundle) | **Vercel** | Free, auto-builds Vite, instant HTTPS. |

End result: `your-frontend.vercel.app` (the dashboard) talking to
`your-user-nids-api.hf.space` (the API) over HTTPS + WSS.

> The repo is already prepared for this: the backend honors `$PORT`, and it no
> longer needs the 149 MB `X_dcnn.npy` (it synthesizes a SHAP background when the
> file is absent). So a clean clone deploys as-is.

---

## Part A — Backend on Hugging Face Spaces

1. **Create a Space.** Sign in at <https://huggingface.co> → **New → Space**.
   - Name: `nids-api`  ·  SDK: **Docker**  ·  Template: **Blank**  ·  Visibility: **Public**.

2. **Add the Space metadata.** A Docker Space needs a `README.md` whose
   front-matter declares the port. Use the template in
   [`deploy/space-README.md`](deploy/space-README.md) — its key lines are:
   ```yaml
   ---
   title: NIDS API
   sdk: docker
   app_port: 8000
   ---
   ```

3. **Push the backend.** The model file is 34 MB, so use Git LFS:
   ```bash
   # one-time
   git lfs install
   git lfs track "*.h5" "*.tflite"

   # point a remote at your Space and push backend files
   git clone https://huggingface.co/spaces/<your-user>/nids-api hf-space
   cd hf-space
   # copy these from the project: Dockerfile  requirements.txt  scripts/  models/
   # and the deploy/space-README.md as README.md
   git add .gitattributes Dockerfile requirements.txt scripts models README.md
   git commit -m "NIDS backend"
   git push
   ```
   The first build takes ~5–10 min (TensorFlow is large). When it's green, your
   API lives at `https://<your-user>-nids-api.hf.space` — check `/health` and
   `/docs`.

   > **Tip:** you don't have to push `data/`, `frontend/`, `tests/`, or
   > `models/old/`. Only `Dockerfile`, `requirements.txt`, `scripts/`,
   > `models/new/`, and the Space `README.md` are needed.

4. **(Set CORS after Part B.)** In the Space → **Settings → Variables and
   secrets**, add a *variable*:
   ```
   NIDS_CORS_ORIGINS = https://<your-frontend>.vercel.app
   ```
   Saving restarts the Space.

---

## Part B — Frontend on Vercel

The frontend lives in `frontend/` and is its own git repo. Push it to GitHub,
then:

1. <https://vercel.com> → **Add New → Project** → import your frontend repo.
   Vercel auto-detects Vite (settings are also pinned in
   [`frontend/vercel.json`](frontend/vercel.json)).

2. **Environment Variables** → add:
   ```
   VITE_API_BASE = https://<your-user>-nids-api.hf.space
   ```
   (the HTTPS URL of your Space from Part A — no trailing slash).

3. **Deploy.** You get a URL like `https://nids-xyz.vercel.app`.

---

## Part C — Wire them together

1. Copy your Vercel URL → paste it as `NIDS_CORS_ORIGINS` in the HF Space
   settings (Part A step 4). The Space restarts.
2. Open the Vercel URL → click **Start monitoring**. Live alerts now stream from
   the Space over **WSS**, and the **Upload Mode** CSV analysis posts to the
   Space's `/api/analyze`.

That's it — a fully hosted, shareable demo on free infrastructure.

---

## Good to know (free-tier quirks)

- **Cold starts:** free Spaces sleep after ~48 h idle; the first visit wakes it
  (~30 s). Fine for a demo — just open it a minute before presenting.
- **Live demo data:** the raw CICIDS CSVs aren't uploaded, so the stream runs the
  model over a synthetic background. Alerts still flow and look realistic; for
  *real* predictions use **Upload Mode** with `test.csv` (or any CICIDS CSV).
- **Geo map:** without a MaxMind GeoLite2 DB the backend returns no coordinates,
  so the frontend places attacks on a stable synthetic location — the map stays
  populated.
- **Even simpler fallback:** if hosting is fiddly, just run `docker compose up`
  locally during your demo and record a short screen capture as backup.
