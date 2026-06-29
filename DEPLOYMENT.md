# Deploying the NIDS demo (free, shareable link)

A public URL you can share or drop into a report, on free tiers only.

| Piece | Host | Why |
|-------|------|-----|
| Backend (FastAPI + TensorFlow + WebSocket) | Hugging Face Spaces (Docker) | Free tier has 16 GB RAM, enough for TensorFlow, and supports Docker + WebSockets. |
| Frontend (React/Vite static bundle) | Vercel | Free, auto-builds Vite, instant HTTPS. |

You end up with `your-frontend.vercel.app` (the dashboard) talking to
`your-user-nids-api.hf.space` (the API) over HTTPS + WSS.

The repo is already set up for this. The backend honors `$PORT` and no longer
needs the 149 MB `X_dcnn.npy` (it synthesizes a SHAP background when the file is
absent), so a clean clone deploys as-is.

---

## Part A: Backend on Hugging Face Spaces

1. Create a Space. Sign in at <https://huggingface.co>, then New > Space.
   - Name: `nids-api`, SDK: Docker, Template: Blank, Visibility: Public.

2. Add the Space metadata. A Docker Space needs a `README.md` whose front-matter
   declares the port. Use the template in
   [`deploy/space-README.md`](deploy/space-README.md); the lines that matter are:
   ```yaml
   ---
   title: NIDS API
   sdk: docker
   app_port: 8000
   ---
   ```

3. Push the backend. Install Git LFS once, since the model is 34 MB and HF
   rejects non-LFS files over 10 MB:
   ```bash
   sudo apt install git-lfs   # Ubuntu/Debian
   ```
   Then clone the Space, assemble the files with the helper script, and push:
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/nids-api hf-space
   ./deploy/prepare_hf_space.sh hf-space      # copies backend + sets up LFS
   cd hf-space
   git add -A && git commit -m "NIDS backend" && git push
   ```
   `prepare_hf_space.sh` copies what the API needs (`Dockerfile`,
   `requirements.txt`, `scripts/`, `models/new/`, `data/demo_flows.npy`, and the
   Space `README.md`) and runs `git lfs track` for the model. The first build
   takes 10-15 min because TensorFlow is large. Once it's green your API lives at
   `https://<your-user>-nids-api.hf.space`; check `/health` and `/docs`.

   Auth: the push prompts for credentials. Use your HF username and a write token
   from <https://huggingface.co/settings/tokens> as the password, or run
   `huggingface-cli login` first.

4. Set CORS after Part B. In the Space, go to Settings > Variables and secrets
   and add a variable:
   ```
   NIDS_CORS_ORIGINS = https://<your-frontend>.vercel.app
   ```
   Saving restarts the Space.

---

## Part B: Frontend on Vercel

The frontend lives in `frontend/` and is its own git repo. Push it to GitHub,
then:

1. <https://vercel.com>, Add New > Project, import your frontend repo. Vercel
   auto-detects Vite (the settings are also pinned in
   [`frontend/vercel.json`](frontend/vercel.json)).

2. Under Environment Variables, add:
   ```
   VITE_API_BASE = https://<your-user>-nids-api.hf.space
   ```
   That's the HTTPS URL of your Space from Part A, with no trailing slash.

3. Deploy. You get a URL like `https://nids-xyz.vercel.app`.

---

## Part C: Wire them together

1. Copy your Vercel URL and paste it as `NIDS_CORS_ORIGINS` in the HF Space
   settings (Part A step 4). The Space restarts.
2. Open the Vercel URL and click Start monitoring. Live alerts now stream from
   the Space over WSS, and the Upload Mode CSV analysis posts to the Space's
   `/api/analyze`.

Done. A fully hosted demo on free infrastructure.

---

## Free-tier quirks

- Cold starts: free Spaces sleep after about 48 h idle, and the first visit wakes
  it in roughly 30 s. Fine for a demo, just open it a minute before presenting.
- Live demo data: the raw CICIDS CSVs aren't uploaded, so the stream runs the
  model over a synthetic background. Alerts still flow and look realistic; for
  real predictions use Upload Mode with `test.csv` (or any CICIDS CSV).
- Geo map: without a MaxMind GeoLite2 DB the backend returns no coordinates, so
  the frontend places attacks at a stable synthetic location and the map stays
  populated.
- Simpler fallback: if hosting is fiddly, run `docker compose up` locally during
  the demo and record a short screen capture as backup.
