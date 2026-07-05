# Sentinel — NIDS Dashboard

The React + Vite frontend for the [Network Intrusion Detection System](../README.md).
It renders a live security console over the FastAPI backend: a streaming alert
feed, a world threat map, and a per-flow "why was this flagged" panel backed by
the model's SHAP explanations.

## Features

- **Live alert stream** over a WebSocket (`/ws/alerts`) with severity, confidence,
  source/destination, and protocol.
- **Threat map** (Leaflet + three-globe) plotting attack origins.
- **Upload mode** — post a CICIDS CSV to `/api/analyze` and browse per-flow
  predictions with SHAP feature contributions.
- **Keep-alive** ping so a sleeping free-tier backend wakes while the dashboard
  is open.

## Tech stack

React 18, TypeScript, Vite, Tailwind + shadcn/ui, TanStack Query, Recharts,
Leaflet, and React Three Fiber.

## Development

```bash
npm install
echo "VITE_API_BASE=http://localhost:8000" > .env   # point at your backend
npm run dev                                          # http://localhost:8080
```

`VITE_API_BASE` is the backend origin (no trailing slash). Leave it empty to talk
to the same origin — that's how the production Docker image works, with nginx
reverse-proxying `/api`, `/ws`, and `/health` to the backend.

## Scripts

| Script | Purpose |
|--------|---------|
| `npm run dev` | Vite dev server with HMR |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | ESLint |
| `npm test` | Vitest unit tests |

## Build & deploy

`npm run build` emits a static bundle to `dist/`. Deploy it anywhere that serves
static files (Vercel, Netlify, nginx). For a one-command full stack, use the
[root `docker-compose.yml`](../docker-compose.yml), which builds this app behind
nginx and wires it to the backend. Hosted-deploy steps are in
[DEPLOYMENT.md](../DEPLOYMENT.md).
