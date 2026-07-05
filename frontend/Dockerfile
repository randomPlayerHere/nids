# Build the Vite app, then serve the static bundle from nginx.
FROM node:20-slim AS builder
WORKDIR /app

# Install deps first for better layer caching. Retry on flaky networks
# (npm ci can hit ECONNRESET when pulling large deps like three.js).
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund \
      --fetch-retries=5 \
      --fetch-retry-mintimeout=20000 \
      --fetch-retry-maxtimeout=120000

COPY . .
# VITE_API_BASE is baked in at build time. Leave empty so the app talks to the
# same origin and nginx (below) reverse-proxies the API. Override for split
# deployments: --build-arg VITE_API_BASE=https://api.example.com
ARG VITE_API_BASE=""
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build


FROM nginx:1.27-alpine AS final
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
