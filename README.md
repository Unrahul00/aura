# AuraStream 🎵

Ad-free YouTube Music frontend. Self-hosted on your homelab via Docker + Nginx + Cloudflare.

## Stack
- **Backend** — FastAPI + yt-dlp (audio stream proxy)
- **Frontend** — Vanilla HTML/JS/CSS (zero build step)
- **Proxy** — Nginx with chunked streaming tuned for audio

## Deploy via Portainer

1. Portainer → **Stacks** → **Add Stack** → **Repository**
2. Paste your GitHub repo URL
3. Set compose path to `docker-compose.yml`
4. Deploy

## Nginx config

Copy `aurastream.editwithrahul.xyz.conf` to `/etc/nginx/sites-available/` on your server and symlink to `sites-enabled/`.

## Cloudflare DNS

Add a CNAME → `aurastream` pointing to your domain/IP, orange cloud proxied.

See `DEPLOY.md` for full step-by-step instructions.
