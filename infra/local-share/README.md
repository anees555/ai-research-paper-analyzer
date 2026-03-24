# Local Share (One Public Link, No Core Changes)

This folder is isolated helper tooling so you can share frontend + backend from one URL while both run locally.

## What This Does

- Runs a local reverse proxy on `http://localhost:8080`
- Routes:
  - `/api/*`, `/docs`, `/openapi.json`, `/redoc`, `/static/*` -> backend `http://localhost:8003`
  - everything else -> frontend `http://localhost:3000`
- Exposes only the proxy using a tunnel tool (example: ngrok)

## Prerequisites

- Frontend running on `localhost:3000`
- Backend running on `localhost:8003`
- Caddy installed:
  - `winget install CaddyServer.Caddy`
- ngrok installed (optional, for public URL):
  - `winget install Ngrok.Ngrok`
- cloudflared installed (optional alternative for public URL):
  - `winget install Cloudflare.cloudflared`

## Start Local One-Link Proxy

From project root:

```powershell
Set-Location infra/local-share
./start-proxy.ps1
```

Local combined app URL:

- `http://localhost:8080`

## Expose Public URL (ngrok)

Open a second terminal:

```powershell
Set-Location infra/local-share
./start-ngrok.ps1
```

ngrok will show one public URL forwarding to `http://localhost:8080`.

## Expose Public URL (Cloudflare Tunnel)

Open a second terminal:

```powershell
Set-Location infra/local-share
./start-cloudflare.ps1
```

Cloudflare will print a public `trycloudflare.com` URL forwarding to `http://localhost:8080`.

## Optional: Named Cloudflare Tunnel (Custom Domain)

If you want a stable custom subdomain, use `cloudflared-config.example.yml` as a template.

High-level flow:

1. `cloudflared tunnel login`
2. `cloudflared tunnel create research-summary-local`
3. Update values in `cloudflared-config.example.yml`
4. Run:

```powershell
cloudflared tunnel --config ./cloudflared-config.example.yml run
```

## Notes

- This does **not** modify core frontend/backend code.
- If your frontend uses a hardcoded backend URL, prefer setting frontend API base URL to `/api/v1` while sharing.
- To stop sharing, just stop the proxy/tunnel terminals.
