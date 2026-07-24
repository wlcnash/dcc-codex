# DCC Codex — Deploy Runbook

## Prerequisites

Before running, add these secrets to Bitwarden Secrets Manager:

| BWS Key | Value |
|---------|-------|
| `dcc-codex-postgres-password` | random strong password |
| `dcc-codex-gemini-api-key` | Gemini API key from Google AI Studio |
| `dcc-codex-minio-access-key` | MinIO access key (create a dedicated user in MinIO) |
| `dcc-codex-minio-secret-key` | MinIO secret key |

Also create a `dcc-codex` bucket in MinIO (or the seeder will create it automatically).

## Step 1 — Push code to GitHub

```bash
git init
git remote add origin https://github.com/wlcnash/dcc-codex
git add .
git commit -m "feat: initial DCC Codex build"
git push -u origin main
```

GitHub Actions will build and push:
- `ghcr.io/wlcnash/dcc-codex:main`
- `ghcr.io/wlcnash/dcc-codex-seeder:main`

## Step 2 — Add manifests to homelab GitOps repo

Copy `manifests/` to `wlcnash/homelab/manifests/dcc-codex/`  
Copy `manifests/argocd-app.yaml` to `wlcnash/homelab/argocd/apps/dcc-codex.yaml`

```bash
# In wlcnash/homelab repo:
cp -r /path/to/dcc-codex/manifests/* manifests/dcc-codex/
cp /path/to/dcc-codex/manifests/argocd-app.yaml argocd/apps/dcc-codex.yaml
git add .
git commit -m "feat: add dcc-codex service manifests"
git push
```

## Step 3 — Bootstrap ArgoCD app

```bash
kubectl apply -f argocd/apps/dcc-codex.yaml
```

ArgoCD will sync and deploy the namespace, postgres, and app within ~2 minutes.

## Step 4 — Out-of-band setup

### UniFi DNS
Add A record: `dcc-codex.home → 192.168.2.50`

### Authentik SSO
Create provider + application via ak shell in authentik worker pod:

```python
from authentik.providers.proxy.models import ProxyProvider
from authentik.core.models import Application

provider = ProxyProvider.objects.create(
    name="dcc-codex",
    external_host="https://dcc-codex.home",
    mode="forward_single",
    authorization_flow=Flow.objects.get(slug="default-provider-authorization-implicit-consent"),
)
app = Application.objects.create(
    name="DCC Codex",
    slug="dcc-codex",
    provider=provider,
)
# Then assign to embedded outpost
```

### Uptime Kuma
- Add HTTPS monitor for `https://dcc-codex.home/health`, 60s interval, ignore TLS
- Assign to "Media Services" status page group

### Homepage widget
Edit `configmap-config.yaml` in homelab/manifests/homepage/:

```yaml
- DCC Codex:
    icon: si-dungeonsanddragons
    href: https://dcc-codex.home
    description: DCC Compendium
    server: my-docker
    container: dcc-codex
    widget:
      type: customapi
      url: http://dcc-codex.dcc-codex.svc.cluster.local:8000/health
      refreshInterval: 60000
```

## Step 5 — Run the seeder

Once app + postgres pods are Running and schema is applied:

```bash
# Apply the seeder job
kubectl apply -f manifests/dcc-codex/job-seeder.yaml

# Follow logs
kubectl logs -f job/dcc-codex-seeder -n dcc-codex
```

The seeder pipeline takes ~2-6 hours to complete all 3 steps (scrape → extract → images).
You can run steps individually:

```bash
# Scrape only
kubectl set args job/dcc-codex-seeder -- --step scrape

# Extract in batches of 20 chapters at a time
kubectl set args job/dcc-codex-seeder -- --step extract --batch 20
```

## Step 6 — Post-deploy verification

```bash
kubectl get pods -n dcc-codex
kubectl get certificate -n dcc-codex
nslookup dcc-codex.home 192.168.2.1
curl -k -H "Host: dcc-codex.home" https://192.168.2.50
curl -k https://dcc-codex.home
```

All 5 should pass. Then browse to `https://dcc-codex.home`.
