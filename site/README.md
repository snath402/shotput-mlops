# Shotput Motion Lab website

This is a static project overview. It contains no athlete videos, athlete-specific measurements, or report links.

## Run locally with Docker

From the repository root in PowerShell:

```powershell
docker compose -f .\site\compose.yaml up --build -d
```

Open <http://localhost:8080> in a browser.

Stop the website when you are done:

```powershell
docker compose -f .\site\compose.yaml down
```
