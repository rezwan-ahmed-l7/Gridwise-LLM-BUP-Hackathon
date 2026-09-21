"""FastAPI route handlers.

Endpoints exposed:
    GET  /                  — Dashboard (static/index.html)
    GET  /docs              — Custom-themed Swagger UI
    GET  /health            — Readiness probe (JSON or HTML)
    GET  /api/status        — Runtime status & metadata
    GET  /api/presets       — Public sample scenarios for the UI
    POST /optimize-energy   — Main API: interpretation + optimization
    POST /api/analyze       — Optimization + analytics
    POST /api/export-csv    — CSV download of the schedule
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi import Request

from app.core.config import (
    FALLBACK_PRESETS_FILE,
    QUESTION_DIR,
    SAMPLE_CASES_FILE,
    settings,
)
from app.models.schemas import (
    AnalyzeResponse,
    OptimizeRequest,
    OptimizeResponse,
)
from app.services.optimizer import InfeasibleScheduleError
from app.services.pipeline import compute_analytics, run_pipeline


logger = logging.getLogger("gridwise.routes")


router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _execute(req: OptimizeRequest) -> Dict[str, Any]:
    """Run the pipeline with uniform error handling."""
    try:
        return run_pipeline(req)
    except HTTPException:
        raise
    except InfeasibleScheduleError as exc:
        logger.warning("Infeasible scenario '%s': %s", req.scenario_id, exc)
        raise HTTPException(
            status_code=422,
            detail=(
                "Infeasible scenario: no schedule satisfies all constraints "
                "(check battery limits, reserve and grid caps)."
            ),
        )
    except Exception:
        logger.exception("Error processing optimization request")
        raise HTTPException(
            status_code=500,
            detail="Controlled internal server error during optimization.",
        )


def _load_presets() -> Dict[str, Any]:
    """Load sample cases from the Question/ folder, with a fallback."""
    if SAMPLE_CASES_FILE.exists():
        with SAMPLE_CASES_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            case["id"]: {"label": case["label"], "input": case["input"]}
            for case in data.get("cases", [])
        }
    if FALLBACK_PRESETS_FILE.exists():
        with FALLBACK_PRESETS_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> Response:
    """Serve the live dashboard from the static directory."""
    from app.core.config import STATIC_DIR  # local import to avoid cycles

    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>GridWise LLM Service Online</h1>")


@router.get("/docs", include_in_schema=False)
def swagger_docs(request: Request) -> HTMLResponse:
    """Custom-themed Swagger UI."""
    from fastapi import FastAPI  # local for typing
    app: FastAPI = request.app  # type: ignore[assignment]

    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="GridWise API · Swagger",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )
    content = response.body.decode("utf-8")
    theme = _SWAGGER_THEME
    return HTMLResponse(content=content.replace("</head>", f"{theme}</head>"))


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------
@router.get("/health")
def health(ui: bool = False) -> Response:
    """JSON readiness probe (or rendered HTML if ?ui=1)."""
    if ui:
        return HTMLResponse(content=_HEALTH_HTML)
    return {"status": "ok"}


@router.get("/api/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "solver": "HiGHS (scipy.optimize.linprog)",
        "llm_configured": settings.llm_configured,
        "llm_models": settings.candidate_models(),
    }


@router.get("/api/presets")
def presets() -> Dict[str, Any]:
    return _load_presets()


# ---------------------------------------------------------------------------
# Optimization endpoints (the official API contract)
# ---------------------------------------------------------------------------
@router.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: OptimizeRequest) -> OptimizeResponse:
    """Interpret operator notes and optimize the 24-hour schedule."""
    return _execute(req)["response"]


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: OptimizeRequest) -> AnalyzeResponse:
    """Optimization plus savings, warnings, and hourly insights."""
    result = _execute(req)
    return AnalyzeResponse(
        optimization=result["response"],
        analytics=compute_analytics(req, result),
    )


@router.post("/api/export-csv")
def export_csv(req: OptimizeRequest) -> Response:
    """Download the optimized 24-hour schedule as CSV."""
    result = _execute(req)
    plan = result["response"].hourly_plan

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "hour",
            "tariff_bdt_per_kwh",
            "demand_kwh",
            "solar_used_kwh",
            "grid_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
            "cost_bdt",
        ]
    )
    for p in plan:
        tariff = req.hours[p.hour].tariff_bdt_per_kwh
        writer.writerow(
            [
                p.hour,
                tariff,
                req.hours[p.hour].demand_kwh,
                p.solar_used_kwh,
                p.grid_kwh,
                p.battery_action,
                p.battery_kwh,
                p.battery_energy_after_kwh,
                round(p.grid_kwh * tariff, 4),
            ]
        )

    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", req.scenario_id) or "scenario"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_id}_schedule.csv"'
        },
    )


# ---------------------------------------------------------------------------
# Inline CSS for theme / health pages
# ---------------------------------------------------------------------------
# Kept here rather than as static assets because these endpoints must work
# even when the user's static directory is empty.
_SWAGGER_THEME = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: dark;
  --bg-deep: #04060f; --bg-base: #070b14;
  --bg-surface: rgba(13, 19, 33, 0.72);
  --bg-elevated: rgba(20, 28, 46, 0.78);
  --bg-inset: rgba(6, 11, 22, 0.55);
  --border-soft: rgba(255, 255, 255, 0.06);
  --border-mid: rgba(255, 255, 255, 0.1);
  --primary: #10b981; --primary-bright: #34d399;
  --secondary: #06b6d4; --secondary-bright: #22d3ee;
  --text: #f8fafc; --text-muted: #94a3b8; --text-dim: #64748b; --text-faint: #475569;
  --font-main: 'Plus Jakarta Sans', sans-serif;
  --font-display: 'Fraunces', serif;
  --font-mono: 'JetBrains Mono', monospace;
}
* { box-sizing: border-box; }
html, body { margin:0; min-width:320px; background:var(--bg-deep); color:var(--text); font-family:var(--font-main); -webkit-font-smoothing:antialiased; background-attachment: fixed;
  background-image:
    radial-gradient(ellipse 70% 50% at 50% -10%, rgba(6, 182, 212, 0.18), transparent 65%),
    radial-gradient(ellipse 50% 40% at 90% 10%, rgba(16, 185, 129, 0.12), transparent 60%),
    radial-gradient(circle 900px at 5% 100%, rgba(196, 181, 253, 0.08), transparent 65%); }
body::after { content:''; position:fixed; inset:0; pointer-events:none; background-image: linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px); background-size: 60px 60px; mask-image: radial-gradient(ellipse at center, black 0%, transparent 80%); z-index:0; }
.gw-brand-header { position:sticky; top:0; z-index:50; backdrop-filter: blur(28px) saturate(180%); background: rgba(4, 8, 18, 0.65); border-bottom: 1px solid var(--border-soft); padding: 1rem 2rem; display:flex; align-items:center; justify-content:space-between; }
.gw-brand-wrap { display:flex; align-items:center; gap:1rem; }
.gw-brand-icon { width:46px; height:46px; border-radius:13px; background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #0ea5e9 100%); display:grid; place-items:center; box-shadow: 0 0 30px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3); color:#fff; }
.gw-brand-title { font-family:var(--font-display); font-size:1.45rem; font-weight:500; letter-spacing:-0.025em; }
.gw-status-pill { display:flex; align-items:center; gap:.55rem; background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(16,185,129,.06)); border:1px solid rgba(52,211,153,.28); color:#6ee7b7; font-size:.74rem; font-weight:600; padding:6px 13px; border-radius:999px; }
.gw-status-pill .gw-pulse { position:relative; width:8px; height:8px; border-radius:50%; background:#34d399; }
.gw-status-pill .gw-pulse::before { content:''; position:absolute; inset:-4px; border-radius:50%; background:#34d399; opacity:.4; animation: gw-pulse 2.2s ease-in-out infinite; }
@keyframes gw-pulse { 0%,100%{transform:scale(.8);opacity:.5} 50%{transform:scale(1.4);opacity:0} }
.gw-nav { display:flex; gap:.65rem; }
.gw-nav a { background: rgba(255,255,255,.04); border:1px solid var(--border-soft); color:var(--text-muted); font-size:.78rem; font-weight:600; padding:7px 14px; border-radius:10px; text-decoration:none; display:inline-flex; align-items:center; gap:.45rem; }
.gw-nav a:hover { background: rgba(255,255,255,.08); color:var(--text); border-color: rgba(34,211,238,.4); }
.swagger-ui { position:relative; z-index:1; max-width:1380px; margin:0 auto; padding: 32px clamp(20px,4vw,56px) 80px; }
.swagger-ui .topbar { display:none; }
.swagger-ui .info { margin: 0 0 32px; }
.swagger-ui .info .title { color:var(--text); font-family:var(--font-display); font-weight:500; font-size:36px; letter-spacing:-0.035em; }
.swagger-ui .info p, .swagger-ui .info li { color:var(--text-muted); }
.swagger-ui .opblock { overflow:hidden; border:1px solid var(--border-soft); border-radius:16px; background:var(--bg-surface); backdrop-filter: blur(24px) saturate(150%); box-shadow: 0 16px 40px rgba(0,0,0,.35); margin-bottom:14px; }
.swagger-ui .opblock-tag { color:var(--text); border-bottom-color:var(--border-soft); font-family:var(--font-display); font-weight:500; font-size:22px; }
.swagger-ui .opblock-summary { border-bottom-color:var(--border-soft); padding:14px 20px; background:transparent; }
.swagger-ui .opblock-summary-method { border-radius:999px !important; font: 800 11px var(--font-mono); padding:7px 16px; }
.swagger-ui .opblock-summary-method-get { background: linear-gradient(135deg, #22d3ee 0%, #0ea5e9 100%) !important; color:#f0f9ff !important; border:1px solid rgba(103,232,249,.5) !important; }
.swagger-ui .opblock-summary-method-post { background: linear-gradient(135deg, #34d399 0%, #059669 100%) !important; color:#ecfdf5 !important; border:1px solid rgba(110,231,183,.55) !important; }
.swagger-ui .btn, .swagger-ui select, .swagger-ui input, .swagger-ui textarea { border-radius:10px; border-color:var(--border-soft); background:rgba(4,8,16,.72); color:var(--text); }
.swagger-ui .btn.execute { background: linear-gradient(135deg, #10b981 0%, #06b6d4 60%, #0ea5e9 100%); color:#021014; font-weight:700; border-radius:999px !important; padding:10px 26px; border:0; }
.swagger-ui .highlight-code, .swagger-ui .microlight { background:#040810 !important; color:#7dd3fc !important; border-radius:8px; }
.swagger-ui .response-col_status { font-family:var(--font-mono); font-weight:700; }
.gw-footer { text-align:center; padding:2rem 1rem 1rem; font-size:.74rem; color:var(--text-faint); position:relative; z-index:1; }
.gw-footer span { background: linear-gradient(135deg, #34d399, #22d3ee); -webkit-background-clip:text; background-clip:text; color:transparent; font-weight:700; }
@media (max-width:720px) { .gw-brand-header { padding:.85rem 1rem; } .gw-brand-title { font-size:1.15rem; } }
</style>
<header class="gw-brand-header">
  <div class="gw-brand-wrap">
    <div class="gw-brand-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg></div>
    <div><div class="gw-brand-title">GridWise<span style="font-family:var(--font-mono);font-size:.65rem;font-weight:700;letter-spacing:.14em;background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(6,182,212,.18));color:#5eead4;padding:3px 9px;border-radius:6px;border:1px solid rgba(94,234,212,.25);margin-left:.5rem;">LLM</span></div>
    <div style="font-size:.72rem;color:var(--text-muted);font-weight:500;letter-spacing:.04em;margin-top:3px;text-transform:uppercase;">Campus Energy Optimization · API Reference</div></div>
  </div>
  <div class="gw-nav">
    <div class="gw-status-pill"><span class="gw-pulse"></span><span>OpenAPI 3 · Live</span></div>
    <a href="/" target="_blank"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12l9-9 9 9"></path><path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"></path></svg> Dashboard</a>
    <a href="/health?ui=1" target="_blank"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg> Health</a>
  </div>
</header>
<div class="gw-footer">Crafted for <span>BUP CSE Fest 2026</span> · GridWise LLM Optimization Engine</div>
"""

_HEALTH_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GridWise · Health Probe</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Fraunces:opsz,wght@9..144,400;9..144,500&display=swap" rel="stylesheet">
<style>
:root{--bg-deep:#04060f;--surface:rgba(13,19,33,.72);--border:rgba(255,255,255,.08);--text:#f8fafc;--muted:#94a3b8;--dim:#64748b;--green:#10b981;--cyan:#06b6d4;--violet:#c4b5fd;--font-main:'Plus Jakarta Sans',sans-serif;--font-display:'Fraunces',serif;--font-mono:'JetBrains Mono',monospace}
*{box-sizing:border-box}html,body{margin:0;min-height:100vh}body{display:grid;place-items:center;padding:clamp(20px,4vw,48px);color:var(--text);font-family:var(--font-main);background:var(--bg-deep);background-image:radial-gradient(ellipse 70% 50% at 50% -10%,rgba(6,182,212,.18),transparent 65%),radial-gradient(ellipse 50% 40% at 90% 10%,rgba(16,185,129,.12),transparent 60%);background-attachment:fixed;-webkit-font-smoothing:antialiased}
.shell{width:min(820px,100%);position:relative;z-index:1}
.brand{display:flex;align-items:center;gap:16px;margin-bottom:26px}
.mark{width:48px;height:48px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(135deg,#10b981,#06b6d4 60%,#0ea5e9);box-shadow:0 0 30px rgba(16,185,129,.4),inset 0 1px 0 rgba(255,255,255,.3);color:#fff}
.mark svg{width:24px;height:24px}
h1{margin:0;font-family:var(--font-display);font-weight:500;font-size:clamp(1.85rem,4vw,2.6rem);letter-spacing:-0.03em;color:#fff;line-height:1.1}
.eyebrow{margin:6px 0 0;color:var(--muted);font-size:.84rem;letter-spacing:.02em}
.card{padding:clamp(28px,5vw,48px);border:1px solid var(--border);border-radius:22px;background:var(--surface);box-shadow:0 24px 60px -12px rgba(0,0,0,.6);backdrop-filter:blur(28px) saturate(150%)}
.status{display:flex;align-items:center;gap:14px;padding:18px 22px;border:1px solid rgba(16,185,129,.28);border-radius:14px;background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(16,185,129,.04));color:#6ee7b7;font-weight:600}
.dot{position:relative;width:11px;height:11px;border-radius:50%;background:#34d399}
.dot::before{content:'';position:absolute;inset:-5px;border-radius:50%;background:#34d399;opacity:.4;animation:pulse 2.2s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(.8);opacity:.5}50%{transform:scale(1.4);opacity:0}}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:22px}
.metric{padding:18px 20px;border:1px solid var(--border);border-radius:14px;background:rgba(6,11,22,.55);position:relative;overflow:hidden}
.metric::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--cyan),var(--violet));opacity:.5}
.label{color:var(--dim);font-family:var(--font-mono);font-size:.65rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
.value{margin-top:8px;color:var(--text);font-family:var(--font-mono);font-weight:700;font-size:1.02rem;overflow-wrap:anywhere}
.links{display:flex;flex-wrap:wrap;gap:10px;margin-top:28px}
a{padding:10px 16px;border:1px solid var(--border);border-radius:10px;color:#cbd5e1;text-decoration:none;font-size:.82rem;font-weight:600;background:rgba(255,255,255,.04)}
a:hover{border-color:rgba(34,211,238,.4);color:#fff;background:rgba(34,211,238,.1)}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
</style></head>
<body><main class="shell">
<div class="brand"><div class="mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg></div>
<div><h1>GridWise Health Probe</h1><p class="eyebrow">Live service readiness and runtime status</p></div></div>
<section class="card">
<div class="status"><span class="dot"></span><span id="health-status">Checking service health…</span></div>
<div class="grid">
<div class="metric"><div class="label">API Status</div><div class="value" id="api-status">—</div></div>
<div class="metric"><div class="label">Version</div><div class="value" id="version">—</div></div>
<div class="metric"><div class="label">Solver</div><div class="value" id="solver">—</div></div>
<div class="metric"><div class="label">LLM Configured</div><div class="value" id="llm">—</div></div>
</div>
<nav class="links"><a href="/">← Live Dashboard</a><a href="/docs">Swagger Docs</a><a href="/health">JSON Response</a></nav>
</section></main>
<script>
Promise.all([fetch('/health'), fetch('/api/status')]).then(async ([h, s]) => {
  const health = await h.json(), status = await s.json();
  document.getElementById('health-status').textContent = health.status==='ok'?'Service is healthy and ready':'Service reported an issue';
  document.getElementById('api-status').textContent = health.status.toUpperCase();
  document.getElementById('version').textContent = status.version;
  document.getElementById('solver').textContent = status.solver;
  document.getElementById('llm').textContent = status.llm_configured?'Configured':'Offline fallback';
}).catch(()=>{document.getElementById('health-status').textContent='Unable to reach service'});
</script></body></html>"""
