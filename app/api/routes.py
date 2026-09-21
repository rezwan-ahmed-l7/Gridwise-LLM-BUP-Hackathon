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

from fastapi import APIRouter, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.core.config import (
    FALLBACK_PRESETS_FILE,
    SAMPLE_CASES_FILE,
    STATIC_DIR,
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
# Reusable HTML fragments  (so /docs and /health share the same brand bar)
# ---------------------------------------------------------------------------
_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&"
    "family=JetBrains+Mono:wght@300;400;500;600;700&"
    "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700"
    '" rel="stylesheet">'
)


def _topbar(active: str = "") -> str:
    """Render the unified brand header used by /docs and /health.

    `active` highlights the current page (one of: '', 'dashboard', 'docs',
    'health') with the gradient pill treatment.
    """
    def _cls(target: str) -> str:
        return " class=\"active\"" if active == target else ""

    dashboard_link = (
        f'<a href="/" {_cls("dashboard")}>'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<path d="M3 12l9-9 9 9"/>'
        '<path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"/>'
        '</svg> Dashboard</a>'
    )
    docs_link = (
        f'<a href="/docs" {_cls("docs")}>'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/><path d="M9 13h6"/><path d="M9 17h6"/>'
        '</svg> Swagger</a>'
    )
    health_link = (
        f'<a href="/health?ui=1" {_cls("health")}>'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Health</a>'
    )

    return (
        '<header class="gw-topbar">'
        '<div class="gw-brand">'
        '<div class="gw-brand-mark" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'
        '</svg></div>'
        '<div class="gw-brand-text">'
        '<h1>GridWise<span class="gw-badge">LLM</span></h1>'
        '<p id="gw-subtitle">Campus Energy Optimization</p>'
        '</div>'
        '</div>'
        '<nav class="gw-nav">'
        '<div class="gw-pill"><span class="gw-pulse"></span>'
        '<span id="gw-pill-text">Live</span></div>'
        f'{dashboard_link}{docs_link}{health_link}'
        '</nav>'
        '</header>'
    )


def _footer() -> str:
    return (
        '<div class="gw-footer">Crafted for '
        '<span>BUP CSE Fest 2026</span> · GridWise LLM Optimization Engine</div>'
    )


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> Response:
    """Serve the live dashboard from the static directory."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>GridWise LLM Service Online</h1>")


@router.get("/docs", include_in_schema=False)
def swagger_docs(request: Request) -> HTMLResponse:
    """Custom-themed Swagger UI matching the dashboard design."""
    app = request.app
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="GridWise API · Swagger",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )
    content = response.body.decode("utf-8")

    # Inject: Google Fonts + shared page.css + dedicated api.css.
    # The shared page.css supplies the brand bar/footer; api.css themes the
    # Swagger UI body.
    head_inject = (
        _FONT_LINK
        + '<link rel="stylesheet" href="/static/page.css">'
        + '<link rel="stylesheet" href="/static/api.css">'
    )
    content = content.replace("</head>", f"{head_inject}</head>")

    # Inject header + footer at the top/bottom of <body> ... </body>.
    inject_top = _topbar("docs")
    inject_bottom = _footer()
    body_open = content.find("<body>")
    if body_open != -1:
        body_open_end = body_open + len("<body>")
        content = (
            content[:body_open_end]
            + inject_top
            + content[body_open_end:]
        )
    body_close = content.rfind("</body>")
    if body_close != -1:
        content = content[:body_close] + inject_bottom + content[body_close:]

    # Update brand bar subtitle + pill for this page.
    content = content.replace(
        "<p id=\"gw-subtitle\">Campus Energy Optimization</p>",
        '<p id="gw-subtitle">Campus Energy Optimization · API Reference</p>',
    )
    content = content.replace(
        '<span id="gw-pill-text">Live</span>',
        '<span id="gw-pill-text">OpenAPI 3 · Live</span>',
    )
    return HTMLResponse(content=content)


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------
@router.get("/health")
def health(ui: bool = False) -> Response:
    """JSON readiness probe (or rendered HTML if ?ui=1)."""
    if ui:
        return HTMLResponse(content=_render_health_html())
    return {"status": "ok"}


@router.get("/api/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "solver": "HiGHS (scipy.optimize.linprog)",
        "runtime": "Python 3.11 · FastAPI · Uvicorn",
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
# Health page renderer
# ---------------------------------------------------------------------------
def _render_health_html() -> str:
    """Premium dark glass-morphism health probe page."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GridWise · Health Probe</title>
{_FONT_LINK}
<link rel="stylesheet" href="/static/page.css">
<link rel="stylesheet" href="/static/health.css">
</head>
<body>
{_topbar('health')}
<main class="gw-container">
  <div class="health-shell">
    <header style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
      <div class="gw-brand-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </div>
      <div>
        <h1 style="margin:0;font-family:var(--font-display);font-weight:500;font-size:clamp(1.85rem,4vw,2.6rem);letter-spacing:-0.03em;color:#fff;line-height:1.1;">GridWise Health Probe</h1>
        <p class="health-eyebrow">Live service readiness and runtime status</p>
      </div>
    </header>

    <section class="health-hero">
      <div class="health-status" id="health-status">
        <span class="health-dot"></span>
        <span id="health-status-text">Checking service health…</span>
        <span class="health-status-meta" id="health-status-meta">Probe in progress</span>
      </div>

      <div class="health-grid">
        <div class="health-metric brand">
          <div class="health-label">API Status</div>
          <div class="health-value" id="api-status">—</div>
          <div class="health-value-sub" id="api-status-sub">Initial readiness probe</div>
        </div>
        <div class="health-metric info">
          <div class="health-label">Service</div>
          <div class="health-value" id="service-name">{settings.APP_NAME}</div>
          <div class="health-value-sub" id="service-runtime">Python · FastAPI · Uvicorn</div>
        </div>
        <div class="health-metric violet">
          <div class="health-label">Version</div>
          <div class="health-value" id="version">—</div>
          <div class="health-value-sub">Released for BUP CSE Fest 2026</div>
        </div>
        <div class="health-metric info">
          <div class="health-label">Solver</div>
          <div class="health-value" id="solver">—</div>
          <div class="health-value-sub">Linear programming backend</div>
        </div>
        <div class="health-metric brand">
          <div class="health-label">LLM Mode</div>
          <div class="health-value" id="llm-mode">—</div>
          <div class="health-value-sub" id="llm-mode-sub">Awaiting probe response</div>
        </div>
        <div class="health-metric solar">
          <div class="health-label">Active Models</div>
          <div class="health-value" id="llm-models">—</div>
          <div class="health-models" id="llm-models-chips"></div>
        </div>
      </div>

      <div class="health-meta">
        <span><strong>Endpoint</strong> <code>GET /health</code></span>
        <span><strong>Status</strong> <code>GET /api/status</code></span>
        <span><strong>JSON probe</strong> <a href="/health" style="color:#22d3ee;text-decoration:none;font-weight:600;">View raw JSON →</a></span>
      </div>

      <nav class="health-action-row">
        <a href="/" class="primary">← Live Dashboard</a>
        <a href="/docs">Swagger Docs</a>
        <a href="/api/status" target="_blank" rel="noopener">Open JSON Status</a>
        <a href="/optimize-energy" onclick="event.preventDefault();alert('POST /optimize-energy — see Swagger for usage.');">POST /optimize-energy</a>
      </nav>
    </section>

    <h2 class="health-section-title">
      Endpoint Reference
      <span class="health-eyebrow-inline" id="health-refreshed">Live · auto-refresh every 10s</span>
    </h2>
    <div class="health-endpoints">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Path</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="method-pill get">GET</span></td>
            <td>
              <span class="path"><a href="/health">/health</a></span>
              <div class="desc">JSON readiness probe · <code style="color:#22d3ee;">{{"status":"ok"}}</code></div>
            </td>
            <td class="desc-inline">Used by judges and uptime monitors.</td>
          </tr>
          <tr>
            <td><span class="method-pill get">GET</span></td>
            <td>
              <span class="path"><a href="/health?ui=1">/health?ui=1</a></span>
              <div class="desc">This branded health probe page</div>
            </td>
            <td class="desc-inline">Matches dashboard design system.</td>
          </tr>
          <tr>
            <td><span class="method-pill get">GET</span></td>
            <td>
              <span class="path"><a href="/api/status">/api/status</a></span>
              <div class="desc">Runtime metadata (version, solver, LLM mode)</div>
            </td>
            <td class="desc-inline">Includes configured LLM models.</td>
          </tr>
          <tr>
            <td><span class="method-pill get">GET</span></td>
            <td>
              <span class="path"><a href="/docs">/docs</a></span>
              <div class="desc">Themed interactive API explorer</div>
            </td>
            <td class="desc-inline">Try every endpoint from the browser.</td>
          </tr>
          <tr>
            <td><span class="method-pill post">POST</span></td>
            <td>
              <span class="path"><a href="/optimize-energy">/optimize-energy</a></span>
              <div class="desc">Main API: interpret directives and optimize schedule</div>
            </td>
            <td class="desc-inline">Accepts scenario + operator notes + battery spec.</td>
          </tr>
          <tr>
            <td><span class="method-pill post">POST</span></td>
            <td>
              <span class="path"><a href="/api/analyze">/api/analyze</a></span>
              <div class="desc">Optimization plus savings, warnings, and hourly insights</div>
            </td>
            <td class="desc-inline">Extended analytics response.</td>
          </tr>
          <tr>
            <td><span class="method-pill post">POST</span></td>
            <td>
              <span class="path"><a href="/api/export-csv">/api/export-csv</a></span>
              <div class="desc">Download the optimized 24-hour schedule as CSV</div>
            </td>
            <td class="desc-inline">Returns <code style="color:#22d3ee;">text/csv</code> attachment.</td>
          </tr>
          <tr>
            <td><span class="method-pill get">GET</span></td>
            <td>
              <span class="path"><a href="/api/presets">/api/presets</a></span>
              <div class="desc">Public sample scenarios for the dashboard</div>
            </td>
            <td class="desc-inline">Loads preset cards on the Compose tab.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</main>
{_footer()}
<script>
(async function () {{
  const err = (msg) => {{
    const statusEl = document.getElementById('health-status-text');
    const pillEl = document.getElementById('health-status');
    const apiEl = document.getElementById('api-status');
    if (statusEl) statusEl.textContent = msg;
    if (pillEl) pillEl.classList.add('error');
    if (apiEl) {{ apiEl.textContent = 'ERROR'; apiEl.classList.remove('ok'); apiEl.classList.add('warn'); }}
  }};
  const setMeta = (txt) => {{
    const el = document.getElementById('health-status-meta');
    if (el) el.textContent = txt;
  }};
  try {{
    const probeStart = performance.now();
    const [h, s] = await Promise.all([
      fetch('/health').then((r) => r.json()),
      fetch('/api/status').then((r) => r.json()),
    ]);
    const probeMs = Math.round(performance.now() - probeStart);
    const ok = h && h.status === 'ok';
    const statusEl = document.getElementById('health-status-text');
    const pillEl = document.getElementById('health-status');
    const apiEl = document.getElementById('api-status');
    const apiSubEl = document.getElementById('api-status-sub');
    const runtimeEl = document.getElementById('service-runtime');
    const verEl = document.getElementById('version');
    const solverEl = document.getElementById('solver');
    const llmEl = document.getElementById('llm-mode');
    const llmSubEl = document.getElementById('llm-mode-sub');
    const modelsEl = document.getElementById('llm-models');
    const chipsEl = document.getElementById('llm-models-chips');

    if (statusEl) statusEl.textContent = ok ? 'Service is healthy and ready' : 'Service reported an issue';
    if (pillEl) pillEl.classList.toggle('error', !ok);
    if (apiEl) {{
      apiEl.textContent = ok ? 'OK' : String(h?.status || '—').toUpperCase();
      apiEl.classList.toggle('ok', ok);
    }}
    if (apiSubEl) apiSubEl.textContent = 'Responded in ' + probeMs + ' ms · v' + (s.version || '—');
    if (runtimeEl) runtimeEl.textContent = (s.runtime || 'Python · FastAPI · Uvicorn');
    if (verEl) verEl.textContent = s.version || '—';
    if (solverEl) solverEl.textContent = s.solver || '—';
    if (llmEl) {{
      if (s.llm_configured) {{
        llmEl.textContent = 'Online · Gemini';
        llmEl.classList.add('ok');
        llmEl.classList.remove('warn');
      }} else {{
        llmEl.textContent = 'Offline';
        llmEl.classList.add('warn');
        llmEl.classList.remove('ok');
      }}
    }}
    if (llmSubEl) llmSubEl.textContent = s.llm_configured
      ? 'Directives interpreted via Gemini LLM'
      : 'Deterministic regex fallback parser';
    if (modelsEl) modelsEl.textContent = (s.llm_models || []).join(', ') || 'fallback parser';
    if (chipsEl) {{
      const models = (s.llm_models || []);
      chipsEl.innerHTML = '';
      if (models.length === 0) {{
        const chip = document.createElement('span');
        chip.className = 'health-chip fallback';
        chip.textContent = 'fallback parser';
        chipsEl.appendChild(chip);
      }} else {{
        models.forEach((m) => {{
          const chip = document.createElement('span');
          chip.className = 'health-chip';
          chip.textContent = m;
          chipsEl.appendChild(chip);
        }});
      }}
    }}
    setMeta('Last probed · ' + probeMs + ' ms');
  }} catch (e) {{
    err('Unable to reach service');
    setMeta('Probe failed');
  }}
  // Auto refresh timestamp label so the UI feels alive.
  let seconds = 0;
  setInterval(() => {{
    seconds += 1;
    const el = document.getElementById('health-refreshed');
    if (el) el.textContent = 'Live · auto-refresh every 10s · +' + seconds + 's';
  }}, 1000);
}})();
</script>
</body>
</html>"""
