# GridWise LLM

> **BUP CSE Fest 2026 · Hackathon** — A premium FastAPI service for LLM-assisted campus microgrid energy optimization.

GridWise interprets natural-language operator directives, validates them through structured guardrails, and solves the resulting 24-hour battery scheduling problem with the **HiGHS** linear-programming solver — all wrapped in a glass-morphism dashboard built for the hackathon.

**Live Demo:** [https://gridwise-llm-bup.onrender.com](https://gridwise-llm-bup.onrender.com)

---

## Screenshots

### Live Dashboard
![Dashboard]<img width="1920" height="1391" alt="screencapture-gridwise-llm-bup-onrender-2026-09-22-02_06_03" src="https://github.com/user-attachments/assets/d51c0355-190e-4ea2-b190-0f8f6244c13c" />

### Health Probe
![Health]<img width="1920" height="1922" alt="screencapture-gridwise-llm-bup-onrender-health-2026-09-22-02_06_34" src="https://github.com/user-attachments/assets/4397e520-b4ec-4acf-8c78-665fe90837a5" />

### Swagger API Reference
![Swagger]<img width="1920" height="1156" alt="screencapture-gridwise-llm-bup-onrender-docs-2026-09-22-02_06_19" src="https://github.com/user-attachments/assets/d4b89097-92ee-4658-8f08-7b6d3f928b2d" />

---

## Highlights

- **LLM-powered directive interpretation** (Google Gemini) with deterministic NLP fallback (no API key required).
- **Exact linear optimization** through `scipy.optimize.linprog` (HiGHS method).
- **Refactored, professional codebase** with clear separation of concerns (`core`, `services`, `models`, `api`).
- **Premium dark dashboard** with glass-morphism, animated charts, and responsive design.
- **Guardrail validation** for directive types, time windows, battery limits, and grid caps.
- **Multiple API surfaces**: live dashboard, Swagger UI, health probe, CSV export, and analytics.

---

## Quick Start

**Requirements**: Python 3.10 or newer. A Google Gemini API key is **optional** — the service includes a deterministic NLP fallback so the dashboard and API remain fully usable without one.

```bash
git clone https://github.com/rezwan-ahmed-l7/Gridwise-LLM-BUP-Hackathon.git
cd Gridwise-LLM-BUP-Hackathon

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
- **Live Dashboard** → [http://localhost:8000](http://localhost:8000)
- **Swagger API** → [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health UI** → [http://localhost:8000/health?ui=1](http://localhost:8000/health?ui=1)

---

## Project Structure

```
Gridwise-LLM-BUP/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/routes.py              # All endpoints
│   ├── core/                      # Config + LLM
│   ├── services/                  # Guardrails, optimizer, pipeline
│   ├── models/schemas.py          # Pydantic models
│   └── utils/parser.py            # Deterministic NLP fallback
├── static/                        # Dashboard UI (HTML/CSS/JS)
├── docs/screenshots/              # README screenshots
├── requirements.txt
├── Dockerfile
├── .env.example
├── sample_request.json
└── README.md
```

---

## Live Surfaces

| Surface | URL | Purpose |
| --- | --- | --- |
| Live Dashboard | `/` | Configure scenarios, run optimization, inspect charts |
| Swagger API | `/docs` | Explore and execute OpenAPI operations |
| Health UI | `/health?ui=1` | Branded readiness + runtime status |
| Health JSON | `/health` | Machine-readable `{"status":"ok"}` |
| Runtime Status | `/api/status` | Non-secret service metadata |

---

## API Contract

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service readiness probe |
| `POST` | `/optimize-energy` | Interpret notes and optimize 24-hour schedule |
| `POST` | `/api/analyze` | Optimization + savings, warnings, insights |
| `POST` | `/api/export-csv` | Download optimized schedule as CSV |

### Example request

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## Configuration

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.0-flash,gemini-1.5-flash
LLM_TIMEOUT_SECONDS=12
PORT=8000
```

> Deterministic NLP fallback works without a Gemini key. **Never commit `.env` or real API keys.**

---

## Processing Pipeline

1. **Interpretation** — Gemini (or fallback parser) turns operator notes into structured directives
2. **Guardrails** — Validate directive types, hours, battery limits, grid caps
3. **Optimization** — Exact 24-hour LP with `scipy.optimize.linprog` (HiGHS)
4. **Post-processing** — Energy balance, end-of-day neutrality, response validation

---

## Supported Directive Types

- `solar_reduction`
- `minimum_battery_reserve`
- `no_charge_window`
- `no_discharge_window`
- `max_grid_window`
- `no_op`

---

## Docker

```bash
docker build -t gridwise-llm:latest .
docker run -d --name gridwise-service -p 8000:8000 \
  -e GEMINI_API_KEY="" \
  gridwise-llm:latest
```

---

## License

Released under the [MIT License](./LICENSE).
