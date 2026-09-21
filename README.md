# GridWise LLM

> **BUP CSE Fest 2026 · Hackathon** — A premium FastAPI service for LLM-assisted campus microgrid energy optimization.

GridWise interprets natural-language operator directives, validates them through structured guardrails, and solves the resulting 24-hour battery scheduling problem with the **HiGHS** linear-programming solver — all wrapped in a glass-morphism dashboard built for the hackathon.

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
git clone <YOUR_REPO_URL>
cd Gridwise-LLM-BUP

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
- **Health probe** → [http://localhost:8000/health](http://localhost:8000/health)

---

## Project Structure

```
Gridwise-LLM-BUP/
├── app/
│   ├── __init__.py                # Package marker (version)
│   ├── main.py                    # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # All endpoints (health, optimize-energy, etc.)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings & environment variables
│   │   └── llm.py                 # Gemini LLM interpretation logic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── guardrails.py          # Directive validation & cleaning
│   │   ├── optimizer.py           # HiGHS LP-based energy optimizer
│   │   └── pipeline.py            # Orchestrates interpretation → optimization
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # All Pydantic models (API contract)
│   └── utils/
│       ├── __init__.py
│       └── parser.py              # Deterministic NLP fallback
├── static/
│   ├── index.html                 # Dashboard markup
│   ├── style.css                  # Glass-morphism dark theme
│   └── script.js                  # Charts, table rendering, API calls
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
| Live Dashboard | `/` | Configure scenarios, run optimization, inspect charts and hourly results |
| Swagger API Reference | `/docs` | Explore and execute OpenAPI operations |
| Health UI | `/health?ui=1` | Service readiness, runtime version, solver, and LLM configuration |
| Health JSON | `/health` | Machine-readable readiness probe returning `{"status":"ok"}` |
| Runtime Status | `/api/status` | Non-secret service and solver metadata |

---

## API Contract

| Method | Endpoint | Description | Success |
| --- | --- | --- | --- |
| `GET`  | `/health` | Service readiness probe | `200 OK` |
| `POST` | `/optimize-energy` | Interpret notes and optimize the 24-hour schedule | `200 OK` |
| `POST` | `/api/analyze` | Optimization plus savings, warnings, and hourly insights | `200 OK` |
| `POST` | `/api/export-csv` | Download the optimized schedule as CSV | `200 OK` |

Invalid payloads return `422`. Controlled processing failures return `500` without credentials or stack traces.

### Example request

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The sample request contains a 24-hour demand and solar profile, operator notes, and battery limits. The response includes interpreted directives, hourly battery actions, grid purchases, total cost, and peak grid import.

---

## Configuration

Create a `.env` file in the project root (copy `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.0-flash,gemini-1.5-flash
LLM_TIMEOUT_SECONDS=12
PORT=8000
```

| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | Enables LLM-driven directive interpretation. |
| `GEMINI_MODEL` | Primary Gemini model name. |
| `GEMINI_FALLBACK_MODELS` | Comma-separated fallback models. |
| `LLM_TIMEOUT_SECONDS` | Per-request LLM timeout (default `12`). |
| `HOST` / `PORT` | Server bind address and port (default `0.0.0.0:8000`). |

> The service has a deterministic NLP fallback, so the dashboard and API remain usable without a Gemini key. **Never commit `.env` or real API keys.**

---

## Processing Pipeline

1. **Interpretation** — Gemini parses each operator note into a structured directive, with a deterministic NLP fallback.
2. **Guardrails** — Directive types, hour windows, battery consistency, and grid caps are validated.
3. **Optimization** — A 24-hour linear program is solved exactly with `scipy.optimize.linprog` (HiGHS).
4. **Post-processing** — Energy balance, mutual exclusivity, day-end neutrality, and response validation.

---

## Supported Directive Types

- `solar_reduction` — Reduce rooftop solar output during specific hours.
- `minimum_battery_reserve` — Enforce a minimum battery reserve during specific hours.
- `no_charge_window` — Disable battery charging during specific hours.
- `no_discharge_window` — Disable battery discharging during specific hours.
- `max_grid_window` — Cap grid import during specific hours.
- `no_op` — Note does not affect the schedule (cafeteria, library, seminar, etc.).

---

## Docker

```bash
docker build -t gridwise-llm:latest .
docker run -d --name gridwise-service -p 8000:8000 \
  -e GEMINI_API_KEY="" \
  gridwise-llm:latest

curl http://localhost:8000/health
```

---

## Code Quality Notes

- **Type hints everywhere** — every public function and method is fully annotated.
- **Docstrings** on every module, class, and public function explaining purpose, parameters, and return values.
- **No circular imports** — `core/config.py` and `core/llm.py` are leaves; `services/` depends only on `models/` and `core/`; `api/routes.py` is the orchestrator.
- **Pydantic v2** for all input/output validation, including ship-ready `response_model` annotations on every endpoint.

---

## Dependencies

- **FastAPI** and **Uvicorn** — HTTP server and ASGI runtime.
- **Pydantic v2** — Request and response validation.
- **Google Generative AI SDK** — LLM directive interpretation (optional).
- **NumPy** and **SciPy** — Linear programming with the HiGHS solver.
- **python-dotenv** — Local environment variable loading.

---

## License

Released under the [MIT License](./LICENSE).
