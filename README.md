# GridWise LLM

A premium FastAPI service for LLM-assisted campus microgrid energy optimization. GridWise interprets natural-language operator directives, validates them through structured guardrails, and solves the resulting 24-hour battery scheduling problem with the HiGHS linear-programming solver. Built for the **BUP CSE Fest 2026 Hackathon**.

## Highlights

- **LLM-powered directive interpretation** with deterministic NLP fallback (no API key required).
- **Exact linear optimization** through `scipy.optimize.linprog` (HiGHS method).
- **Premium dark dashboard** with glass-morphism, animated accents, and refined typography.
- **Guardrail validation** for directive types, time windows, battery limits, and grid caps.
- **Multiple API surfaces**: live dashboard, Swagger UI, health probe, CSV export, and analytics.

## Live Surfaces

| Surface | URL | Purpose |
| --- | --- | --- |
| Live Dashboard | `/` | Configure scenarios, run optimization, inspect charts and hourly results |
| Swagger API Reference | `/docs` | Explore and execute OpenAPI operations |
| Health UI | `/health?ui=1` | Service readiness, runtime version, solver, and LLM configuration |
| Health JSON | `/health` | Machine-readable readiness probe returning `{"status":"ok"}` |
| Runtime Status | `/api/status` | Non-secret service and solver metadata |

All surfaces share a unified visual system: deep navy glass panels, green-to-cyan accents, monospace metrics, and responsive layouts.

## API Contract

| Method | Endpoint | Description | Success |
| --- | --- | --- | --- |
| `GET` | `/health` | Service readiness probe | `200 OK` |
| `POST` | `/optimize-energy` | Interpret notes and optimize the 24-hour schedule | `200 OK` |
| `POST` | `/api/analyze` | Optimization plus savings, warnings, and hourly insights | `200 OK` |
| `POST` | `/api/export-csv` | Download the optimized schedule as CSV | `200 OK` |

Invalid payloads return `422`. Controlled processing failures return `500` without credentials or stack traces.

## Local Setup

**Requirements**: Python 3.10 or newer. A Google Gemini API key is optional — the service includes a deterministic NLP fallback so the dashboard and API remain fully usable without one.

```bash
git clone <YOUR_REPO_URL>
cd Gridwise-LLM-BUP
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

Install dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the live dashboard at [http://localhost:8000](http://localhost:8000).

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
PORT=8000
```

Available tunables:

- `GEMINI_API_KEY` — Enables LLM-driven directive interpretation.
- `GEMINI_MODEL` — Primary Gemini model name.
- `GEMINI_FALLBACK_MODELS` — Comma-separated fallback models.
- `LLM_TIMEOUT_SECONDS` — Per-request LLM timeout (default `12`).
- `PORT` — HTTP port for the service (default `8000`).

The service has a deterministic NLP fallback, so the dashboard and API remain usable without a Gemini key. Never commit `.env` or API keys.

## Example Requests

Readiness probe:

```bash
curl http://localhost:8000/health
```

Optimization:

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The sample request contains a 24-hour demand and solar profile, operator notes, and battery limits. The response includes interpreted directives, hourly battery actions, grid purchases, total cost, and peak grid import.

## Processing Pipeline

1. **Interpretation** — Gemini parses each operator note into a structured directive, with a deterministic NLP fallback.
2. **Guardrails** — Directive types, hour windows, battery consistency, and grid caps are validated.
3. **Optimization** — A 24-hour linear program is solved exactly with `scipy.optimize.linprog` (HiGHS).
4. **Post-processing** — Energy balance, mutual exclusivity, day-end neutrality, and response validation.

## Supported Directive Types

- `solar_reduction` — Reduce rooftop solar output during specific hours.
- `minimum_battery_reserve` — Enforce a minimum battery reserve during specific hours.
- `no_charge_window` — Disable battery charging during specific hours.
- `no_discharge_window` — Disable battery discharging during specific hours.
- `max_grid_window` — Cap grid import during specific hours.
- `no_op` — Note does not affect the schedule (cafeteria, library, seminar, etc.).

## Docker

```bash
docker build -t gridwise-llm:latest .
docker run -d --name gridwise-service -p 8000:8000 -e GEMINI_API_KEY="" -e PORT=8000 gridwise-llm:latest
curl http://localhost:8000/health
```

## Project Structure

```
Gridwise-LLM-BUP/
├── main.py                     # FastAPI app, LP solver, directive parsing
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
├── sample_request.json         # Example payload for /optimize-energy
├── static_presets.json         # Fallback preset definitions
├── static/
│   └── index.html              # Premium dashboard UI
└── Question/
    └── BUP_CSE_FEST_2026_*.json # Public sample cases
```

## Dependencies

- **FastAPI** and **Uvicorn** — HTTP server and ASGI runtime.
- **Pydantic v2** — Request and response validation.
- **Google Generative AI SDK** — LLM directive interpretation.
- **NumPy** and **SciPy** — Linear programming with the HiGHS solver.
- **python-dotenv** — Local environment variable loading.

## License

Released under the [MIT License](./LICENSE).
