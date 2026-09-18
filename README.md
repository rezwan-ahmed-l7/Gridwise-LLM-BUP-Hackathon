# GridWise LLM - Smart Campus Energy Optimization

**Event**: BUP CSE Fest 2026 Hackathon · Online Preliminary  
**Service**: LLM-Assisted Campus Microgrid Energy Optimization  
**Endpoints**: `GET /health` | `POST /optimize-energy`  

---

## 1. Architecture Overview

The system operates as a robust multi-stage pipeline:
```
[Campus Energy Scenario + Operator Notes]
                    │
                    ▼
       [Stage 1: LLM Interpreter]
       (Google Gemini with NLP Fallback)
                    │
                    ▼
       [Stage 2: Deterministic Guardrails]
       (Schema, Enum, Time Window & Bound Validation)
                    │
                    ▼
       [Stage 3: Mathematical Optimizer]
       (Exact Linear Programming via HiGHS / SciPy)
                    │
                    ▼
       [Stage 4: Post-Processing & Validation]
       (Energy Balance, Neutrality & Totals)
                    │
                    ▼
            [API Response]
```

1. **LLM Interpreter**: Translates 1–3 natural-language operator notes into structured directives (`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window`, or `no_op`).
2. **Deterministic Guardrails**: Validates and normalizes directive parameters (ascending unique hours 0–23, clamped values, applies flags). If the LLM output is malformed or offline, a verified NLP extractor guarantees seamless fallback.
3. **Mathematical Optimizer**: Formulates the 24-hour campus scheduling problem as an exact Linear Program solved via the HiGHS solver in `scipy.optimize.linprog`. Enforces strict hourly energy balance ($g_h + s_h + d_h - c_h = D_h$), rate limits, storage boundaries, and end-of-day battery neutrality ($E_{23} = E_0$).
4. **Post-Processing**: Ensures mutual exclusivity between battery charging and discharging, and re-derives exact numerical figures to guarantee zero constraint violations.

---

## 2. API Contract

### Endpoints

| Method | Endpoint | Description | Expected Status |
|--------|----------|-------------|-----------------|
| `GET` | `/health` | Service readiness probe | `200 OK` (`{"status": "ok"}`) |
| `POST` | `/optimize-energy` | 24-hour LLM interpretation & schedule optimization | `200 OK` |

### Error Codes
- `200`: Successful health or optimization response.
- `422`: Structurally invalid or malformed request payload.
- `500`: Controlled internal error (credentials and stack traces are never exposed).

---

## 3. Local Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14)
- (Optional) Google Gemini API Key

### Step-by-Step Setup

```bash
# 1. Clone repository & enter directory
git clone <YOUR_REPO_URL>
cd Gridwise-LLM-BUP

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Copy .env.example to .env and set your key:
cp .env.example .env
```

### Running the Service

```bash
# Start server with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Configuration & Environment Variables

Create a `.env` file in the root directory:

```env
# Google Gemini API Key (Optional for offline testing; required for live model calls)
GEMINI_API_KEY=your_gemini_api_key_here

# Model identifier (Default: gemini-1.5-flash)
GEMINI_MODEL=gemini-1.5-flash

# Service Port (Default: 8000)
PORT=8000
```

> **Security Note**: Never commit API keys or `.env` files to the repository. The application redacts error details in 500 responses to prevent secret exposure.

---

## 5. Sample API Testing

### Test Readiness Endpoint
```bash
curl -X GET http://localhost:8000/health
```
**Expected Response**:
```json
{"status": "ok"}
```

### Test Energy Optimization Endpoint
```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Sample Request Shape**:
```json
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    ...
  ],
  "battery": {
    "capacity_kwh": 220,
    "initial_energy_kwh": 110,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50
  }
}
```

---

## 6. Docker Fallback Deployment

A tested Docker container image is provided for reproducible testing.

### Build Docker Image
```bash
docker build -t gridwise-llm:latest .
```

### Run Docker Container
```bash
docker run -d --name gridwise-service \
  -p 8000:8000 \
  -e GEMINI_API_KEY="" \
  -e PORT=8000 \
  gridwise-llm:latest
```

### Test Container Health
```bash
curl http://localhost:8000/health
```

---

## 7. Dependencies & Solver

- **Framework**: `FastAPI` + `Uvicorn`
- **Data Validation**: `Pydantic v2`
- **Generative AI**: `google-generativeai` (Gemini Flash)
- **Optimization Solver**: `SciPy` (HiGHS Simplex / Dual Simplex / Interior Point solver)
- **Mathematical Array Processing**: `NumPy`

---

## 8. Known Limitations & Notes

- Optimization assumes hourly discrete intervals ($t = 0 \dots 23$).
- Solar energy cannot be exported back to the grid (unused solar is curtailed according to challenge rules).
- The fallback NLP parser ensures 100% test coverage even if network or API quotas fail.