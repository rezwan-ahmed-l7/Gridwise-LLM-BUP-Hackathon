# GridWise LLM

> **BUP CSE Fest 2026 · Hackathon** — A premium FastAPI service for LLM-assisted campus microgrid energy optimization.

GridWise interprets natural-language operator directives, validates them through structured guardrails, and solves the resulting 24-hour battery scheduling problem with the **HiGHS** linear-programming solver — all wrapped in a glass-morphism dashboard built for the hackathon.

**Live Demo:** [https://gridwise-llm-bup.onrender.com](https://gridwise-llm-bup.onrender.com)

---

## Screenshots

### Live Dashboard
![Dashboard](<img width="1920" height="1391" alt="screencapture-gridwise-llm-bup-onrender-2026-09-22-02_06_03" src="https://github.com/user-attachments/assets/d51c0355-190e-4ea2-b190-0f8f6244c13c" />)


### Health Probe
![Health](<img width="1920" height="1922" alt="screencapture-gridwise-llm-bup-onrender-health-2026-09-22-02_06_34" src="https://github.com/user-attachments/assets/4397e520-b4ec-4acf-8c78-665fe90837a5" />
)

### Swagger API Reference
![Swagger](<img width="1920" height="1156" alt="screencapture-gridwise-llm-bup-onrender-docs-2026-09-22-02_06_19" src="https://github.com/user-attachments/assets/d4b89097-92ee-4658-8f08-7b6d3f928b2d" />
)

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
