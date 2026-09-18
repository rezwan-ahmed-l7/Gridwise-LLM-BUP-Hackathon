# GridWise LLM - BUP CSE Fest 2026 Hackathon Preliminary

Smart Campus Energy Optimization with **LLM-assisted Operator Directive Interpretation**.

## Overview

This service accepts a 24-hour campus energy scenario + natural language operator notes, interprets the notes using an LLM (Google Gemini), applies deterministic guardrails, and produces a valid cost-minimizing schedule.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness check → `{"status": "ok"}` |
| POST | `/optimize-energy` | Main optimization endpoint |

## Quick Start (Local)

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key (free tier works)

### 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt