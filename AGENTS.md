# Repository Guidelines

## Overview
This repo hosts an OpenAI-based prompt-chaining pipeline that generates Canvas LMS-ready HTML accessible Pages with ImageKit-hosted images. The primary entry point is `openai_lms_pipeline.py`, which must use the OpenAI Responses API for text generation and the Images API for native image generation.

## Project Structure & Module Organization
- `openai_lms_pipeline.py`: OpenAI pipeline (Agents A–D) using Responses + Images APIs.
- `gemini_lms_pipeline.py`: Legacy Gemini pipeline (reference only).
- `prompts/`: Agent prompts shared across pipelines.
- `lms_output_*/`: Generated output artifacts (HTML, images, logs).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create/activate venv.
- `pip install -r requirements.txt`: Install dependencies.
- `python openai_lms_pipeline.py`: Run the OpenAI pipeline (interactive).

## OpenAI API Usage (Required)
- Use `from openai import OpenAI` and `client = OpenAI()`.
- All text stages must call `client.responses.create(...)` and extract text via a robust helper (see `text_from_response` in `openai_lms_pipeline.py`).
- Image generation must call `client.images.generate(model="gpt-image-1.5", ...)` and decode `b64_json`.
- Agent models are fixed:
  - Agent A: `gpt-5.2` with `reasoning={"effort":"high"}`.
  - Agents B/C/D: `gpt-5-mini` with no temperature/top_p.

## Note (For Agents)
The OpenAI pipeline uses the Responses API, not Chat Completions. Use client.responses.create(…) for all text agents and client.images.generate(…) for image generation. Do not attempt to fetch platform.openai.com/docs from the agent; those pages may be blocked by Cloudflare. For API fields and request/response shapes, consult vendor/openai.openapi.documented.yml (downloaded from the OpenAI OpenAPI spec). For GPT-5.2: if reasoning.effort is set to low/medium/high/xhigh, do not pass temperature/top_p/logprobs, as those are only supported when reasoning.effort is none.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; PEP 8 conventions.
- Keep prompts in `prompts/` with names like `agent_a_textbook.txt`.

## Testing Guidelines
- No automated tests yet. For changes, do a short manual run and confirm outputs in `lms_output_*/content/`.

## Configuration & Secrets
- Required env vars: `OPENAI_API_KEY`, `IMAGEKIT_PRIVATE_KEY`, `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_URL_ENDPOINT`.
