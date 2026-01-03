# Repository Guidelines

## Project Structure & Module Organization
- `anthropic/`: Main pipeline scripts (`lms_pipeline.py`, `anthropic_lms_chain.py`, `lms_image_enhancer.py`), templates, and sample outputs.
- `prompts/`: Prompt text files for each agent and content type.
- `gemini_lms_pipeline.py`: Alternate entry point for Gemini-backed runs.
- `lms_output_*/`: Generated run artifacts (HTML, images, logs). Treat as build outputs.

## Build, Test, and Development Commands
- `python anthropic/lms_pipeline.py`: Run the interactive LMS pipeline.
- `python anthropic/lms_pipeline.py --type textbook --topic "..."`: CLI mode for a specific content type.
- `python anthropic/anthropic_lms_chain.py`: Run Agents A–C only (no image generation).
- `python anthropic/lms_image_enhancer.py input.html output.html`: Add images to existing HTML.
- `pip install -r anthropic/requirements.txt`: Install Python dependencies.

## Coding Style & Naming Conventions
- Python 3.10+ code; follow PEP 8 style and 4-space indentation.
- Prefer descriptive, snake_case function and variable names.
- Keep prompts in `prompts/` with names like `agent_a_textbook.txt`.

## Testing Guidelines
- No automated test suite is present. If you add tests, place them under `tests/` and name files `test_*.py`.
- When modifying pipelines, do a local smoke run with a short topic to validate output files.

## Commit & Pull Request Guidelines
- Git history is minimal (initial commit only), so no established commit format.
- Use concise, imperative commit messages (e.g., "Add image enhancer CLI flags").
- PRs should include: summary of changes, example command used, and sample output path if it affects HTML generation.

## Configuration & Secrets
- Store API keys in environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `IMAGEKIT_PRIVATE_KEY`).
- Avoid committing generated outputs or secrets; prefer `.env` locally.
