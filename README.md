# LMS Prompt Chain

OpenAI-based prompt-chaining pipeline that generates Canvas LMS-ready HTML pages,
optionally with ImageKit-hosted illustrations. The primary entry point is
`openai_lms_pipeline.py` and it uses the OpenAI Responses API for text and the
Images API for image generation.

## Requirements

- Python 3.10+
- Environment variables:
  - `OPENAI_API_KEY`
  - `IMAGEKIT_PRIVATE_KEY`
  - `IMAGEKIT_PUBLIC_KEY`
  - `IMAGEKIT_URL_ENDPOINT`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Interactive run:

```bash
python openai_lms_pipeline.py
```

Non-interactive run:

```bash
python openai_lms_pipeline.py --type textbook --topic "Machine Learning fundamentals"
```

Resume from an existing Step 1 draft (skips Agent A):

```bash
python openai_lms_pipeline.py --input-file lms_output_YYYYMMDD-HHMMSS/content/Step1_Content.txt
```

Options:

- `--no-images` to skip image generation.
- `--force-images` to enable images for non-textbook content types.

## Outputs

Runs create `lms_output_YYYYMMDD-HHMMSS/` with:

- `content/Step1_Content.txt`, `Step2_Structured.html`, `Step3_Styled.html`,
  `Step4_Final.html` (if images enabled)
- `images/` (if images enabled)
- `pipeline.log`

A convenience copy of the final HTML is also saved to
`lms_output_YYYYMMDD-HHMMSS.html` in the repo root.
