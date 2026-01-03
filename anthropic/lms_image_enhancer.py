#!/usr/bin/env python3
"""
LMS Image Enhancement Pipeline (Agent D)
=========================================
Post-processing script to add AI-generated images to vetted HTML content.

Pipeline:
  1. Claude brainstorms up to 3 optimal image insertion points
  2. Gemini Image Pro generates one image per insertion point
  3. ImageKit.io hosts the generated images via official SDK
  4. Images are injected into the HTML with proper figure/caption markup

Usage:
  python lms_image_enhancer.py [input_file] [output_file]

  Defaults:
    input_file:  agent_c_output.html (or anthropic_*_Step3_Final.html)
    output_file: agent_d_output_YYYYMMDD-HHMMSS.html

Requires Environment Variables:
  - ANTHROPIC_API_KEY
  - GEMINI_API_KEY (or GOOGLE_API_KEY)
  - IMAGEKIT_PRIVATE_KEY

Optional Environment Variables:
  - IMAGEKIT_FOLDER (default: /lms-content/)

Dependencies:
  pip install anthropic google-genai imagekitio

Author: Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import anthropic
from google import genai
from google.genai import types
from imagekitio import ImageKit

# =============================================================================
# CONFIGURATION
# =============================================================================

# Models
CLAUDE_MODEL: Final[str] = "claude-sonnet-4-5"
GEMINI_IMAGE_MODEL: Final[str] = "gemini-3-pro-image-preview"

# Limits
MAX_IMAGES: Final[int] = 3

# Local backup directory (images saved locally before upload)
LOCAL_IMAGE_DIR: Final[Path] = Path("generated_images")

# ImageKit folder for organization (can be overridden via env var)
DEFAULT_IMAGEKIT_FOLDER: Final[str] = "/lms-content/"

# Styling for injected figures
FIGURE_STYLES: Final[dict[str, str]] = {
    "figure": "margin: 2em 0; max-width: 75ch;",
    "img": "width: 100%; height: auto; border: 1px solid #E0E0E0; border-radius: 4px;",
    "figcaption": (
        "padding: 0.75em 1em; font-style: italic; background-color: #F5F5F5; "
        "text-align: center; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; "
        "font-size: 16px; color: #333333; border-bottom-left-radius: 4px; "
        "border-bottom-right-radius: 4px;"
    ),
}

# =============================================================================
# CLAUDE BRAINSTORMING PROMPT
# =============================================================================

AGENT_D_SYSTEM_PROMPT: Final[str] = """
You are Agent D, an expert educational illustrator and instructional designer.
Your goal is to analyze HTML educational content and plan visual assets that enhance learning.

Task:
1. Read the provided HTML content carefully.
2. Identify up to 3 optimal locations where a visual aid would significantly enhance comprehension.
3. Prioritize locations where:
   - Abstract concepts could benefit from visualization
   - Processes or workflows could be diagrammed
   - Comparisons could be illustrated
   - Technical architectures could be shown

For each location, provide:
- "insertion_context": A UNIQUE, VERBATIM snippet of 10-20 words from the HTML content 
  IMMEDIATELY AFTER which the image should appear. This MUST be an exact character-for-character 
  copy of text found in the source. Do not rephrase or paraphrase. Include enough words to be unique.

- "image_prompt": A detailed, specific prompt for image generation. Include:
  - Subject matter and key elements to show
  - Visual style (diagram, illustration, infographic, etc.)
  - Color preferences if relevant
  - Any text labels that should appear
  - Composition guidance

- "alt_text": Descriptive alt text for accessibility (max 125 characters).
  Describe what the image shows, not what it represents conceptually.

- "caption": A figure caption starting with "Figure [N]:" that explains 
  what the image illustrates and how it relates to the content.

Output Format:
Return ONLY a valid JSON object with this exact structure:
{
  "images": [
    {
      "insertion_context": "exact verbatim text from source",
      "image_prompt": "detailed generation prompt",
      "alt_text": "accessibility description",
      "caption": "Figure 1: Explanation of the visual"
    }
  ]
}

Important:
- Return valid JSON only, no markdown code fences
- Maximum 3 images
- Each insertion_context must be unique and findable in the source
- Focus on educational value, not decoration
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImagePlan:
    """Represents a planned image with its metadata."""
    insertion_context: str
    image_prompt: str
    alt_text: str
    caption: str
    figure_number: int
    local_path: Path | None = None
    hosted_url: str | None = None


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def validate_environment() -> tuple[bool, list[str]]:
    """Check all required environment variables are set."""
    required = [
        "ANTHROPIC_API_KEY",
        "IMAGEKIT_PRIVATE_KEY",
    ]
    # Gemini accepts either key name
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    missing = [key for key in required if not os.environ.get(key)]
    if not gemini_key:
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")

    return len(missing) == 0, missing


def find_input_file() -> Path | None:
    """Find the most appropriate input file."""
    candidates = [
        "agent_c_output.html",
    ]

    # Add any anthropic_*_Step3_Final.html files
    for f in Path.cwd().glob("anthropic_*_Step3_Final.html"):
        candidates.append(str(f))

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    return None


def clean_json_response(text: str) -> str:
    """Remove markdown code fences from JSON response."""
    text = text.strip()
    # Remove ```json or ``` prefix
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def brainstorm_images(
        client: anthropic.Anthropic,
        html_content: str,
) -> list[ImagePlan] | None:
    """Use Claude to identify optimal image insertion points."""
    print(f"\n🧠 [1/3] Brainstorming image concepts with Claude...")

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.3,
            system=AGENT_D_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this HTML content and identify up to {MAX_IMAGES} optimal locations for educational images:\n\n{html_content}",
                }
            ],
        )

        # Extract text from response
        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        # Parse JSON
        cleaned = clean_json_response(response_text)
        data = json.loads(cleaned)

        # Convert to ImagePlan objects
        plans = []
        for i, item in enumerate(data.get("images", [])[:MAX_IMAGES], start=1):
            plan = ImagePlan(
                insertion_context=item["insertion_context"],
                image_prompt=item["image_prompt"],
                alt_text=item["alt_text"][:125],  # Ensure alt text limit
                caption=item["caption"],
                figure_number=i,
            )
            plans.append(plan)

        print(f"    ✓ Identified {len(plans)} image insertion points")
        for plan in plans:
            print(f"      • Figure {plan.figure_number}: {plan.caption[:50]}...")

        return plans

    except json.JSONDecodeError as e:
        print(f"    ❌ Failed to parse Claude's response as JSON: {e}")
        print(f"    Response was: {response_text[:500]}...")
        return None
    except anthropic.APIError as e:
        print(f"    ❌ Claude API error: {e.message}")
        return None
    except Exception as e:
        print(f"    ❌ Brainstorming failed: {e}")
        return None


def upload_to_imagekit(
        imagekit_client: ImageKit,
        image_data: bytes,
        filename: str,
        folder: str,
) -> str | None:
    """
    Upload an image to ImageKit.io using the official SDK v5.

    The SDK's files.upload() method accepts:
    - file: bytes, Path, IO[bytes], or base64 string
    - file_name: desired filename
    - folder: target folder path
    - tags: list of tags
    - use_unique_file_name: add random suffix to prevent overwrites

    Returns the hosted URL or None if upload failed.
    """
    try:
        # Upload using the SDK v5 API: client.files.upload()
        result = imagekit_client.files.upload(
            file=image_data,
            file_name=filename,
            folder=folder,
            use_unique_file_name=True,
            is_private_file=False,
            tags=["lms", "educational", "ai-generated"],
        )

        # The result is a FileUploadResponse object with a 'url' attribute
        if result and hasattr(result, 'url') and result.url:
            return result.url

        print(f"        ⚠️ Upload completed but URL not found in response")
        return None

    except Exception as e:
        error_type = type(e).__name__
        print(f"        ❌ ImageKit upload failed ({error_type}): {e}")
        return None


def generate_images(
        gemini_client: genai.Client,
        imagekit_client: ImageKit,
        plans: list[ImagePlan],
        imagekit_folder: str,
        timestamp: str,
) -> list[ImagePlan]:
    """Generate images with Gemini and upload to ImageKit."""
    print(f"\n🎨 [2/3] Generating and hosting {len(plans)} images...")

    # Ensure local backup directory exists
    LOCAL_IMAGE_DIR.mkdir(exist_ok=True)

    successful_plans = []

    for plan in plans:
        print(f"\n    [{plan.figure_number}/{len(plans)}] Generating Figure {plan.figure_number}...")

        try:
            # Build the generation prompt
            full_prompt = (
                f"Generate a high-quality educational illustration: {plan.image_prompt}. "
                f"Style: Professional, clean, minimal text overlay, suitable for a textbook."
            )

            # Generate with Gemini
            response = gemini_client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Extract image data
            image_data = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        image_data = part.inline_data.data
                        break

            if not image_data:
                print(f"        ❌ No image data in Gemini response")
                continue

            # Save locally as backup
            # Filename uses only allowed chars: a-z, A-Z, 0-9, ., -
            local_filename = f"fig{plan.figure_number}-{timestamp}.png"
            local_path = LOCAL_IMAGE_DIR / local_filename
            with open(local_path, "wb") as f:
                f.write(image_data)
            plan.local_path = local_path
            print(f"        ✓ Saved locally: {local_filename}")

            # Upload to ImageKit
            print(f"        ↑ Uploading to ImageKit...")
            hosted_url = upload_to_imagekit(
                imagekit_client=imagekit_client,
                image_data=image_data,
                filename=local_filename,
                folder=imagekit_folder,
            )

            if hosted_url:
                plan.hosted_url = hosted_url
                print(f"        ✓ Hosted: {hosted_url}")
                successful_plans.append(plan)
            else:
                print(f"        ⚠️ Upload failed, using local path as fallback")
                # Still add to successful plans with local path
                successful_plans.append(plan)

        except Exception as e:
            print(f"        ❌ Generation failed: {type(e).__name__}: {e}")
            continue

    print(f"\n    Summary: {len(successful_plans)}/{len(plans)} images processed")
    hosted_count = sum(1 for p in successful_plans if p.hosted_url)
    print(f"    Hosted on ImageKit: {hosted_count}")
    print(f"    Local backups: {len(successful_plans)}")

    return successful_plans


def word_to_pattern(word: str) -> str:
    """
    Convert a word to a regex pattern that allows HTML entities for special chars.

    This handles common Unicode characters that might appear as HTML entities
    in the source HTML, such as em-dashes, curly quotes, etc.
    """
    result = []
    for char in word:
        if char in '—–':  # em-dash, en-dash
            result.append(r'(?:&mdash;|&ndash;|&#8212;|&#8211;|—|–|-)')
        elif char == '"':  # left double quote
            result.append(r'(?:&ldquo;|&#8220;|"|")')
        elif char == '"':  # right double quote
            result.append(r'(?:&rdquo;|&#8221;|"|")')
        elif char == ''':  # left single quote
            result.append(r"(?:&lsquo;|&#8216;|'|')")
        elif char == ''':  # right single quote / apostrophe
            result.append(r"(?:&rsquo;|&#8217;|'|')")
        elif char == '…':  # ellipsis
            result.append(r'(?:&hellip;|&#8230;|…|\.\.\.)')
        elif char == '&':
            result.append(r'(?:&amp;|&)')
        elif char == '<':
            result.append(r'(?:&lt;|<)')
        elif char == '>':
            result.append(r'(?:&gt;|>)')
        elif char == ' ':
            # Allow flexible whitespace
            result.append(r'\s+')
        else:
            result.append(re.escape(char))
    return ''.join(result)


def inject_images_into_html(
        html_content: str,
        plans: list[ImagePlan],
) -> str:
    """Inject figure elements into HTML at the specified locations."""
    print(f"\n📝 [3/3] Injecting {len(plans)} images into HTML...")

    modified_html = html_content

    for plan in plans:
        # Use hosted URL if available, otherwise fall back to local path
        img_src = plan.hosted_url if plan.hosted_url else str(plan.local_path)

        # Build the figure HTML block
        figure_html = f"""
</p>
<figure style="{FIGURE_STYLES['figure']}">
  <img style="{FIGURE_STYLES['img']}"
       src="{img_src}"
       alt="{plan.alt_text}"
       loading="lazy" />
  <figcaption style="{FIGURE_STYLES['figcaption']}">
    {plan.caption} <em>(Generated by AI)</em>
  </figcaption>
</figure>
<p style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 18px; line-height: 1.8; letter-spacing: 0.02em; word-spacing: 0.05em; margin-bottom: 1.5em; max-width: 75ch; color: #000000;">
"""

        # Get the insertion context and normalize it (decode HTML entities)
        context = plan.insertion_context.strip()
        normalized_context = html.unescape(context)

        # Split into words and build entity-aware patterns for each
        words = normalized_context.split()

        if not words:
            print(f"    ❌ Empty insertion context for Figure {plan.figure_number}")
            continue

        # Build pattern pieces for each word, allowing entities within words
        word_patterns = [word_to_pattern(w) for w in words]

        # Join with pattern allowing whitespace and HTML tags between words
        separator = r'[\s\n]*(?:<[^>]+>)*[\s\n]*'
        pattern = r'(?si)' + separator.join(word_patterns)

        match = re.search(pattern, modified_html)

        if match:
            # Insert the figure after the matched context
            end_pos = match.end()
            modified_html = modified_html[:end_pos] + figure_html + modified_html[end_pos:]
            print(f"    ✓ Inserted Figure {plan.figure_number}: {plan.caption[:40]}...")
        else:
            print(f"    ❌ Context not found for Figure {plan.figure_number}")
            print(f"       Looking for: '{context[:60]}...'")
            # Debug: show first few words we're trying to match
            if len(word_patterns) >= 3:
                print(f"       Pattern start: {word_patterns[0]}...{word_patterns[1]}...{word_patterns[2]}")

    return modified_html


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  LMS IMAGE ENHANCEMENT PIPELINE (Agent D)")
    print("=" * 60)

    # Generate timestamp early so it can be reused for images and output file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Validate environment
    valid, missing = validate_environment()
    if not valid:
        print(f"\n❌ Missing environment variables:")
        for var in missing:
            print(f"   • {var}")
        return 1

    # Determine input/output files
    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1])
    else:
        input_file = find_input_file()

    if not input_file or not input_file.exists():
        print(f"\n❌ Input file not found.")
        print(f"   Usage: python {sys.argv[0]} [input.html] [output.html]")
        print(f"   Default input: agent_c_output.html or anthropic_*_Step3_Final.html")
        return 1

    # Output file: use provided name or generate timestamped default
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path(f"agent_d_output_{timestamp}.html")

    print(f"\n📄 Input:  {input_file}")
    print(f"📄 Output: {output_file}")
    print(f"🕐 Timestamp: {timestamp}")

    # Load HTML content
    try:
        html_content = input_file.read_text(encoding="utf-8")
        print(f"   Loaded {len(html_content):,} characters")
    except Exception as e:
        print(f"\n❌ Failed to read input file: {e}")
        return 1

    # Get ImageKit folder from environment or use default
    imagekit_folder = os.environ.get("IMAGEKIT_FOLDER", DEFAULT_IMAGEKIT_FOLDER)

    # Initialize clients
    try:
        print(f"\n⚙️  Initializing API clients...")

        # Anthropic client
        claude_client = anthropic.Anthropic()
        print(f"   ✓ Anthropic client ready")

        # Gemini client
        gemini_client = genai.Client()
        print(f"   ✓ Gemini client ready")

        # ImageKit client - only needs private_key for server-side uploads
        imagekit_client = ImageKit(
            private_key=os.environ.get("IMAGEKIT_PRIVATE_KEY"),
        )
        print(f"   ✓ ImageKit client ready (folder: {imagekit_folder})")

    except Exception as e:
        print(f"\n❌ Failed to initialize clients: {e}")
        return 1

    # Step 1: Brainstorm with Claude
    plans = brainstorm_images(claude_client, html_content)
    if not plans:
        print("\n❌ No image plans generated. Exiting.")
        return 1

    # Step 2: Generate images and upload to ImageKit
    successful_plans = generate_images(
        gemini_client,
        imagekit_client,
        plans,
        imagekit_folder,
        timestamp,
    )
    if not successful_plans:
        print("\n❌ No images were successfully generated. Exiting.")
        return 1

    # Step 3: Inject images into HTML
    final_html = inject_images_into_html(html_content, successful_plans)

    # Save output
    try:
        output_file.write_text(final_html, encoding="utf-8")
        print(f"\n✅ Done! Enhanced HTML saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save output: {e}")
        return 1

    # Summary
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"  • Timestamp: {timestamp}")
    print(f"  • Images planned: {len(plans)}")
    print(f"  • Images generated: {len(successful_plans)}")
    hosted_count = sum(1 for p in successful_plans if p.hosted_url)
    print(f"  • Hosted on ImageKit: {hosted_count}")
    print(f"  • Local backups: {LOCAL_IMAGE_DIR}/")
    print(f"  • Output file: {output_file}")
    print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())