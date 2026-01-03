#!/usr/bin/env python3
"""
LMS Content Pipeline - OpenAI Edition
=====================================
A complete prompt-chaining pipeline for generating Canvas LMS-ready educational content
with optional AI-generated illustrations, powered by OpenAI Responses API + Images API.

Pipeline Stages:
  Agent A: Content Writer    → Drafts educational content
  Agent B: HTML Structurer   → Converts to semantic HTML
  Agent C: Publisher/Styler  → Applies accessibility styles and Knowledge Checks
  Agent D: Illustrator       → Generates, uploads (ImageKit), and embeds AI images

Supported Content Types:
  - Textbook: Full chapters with Knowledge Checks and illustrations
  - Discussion: Interactive discussion prompts with Font Awesome icons
  - Assignment: Step-based assignments with rubrics

Output Structure:
  lms_output_YYYYMMDD-HHMMSS/
  ├── content/
  │   ├── Step1_Content.txt
  │   ├── Step2_Structured.html
  │   ├── Step3_Styled.html
  │   └── Step4_Final.html (with images, if enabled)
  ├── images/
  │   ├── fig1-YYYYMMDD-HHMMSS.png
  │   ├── fig2-YYYYMMDD-HHMMSS.png
  │   └── fig3-YYYYMMDD-HHMMSS.png
  └── pipeline.log

Usage:
  python openai_lms_pipeline.py                    # Interactive mode
  python openai_lms_pipeline.py --no-images        # Skip image generation
  python openai_lms_pipeline.py --topic "AI"       # Provide topic directly
  python openai_lms_pipeline.py --input-file path  # Skip Agent A with existing text

Requires Environment Variables:
  - OPENAI_API_KEY (required)
  - IMAGEKIT_PRIVATE_KEY, IMAGEKIT_PUBLIC_KEY, IMAGEKIT_URL_ENDPOINT (required for images)

Dependencies:
  pip install openai imagekitio
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final, List, Optional, Dict, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai library not found. Please install it via 'pip install openai'")

try:
    from imagekitio import ImageKit
    IMAGEKIT_AVAILABLE = True
except ImportError:
    IMAGEKIT_AVAILABLE = False
    print("Warning: imagekitio library not found. Images will be local only if not installed.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ContentType(Enum):
    """Supported LMS content types."""
    TEXTBOOK = "textbook"
    DISCUSSION = "discussion"
    ASSIGNMENT = "assignment"


# Model Configuration
TEXT_MODEL_AGENT_A: Final[str] = "gpt-5.2"
TEXT_MODEL_AGENT_BCD: Final[str] = "gpt-5-mini"
IMAGE_MODEL: Final[str] = "gpt-image-1.5"

# Limits
MAX_IMAGES: Final[int] = 3

# Structured Outputs schema for Agent D image planning
IMAGE_PLAN_SCHEMA: Final[Dict[str, Any]] = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "minItems": 0,
            "maxItems": MAX_IMAGES,
            "items": {
                "type": "object",
                "properties": {
                    "insertion_context": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "alt_text": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["insertion_context", "image_prompt", "alt_text", "caption"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["images"],
    "additionalProperties": False,
}

# Paths
PROMPTS_DIR = Path("prompts")
DEFAULT_IMAGEKIT_FOLDER: Final[str] = "/lms-content/"


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
    local_path: Optional[Path] = None
    hosted_url: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""
    content_type: ContentType
    topic: str
    timestamp: str
    output_dir: Path
    content_dir: Path
    images_dir: Path
    enable_images: bool
    imagekit_folder: str = DEFAULT_IMAGEKIT_FOLDER


@dataclass
class PipelineState:
    """Holds state during pipeline execution."""
    config: PipelineConfig
    openai_client: Optional[object] = None
    imagekit_client: Optional[object] = None
    step_outputs: Dict[str, str] = field(default_factory=dict)
    image_plans: List[ImagePlan] = field(default_factory=list)
    log_messages: List[str] = field(default_factory=list)

    def log(self, message: str, print_it: bool = True) -> None:
        """Log a message and optionally print it."""
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.log_messages.append(timestamped)
        if print_it:
            print(message)

    def save_log(self) -> None:
        """Save the log to a file."""
        log_path = self.config.output_dir / "pipeline.log"
        log_path.write_text("\n".join(self.log_messages), encoding="utf-8")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_environment(need_images: bool) -> tuple[bool, list[str]]:
    """Check required environment variables."""
    missing = []

    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if need_images:
        if not os.environ.get("IMAGEKIT_PRIVATE_KEY"):
            missing.append("IMAGEKIT_PRIVATE_KEY")
        if not os.environ.get("IMAGEKIT_PUBLIC_KEY"):
            missing.append("IMAGEKIT_PUBLIC_KEY")
        if not os.environ.get("IMAGEKIT_URL_ENDPOINT"):
            missing.append("IMAGEKIT_URL_ENDPOINT")

    return len(missing) == 0, missing


def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def clean_json_response(text: str) -> str:
    """Remove markdown code fences and fix common JSON issues."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try to extract JSON if wrapped in other text
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return text.strip()


def word_to_pattern(word: str) -> str:
    """Convert a word to a regex pattern allowing HTML entities."""
    result = []
    for char in word:
        if char in "—–":
            result.append(r"(?:&mdash;|&ndash;|&#8212;|&#8211;|—|–|-)")
        elif char == '"':
            result.append(r"(?:&rdquo;|&ldquo;|&#8221;|&#8220;|\")")
        elif char == "'":
            result.append(r"(?:&rsquo;|&lsquo;|&#8217;|&#8216;|')")
        elif char == '…':
            result.append(r'(?:&hellip;|&#8230;|…|…)')
        elif char == '&':
            result.append(r'(?:&amp;|&)')
        elif char == '<':
            result.append(r'(?:&lt;|<)')
        elif char == '>':
            result.append(r'(?:&gt;|>)')
        elif char == ' ':
            result.append(r'\s+')
        else:
            result.append(re.escape(char))
    return ''.join(result)


def _extract_text(resp: Any) -> str:
    """Return plain text from a Responses API result across common shapes."""
    if resp is None:
        return ""

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(resp, dict):
        output_text = resp.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")
    if not output:
        return ""

    parts: List[str] = []
    for item in output:
        content_blocks = getattr(item, "content", None)
        if content_blocks is None and isinstance(item, dict):
            content_blocks = item.get("content")
        if content_blocks:
            for block in content_blocks:
                text = None
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)

        item_text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if isinstance(item_text, str) and item_text:
            parts.append(item_text)

    return "".join(parts).strip()


def _extract_json(resp: Any) -> Optional[Dict[str, Any]]:
    """Return JSON content from a Responses API result when available."""
    if resp is None:
        return None

    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")
    if output:
        for item in output:
            content_blocks = getattr(item, "content", None)
            if content_blocks is None and isinstance(item, dict):
                content_blocks = item.get("content")
            if not content_blocks:
                continue
            for block in content_blocks:
                if isinstance(block, dict):
                    json_payload = block.get("json") or block.get("parsed")
                else:
                    json_payload = getattr(block, "json", None) or getattr(block, "parsed", None)
                if isinstance(json_payload, dict):
                    return json_payload

    output_text = getattr(resp, "output_text", None)
    if output_text is None and isinstance(resp, dict):
        output_text = resp.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        try:
            data = json.loads(clean_json_response(output_text))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data

    text = _extract_text(resp)
    if text:
        try:
            data = json.loads(clean_json_response(text))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data

    return None


def generate_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str] = None,
    text_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate text via the OpenAI Responses API."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if text_format:
        kwargs["text"] = {"format": text_format}

    response = client.responses.create(**kwargs)
    return _extract_text(response)


def generate_structured_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    schema_name: str,
    schema: Dict[str, Any],
    schema_description: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Generate structured JSON via the OpenAI Responses API."""
    text_format: Dict[str, Any] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": schema,
        "strict": True,
    }
    if schema_description:
        text_format["description"] = schema_description

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
        text={"format": text_format},
    )
    return _extract_json(response), _extract_text(response)


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_a_write_content(state: PipelineState) -> bool:
    """Agent A: Write educational content."""
    state.log("\n📝 [1/4] Agent A: Writing content...")

    prompt_file = f"agent_a_{state.config.content_type.value}.txt"
    try:
        system_prompt = load_prompt(prompt_file)
    except FileNotFoundError as e:
        state.log(f"    ❌ Error: {e}")
        return False

    try:
        response_text = generate_text(
            client=state.openai_client,
            model=TEXT_MODEL_AGENT_A,
            system_prompt=system_prompt,
            user_prompt=state.config.topic,
            max_output_tokens=12000,
            reasoning_effort="high",
        )

        if not response_text.strip():
            state.log("    ❌ Empty response from Agent A")
            return False

        # Save output
        output_path = state.config.content_dir / "Step1_Content.txt"
        output_path.write_text(response_text, encoding="utf-8")
        state.step_outputs["agent_a"] = response_text

        word_count = len(response_text.split())
        state.log(f"    ✓ Generated {word_count:,} words → {output_path.name}")
        return True

    except Exception as e:
        state.log(f"    ❌ Agent A Error: {e}")
        return False


def stage_b_structure_html(state: PipelineState) -> bool:
    """Agent B: Convert to semantic HTML."""
    state.log("\n🏗️  [2/4] Agent B: Structuring HTML...")

    prompt_file = f"agent_b_{state.config.content_type.value}.txt"
    try:
        system_prompt = load_prompt(prompt_file)
    except FileNotFoundError as e:
        state.log(f"    ❌ Error: {e}")
        return False

    try:
        response_text = generate_text(
            client=state.openai_client,
            model=TEXT_MODEL_AGENT_BCD,
            system_prompt=system_prompt,
            user_prompt=state.step_outputs["agent_a"],
            max_output_tokens=14000,
        )

        if not response_text.strip():
            state.log("    ❌ Empty response from Agent B")
            return False

        output_path = state.config.content_dir / "Step2_Structured.html"
        output_path.write_text(response_text, encoding="utf-8")
        state.step_outputs["agent_b"] = response_text

        state.log(f"    ✓ Structured HTML → {output_path.name}")
        return True

    except Exception as e:
        state.log(f"    ❌ Agent B Error: {e}")
        return False


def stage_c_style_publish(state: PipelineState) -> bool:
    """Agent C: Apply styles and accessibility features."""
    state.log("\n🎨 [3/4] Agent C: Styling and publishing...")

    if state.config.content_type == ContentType.TEXTBOOK:
        prompt_file = "agent_c_textbook.txt"
    else:
        prompt_file = "agent_c_discussion_assignment.txt"

    try:
        system_prompt = load_prompt(prompt_file)
    except FileNotFoundError as e:
        state.log(f"    ❌ Error: {e}")
        return False

    try:
        response_text = generate_text(
            client=state.openai_client,
            model=TEXT_MODEL_AGENT_BCD,
            system_prompt=system_prompt,
            user_prompt=state.step_outputs["agent_b"],
            max_output_tokens=16384,
        )

        if not response_text.strip():
            state.log("    ❌ Empty response from Agent C")
            return False

        output_path = state.config.content_dir / "Step3_Styled.html"
        output_path.write_text(response_text, encoding="utf-8")
        state.step_outputs["agent_c"] = response_text

        char_count = len(response_text)
        state.log(f"    ✓ Styled HTML ({char_count:,} chars) → {output_path.name}")
        return True

    except Exception as e:
        state.log(f"    ❌ Agent C Error: {e}")
        return False


def stage_d_brainstorm_images(state: PipelineState) -> bool:
    """Agent D Part 1: Brainstorm image insertion points."""
    state.log("\n🧠 [4a/4] Agent D: Brainstorming images...")

    try:
        system_prompt = load_prompt("agent_d_brainstorm.txt")
    except FileNotFoundError as e:
        state.log(f"    ❌ Error: {e}")
        return False

    try:
        data, raw_text = generate_structured_json(
            client=state.openai_client,
            model=TEXT_MODEL_AGENT_BCD,
            system_prompt=system_prompt,
            user_prompt=(
                f"Analyze this HTML content and identify up to {MAX_IMAGES} "
                f"optimal locations for educational images:\n\n{state.step_outputs['agent_c']}"
            ),
            max_output_tokens=4096,
            schema_name="image_plan",
            schema=IMAGE_PLAN_SCHEMA,
            schema_description="Image planning data for LMS content.",
        )

        if not data:
            state.log("    ❌ Empty or invalid structured output from Agent D")
            if raw_text:
                state.log(f"    Debug - Raw Response: {raw_text[:200]}...")
            return False

        images = data.get("images", [])
        if not isinstance(images, list):
            state.log("    ❌ Structured output missing 'images' array")
            if raw_text:
                state.log(f"    Debug - Raw Response: {raw_text[:200]}...")
            return False

        for i, item in enumerate(images[:MAX_IMAGES], start=1):
            plan = ImagePlan(
                insertion_context=item["insertion_context"],
                image_prompt=item["image_prompt"],
                alt_text=item["alt_text"][:125],
                caption=item["caption"],
                figure_number=i,
            )
            state.image_plans.append(plan)

        state.log(f"    ✓ Identified {len(state.image_plans)} image insertion points")
        return True

    except Exception as e:
        state.log(f"    ❌ Agent D Error: {e}")
        return False


def stage_d_generate_images(state: PipelineState) -> bool:
    """Agent D Part 2: Generate images, save locally, and upload to ImageKit."""
    state.log(f"\n🖼️  [4b/4] Generating {len(state.image_plans)} images...")

    successful_plans = []

    for plan in state.image_plans:
        state.log(f"    Figure {plan.figure_number}: Generating...")

        try:
            full_prompt = (
                f"Generate a high-quality educational illustration: {plan.image_prompt}. "
                f"Style: Professional, clean, minimal text overlay, suitable for a textbook."
            )

            response = state.openai_client.images.generate(
                model=IMAGE_MODEL,
                prompt=full_prompt,
            )

            if not response.data or not response.data[0].b64_json:
                state.log("        ❌ No image data received")
                continue

            image_data = base64.b64decode(response.data[0].b64_json)

            # Save locally
            local_filename = f"fig{plan.figure_number}-{state.config.timestamp}.png"
            local_path = state.config.images_dir / local_filename
            with open(local_path, "wb") as f:
                f.write(image_data)

            plan.local_path = local_path
            state.log(f"        ✓ Saved local: {local_filename}")

            # Upload to ImageKit
            if state.imagekit_client:
                state.log("        ↑ Uploading to ImageKit...")
                result = state.imagekit_client.files.upload(
                    file=image_data,
                    file_name=local_filename,
                    folder=state.config.imagekit_folder,
                    use_unique_file_name=True,
                    is_private_file=False,
                    tags=["lms", "educational", "ai-generated"],
                )

                if result and hasattr(result, "url") and result.url:
                    plan.hosted_url = result.url
                    state.log(f"        ✓ Hosted: {result.url}")
                else:
                    state.log("        ⚠️ Upload failed or no URL returned, using local path")

            successful_plans.append(plan)

        except Exception as e:
            state.log(f"        ❌ Failed: {e}")
            continue

    state.image_plans = successful_plans
    return len(successful_plans) > 0


def stage_d_inject_images(state: PipelineState) -> bool:
    """Agent D Part 3: Inject images into HTML."""
    state.log("\n📎 [4c/4] Injecting images into HTML...")

    html_content = state.step_outputs["agent_c"]

    for plan in state.image_plans:
        # Determine Source URL
        if plan.hosted_url:
            img_src = plan.hosted_url
        elif plan.local_path:
            # Fallback to relative path if upload failed
            img_src = f"../images/{plan.local_path.name}"
        else:
            continue

        figure_html = f"""
<figure style="margin: 2em 0; max-width: 75ch;">
  <img style="width: 100%; height: auto; border: 1px solid #E0E0E0; border-radius: 4px;"
       src="{img_src}"
       alt="{plan.alt_text}"
       loading="lazy" />
  <figcaption style="padding: 0.75em 1em; font-style: italic; background-color: #F5F5F5; text-align: center; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 16px; color: #333333; border-bottom-left-radius: 4px; border-bottom-right-radius: 4px;">
    {plan.caption} <em>(Generated by AI)</em>
  </figcaption>
</figure>
"""

        context = plan.insertion_context.strip()
        normalized_context = html.unescape(context)
        # Split into tokens (words and punctuation) to handle tags inside/between punctuation
        # e.g. "word</strong>." -> "word", "."
        tokens = re.findall(r"\w+|[^\w\s]", normalized_context)

        if not tokens:
            continue

        word_patterns = [word_to_pattern(t) for t in tokens]
        separator = r"[\s\n]*(?:<[^>]+>)*[\s\n]*"
        pattern = r"(?si)" + separator.join(word_patterns)

        match = re.search(pattern, html_content)

        if not match and len(tokens) > 4:
            # Fallback: Try matching just the last 50% of tokens
            half_tokens = tokens[len(tokens)//2:]
            state.log(f"    ⚠️ Exact match failed. Retrying with partial context ({len(half_tokens)} tokens)...")

            word_patterns_fallback = [word_to_pattern(t) for t in half_tokens]
            pattern_fallback = r"(?si)" + separator.join(word_patterns_fallback)
            match = re.search(pattern_fallback, html_content)

        if match:
            end_pos = match.end()
            prefix = html_content[:end_pos]
            last_p_open = None
            last_close_idx = prefix.lower().rfind("</p>")
            for p_match in re.finditer(r"<p\b[^>]*>", prefix, flags=re.IGNORECASE):
                last_p_open = p_match
            if last_p_open and last_p_open.start() > last_close_idx:
                insertion = f"</p>{figure_html}{last_p_open.group(0)}"
            else:
                insertion = figure_html
            html_content = prefix + insertion + html_content[end_pos:]
            state.log(f"    ✓ Inserted Figure {plan.figure_number}")
        else:
            state.log(f"    ❌ Insertion context not found for Figure {plan.figure_number}")
            state.log(f"       Context looked for: '{context}'")

    state.step_outputs["agent_d"] = html_content
    return True


def save_final_output(state: PipelineState) -> Path:
    """Save the final HTML output."""
    if "agent_d" in state.step_outputs:
        final_html = state.step_outputs["agent_d"]
        step_name = "Step4_Final.html"
    else:
        final_html = state.step_outputs["agent_c"]
        step_name = "Step3_Final.html"

    output_path = state.config.content_dir / step_name
    output_path.write_text(final_html, encoding="utf-8")

    # Also save a convenience copy to CWD
    cwd_copy = Path.cwd() / f"lms_output_{state.config.timestamp}.html"
    cwd_copy.write_text(final_html, encoding="utf-8")

    return output_path


# =============================================================================
# USER INTERFACE
# =============================================================================

def display_banner():
    print("\n" + "=" * 65)
    print("  ✨ LMS CONTENT PIPELINE - OpenAI Edition ✨")
    print("=" * 65)


def display_menu() -> ContentType:
    print("\nWhat type of LMS content would you like to generate?\n")
    print("  [1] 📚 Textbook Chapter")
    print("      Full chapter with Knowledge Checks + AI illustrations")
    print()
    print("  [2] 💬 Discussion Prompt")
    print("      Write/Respond/Evaluate sections with icons")
    print()
    print("  [3] 📋 Assignment Instructions")
    print("      Overview/Instructions/Deliverables/Rubric with icons")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return ContentType.TEXTBOOK
        elif choice == "2":
            return ContentType.DISCUSSION
        elif choice == "3":
            return ContentType.ASSIGNMENT
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_topic_prompt(content_type: ContentType) -> str:
    prompts = {
        ContentType.TEXTBOOK: "Enter the topic for your textbook chapter",
        ContentType.DISCUSSION: "Enter the topic/scenario for your discussion",
        ContentType.ASSIGNMENT: "Enter the topic/task for your assignment",
    }
    return prompts[content_type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LMS Content Pipeline (OpenAI Edition)")
    parser.add_argument("--topic", "-t", help="Topic for content generation")
    parser.add_argument("--type", "-c", choices=["textbook", "discussion", "assignment"], help="Content type")
    parser.add_argument("--input-file", "-i", help="Path to Step1_Content.txt to skip Agent A")
    parser.add_argument("--no-images", action="store_true", help="Skip image generation")
    parser.add_argument("--force-images", action="store_true", help="Force image generation for non-textbooks")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    if not OPENAI_AVAILABLE:
        return 1

    args = parse_args()
    display_banner()

    # Content Type
    if args.type:
        content_type = ContentType(args.type)
        print(f"\n📋 Content type: {content_type.value.capitalize()}")
    else:
        content_type = display_menu()

    # Image Logic
    enable_images = (
        not args.no_images
        and (content_type == ContentType.TEXTBOOK or args.force_images)
    )

    if args.no_images:
        print("\n📷 Image generation: Disabled (--no-images)")
    elif enable_images:
        if IMAGEKIT_AVAILABLE:
            print("\n📷 Image generation: Enabled (ImageKit Hosting)")
        else:
            print("\n📷 Image generation: Enabled (Local Only - ImageKit missing)")
    else:
        print("\n📷 Image generation: Skipped (default for this type)")

    # Validate Environment
    valid, missing = validate_environment(enable_images)
    if not valid:
        print("\n❌ Missing environment variables:")
        for var in missing:
            print(f"   • {var}")
        return 1

    # Topic
    if args.topic:
        topic = args.topic
        print(f"\n📝 Topic: {topic}")
    elif args.input_file:
        topic = "Imported content"
        print("\n📝 Topic: (using imported content)")
    else:
        prompt = get_topic_prompt(content_type)
        topic = input(f"\n{prompt}: ").strip()

    if not topic:
        print("❌ Topic cannot be empty.")
        return 1

    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path.cwd() / f"lms_output_{timestamp}"
    content_dir = output_dir / "content"
    images_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(exist_ok=True)
    if enable_images:
        images_dir.mkdir(exist_ok=True)

    # Initialize Clients
    print("\n⚙️  Initializing clients...")
    openai_client = OpenAI()

    imagekit_client = None
    if enable_images and IMAGEKIT_AVAILABLE:
        try:
            # ImageKit 5.x+ initialization
            imagekit_client = ImageKit(
                private_key=os.environ.get("IMAGEKIT_PRIVATE_KEY"),
                # base_url=os.environ.get("IMAGEKIT_URL_ENDPOINT") # Removed to use default API endpoint
            )
            print("   ✓ ImageKit client ready")
        except Exception as e:
            print(f"   ⚠️ ImageKit init failed: {e}")

    config = PipelineConfig(
        content_type=content_type,
        topic=topic,
        timestamp=timestamp,
        output_dir=output_dir,
        content_dir=content_dir,
        images_dir=images_dir,
        enable_images=enable_images,
        imagekit_folder=os.environ.get("IMAGEKIT_FOLDER", DEFAULT_IMAGEKIT_FOLDER),
    )

    state = PipelineState(
        config=config,
        openai_client=openai_client,
        imagekit_client=imagekit_client,
    )

    # Execution
    success = True
    final_path = None
    input_path = Path(args.input_file).expanduser() if args.input_file else None

    if success and input_path:
        if not input_path.exists():
            print(f"\n❌ Input file not found: {input_path}")
            return 1
        try:
            imported_text = input_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"\n❌ Failed to read input file: {e}")
            return 1
        if not imported_text.strip():
            print("\n❌ Input file is empty.")
            return 1
        state.step_outputs["agent_a"] = imported_text
        (state.config.content_dir / "Step1_Content.txt").write_text(imported_text, encoding="utf-8")
        state.log(f"    ✓ Loaded Step1 content from {input_path}")
    elif success:
        success = stage_a_write_content(state)
    if success:
        success = stage_b_structure_html(state)
    if success:
        success = stage_c_style_publish(state)

    if success and enable_images:
        if stage_d_brainstorm_images(state):
            if state.image_plans:
                stage_d_generate_images(state)
                stage_d_inject_images(state)

    if success:
        final_path = save_final_output(state)
        state.log(f"\n✅ Final output: {final_path}")

    state.save_log()

    if success:
        print(f"\n✅ PIPELINE COMPLETE! Check directory: {output_dir}")
        if final_path:
            print(f"📄 Open file: {final_path}")
    else:
        print("\n❌ PIPELINE FAILED. Check log.")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
