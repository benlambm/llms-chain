#!/usr/bin/env python3
"""
LMS Content Pipeline - Unified Edition
=======================================
A complete prompt-chaining pipeline for generating Canvas LMS-ready educational content
with optional AI-generated illustrations.

Pipeline Stages:
  Agent A: Content Writer    → Drafts educational content
  Agent B: HTML Structurer   → Converts to semantic HTML
  Agent C: Publisher/Styler  → Applies accessibility styles and Knowledge Checks
  Agent D: Illustrator       → Generates and embeds AI images (optional)

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
  python lms_pipeline.py                    # Interactive mode
  python lms_pipeline.py --no-images        # Skip image generation
  python lms_pipeline.py --topic "AI"       # Provide topic directly

Requires Environment Variables:
  - ANTHROPIC_API_KEY (required)
  - GEMINI_API_KEY or GOOGLE_API_KEY (required for images)
  - IMAGEKIT_PRIVATE_KEY (required for images)

Optional Environment Variables:
  - IMAGEKIT_FOLDER (default: /lms-content/)

Dependencies:
  pip install anthropic google-genai imagekitio

Author: Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final

import anthropic

# Optional imports for image generation
try:
    from google import genai
    from google.genai import types
    from imagekitio import ImageKit
    IMAGES_AVAILABLE = True
except ImportError:
    IMAGES_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class ContentType(Enum):
    """Supported LMS content types."""
    TEXTBOOK = "textbook"
    DISCUSSION = "discussion"
    ASSIGNMENT = "assignment"


# Model Configuration
CLAUDE_WRITER_MODEL: Final[str] = "claude-haiku-4-5"
CLAUDE_BRAINSTORM_MODEL: Final[str] = "claude-haiku-4-5"
GEMINI_IMAGE_MODEL: Final[str] = "gemini-3-pro-image-preview"

# Limits
MAX_IMAGES: Final[int] = 3

# Default ImageKit folder
DEFAULT_IMAGEKIT_FOLDER: Final[str] = "/lms-content/"

# Shared CSS Constants
FONT_STACK: Final[str] = "system-ui, -apple-system, 'Segoe UI', sans-serif"

SHARED_STYLES: Final[dict[str, str]] = {
    "h2": f"font-family: {FONT_STACK}; font-size: 2.25em; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; border-bottom: 4px solid #000000; padding-bottom: 0.5em; margin-top: 0; margin-bottom: 1em; color: #000000;",
    "h3": f"font-family: {FONT_STACK}; font-size: 1.5em; font-weight: 700; letter-spacing: 0.03em; border-left: 6px solid #000000; padding-left: 0.75em; margin-top: 2em; margin-bottom: 1em; color: #000000;",
    "h4": f"font-family: {FONT_STACK}; font-size: 1.25em; font-weight: 700; font-style: italic; margin-top: 2em; margin-bottom: 0.75em; color: #000000;",
    "p": f"font-family: {FONT_STACK}; font-size: 18px; line-height: 1.8; letter-spacing: 0.02em; word-spacing: 0.05em; margin-bottom: 1.5em; max-width: 75ch; color: #000000;",
    "ul": f"list-style-type: square; margin-left: 1.5em; margin-bottom: 1.5em; line-height: 1.8; font-family: {FONT_STACK}; font-size: 18px; color: #000000; max-width: 75ch;",
    "li": "margin-bottom: 0.75em; padding-left: 0.5em;",
    "dl": f"margin-bottom: 2em; border-left: 4px solid #000000; padding-left: 1.5em; font-family: {FONT_STACK}; max-width: 75ch;",
    "dt": "font-size: 1.1em; font-weight: 800; letter-spacing: 0.02em; margin-top: 1.5em; margin-bottom: 0.25em; color: #000000;",
    "dd": "font-size: 18px; line-height: 1.8; margin-left: 0; margin-bottom: 1em; padding-left: 1em; border-left: 2px solid #E0E0E0; color: #000000;",
    "em": "font-style: italic; letter-spacing: 0.01em;",
    "strong": "font-weight: 800;",
    "hr": "border: none; border-top: 3px solid #000000; margin: 3em auto; width: 70%;",
    "figure": "margin: 2em 0; max-width: 75ch;",
    "img": "width: 100%; height: auto; border: 1px solid #E0E0E0; border-radius: 4px;",
    "figcaption": (
        "padding: 0.75em 1em; font-style: italic; background-color: #F5F5F5; "
        "text-align: center; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; "
        "font-size: 16px; color: #333333; border-bottom-left-radius: 4px; "
        "border-bottom-right-radius: 4px;"
    ),
}

# Font Awesome icons for Discussion/Assignment pages
FA_ICONS: Final[dict[str, str]] = {
    "write": '<i class="fas fa-pencil-alt" style="padding-right: 10px;" aria-hidden="true"></i>',
    "respond": '<i class="fas fa-comments" style="padding-right: 10px;" aria-hidden="true"></i>',
    "evaluate": '<i class="fas fa-check" style="padding-right: 10px;" aria-hidden="true"></i>',
}


# =============================================================================
# AGENT PROMPTS
# =============================================================================

# Agent A: Content Writer
AGENT_A_BASE: Final[str] = """
Writing Standards:
Write like a knowledgeable mentor—professional but approachable, never dry or encyclopedic.
Use current industry terminology from relevant standards or open-source bodies.
Define acronyms on first use. Scaffold from simple to complex.
Always explain historical and theory context ("why") alongside "what" and "how."

Format & Length:
Output plain text prose with minimal formatting. Avoid bullet points, numbered lists,
and excessive headers within the body—write in flowing paragraphs.
Use section headings only to divide major parts of the content.
"""

AGENT_A_PROMPTS: Final[dict[ContentType, str]] = {
    ContentType.TEXTBOOK: f"""
You write textbook chapters for incoming college students. Your audience is intelligent adults
who may have limited technical background—assume no prerequisites beyond basic computer
literacy unless the topic explicitly requires it.

Chapter Requirements:
Every chapter includes:
- Learning Objectives: 2-3 measurable goals using action verbs (explain, identify, compare, evaluate, apply, create)
- Core Content: 5-7 logically organized sections that define key terms on first use, give context,
  explain theory, use concrete examples and analogies, and include at least one detailed walkthrough
  or case study (but avoid case studies of e-mail and spam).
- Summary: Brief recap of key takeaways
- Glossary of Key Terms

{AGENT_A_BASE}

Length: approximately 2,000-2,500 words.
""",

    ContentType.DISCUSSION: f"""
You write discussion board prompts for college-level courses. Your audience is intelligent adults
who may have limited technical background.

Discussion Prompt Requirements:
Every discussion prompt includes these sections (use these exact headings):
- Write/Prompt: The main discussion question or scenario students must respond to (1-2 paragraphs)
- Respond/Reply: Clear instructions for how students should engage with peers' posts
- Evaluation/Rubric: Grading criteria explaining how responses will be assessed

If the topic introduces technical terminology, include a brief:
- Key Terms: Essential vocabulary students need (only if technical terms are introduced)

{AGENT_A_BASE}

Length: approximately 150-250 words total.
Keep prompts focused and actionable. Encourage critical thinking and peer engagement.
""",

    ContentType.ASSIGNMENT: f"""
You write assignment instructions for college-level courses. Your audience is intelligent adults
who may have limited technical background.

Assignment Requirements:
Every assignment includes these sections (use these exact headings):
- Overview: Brief description of the assignment purpose and learning goals
- Instructions: Step-by-step directions for completing the assignment
- Deliverables: Clear list of what students must submit
- Evaluation/Rubric: Detailed grading criteria with point allocations

If the topic introduces technical terminology, include a brief:
- Key Terms: Essential vocabulary students need (only if technical terms are introduced)

{AGENT_A_BASE}

Length: approximately 250-300 words total.
Be specific about expectations, deadlines format, and submission requirements.
""",
}


# Agent B: HTML Structurer
AGENT_B_BASE: Final[str] = """
You are an instructional HTML re-formatter. Your task is to take the provided plain text
and convert it into a clean HTML fragment (no <head> or <body> tags).

Core Rules:
1. Do NOT add any CSS styles, classes, or IDs. Output raw, clean HTML tags only.
2. Do NOT change the content or wording.
3. Return only the HTML code for immediate usage in an LMS Page.

Structure Guidelines:
- Use <h2> for the main title
- Use <h3> for major section headings
- Use <h4> for subsections
- Use <p> for body paragraphs
- Use <ul>/<li> for lists
- Use <dl>/<dt>/<dd> specifically for Glossary or Key Terms
- Use <strong> for key terms when they are first defined in the text
- Use <em> for emphasis where appropriate
- Use <hr> to separate major sections
"""

AGENT_B_PROMPTS: Final[dict[ContentType, str]] = {
    ContentType.TEXTBOOK: f"""
{AGENT_B_BASE}

Textbook-Specific Structure:
- Learning Objectives should be an <h3> followed by a <ul> list
- Core content sections use <h3> for main topics, <h4> for subtopics
- Summary is an <h3> followed by paragraph(s)
- Glossary uses <h3> "Glossary of Key Terms" followed by <dl>/<dt>/<dd>
""",

    ContentType.DISCUSSION: f"""
{AGENT_B_BASE}

Discussion-Specific Structure:
- The main title uses <h2>
- "Write/Prompt" section uses <h3>Write/Prompt</h3>
- "Respond/Reply" section uses <h3>Respond/Reply</h3>
- "Evaluation/Rubric" section uses <h3>Evaluation/Rubric</h3>
- If "Key Terms" exists, use <h3>Key Terms</h3> followed by <dl>/<dt>/<dd>
- Use <hr> between major sections
""",

    ContentType.ASSIGNMENT: f"""
{AGENT_B_BASE}

Assignment-Specific Structure:
- The main title uses <h2>
- "Overview" section uses <h3>Overview</h3>
- "Instructions" section uses <h3>Instructions</h3>
- "Deliverables" section uses <h3>Deliverables</h3>
- "Evaluation/Rubric" section uses <h3>Evaluation/Rubric</h3>
- If "Key Terms" exists, use <h3>Key Terms</h3> followed by <dl>/<dt>/<dd>
- Use <hr> between major sections
""",
}


# Agent C: Publisher/Styler
def _build_agent_c_styles() -> str:
    """Build the styling instructions for Agent C."""
    return f"""
Use only inline CSS (no classes, no external styles).
Use this font stack everywhere: {FONT_STACK}

Apply styles as follows (copy these style attributes verbatim):

h2: style="{SHARED_STYLES['h2']}"
h3: style="{SHARED_STYLES['h3']}"
h4: style="{SHARED_STYLES['h4']}"
p: style="{SHARED_STYLES['p']}"
ul: style="{SHARED_STYLES['ul']}"
li: style="{SHARED_STYLES['li']}"
dl: style="{SHARED_STYLES['dl']}"
dt: style="{SHARED_STYLES['dt']}"
dd: style="{SHARED_STYLES['dd']}"
em: style="{SHARED_STYLES['em']}"
strong: style="{SHARED_STYLES['strong']}"
hr: style="{SHARED_STYLES['hr']}"
"""


AGENT_C_TEXTBOOK: Final[str] = f"""
You are an HTML rewriter to make instructional content accessible. You will be given a single
HTML fragment (no head/body tags). Your task is to produce a new HTML fragment that:

1. Preserves content and structure
- Keep all educational text, ordering, and headings the same.
- Do not delete, shorten, or rephrase existing text.
- Do not use semantic container tags (section, article, main, aside, nav) as Canvas LMS
  does not render them properly.
- You may only: a) add inline CSS styling, and b) insert "Knowledge Check" blocks

{_build_agent_c_styles()}

2. Add interactive "Knowledge Check" blocks using details/summary

At several natural points (after each major section, before <hr> tags) insert a knowledge check.
Use only HTML and CSS, no JavaScript.

Knowledge Check Template:
<details style="background-color: #F5F5F5; border: 3px solid #000000; padding: 1.5em; margin: 2em 0; max-width: 75ch;">
  <summary style="font-family: {FONT_STACK}; font-size: 1.1em; font-weight: 700; cursor: pointer; letter-spacing: 0.02em; color: #000000;">
    ✦ KNOWLEDGE CHECK: [Write the complete question here]</summary>
  <p style="font-family: {FONT_STACK}; font-size: 18px; line-height: 1.8; margin-top: 1em; border-top: 2px solid #E0E0E0; padding-top: 1em; margin-bottom: 0; color: #000000;">
    <strong style="font-weight: 800;">Answer:</strong> [Concise explanatory answer based on the section content]</p>
</details>

3. Output format
- Output only the final transformed HTML fragment.
- Do not add comments, explanations, or any extra text outside the HTML.
- Ensure compliance with WCAG 2.1 AA accessibility standards.
"""


AGENT_C_DISCUSSION_ASSIGNMENT: Final[str] = f"""
You are an HTML rewriter to make instructional content accessible. You will be given a single
HTML fragment (no head/body tags). Your task is to produce a new HTML fragment that:

1. Preserves content and structure
- Keep all educational text, ordering, and headings the same.
- Do not delete, shorten, or rephrase existing text.
- Do not use semantic container tags (section, article, main, aside, nav) as Canvas LMS
  does not render them properly.
- You may only add inline CSS styling.
- Do NOT add Knowledge Check blocks for Discussion/Assignment pages.

{_build_agent_c_styles()}

2. Add Font Awesome icons to specific H3 headings

Prepend Font Awesome icons inside specific H3 tags:
- For "Write/Prompt" or "Overview" or "Instructions": {FA_ICONS['write']}
- For "Respond/Reply" or "Deliverables": {FA_ICONS['respond']}
- For "Evaluation/Rubric": {FA_ICONS['evaluate']}

3. Output format
- Output only the final transformed HTML fragment.
- Ensure compliance with WCAG 2.1 AA accessibility standards.
"""

AGENT_C_PROMPTS: Final[dict[ContentType, str]] = {
    ContentType.TEXTBOOK: AGENT_C_TEXTBOOK,
    ContentType.DISCUSSION: AGENT_C_DISCUSSION_ASSIGNMENT,
    ContentType.ASSIGNMENT: AGENT_C_DISCUSSION_ASSIGNMENT,
}


# Agent D: Image Brainstormer
AGENT_D_PROMPT: Final[str] = """
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
  copy of text found in the source.
  
- "image_prompt": A detailed, specific prompt for image generation.

- "alt_text": Descriptive alt text for accessibility (max 125 characters).

- "caption": A figure caption starting with "Figure [N]:" that explains what the image illustrates.

Output Format:
Return ONLY a valid JSON object:
{
  "images": [
    {
      "insertion_context": "exact verbatim text from source",
      "image_prompt": "detailed generation prompt",
      "alt_text": "accessibility description",
      "caption": "Figure 1: Explanation"
    }
  ]
}

Important:
- Return valid JSON only, no markdown code fences
- Maximum 3 images
- Each insertion_context must be unique and findable in the source
- Ensure all string values are properly escaped (use \" for quotes, \n for newlines)
- Do not include literal newlines inside string values
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
    claude_client: anthropic.Anthropic
    gemini_client: object | None = None
    imagekit_client: object | None = None
    step_outputs: dict[str, str] = field(default_factory=dict)
    image_plans: list[ImagePlan] = field(default_factory=list)
    log_messages: list[str] = field(default_factory=list)

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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if need_images:
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not gemini_key:
            missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
        if not os.environ.get("IMAGEKIT_PRIVATE_KEY"):
            missing.append("IMAGEKIT_PRIVATE_KEY")

    return len(missing) == 0, missing


def clean_json_response(text: str) -> str:
    """Remove markdown code fences and fix common JSON issues."""
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try to extract JSON if wrapped in other text
    # Look for the first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return text.strip()


def word_to_pattern(word: str) -> str:
    """Convert a word to a regex pattern allowing HTML entities."""
    result = []
    for char in word:
        if char in '—–':
            result.append(r'(?:&mdash;|&ndash;|&#8212;|&#8211;|—|–|-)')
        elif char == '"':
            result.append(r'(?:&ldquo;|&#8220;|"|")')
        elif char == '"':
            result.append(r'(?:&rdquo;|&#8221;|"|")')
        elif char == ''':
            result.append(r"(?:&lsquo;|&#8216;|'|')")
        elif char == ''':
            result.append(r"(?:&rsquo;|&#8217;|'|')")
        elif char == '…':
            result.append(r'(?:&hellip;|&#8230;|…|\.\.\.)')
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


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_a_write_content(state: PipelineState) -> bool:
    """Agent A: Write educational content."""
    state.log("\n📝 [1/4] Agent A: Writing content...")

    try:
        message = state.claude_client.messages.create(
            model=CLAUDE_WRITER_MODEL,
            max_tokens=12000,
            temperature=1.0,
            system=AGENT_A_PROMPTS[state.config.content_type],
            messages=[{"role": "user", "content": state.config.topic}],
        )

        response_text = "".join(
            block.text for block in message.content if hasattr(block, "text")
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

    except anthropic.APIError as e:
        state.log(f"    ❌ API Error: {e.message}")
        return False


def stage_b_structure_html(state: PipelineState) -> bool:
    """Agent B: Convert to semantic HTML."""
    state.log("\n🏗️  [2/4] Agent B: Structuring HTML...")

    try:
        message = state.claude_client.messages.create(
            model=CLAUDE_WRITER_MODEL,
            max_tokens=14000,
            temperature=0.2,
            system=AGENT_B_PROMPTS[state.config.content_type],
            messages=[{"role": "user", "content": state.step_outputs["agent_a"]}],
        )

        response_text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )

        if not response_text.strip():
            state.log("    ❌ Empty response from Agent B")
            return False

        output_path = state.config.content_dir / "Step2_Structured.html"
        output_path.write_text(response_text, encoding="utf-8")
        state.step_outputs["agent_b"] = response_text

        state.log(f"    ✓ Structured HTML → {output_path.name}")
        return True

    except anthropic.APIError as e:
        state.log(f"    ❌ API Error: {e.message}")
        return False


def stage_c_style_publish(state: PipelineState) -> bool:
    """Agent C: Apply styles and accessibility features."""
    state.log("\n🎨 [3/4] Agent C: Styling and publishing...")

    try:
        message = state.claude_client.messages.create(
            model=CLAUDE_WRITER_MODEL,
            max_tokens=16384,
            temperature=0.2,
            system=AGENT_C_PROMPTS[state.config.content_type],
            messages=[{"role": "user", "content": state.step_outputs["agent_b"]}],
        )

        response_text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )

        if not response_text.strip():
            state.log("    ❌ Empty response from Agent C")
            return False

        output_path = state.config.content_dir / "Step3_Styled.html"
        output_path.write_text(response_text, encoding="utf-8")
        state.step_outputs["agent_c"] = response_text

        char_count = len(response_text)
        state.log(f"    ✓ Styled HTML ({char_count:,} chars) → {output_path.name}")

        if state.config.content_type == ContentType.TEXTBOOK:
            state.log("    ✓ Added Knowledge Check blocks")
        else:
            state.log("    ✓ Added Font Awesome icons")

        return True

    except anthropic.APIError as e:
        state.log(f"    ❌ API Error: {e.message}")
        return False


def stage_d_brainstorm_images(state: PipelineState) -> bool:
    """Agent D Part 1: Brainstorm image insertion points."""
    state.log("\n🧠 [4a/4] Agent D: Brainstorming images...")

    try:
        message = state.claude_client.messages.create(
            model=CLAUDE_BRAINSTORM_MODEL,
            max_tokens=4096,
            temperature=0.3,
            system=AGENT_D_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this HTML content and identify up to {MAX_IMAGES} optimal locations for educational images:\n\n{state.step_outputs['agent_c']}",
            }],
        )

        response_text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )

        # Parse JSON
        cleaned = clean_json_response(response_text)

        # Debug: Log the response to help diagnose the issue
        state.log(f"    📄 Raw JSON response preview: {cleaned[:200]}...")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as json_err:
            # Log the full response for debugging
            debug_path = state.config.output_dir / "agent_d_failed_response.txt"
            debug_path.write_text(
                f"Raw response:\n{response_text}\n\n"
                f"Cleaned:\n{cleaned}\n\n"
                f"Error: {json_err}\n"
                f"Error position: line {json_err.lineno}, column {json_err.colno}",
                encoding="utf-8"
            )
            state.log(f"    ⚠️  Full response saved to: {debug_path.name}")
            state.log(f"    💡 Tip: Check the JSON for unescaped quotes, newlines, or trailing commas")
            raise json_err

        # Convert to ImagePlan objects
        for i, item in enumerate(data.get("images", [])[:MAX_IMAGES], start=1):
            plan = ImagePlan(
                insertion_context=item["insertion_context"],
                image_prompt=item["image_prompt"],
                alt_text=item["alt_text"][:125],
                caption=item["caption"],
                figure_number=i,
            )
            state.image_plans.append(plan)

        state.log(f"    ✓ Identified {len(state.image_plans)} image insertion points")
        for plan in state.image_plans:
            state.log(f"      • Figure {plan.figure_number}: {plan.caption[:50]}...")

        return True

    except json.JSONDecodeError as e:
        state.log(f"    ❌ Failed to parse JSON: {e}")
        return False
    except anthropic.APIError as e:
        state.log(f"    ❌ API Error: {e.message}")
        return False


def stage_d_generate_images(state: PipelineState) -> bool:
    """Agent D Part 2: Generate and upload images."""
    state.log(f"\n🖼️  [4b/4] Generating {len(state.image_plans)} images...")

    successful_plans = []

    for plan in state.image_plans:
        state.log(f"\n    [{plan.figure_number}/{len(state.image_plans)}] Figure {plan.figure_number}...")

        try:
            # Build generation prompt
            full_prompt = (
                f"Generate a high-quality educational illustration: {plan.image_prompt}. "
                f"Style: Professional, clean, minimal text overlay, suitable for a textbook."
            )

            # Generate with Gemini
            response = state.gemini_client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

            # Extract image data
            image_data = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        image_data = part.inline_data.data
                        break

            if not image_data:
                state.log(f"        ❌ No image data in response")
                continue

            # Save locally
            local_filename = f"fig{plan.figure_number}-{state.config.timestamp}.png"
            local_path = state.config.images_dir / local_filename
            with open(local_path, "wb") as f:
                f.write(image_data)
            plan.local_path = local_path
            state.log(f"        ✓ Saved: {local_filename}")

            # Upload to ImageKit
            state.log(f"        ↑ Uploading to ImageKit...")
            result = state.imagekit_client.files.upload(
                file=image_data,
                file_name=local_filename,
                folder=state.config.imagekit_folder,
                use_unique_file_name=True,
                is_private_file=False,
                tags=["lms", "educational", "ai-generated"],
            )

            if result and hasattr(result, 'url') and result.url:
                plan.hosted_url = result.url
                state.log(f"        ✓ Hosted: {result.url}")
            else:
                state.log(f"        ⚠️ Upload failed, using local path")

            successful_plans.append(plan)

        except Exception as e:
            state.log(f"        ❌ Failed: {type(e).__name__}: {e}")
            continue

    state.image_plans = successful_plans
    hosted = sum(1 for p in successful_plans if p.hosted_url)
    state.log(f"\n    Summary: {len(successful_plans)}/{MAX_IMAGES} images, {hosted} hosted")

    return len(successful_plans) > 0


def stage_d_inject_images(state: PipelineState) -> bool:
    """Agent D Part 3: Inject images into HTML."""
    state.log(f"\n📎 [4c/4] Injecting {len(state.image_plans)} images into HTML...")

    html_content = state.step_outputs["agent_c"]

    for plan in state.image_plans:
        img_src = plan.hosted_url if plan.hosted_url else str(plan.local_path)

        figure_html = f"""
</p>
<figure style="{SHARED_STYLES['figure']}">
  <img style="{SHARED_STYLES['img']}"
       src="{img_src}"
       alt="{plan.alt_text}"
       loading="lazy" />
  <figcaption style="{SHARED_STYLES['figcaption']}">
    {plan.caption} <em>(Generated by AI)</em>
  </figcaption>
</figure>
<p style="{SHARED_STYLES['p']}">
"""

        # Build entity-aware pattern
        context = plan.insertion_context.strip()
        normalized_context = html.unescape(context)
        words = normalized_context.split()

        if not words:
            state.log(f"    ❌ Empty context for Figure {plan.figure_number}")
            continue

        word_patterns = [word_to_pattern(w) for w in words]
        separator = r'[\s\n]*(?:<[^>]+>)*[\s\n]*'
        pattern = r'(?si)' + separator.join(word_patterns)

        match = re.search(pattern, html_content)

        if match:
            end_pos = match.end()
            html_content = html_content[:end_pos] + figure_html + html_content[end_pos:]
            state.log(f"    ✓ Inserted Figure {plan.figure_number}: {plan.caption[:40]}...")
        else:
            state.log(f"    ❌ Context not found for Figure {plan.figure_number}")

    state.step_outputs["agent_d"] = html_content
    return True


def save_final_output(state: PipelineState) -> Path:
    """Save the final HTML output."""
    # Determine which step has the final output
    if "agent_d" in state.step_outputs:
        final_html = state.step_outputs["agent_d"]
        step_name = "Step4_Final.html"
    else:
        final_html = state.step_outputs["agent_c"]
        step_name = "Step3_Final.html"

    # Save to content directory
    output_path = state.config.content_dir / step_name
    output_path.write_text(final_html, encoding="utf-8")

    # Also save a convenience copy to cwd
    cwd_copy = Path.cwd() / f"lms_output_{state.config.timestamp}.html"
    cwd_copy.write_text(final_html, encoding="utf-8")

    return output_path


# =============================================================================
# USER INTERFACE
# =============================================================================

def display_banner() -> None:
    """Display the application banner."""
    print("\n" + "=" * 65)
    print("  ✨ LMS CONTENT PIPELINE - Unified Edition ✨")
    print("=" * 65)


def display_menu() -> ContentType:
    """Display content type menu and return selection."""
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
    """Get the topic prompt for the content type."""
    prompts = {
        ContentType.TEXTBOOK: "Enter the topic for your textbook chapter",
        ContentType.DISCUSSION: "Enter the topic/scenario for your discussion",
        ContentType.ASSIGNMENT: "Enter the topic/task for your assignment",
    }
    return prompts[content_type]


def print_summary(state: PipelineState, success: bool, final_path: Path | None) -> None:
    """Print the pipeline summary."""
    print("\n" + "=" * 65)

    if success:
        print("✅ PIPELINE COMPLETE!")
        print("=" * 65)

        print(f"\n📁 Output Directory: {state.config.output_dir}")
        print(f"\n📂 Content Files ({state.config.content_dir.name}/):")

        for f in sorted(state.config.content_dir.glob("*")):
            size = f.stat().st_size
            print(f"   • {f.name} ({size:,} bytes)")

        if state.image_plans:
            print(f"\n🖼️  Images ({state.config.images_dir.name}/):")
            for plan in state.image_plans:
                if plan.local_path:
                    print(f"   • {plan.local_path.name}")
                if plan.hosted_url:
                    print(f"     ↳ {plan.hosted_url}")

        if final_path:
            cwd_copy = Path.cwd() / f"lms_output_{state.config.timestamp}.html"
            print(f"\n📄 Convenience copy: {cwd_copy}")

        print(f"\n📋 Log saved: {state.config.output_dir / 'pipeline.log'}")

    else:
        print("❌ PIPELINE FAILED")
        print("=" * 65)
        print("\nCheck the error messages above and the log file for details.")

    print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LMS Content Pipeline - Generate Canvas-ready educational content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic", "-t",
        help="Topic for content generation (skips interactive prompt)",
    )
    parser.add_argument(
        "--type", "-c",
        choices=["textbook", "discussion", "assignment"],
        help="Content type (skips interactive menu)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image generation (Agent D)",
    )
    parser.add_argument(
        "--force-images",
        action="store_true",
        help="Generate images even for Discussion/Assignment types",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    display_banner()

    # Determine content type
    if args.type:
        content_type = ContentType(args.type)
        print(f"\n📋 Content type: {content_type.value.capitalize()}")
    else:
        content_type = display_menu()

    # Determine if images should be generated
    enable_images = (
        not args.no_images
        and IMAGES_AVAILABLE
        and (content_type == ContentType.TEXTBOOK or args.force_images)
    )

    # Show image status
    if not IMAGES_AVAILABLE:
        print("\n⚠️  Image generation unavailable (missing google-genai or imagekitio)")
    elif args.no_images:
        print("\n📷 Image generation: Disabled (--no-images)")
    elif content_type != ContentType.TEXTBOOK and not args.force_images:
        print("\n📷 Image generation: Skipped (use --force-images for non-textbook)")
    else:
        print("\n📷 Image generation: Enabled")

    # Validate environment
    valid, missing = validate_environment(enable_images)
    if not valid:
        print(f"\n❌ Missing environment variables:")
        for var in missing:
            print(f"   • {var}")
        return 1

    # Get topic
    if args.topic:
        topic = args.topic
        print(f"\n📝 Topic: {topic}")
    else:
        prompt = get_topic_prompt(content_type)
        topic = input(f"\n{prompt}: ").strip()

    if not topic:
        print("❌ Topic cannot be empty.")
        return 1

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path.cwd() / f"lms_output_{timestamp}"
    content_dir = output_dir / "content"
    images_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(exist_ok=True)
    if enable_images:
        images_dir.mkdir(exist_ok=True)

    # Create configuration
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

    # Initialize clients
    print("\n⚙️  Initializing API clients...")

    claude_client = anthropic.Anthropic()
    print("   ✓ Anthropic client ready")

    gemini_client = None
    imagekit_client = None

    if enable_images:
        gemini_client = genai.Client()
        print("   ✓ Gemini client ready")

        imagekit_client = ImageKit(
            private_key=os.environ.get("IMAGEKIT_PRIVATE_KEY"),
        )
        print(f"   ✓ ImageKit client ready (folder: {config.imagekit_folder})")

    # Create pipeline state
    state = PipelineState(
        config=config,
        claude_client=claude_client,
        gemini_client=gemini_client,
        imagekit_client=imagekit_client,
    )

    state.log(f"Pipeline started: {timestamp}")
    state.log(f"Content type: {content_type.value}")
    state.log(f"Topic: {topic}")
    state.log(f"Output directory: {output_dir}")

    # Execute pipeline stages
    success = True
    final_path = None

    # Stage A: Write content
    if success:
        success = stage_a_write_content(state)

    # Stage B: Structure HTML
    if success:
        success = stage_b_structure_html(state)

    # Stage C: Style and publish
    if success:
        success = stage_c_style_publish(state)

    # Stage D: Images (optional)
    if success and enable_images:
        if stage_d_brainstorm_images(state):
            if state.image_plans:
                stage_d_generate_images(state)
                if state.image_plans:
                    stage_d_inject_images(state)

    # Save final output
    if success:
        final_path = save_final_output(state)
        state.log(f"\n✅ Final output: {final_path}")

    # Save log
    state.save_log()

    # Print summary
    print_summary(state, success, final_path)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
