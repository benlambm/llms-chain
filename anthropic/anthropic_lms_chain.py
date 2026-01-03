#!/usr/bin/env python3
"""
Anthropic LMS Content Generator - Plus Edition
===============================================
A prompt-chaining pipeline for generating Canvas LMS-ready educational content.

Supports three content types:
  - Textbook: Full chapters with Knowledge Checks
  - Discussion: Interactive discussion prompts with Font Awesome icons
  - Assignment: Step-based assignments with rubrics

Usage:
  python anthropic_lms_chain.py

Requires:


  - ANTHROPIC_API_KEY environment variable

Author: Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final

import anthropic

# =============================================================================
# CONFIGURATION LAYER - Centralized for easy maintainability
# =============================================================================

class ContentType(Enum):
    """Supported LMS content types."""
    TEXTBOOK = "textbook"
    DISCUSSION = "discussion"
    ASSIGNMENT = "assignment"


# Shared CSS Constants (extracted for DRY principle)
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
}

# Font Awesome icons for Discussion/Assignment pages
FA_ICONS: Final[dict[str, str]] = {
    "write": '<i class="fas fa-pencil-alt" style="padding-right: 10px;" aria-hidden="true"></i>',
    "respond": '<i class="fas fa-comments" style="padding-right: 10px;" aria-hidden="true"></i>',
    "evaluate": '<i class="fas fa-check" style="padding-right: 10px;" aria-hidden="true"></i>',
}


# =============================================================================
# AGENT PROMPTS - Organized by content type and agent role
# =============================================================================

# -----------------------------------------------------------------------------
# AGENT A: Content Writer Prompts
# -----------------------------------------------------------------------------

AGENT_A_BASE_WRITING_STANDARDS: Final[str] = """
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

AGENT_A_TEXTBOOK: Final[str] = f"""
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

{AGENT_A_BASE_WRITING_STANDARDS}

Length: approximately 3,000-3,500 words.
"""

AGENT_A_DISCUSSION: Final[str] = f"""
You write discussion board prompts for college-level courses. Your audience is intelligent adults
who may have limited technical background.

Discussion Prompt Requirements:
Every discussion prompt includes these sections (use these exact headings):
- Write/Prompt: The main discussion question or scenario students must respond to (1-2 paragraphs)
- Respond/Reply: Clear instructions for how students should engage with peers' posts
- Evaluation/Rubric: Grading criteria explaining how responses will be assessed

If the topic introduces technical terminology, include a brief:
- Key Terms: Essential vocabulary students need (only if technical terms are introduced)

{AGENT_A_BASE_WRITING_STANDARDS}

Length: approximately 400-600 words total.
Keep prompts focused and actionable. Encourage critical thinking and peer engagement.
"""

AGENT_A_ASSIGNMENT: Final[str] = f"""
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

{AGENT_A_BASE_WRITING_STANDARDS}

Length: approximately 500-800 words total.
Be specific about expectations, deadlines format, and submission requirements.
"""


# -----------------------------------------------------------------------------
# AGENT B: HTML Structurer Prompts
# -----------------------------------------------------------------------------

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

AGENT_B_TEXTBOOK: Final[str] = f"""
{AGENT_B_BASE}

Textbook-Specific Structure:
- Learning Objectives should be an <h3> followed by a <ul> list
- Core content sections use <h3> for main topics, <h4> for subtopics
- Summary is an <h3> followed by paragraph(s)
- Glossary uses <h3> "Glossary of Key Terms" followed by <dl>/<dt>/<dd>
"""

AGENT_B_DISCUSSION: Final[str] = f"""
{AGENT_B_BASE}

Discussion-Specific Structure:
- The main title uses <h2>
- "Write/Prompt" section uses <h3>Write/Prompt</h3>
- "Respond/Reply" section uses <h3>Respond/Reply</h3>
- "Evaluation/Rubric" section uses <h3>Evaluation/Rubric</h3>
- If "Key Terms" exists, use <h3>Key Terms</h3> followed by <dl>/<dt>/<dd>
- Use <hr> between major sections
"""

AGENT_B_ASSIGNMENT: Final[str] = f"""
{AGENT_B_BASE}

Assignment-Specific Structure:
- The main title uses <h2>
- "Overview" section uses <h3>Overview</h3>
- "Instructions" section uses <h3>Instructions</h3>
- "Deliverables" section uses <h3>Deliverables</h3>
- "Evaluation/Rubric" section uses <h3>Evaluation/Rubric</h3>
- If "Key Terms" exists, use <h3>Key Terms</h3> followed by <dl>/<dt>/<dd>
- Use <hr> between major sections
"""


# -----------------------------------------------------------------------------
# AGENT C: Publisher/Styler Prompts
# -----------------------------------------------------------------------------

def _build_agent_c_base_styles() -> str:
    """Build the base styling instructions for Agent C."""
    return f"""
Use only inline CSS (no classes, no external styles).
Use this font stack everywhere: {FONT_STACK}

Apply styles as follows (copy these style attributes verbatim):

h2 (module/page title):
style="{SHARED_STYLES['h2']}"

h3 (major section headings):
style="{SHARED_STYLES['h3']}"

h4 (subsection headings):
style="{SHARED_STYLES['h4']}"

p (body text):
style="{SHARED_STYLES['p']}"

ul:
style="{SHARED_STYLES['ul']}"

li:
style="{SHARED_STYLES['li']}"

dl (for glossaries/definitions):
style="{SHARED_STYLES['dl']}"

dt:
style="{SHARED_STYLES['dt']}"

dd:
style="{SHARED_STYLES['dd']}"

em:
style="{SHARED_STYLES['em']}"

strong:
style="{SHARED_STYLES['strong']}"

hr:
Replace any bare <hr> with:
<hr style="{SHARED_STYLES['hr']}">
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

{_build_agent_c_base_styles()}

Usage guidelines for strong and em:
- Use sparingly: Avoid excessive bold/italics for long passages
- Don't rely on visual formatting alone: Screen readers don't announce formatting changes

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

{_build_agent_c_base_styles()}

2. Add Font Awesome icons to specific H3 headings

For Discussion and Assignment pages, prepend Font Awesome icons inside specific H3 tags:

- For "Write/Prompt" or "Overview" or "Instructions" headings:
  {FA_ICONS['write']}

- For "Respond/Reply" or "Deliverables" headings:
  {FA_ICONS['respond']}

- For "Evaluation/Rubric" headings:
  {FA_ICONS['evaluate']}

Example transformation:
BEFORE: <h3>Write/Prompt</h3>
AFTER: <h3 style="...">{FA_ICONS['write']}Write/Prompt</h3>

3. Glossary/Key Terms handling
- Only include a styled Glossary/Key Terms section if one exists in the input HTML
- If present, style it using the dl/dt/dd styles above

4. Output format
- Output only the final transformed HTML fragment.
- Do not add comments, explanations, or any extra text outside the HTML.
- Ensure compliance with WCAG 2.1 AA accessibility standards.
"""


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AgentStep:
    """Configuration for a single agent step in the chain."""
    name: str
    description: str
    model: str
    max_tokens: int
    temperature: float
    output_suffix: str

    def get_prompt(self, content_type: ContentType) -> str:
        """Get the appropriate system prompt for this agent and content type."""
        prompts = {
            "Agent A": {
                ContentType.TEXTBOOK: AGENT_A_TEXTBOOK,
                ContentType.DISCUSSION: AGENT_A_DISCUSSION,
                ContentType.ASSIGNMENT: AGENT_A_ASSIGNMENT,
            },
            "Agent B": {
                ContentType.TEXTBOOK: AGENT_B_TEXTBOOK,
                ContentType.DISCUSSION: AGENT_B_DISCUSSION,
                ContentType.ASSIGNMENT: AGENT_B_ASSIGNMENT,
            },
            "Agent C": {
                ContentType.TEXTBOOK: AGENT_C_TEXTBOOK,
                ContentType.DISCUSSION: AGENT_C_DISCUSSION_ASSIGNMENT,
                ContentType.ASSIGNMENT: AGENT_C_DISCUSSION_ASSIGNMENT,
            },
        }
        return prompts[self.name][content_type]


# Pipeline steps definition
PIPELINE_STEPS: Final[list[AgentStep]] = [
    AgentStep(
        name="Agent A",
        description="Writer - drafting content",
        model="claude-sonnet-4-20250514",
        max_tokens=12000,
        temperature=1.0,
        output_suffix="_Step1_Content.txt",
    ),
    AgentStep(
        name="Agent B",
        description="Structurer - converting to semantic HTML",
        model="claude-sonnet-4-20250514",
        max_tokens=14000,
        temperature=0.2,
        output_suffix="_Step2_Structured.html",
    ),
    AgentStep(
        name="Agent C",
        description="Publisher - applying styles and accessibility features",
        model="claude-sonnet-4-20250514",
        max_tokens=16384,
        temperature=0.2,
        output_suffix="_Step3_Final.html",
    ),
]


# =============================================================================
# CORE PIPELINE FUNCTIONS
# =============================================================================

@dataclass
class PipelineContext:
    """Holds state and configuration for a pipeline run."""
    content_type: ContentType
    topic: str
    output_dir: Path
    client: anthropic.Anthropic
    outputs: dict[str, str] = field(default_factory=dict)


def display_menu() -> ContentType:
    """Display content type selection menu and return user's choice."""
    print("\n" + "=" * 60)
    print("  ANTHROPIC LMS CONTENT GENERATOR - Plus Edition")
    print("=" * 60)
    print("\nWhat type of LMS Page content do you need to generate?\n")
    print("  [1] Textbook Chapter")
    print("      → Full educational chapter with Learning Objectives,")
    print("        Core Content, Summary, Glossary, and Knowledge Checks")
    print()
    print("  [2] Discussion Board Prompt")
    print("      → Interactive discussion with Write/Respond/Evaluate sections")
    print("        and Font Awesome icons")
    print()
    print("  [3] Assignment Instructions")
    print("      → Step-based assignment with Overview, Instructions,")
    print("        Deliverables, and Rubric with Font Awesome icons")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return ContentType.TEXTBOOK
        elif choice == "2":
            return ContentType.DISCUSSION
        elif choice == "3":
            return ContentType.ASSIGNMENT
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def get_topic_prompt(content_type: ContentType) -> str:
    """Get the appropriate topic prompt based on content type."""
    prompts = {
        ContentType.TEXTBOOK: "Enter the topic for your textbook chapter: ",
        ContentType.DISCUSSION: "Enter the topic/scenario for your discussion prompt: ",
        ContentType.ASSIGNMENT: "Enter the topic/task for your assignment: ",
    }
    return prompts[content_type]


def call_claude(
    ctx: PipelineContext,
    step: AgentStep,
    input_text: str,
) -> str:
    """Make an API call to Claude with the given step configuration."""
    system_prompt = step.get_prompt(ctx.content_type)

    message = ctx.client.messages.create(
        model=step.model,
        max_tokens=step.max_tokens,
        temperature=step.temperature,
        system=system_prompt,
        messages=[
            {"role": "user", "content": input_text}
        ],
    )

    # Extract text from response
    response_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            response_text += block.text

    return response_text


def save_output(ctx: PipelineContext, step: AgentStep, content: str) -> Path:
    """Save the output of a pipeline step to a file."""
    # Create filename based on content type and step
    type_prefix = ctx.content_type.value.capitalize()
    filename = f"{type_prefix}{step.output_suffix}"
    filepath = ctx.output_dir / filename

    filepath.write_text(content, encoding="utf-8")

    # Also save a copy to current directory for convenience
    cwd_filepath = Path.cwd() / f"anthropic_{filename}"
    cwd_filepath.write_text(content, encoding="utf-8")

    return filepath


def run_pipeline(ctx: PipelineContext) -> bool:
    """Execute the full pipeline and return success status."""
    print(f"\n🔗 Starting {ctx.content_type.value.upper()} pipeline...")
    print(f"   Topic: {ctx.topic}")
    print(f"   Output: {ctx.output_dir}\n")

    current_input = ctx.topic

    for idx, step in enumerate(PIPELINE_STEPS, start=1):
        try:
            print(f"[{idx}/{len(PIPELINE_STEPS)}] {step.name} ({step.description})...")

            # Call Claude
            output = call_claude(ctx, step, current_input)

            if not output.strip():
                print(f"    ✗ FAILED: Empty response from {step.name}")
                return False

            # Save output
            filepath = save_output(ctx, step, output)
            ctx.outputs[step.name] = output

            print(f"    ✓ Saved: {filepath.name}")

            # Output becomes input for next step
            current_input = output

        except anthropic.APIError as e:
            print(f"    ✗ API Error in {step.name}: {e.message}")
            return False
        except Exception as e:
            print(f"    ✗ Error in {step.name}: {str(e)[:200]}")
            return False

    return True


def print_summary(ctx: PipelineContext, success: bool) -> None:
    """Print a summary of the pipeline run."""
    print("\n" + "=" * 60)

    if success:
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nContent Type: {ctx.content_type.value.capitalize()}")
        print(f"Topic: {ctx.topic}")
        print(f"\nGenerated Files (in {ctx.output_dir}):")

        type_prefix = ctx.content_type.value.capitalize()
        for step in PIPELINE_STEPS:
            filename = f"{type_prefix}{step.output_suffix}"
            print(f"  • {filename}")

        print(f"\nConvenience copies also saved to current directory with 'anthropic_' prefix.")

        if ctx.content_type == ContentType.TEXTBOOK:
            print("\n📚 Textbook chapter includes Knowledge Check blocks.")
        else:
            print(f"\n🎨 {ctx.content_type.value.capitalize()} page includes Font Awesome icons.")
            print("   Ensure Font Awesome CSS is loaded in your LMS theme.")
    else:
        print("❌ PIPELINE FAILED")
        print("=" * 60)
        print("\nCheck the error messages above for details.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point for the LMS content generator."""
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your API key and try again.")
        return 1

    # Display menu and get content type
    content_type = display_menu()

    # Get topic from user
    topic_prompt = get_topic_prompt(content_type)
    topic = input(f"\n{topic_prompt}").strip()

    if not topic:
        print("ERROR: Topic cannot be empty.")
        return 1

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "chain_outputs" / f"{content_type.value}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Create pipeline context
    ctx = PipelineContext(
        content_type=content_type,
        topic=topic,
        output_dir=output_dir,
        client=client,
    )

    # Run the pipeline
    success = run_pipeline(ctx)

    # Print summary
    print_summary(ctx, success)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
