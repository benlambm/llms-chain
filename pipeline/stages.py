from __future__ import annotations

import base64
import re
from typing import List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - handled at runtime
    BeautifulSoup = None
    _bs4_import_error = exc
else:
    _bs4_import_error = None

from .config import (
    IMAGE_MODEL,
    IMAGE_PLAN_SCHEMA,
    MAX_IMAGES,
    PROMPTS_DIR,
    TEXT_MODEL_AGENT_A,
    TEXT_MODEL_AGENT_BCD,
    ContentType,
    ImagePlan,
    PipelineState,
)
from .io import load_prompt, normalize_text, validate_html_fragment
from .openai_client import generate_structured_json, generate_text


def stage_a_write_content(state: PipelineState) -> bool:
    """Agent A: Write educational content."""
    state.log("\n📝 [1/4] Agent A: Writing content...")

    prompt_path = PROMPTS_DIR / f"agent_a_{state.config.content_type.value}.txt"
    try:
        system_prompt = load_prompt(prompt_path)
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

    prompt_path = PROMPTS_DIR / f"agent_b_{state.config.content_type.value}.txt"
    try:
        system_prompt = load_prompt(prompt_path)
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
        prompt_path = PROMPTS_DIR / "agent_c_textbook.txt"
    else:
        prompt_path = PROMPTS_DIR / "agent_c_discussion_assignment.txt"

    try:
        system_prompt = load_prompt(prompt_path)
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
        system_prompt = load_prompt(PROMPTS_DIR / "agent_d_brainstorm.txt")
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

    successful_plans: List[ImagePlan] = []

    for plan in state.image_plans:
        state.log(f"    Figure {plan.figure_number}: Generating...")

        try:
            full_prompt = (
                f"Generate a high-quality educational illustration: {plan.image_prompt}. "
                f"Style: Professional, clean, no text or labels, suitable for a textbook."
            )

            response = state.openai_client.images.generate(
                model=IMAGE_MODEL,
                prompt=full_prompt,
            )

            if not response.data or not response.data[0].b64_json:
                state.log("        ❌ No image data received")
                continue

            image_data = base64.b64decode(response.data[0].b64_json)

            local_filename = f"fig{plan.figure_number}-{state.config.timestamp}.png"
            local_path = state.config.images_dir / local_filename
            with open(local_path, "wb") as f:
                f.write(image_data)

            plan.local_path = local_path
            state.log(f"        ✓ Saved local: {local_filename}")

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
    if not successful_plans:
        state.log("    ⚠️ No images generated; continuing without images")
    return True


def _find_heading_index(tags: List[object], heading_text: str) -> Optional[int]:
    target = heading_text.lower()
    for idx, tag in enumerate(tags):
        if not hasattr(tag, "name") or not tag.name:
            continue
        if re.fullmatch(r"h[1-6]", tag.name, flags=re.IGNORECASE):
            if normalize_text(tag.get_text(" ", strip=True)).lower() == target:
                return idx
    return None


def _candidate_tags(tags: List[object]) -> List[object]:
    candidates = []
    for tag in tags:
        if not hasattr(tag, "name") or tag.name != "p":
            continue
        if tag.find_parent(["details", "dl"]):
            continue
        candidates.append(tag)
    return candidates


def _soup_to_html(soup: BeautifulSoup) -> str:
    if soup.body:
        return "".join(str(item) for item in soup.body.contents)
    return str(soup)


def stage_d_inject_images(state: PipelineState) -> bool:
    """Agent D Part 3: Inject images into HTML."""
    state.log("\n📎 [4c/4] Injecting images into HTML...")

    if not state.image_plans:
        state.log("    ⚠️ No images to inject")
        state.step_outputs["agent_d"] = state.step_outputs["agent_c"]
        return True

    if BeautifulSoup is None:
        raise RuntimeError(
            "BeautifulSoup is required for DOM-aware insertion. "
            "Install it with 'pip install beautifulsoup4'."
        ) from _bs4_import_error

    html_content = state.step_outputs["agent_c"]
    soup = BeautifulSoup(html_content, "html.parser")

    all_tags = list(soup.find_all(True))
    core_idx = _find_heading_index(all_tags, "Core Content")
    end_candidates = [
        idx
        for idx in (
            _find_heading_index(all_tags, "Summary"),
            _find_heading_index(all_tags, "Glossary of Key Terms"),
        )
        if idx is not None
    ]
    end_idx = min(end_candidates) if end_candidates else None

    start = core_idx + 1 if core_idx is not None else 0
    end = end_idx if end_idx is not None else len(all_tags)

    core_tags = _candidate_tags(all_tags[start:end])
    fallback_tags = _candidate_tags(all_tags)
    tag_positions = {id(tag): idx for idx, tag in enumerate(all_tags)}

    placements = []
    for plan in state.image_plans:
        img_src = plan.hosted_url or (f"../images/{plan.local_path.name}" if plan.local_path else None)
        if not img_src:
            state.log(f"    ⚠️ Missing image source for Figure {plan.figure_number}; skipping")
            continue
        context_norm = normalize_text(plan.insertion_context)
        match_tag = None
        for tag in core_tags:
            if context_norm and context_norm in normalize_text(tag.get_text(" ", strip=True)):
                match_tag = tag
                break
        if match_tag is None:
            for tag in fallback_tags:
                if context_norm and context_norm in normalize_text(tag.get_text(" ", strip=True)):
                    match_tag = tag
                    break
        if match_tag is None:
            state.log(f"    ❌ Insertion context not found for Figure {plan.figure_number}")
            state.log(f"       Context looked for: '{plan.insertion_context.strip()}'")
            continue
        placements.append((tag_positions.get(id(match_tag), 0), match_tag, plan, img_src))

    if not placements:
        state.log("    ⚠️ No valid insertion points found; skipping image injection")
        state.step_outputs["agent_d"] = state.step_outputs["agent_c"]
        return True

    placements.sort(key=lambda item: item[0])

    for display_number, (_, tag, plan, img_src) in enumerate(placements, start=1):

        caption_text = re.sub(r"^Figure\s+\d+\s*:\s*", "", plan.caption.strip(), flags=re.IGNORECASE)
        caption_full = f"Figure {display_number}: {caption_text}" if caption_text else f"Figure {display_number}"

        figure = soup.new_tag("figure")
        figure["style"] = "margin: 2em 0; max-width: 75ch;"

        img = soup.new_tag("img")
        img["style"] = "width: 100%; height: auto; border: 1px solid #E0E0E0; border-radius: 4px;"
        img["src"] = img_src
        img["alt"] = plan.alt_text
        img["loading"] = "lazy"
        figure.append(img)

        figcaption = soup.new_tag("figcaption")
        figcaption["style"] = (
            "padding: 0.75em 1em; font-style: italic; background-color: #F5F5F5; "
            "text-align: center; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; "
            "font-size: 16px; color: #333333; border-bottom-left-radius: 4px; "
            "border-bottom-right-radius: 4px;"
        )
        figcaption.append(caption_full + " ")
        generated = soup.new_tag("em")
        generated.string = "(Generated by AI)"
        figcaption.append(generated)
        figure.append(figcaption)

        tag.insert_after(figure)
        state.log(f"    ✓ Inserted Figure {display_number}")

    html_content = _soup_to_html(soup)
    sanity_issues = validate_html_fragment(html_content)
    for issue in sanity_issues:
        state.log(f"    ⚠️ HTML sanity: {issue}")

    state.step_outputs["agent_d"] = html_content
    return True
