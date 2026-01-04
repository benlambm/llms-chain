from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

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

from .config import ContentType, PipelineConfig, PipelineState, DEFAULT_IMAGEKIT_FOLDER, PROMPTS_DIR
from .io import save_final_output, save_run_manifest, serve_final_output, validate_environment
from .stages import (
    stage_a_write_content,
    stage_b_structure_html,
    stage_c_style_publish,
    stage_d_brainstorm_images,
    stage_d_generate_images,
    stage_d_inject_images,
)


def display_banner() -> None:
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
        if choice == "2":
            return ContentType.DISCUSSION
        if choice == "3":
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
    parser.add_argument("--test", action="store_true", help="Run in test mode with reduced max_output_tokens")
    return parser.parse_args()


def _format_timestamp(now: datetime) -> str:
    hour_12 = now.hour % 12 or 12
    return f"{now.month}-{now.day}-{now.year}_{hour_12}-{now.minute:02d}"


def _prompt_paths(content_type: ContentType, test_mode: bool) -> dict[str, Path]:
    if test_mode:
        return {
            "agent_a": PROMPTS_DIR / "agent_a_test.txt",
            "agent_b": PROMPTS_DIR / "agent_b_test.txt",
            "agent_c": PROMPTS_DIR / "agent_c_test.txt",
            "agent_d_brainstorm": PROMPTS_DIR / "agent_d_brainstorm_test.txt",
        }

    prompt_paths = {
        "agent_a": PROMPTS_DIR / f"agent_a_{content_type.value}.txt",
        "agent_b": PROMPTS_DIR / f"agent_b_{content_type.value}.txt",
        "agent_d_brainstorm": PROMPTS_DIR / "agent_d_brainstorm.txt",
    }
    if content_type == ContentType.TEXTBOOK:
        prompt_paths["agent_c"] = PROMPTS_DIR / "agent_c_textbook.txt"
    else:
        prompt_paths["agent_c"] = PROMPTS_DIR / "agent_c_discussion_assignment.txt"
    return prompt_paths


def main() -> int:
    if not OPENAI_AVAILABLE:
        return 1

    args = parse_args()
    display_banner()

    if args.type:
        content_type = ContentType(args.type)
        print(f"\n📋 Content type: {content_type.value.capitalize()}")
    else:
        content_type = display_menu()

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

    if args.test:
        print("\n🧪 Test mode: max_output_tokens scaled to 10%")

    valid, missing = validate_environment(enable_images)
    if not valid:
        print("\n❌ Missing environment variables:")
        for var in missing:
            print(f"   • {var}")
        return 1

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

    timestamp = _format_timestamp(datetime.now())
    base_output_dir = Path.cwd() / "lms_output"
    output_dir = base_output_dir / timestamp
    content_dir = output_dir / "content"
    images_dir = output_dir / "images"

    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(exist_ok=True)
    if enable_images:
        images_dir.mkdir(exist_ok=True)

    print("\n⚙️  Initializing clients...")
    openai_client = OpenAI()

    imagekit_client = None
    if enable_images and IMAGEKIT_AVAILABLE:
        try:
            imagekit_client = ImageKit(
                private_key=os.environ.get("IMAGEKIT_PRIVATE_KEY"),
            )
            print("   ✓ ImageKit client ready")
        except Exception as e:
            print(f"   ⚠️ ImageKit init failed: {e}")

    input_path = Path(args.input_file).expanduser() if args.input_file else None

    config = PipelineConfig(
        content_type=content_type,
        topic=topic,
        timestamp=timestamp,
        output_dir=output_dir,
        content_dir=content_dir,
        images_dir=images_dir,
        enable_images=enable_images,
        max_output_token_scale=0.1 if args.test else 1.0,
        test_mode=args.test,
        imagekit_folder=os.environ.get("IMAGEKIT_FOLDER", DEFAULT_IMAGEKIT_FOLDER),
        input_file=input_path,
    )

    state = PipelineState(
        config=config,
        openai_client=openai_client,
        imagekit_client=imagekit_client,
    )

    success = True
    final_path = None

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
        if stage_d_brainstorm_images(state) and state.image_plans:
            stage_d_generate_images(state)
            stage_d_inject_images(state)
        else:
            state.log("    ⚠️ Skipping images due to brainstorming failure or no plans")

    if success:
        final_path = save_final_output(state)
        state.log(f"\n✅ Final output: {final_path}")

    state.save_log()

    if success:
        prompt_paths = _prompt_paths(content_type, args.test)
        save_run_manifest(state, final_path, prompt_paths)
        print(f"\n✅ PIPELINE COMPLETE! Check directory: {output_dir}")
        if final_path:
            print(f"📄 Open file: {final_path}")
            serve_final_output(output_dir, final_path)
    else:
        print("\n❌ PIPELINE FAILED. Check log.")

    return 0 if success else 1
