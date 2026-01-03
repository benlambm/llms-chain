from __future__ import annotations

import hashlib
import html
import http.server
import json
import os
import re
import sys
import threading
import webbrowser
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

from .config import IMAGE_MODEL, TEXT_MODEL_AGENT_A, TEXT_MODEL_AGENT_BCD, PipelineState


VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class TagBalanceChecker(HTMLParser):
    """Basic tag balance checker to catch obvious HTML issues."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: List[str] = []
        self.issues: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in VOID_TAGS:
            return
        if tag == "p" and self.stack and self.stack[-1] == "p":
            self.stack.pop()
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.issues.append(f"Unexpected closing tag </{tag}>.")
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
        else:
            self.issues.append(f"Unmatched closing tag </{tag}>.")

    def close(self) -> None:
        super().close()
        if self.stack:
            leftover = ", ".join(self.stack[-5:])
            self.issues.append(f"Unclosed tags remain: {leftover}.")


def validate_environment(need_images: bool) -> tuple[bool, List[str]]:
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


def load_prompt(path: Path) -> str:
    """Load a prompt from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    """Normalize whitespace and entities for matching."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def validate_html_fragment(html_content: str) -> List[str]:
    """Return a list of HTML sanity issues (empty if none)."""
    checker = TagBalanceChecker()
    checker.feed(html_content)
    checker.close()
    return checker.issues


def _hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_prompt_metadata(prompt_paths: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for key, path in prompt_paths.items():
        digest = _hash_file(path)
        if digest:
            metadata[key] = {"path": str(path), "sha256": digest}
    return metadata


def build_run_manifest(
    state: PipelineState,
    final_path: Optional[Path],
    prompt_paths: Dict[str, Path],
) -> Dict[str, object]:
    openai_version = None
    try:
        openai_version = metadata.version("openai")
    except metadata.PackageNotFoundError:
        openai_version = None

    step_output_paths = {}
    if "agent_a" in state.step_outputs:
        step_output_paths["agent_a"] = str(state.config.content_dir / "Step1_Content.txt")
    if "agent_b" in state.step_outputs:
        step_output_paths["agent_b"] = str(state.config.content_dir / "Step2_Structured.html")
    if "agent_c" in state.step_outputs:
        step_output_paths["agent_c"] = str(state.config.content_dir / "Step3_Styled.html")
    if "agent_d" in state.step_outputs:
        step_output_paths["agent_d"] = str(state.config.content_dir / "Step4_Final.html")

    manifest = {
        "timestamp": state.config.timestamp,
        "content_type": state.config.content_type.value,
        "topic": state.config.topic,
        "output_dir": str(state.config.output_dir),
        "content_dir": str(state.config.content_dir),
        "images_dir": str(state.config.images_dir),
        "enable_images": state.config.enable_images,
        "imagekit_folder": state.config.imagekit_folder,
        "input_file": str(state.config.input_file) if state.config.input_file else None,
        "pipeline_log": str(state.config.output_dir / "pipeline.log"),
        "models": {
            "agent_a": TEXT_MODEL_AGENT_A,
            "agent_bcd": TEXT_MODEL_AGENT_BCD,
            "image": IMAGE_MODEL,
        },
        "prompts": collect_prompt_metadata(prompt_paths),
        "step_outputs": step_output_paths,
        "final_output": str(final_path) if final_path else None,
        "images": [
            {
                "figure_number": plan.figure_number,
                "insertion_context": plan.insertion_context,
                "image_prompt": plan.image_prompt,
                "alt_text": plan.alt_text,
                "caption": plan.caption,
                "local_path": str(plan.local_path) if plan.local_path else None,
                "hosted_url": plan.hosted_url,
            }
            for plan in state.image_plans
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "openai": openai_version,
        },
    }

    return manifest


def save_run_manifest(
    state: PipelineState,
    final_path: Optional[Path],
    prompt_paths: Dict[str, Path],
) -> Optional[Path]:
    """Save a manifest JSON describing the run."""
    manifest = build_run_manifest(state, final_path, prompt_paths)
    manifest_path = state.config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


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

    root_copy_dir = state.config.output_dir.parent
    root_copy = root_copy_dir / f"lms_output_{state.config.timestamp}.html"
    root_copy.write_text(final_html, encoding="utf-8")

    return output_path


def serve_final_output(output_dir: Path, final_path: Path) -> Optional[str]:
    """Serve the final HTML output and return the preview URL if available."""
    if not final_path.exists():
        return None

    try:
        handler = http.server.SimpleHTTPRequestHandler
        httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: handler(*args, directory=str(output_dir), **kwargs),
        )
    except OSError as e:
        print(f"\n⚠️  Preview server failed to start: {e}")
        return None

    host, port = httpd.server_address
    rel_path = final_path.relative_to(output_dir).as_posix()
    url = f"http://{host}:{port}/{rel_path}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    opened = False
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False

    if opened:
        print(f"\n🌐 Preview opened in your browser: {url}")
    else:
        print(f"\n🌐 Preview server running: {url}")

    if sys.stdin.isatty():
        try:
            input("Press Enter to stop the preview server...")
        except KeyboardInterrupt:
            pass
        httpd.shutdown()
        httpd.server_close()

    return url
