from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional


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

# Paths
PROMPTS_DIR = Path("prompts")
DEFAULT_IMAGEKIT_FOLDER: Final[str] = "/lms-content/"

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
    input_file: Optional[Path] = None


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
