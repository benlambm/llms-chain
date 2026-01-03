# =============================================================================
# LMS Content Generator - Configuration File
# =============================================================================
# This file contains customizable settings for the prompt chain.
# To use: Rename to 'config.py' and import in the main script.
# =============================================================================

from typing import Final

# -----------------------------------------------------------------------------
# STYLING CONFIGURATION
# -----------------------------------------------------------------------------

# Font stack used throughout all generated content
FONT_STACK: Final[str] = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Primary color scheme (Bauhaus-inspired minimalist black)
PRIMARY_COLOR: Final[str] = "#000000"
SECONDARY_COLOR: Final[str] = "#E0E0E0"
BACKGROUND_ACCENT: Final[str] = "#F5F5F5"

# Typography settings
BODY_FONT_SIZE: Final[str] = "18px"
LINE_HEIGHT: Final[str] = "1.8"
MAX_WIDTH: Final[str] = "75ch"

# -----------------------------------------------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------------------------------------------

# Available Claude models (as of 2025):
# - claude-opus-4-5-20251101     (Most capable, highest cost)
# - claude-sonnet-4-20250514     (Balanced capability/cost) 
# - claude-haiku-4-5-20251001    (Fastest, lowest cost)

# Model assignments per agent
AGENT_A_MODEL: Final[str] = "claude-sonnet-4-20250514"  # Writer needs creativity
AGENT_B_MODEL: Final[str] = "claude-sonnet-4-20250514"  # Structurer needs precision
AGENT_C_MODEL: Final[str] = "claude-sonnet-4-20250514"  # Publisher needs precision

# Temperature settings (0.0 = deterministic, 1.0 = creative)
AGENT_A_TEMPERATURE: Final[float] = 1.0    # Creative writing
AGENT_B_TEMPERATURE: Final[float] = 0.2    # Precise conversion
AGENT_C_TEMPERATURE: Final[float] = 0.2    # Precise styling

# Token limits
AGENT_A_MAX_TOKENS: Final[int] = 12000
AGENT_B_MAX_TOKENS: Final[int] = 14000
AGENT_C_MAX_TOKENS: Final[int] = 16384

# -----------------------------------------------------------------------------
# CONTENT LENGTH TARGETS
# -----------------------------------------------------------------------------

TEXTBOOK_WORD_COUNT: Final[str] = "3,000-3,500"
DISCUSSION_WORD_COUNT: Final[str] = "400-600"
ASSIGNMENT_WORD_COUNT: Final[str] = "500-800"

# -----------------------------------------------------------------------------
# FONT AWESOME ICONS (for Discussion/Assignment pages)
# -----------------------------------------------------------------------------

# Icon definitions - change these to customize the visual style
FA_ICON_WRITE: Final[str] = '<i class="fas fa-pencil-alt" style="padding-right: 10px;" aria-hidden="true"></i>'
FA_ICON_RESPOND: Final[str] = '<i class="fas fa-comments" style="padding-right: 10px;" aria-hidden="true"></i>'
FA_ICON_EVALUATE: Final[str] = '<i class="fas fa-check" style="padding-right: 10px;" aria-hidden="true"></i>'

# Alternative icon suggestions:
# - Write: fa-pen, fa-edit, fa-keyboard
# - Respond: fa-reply, fa-share, fa-message
# - Evaluate: fa-star, fa-clipboard-check, fa-tasks

# -----------------------------------------------------------------------------
# KNOWLEDGE CHECK STYLING
# -----------------------------------------------------------------------------

KNOWLEDGE_CHECK_PREFIX: Final[str] = "✦ KNOWLEDGE CHECK:"
KNOWLEDGE_CHECK_BG: Final[str] = "#F5F5F5"
KNOWLEDGE_CHECK_BORDER: Final[str] = "3px solid #000000"

# -----------------------------------------------------------------------------
# OUTPUT FILE NAMING
# -----------------------------------------------------------------------------

# Suffixes for output files
STEP1_SUFFIX: Final[str] = "_Step1_Content.txt"
STEP2_SUFFIX: Final[str] = "_Step2_Structured.html"
STEP3_SUFFIX: Final[str] = "_Step3_Final.html"

# Prefix for convenience copies
CONVENIENCE_PREFIX: Final[str] = "anthropic_"

# -----------------------------------------------------------------------------
# ACCESSIBILITY SETTINGS
# -----------------------------------------------------------------------------

# Minimum contrast ratio for WCAG 2.1 AA is 4.5:1 for normal text
# Black (#000000) on white has contrast ratio of 21:1 (exceeds requirements)

# These settings help ensure accessibility:
HEADING_LETTER_SPACING: Final[str] = "0.05em"
BODY_LETTER_SPACING: Final[str] = "0.02em"
BODY_WORD_SPACING: Final[str] = "0.05em"
