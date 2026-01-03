# LMS Content Pipeline

A powerful prompt-chaining pipeline that generates Canvas LMS-ready educational content with AI-generated illustrations. Built with Claude (Anthropic), Gemini, and ImageKit.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Four-Agent Pipeline**: Specialized AI agents handle writing, structuring, styling, and illustration
- **Three Content Types**: Textbook chapters, discussion prompts, and assignment instructions
- **Accessibility First**: WCAG 2.1 AA compliant output with proper semantic structure
- **AI Illustrations**: Automatic image generation and hosting for textbook chapters
- **Knowledge Checks**: Interactive expandable Q&A blocks for textbook content
- **Canvas LMS Ready**: Inline-styled HTML that works perfectly in Canvas Pages

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LMS Content Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [Topic] ──► Agent A ──► Agent B ──► Agent C ──► Agent D       │
│               Writer     Structurer   Publisher   Illustrator    │
│               (Claude)   (Claude)     (Claude)    (Claude+Gemini)│
│                  │           │            │            │         │
│                  ▼           ▼            ▼            ▼         │
│              .txt        .html        .html        .html         │
│            (prose)    (semantic)   (styled)    (+ images)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Role | Model | Output |
|-------|------|-------|--------|
| **A** | Content Writer | Claude Sonnet | Educational prose with proper pedagogy |
| **B** | HTML Structurer | Claude Sonnet | Semantic HTML without styling |
| **C** | Publisher/Styler | Claude Sonnet | Accessible, styled HTML + Knowledge Checks |
| **D** | Illustrator | Claude + Gemini | AI-generated images embedded in HTML |

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- API keys for Anthropic, Google (Gemini), and ImageKit

### Install Dependencies

```bash
pip install anthropic google-genai imagekitio
```

Or with uv:

```bash
uv pip install anthropic google-genai imagekitio
```

### Environment Variables

Create a `.env` file or export these variables:

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."

# Required for image generation (Agent D)
export GEMINI_API_KEY="..."          # or GOOGLE_API_KEY
export IMAGEKIT_PRIVATE_KEY="..."

# Optional
export IMAGEKIT_FOLDER="/lms-content/"  # Default folder for hosted images
```

## 🚀 Quick Start

### Interactive Mode

```bash
python lms_pipeline.py
```

You'll see a menu to select content type and enter your topic:

```
============================================================
  ✨ LMS CONTENT PIPELINE - Unified Edition ✨
============================================================

What type of LMS content would you like to generate?

  [1] 📚 Textbook Chapter
      Full chapter with Knowledge Checks + AI illustrations

  [2] 💬 Discussion Prompt
      Write/Respond/Evaluate sections with icons

  [3] 📋 Assignment Instructions
      Overview/Instructions/Deliverables/Rubric with icons

Enter your choice (1-3): 1

Enter the topic for your textbook chapter: Vector Databases and Semantic Search
```

### Command Line Mode

```bash
# Generate a textbook chapter with images
python lms_pipeline.py --type textbook --topic "Introduction to Neural Networks"

# Generate a discussion prompt (no images by default)
python lms_pipeline.py --type discussion --topic "Ethics of AI in Healthcare"

# Generate an assignment
python lms_pipeline.py -c assignment -t "Build a REST API with Flask"

# Skip image generation
python lms_pipeline.py --type textbook --topic "Cloud Computing" --no-images

# Force images on non-textbook content
python lms_pipeline.py --type discussion --topic "AI Ethics" --force-images
```

## 📁 Output Structure

Each pipeline run creates a timestamped directory:

```
lms_output_20251224-101530/
├── content/
│   ├── Step1_Content.txt       # Agent A: Raw educational prose
│   ├── Step2_Structured.html   # Agent B: Semantic HTML
│   ├── Step3_Styled.html       # Agent C: Styled + Knowledge Checks
│   └── Step4_Final.html        # Agent D: With embedded images
├── images/
│   ├── fig1-20251224-101530.png
│   ├── fig2-20251224-101530.png
│   └── fig3-20251224-101530.png
└── pipeline.log                # Execution log with timestamps
```

A convenience copy of the final HTML is also saved to your current directory:
```
./lms_output_20251224-101530.html
```

## 📚 Content Types

### Textbook Chapter

Full educational chapters including:
- **Learning Objectives**: 2-3 measurable goals with action verbs
- **Core Content**: 5-7 sections with definitions, context, examples
- **Knowledge Checks**: Interactive Q&A blocks (expandable `<details>` elements)
- **Summary**: Key takeaways
- **Glossary**: Definition list of key terms
- **AI Illustrations**: Up to 3 contextual images (optional)

### Discussion Prompt

Interactive discussion board content with Font Awesome icons:
- **Write/Prompt**: Main discussion question or scenario
- **Respond/Reply**: Peer engagement instructions
- **Evaluation/Rubric**: Grading criteria
- **Key Terms**: (if technical vocabulary is introduced)

### Assignment Instructions

Step-based assignment documentation:
- **Overview**: Purpose and learning goals
- **Instructions**: Step-by-step directions
- **Deliverables**: Submission requirements
- **Evaluation/Rubric**: Point allocations and criteria
- **Key Terms**: (if technical vocabulary is introduced)

## 🎨 Styling

All output uses inline CSS for maximum LMS compatibility:

- **Typography**: System font stack (`system-ui, -apple-system, 'Segoe UI', sans-serif`)
- **Readability**: 18px body text, 1.8 line height, max-width 75ch
- **Hierarchy**: Bold headers with decorative borders
- **Accessibility**: High contrast (#000000 on white), semantic HTML
- **Icons**: Font Awesome 5 for Discussion/Assignment pages

### Font Awesome Requirement

For Discussion and Assignment pages, ensure your LMS theme includes Font Awesome:

```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
```

## 🖼️ Image Generation

Agent D handles intelligent image placement:

1. **Brainstorming**: Claude analyzes content and identifies up to 3 optimal insertion points
2. **Generation**: Gemini creates educational illustrations
3. **Hosting**: Images are uploaded to ImageKit CDN
4. **Injection**: Images are embedded with proper `<figure>`, `<img>`, and `<figcaption>` markup

### Image Features

- Automatic alt text for accessibility
- Lazy loading for performance
- Responsive sizing (100% width, auto height)
- Styled captions with "(Generated by AI)" attribution
- Local backups saved to `images/` directory

### Disabling Images

```bash
# Skip image generation entirely
python lms_pipeline.py --no-images

# Images are automatically skipped for Discussion/Assignment
# unless you force them:
python lms_pipeline.py --type discussion --force-images
```

## 🔧 Advanced Usage

### Using Individual Scripts

The pipeline is also available as separate scripts for modularity:

```bash
# Generate content only (Agents A-C)
python anthropic_lms_chain.py

# Add images to existing HTML (Agent D only)
python lms_image_enhancer.py input.html output.html
```

### Customizing Prompts

Edit the `AGENT_*_PROMPTS` dictionaries in `lms_pipeline.py` to customize:
- Writing style and tone
- Section requirements
- Length guidelines
- Formatting rules

### Customizing Styles

Modify the `SHARED_STYLES` dictionary to change:
- Font families
- Colors and borders
- Spacing and margins
- Knowledge Check appearance

## 🐛 Troubleshooting

### "Context not found for Figure X"

The image insertion uses regex matching that handles HTML entities. If you still see this error:
- Check if the insertion context contains unusual Unicode characters
- Verify the context text actually exists in the HTML
- The pattern debug output shows what's being searched

### "Missing environment variables"

Ensure all required API keys are set:
```bash
echo $ANTHROPIC_API_KEY
echo $GEMINI_API_KEY
echo $IMAGEKIT_PRIVATE_KEY
```

### "Image generation unavailable"

Install the optional dependencies:
```bash
pip install google-genai imagekitio
```

### Canvas LMS Display Issues

- Ensure you're pasting into the HTML editor, not the rich text editor
- Check that Font Awesome CSS is loaded in your theme
- Verify no conflicting CSS in your Canvas theme

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Anthropic Claude**: Content writing, structuring, styling, and image brainstorming
- **Google Gemini**: Image generation
- **ImageKit**: Image hosting CDN
- **Font Awesome**: Icons for Discussion/Assignment pages

## 📬 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

Built with ❤️ by Claude (Anthropic) for educators everywhere.
