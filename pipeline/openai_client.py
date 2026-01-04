from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from openai import OpenAI


def clean_json_response(text: str) -> str:
    """Remove markdown code fences and fix common JSON issues."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return text.strip()


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
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)

        if isinstance(item, dict):
            item_text = item.get("text")
        else:
            item_text = getattr(item, "text", None)
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
                # Prefer the parsed field returned by Structured Outputs.
                if isinstance(block, dict):
                    json_payload = block.get("parsed") or block.get("json")
                else:
                    parsed_val = getattr(block, "parsed", None)
                    json_val = getattr(block, "json", None)
                    json_payload = parsed_val or json_val
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
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """Generate structured JSON via the OpenAI Responses API."""
    text_format: Dict[str, Any] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": schema,
        "strict": True,
    }
    if schema_description:
        text_format["description"] = schema_description

    responses_client = cast(Any, client.responses)
    response = responses_client.create(  # type: ignore[arg-type]
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
        text={"format": text_format},
    )
    parsed = _extract_json(response)
    raw_text = _extract_text(response)
    parse_error: Optional[str] = None
    if parsed is None and raw_text:
        try:
            json.loads(clean_json_response(raw_text))
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    return parsed, raw_text, parse_error
