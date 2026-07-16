import json
import re
import base64
import os
import time
from typing import Optional, List, Tuple

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

from config.settings import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL

_RETRYABLE = (RateLimitError, APIConnectionError)
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds


class LLMService:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or OPENAI_MODEL
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY, base_url=OPENAI_API_BASE or None)

    def _build_messages(self, prompt: str, images: Optional[List[str]] = None) -> list:
        if images:
            content: list = [{"type": "text", "text": prompt}]
            for img_path in images:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            return [{"role": "user", "content": content}]
        return [{"role": "user", "content": prompt}]

    def _make_api_call(self, messages: list) -> Tuple[str, int, int]:
        """Returns (content, input_tokens, output_tokens)."""
        call_type = "vision" if isinstance(messages[0]["content"], list) else "text"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                print(f"[OpenAI] Calling {self.model} ({call_type})...", flush=True)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                print(f"[OpenAI] Done ({self.model})", flush=True)
                content = response.choices[0].message.content or ""
                usage = response.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                return content, in_tok, out_tok

            except _RETRYABLE as e:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY * attempt
                    print(f"[OpenAI] {type(e).__name__} — retrying in {wait}s (attempt {attempt}/{_MAX_RETRIES})", flush=True)
                    time.sleep(wait)
                else:
                    print(f"[OpenAI] Failed after {_MAX_RETRIES} attempts: {e}", flush=True)
                    raise

            except APIStatusError as e:
                print(f"[OpenAI] API error {e.status_code}: {e.message}", flush=True)
                raise

            except Exception as e:
                print(f"[OpenAI] Unexpected error: {e}", flush=True)
                raise

        return "", 0, 0

    def generate(self, prompt: str, images: Optional[List[str]] = None) -> str:
        messages = self._build_messages(prompt, images)
        content, _, _ = self._make_api_call(messages)
        return content

    def generate_tracked(self, prompt: str, images: Optional[List[str]] = None) -> Tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        messages = self._build_messages(prompt, images)
        return self._make_api_call(messages)

    def generate_json(self, prompt: str, images: Optional[List[str]] = None) -> dict:
        full_prompt = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown code blocks, no explanation, just the JSON object."
        )
        messages = self._build_messages(full_prompt, images)
        content, _, _ = self._make_api_call(messages)
        return self._parse_json(content)

    def generate_json_tracked(
        self, prompt: str, images: Optional[List[str]] = None
    ) -> Tuple[dict, str, int, int]:
        """Returns (parsed_dict, raw_text, input_tokens, output_tokens)."""
        full_prompt = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown code blocks, no explanation, just the JSON object."
        )
        messages = self._build_messages(full_prompt, images)
        content, in_tok, out_tok = self._make_api_call(messages)
        return self._parse_json(content), content, in_tok, out_tok

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}