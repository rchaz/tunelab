"""Fake openai SDK (Responses API surface) for offline tests of
distill_generate.py / judge_eval.py — same conventions as tests/shims/anthropic.py.

Shadows the real `openai` package when tests/shims is on PYTHONPATH
(PYTHONPATH entries precede site-packages in sys.path). Deterministic and
input-aware:

  - input containing "REFUSE_ME"   -> message item with a refusal content
                                      part (refusal="(refusing)"), no text
  - input containing "ERROR_ME"    -> raises openai.APIError
  - input containing "TRUNCATE_ME" -> status="incomplete",
                                      incomplete_details.reason=
                                      "max_output_tokens", no output text
  - input containing "NOUSAGE_ME"  -> normal response but usage=None
                                      (Response.usage is Optional in the SDK)
  - input containing "BANANA_ME"   -> judge verdict with out-of-enum
                                      winner "banana" (defense-in-depth path)
  - classify format (label schema) -> {"label": <enum[crc32(input) % len]>}
  - judge format (winner schema)   -> {"winner": "first", "reason": "shim"}
  - no text.format (generate)      -> output_text "GEN::" + input

Mirrors the real API's requirement that a json_schema text format carry a
format-level "name" (400 -> APIError here). Every successful response reports
usage input_tokens=10, output_tokens=5. Set SHIM_DEBUG=1 to print
"SHIM max_output_tokens=<n> store=<v> effort=<e>" per call to stderr.
"""

import json
import os
import sys
import zlib
from types import SimpleNamespace


class APIError(Exception):
    pass


def _part(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _response(parts, text, status="completed", reason=None, usage=True):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", content=parts)],
        output_text=text,
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5) if usage else None,
    )


class _Responses:
    def create(self, **kwargs):
        if os.environ.get("SHIM_DEBUG"):
            effort = (kwargs.get("reasoning") or {}).get("effort")
            print(
                f"SHIM max_output_tokens={kwargs.get('max_output_tokens')} "
                f"store={kwargs.get('store')} effort={effort}",
                file=sys.stderr,
            )
        text = kwargs["input"]
        if "ERROR_ME" in text:
            raise APIError("shim-injected api error")
        if "REFUSE_ME" in text:
            return _response([_part("refusal", refusal="(refusing)")], "")
        if "TRUNCATE_ME" in text:
            return _response([], "", status="incomplete", reason="max_output_tokens")

        fmt = (kwargs.get("text") or {}).get("format") or {}
        if fmt and (fmt.get("type") != "json_schema" or not fmt.get("name")):
            raise APIError("shim: text.format requires type=json_schema and a name")
        props = (fmt.get("schema") or {}).get("properties", {})
        if "label" in props:
            enum = props["label"]["enum"]
            body = json.dumps({"label": enum[zlib.crc32(text.encode("utf-8")) % len(enum)]})
        elif "winner" in props:
            winner = "banana" if "BANANA_ME" in text else "first"
            body = json.dumps({"winner": winner, "reason": "shim"})
        else:
            body = "GEN::" + text
        return _response([_part("output_text", text=body)], body,
                         usage="NOUSAGE_ME" not in text)


class OpenAI:
    def __init__(self, **kwargs):
        self.responses = _Responses()
