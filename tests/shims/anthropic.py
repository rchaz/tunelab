"""Fake anthropic SDK for offline tests of distill_generate.py / judge_eval.py.

Shadows the real `anthropic` package when tests/shims is on PYTHONPATH
(PYTHONPATH entries precede site-packages in sys.path). Deterministic and
input-aware:

  - user text containing "REFUSE_ME"   -> response with NO text block,
                                          stop_reason="refusal" (thinking only)
  - user text containing "ERROR_ME"    -> raises anthropic.APIError
  - user text containing "BANANA_ME"   -> judge verdict with out-of-enum
                                          winner "banana" (defense-in-depth path)
  - classify request (schema w/ label) -> {"label": <enum[crc32(text) % len]>}
  - judge request (schema w/ winner)   -> {"winner": "first", "reason": "shim"}
  - generate request (no output_config)-> text "GEN::" + user text

Every successful response reports usage input_tokens=10, output_tokens=5.
Set SHIM_DEBUG=1 to print "SHIM max_tokens=<n>" per call to stderr.
"""

import json
import os
import sys
import zlib
from types import SimpleNamespace


class APIError(Exception):
    pass


def _block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _response(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class _Messages:
    def create(self, **kwargs):
        if os.environ.get("SHIM_DEBUG"):
            print(f"SHIM max_tokens={kwargs.get('max_tokens')}", file=sys.stderr)
        text = kwargs["messages"][0]["content"]
        if "ERROR_ME" in text:
            raise APIError("shim-injected api error")
        if "REFUSE_ME" in text:
            return _response([_block("thinking", thinking="(refusing)")], "refusal")

        schema = (kwargs.get("output_config") or {}).get("format", {}).get("schema", {})
        props = schema.get("properties", {})
        if "label" in props:
            enum = props["label"]["enum"]
            label = enum[zlib.crc32(text.encode("utf-8")) % len(enum)]
            body = json.dumps({"label": label})
        elif "winner" in props:
            winner = "banana" if "BANANA_ME" in text else "first"
            body = json.dumps({"winner": winner, "reason": "shim"})
        else:
            body = "GEN::" + text
        return _response([_block("text", text=body)], "end_turn")


class Anthropic:
    def __init__(self, **kwargs):
        self.messages = _Messages()
