#!/usr/bin/env python3
"""Chunk raw domain text/markdown into {"text": ...} JSONL for continued pretraining. Stdlib only.

  python3 chunk_text.py --input corpus/ extra_notes.md --output chunks.jsonl \
      [--target-tokens 2000] [--max-tokens 4000] [--min-tokens 50]

--input takes files and/or directories (directories recurse for .txt/.md;
explicitly named files are taken whatever the extension). Tokens are
approximated as chars/4 throughout.

CPT wants 1-4k-token chunks broken on natural boundaries:
splits happen at markdown headings first, then blank-line paragraphs; a
single paragraph is only hard-split when it alone exceeds --max-tokens.
Adjacent small pieces merge up toward --target-tokens, and leftover
fragments under --min-tokens are dropped. Merging flushes only after
crossing target, so routine chunks land between --target and --max
tokens (~2x target is common, not near target) — at the defaults that
exceeds mlx_lm's --max-seq-length default of 2048; raise it when
training. Each record carries a "source" key (relative path) — mlx-lm
ignores unknown keys, and it keeps the DATACARD traceable back to the
original files.

Dedupe afterwards is mandatory (repetition is the dominant memorization
cause): run dedupe.py on the output before splitting.
"""

import argparse
import json
import os
import re
import statistics
import sys

HEADING_RE = re.compile(r"^#{1,6} ")
FENCE_RE = re.compile(r"^(```|~~~)")


def split_sections(text):
    # Markdown headings are the primary boundary. Fence state is tracked so a
    # '# comment' line inside a code block doesn't masquerade as a heading.
    sections, cur, in_fence = [], [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and HEADING_RE.match(line) and any(l.strip() for l in cur):
            sections.append("\n".join(cur))
            cur = []
        cur.append(line)
    if any(l.strip() for l in cur):
        sections.append("\n".join(cur))
    return sections


def split_paragraphs(section):
    # Blank lines are the secondary boundary — except inside code fences,
    # where a blank line is content, not a break.
    paras, cur, in_fence = [], [], False
    for line in section.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if cur:
                paras.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        paras.append("\n".join(cur))
    return paras


def hard_split(para, target_chars, max_chars):
    # Last resort for a single paragraph over --max-tokens: cut near target,
    # preferring a newline then a space so we never split mid-word.
    pieces = []
    while len(para) > max_chars:
        cut = para.rfind("\n", target_chars // 2, target_chars)
        if cut == -1:
            cut = para.rfind(" ", target_chars // 2, target_chars)
        if cut == -1:
            # No boundary in the window (minified text): cut cold, keep all chars.
            pieces.append(para[:target_chars])
            para = para[target_chars:]
            continue
        pieces.append(para[:cut])
        # Drop only the matched separator char — lstrip() here would eat the
        # next line's indentation, and CPT trains on the mangled text verbatim.
        para = para[cut + 1:]
    if para:
        pieces.append(para)
    return pieces


def file_pieces(text, target_chars, max_chars):
    pieces = []
    for section in split_sections(text):
        if len(section) <= max_chars:
            pieces.append(section)
            continue
        for para in split_paragraphs(section):
            if len(para) > max_chars:
                pieces.extend(hard_split(para, target_chars, max_chars))
            else:
                pieces.append(para)
    return pieces


def merge_pieces(pieces, target_chars, max_chars):
    chunks, cur, cur_len = [], [], 0
    for p in pieces:
        if cur and cur_len + 2 + len(p) > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += (2 if cur_len else 0) + len(p)
        if cur_len >= target_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def collect_inputs(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            found = []
            for root, dirs, names in os.walk(inp):
                dirs.sort()
                for name in sorted(names):
                    if name.lower().endswith((".txt", ".md")):
                        found.append(os.path.join(root, name))
            if not found:
                print(f"warning: no .txt/.md files under {inp}", file=sys.stderr)
            files.extend(found)
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            sys.exit(f"--input {inp}: not found")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True, help="files and/or directories (dirs recurse for .txt/.md)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--min-tokens", type=int, default=50)
    args = ap.parse_args()

    if not 0 < args.min_tokens <= args.target_tokens <= args.max_tokens:
        sys.exit("require 0 < --min-tokens <= --target-tokens <= --max-tokens")
    target_chars = args.target_tokens * 4
    max_chars = args.max_tokens * 4
    min_chars = args.min_tokens * 4

    files = collect_inputs(args.input)
    if not files:
        sys.exit("no input files found")

    records, dropped = [], 0
    for path in files:
        source = os.path.relpath(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        pieces = file_pieces(text, target_chars, max_chars)
        chunks = merge_pieces(pieces, target_chars, max_chars)
        kept = [c for c in chunks if len(c) >= min_chars]
        dropped += len(chunks) - len(kept)
        for c in kept:
            records.append({"text": c, "source": source})
        note = f" ({len(chunks) - len(kept)} dropped)" if len(chunks) > len(kept) else ""
        print(f"{source}: {len(kept)} chunks{note}", file=sys.stderr)

    if not records:
        sys.exit(f"no chunks produced (inputs empty or everything under --min-tokens {args.min_tokens})")

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sizes = [len(r["text"]) // 4 for r in records]
    print(
        f"total: {len(records)} chunks from {len(files)} files -> {args.output}"
        f" ({dropped} sub-{args.min_tokens}-token fragments dropped)",
        file=sys.stderr,
    )
    print(
        f"chunk tokens (~chars/4): min {min(sizes)} / median {int(statistics.median(sizes))} / max {max(sizes)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
