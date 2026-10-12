"""Change captions for adjacent frame pairs of the OpenVid sequences through an OpenAI-compatible
chat API (the released captions used Qwen3-VL-30B-A3B-Instruct with the prompt below).

  export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=...
  python data_preparation/openvid/caption_sequences.py --root data --model Qwen/Qwen3-VL-30B-A3B-Instruct --workers 32

Writes <root>/openvid_raw/outputs/captions.jsonl (one line per pair, resumable) and captions.errors.jsonl.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import add_root_arg, data_root  # noqa: E402

PROMPT_VERSION = "literal_v3"
NO_CHANGE_CAPTION = "No discernible change is visible."
SYSTEM_PROMPT = ("You compare two video frames and write a literal change caption. Report only directly observable "
                 "visual differences. Never infer speech, intention, emotion, cause, identity, or events outside the images.")
USER_PROMPT = """The following images are two chronological video frames approximately half a second apart. The first image is before the second. Most pairs come from one shot, but a small minority may cross a hard scene cut.

Write one compact caption describing only what visibly changes from the first image to the second.

Rules:
- Prioritize movement, position, pose, expression, and state changes of visible people and objects.
- Do not describe anything that remains unchanged.
- Do not mention the camera, framing, or background unless a global shift of the whole view is the main visible change.
- Do not infer speech, intentions, emotions, causes, relationships, or future actions.
- Avoid speculative phrases such as appears, seems, suggests, as if, likely, possibly, indicating, or implying.
- Never quote or transcribe visible text, subtitles, labels, logos, hashtags, dates, counters, timers, or numbers. If such an element changes, describe the change generically in English, such as a subtitle changes, a banner disappears, or a countdown decreases.
- If the images clearly show unrelated scenes, begin with: A scene cut changes the view from
- Look carefully for small but reliable changes in the mouth, eyes, head, hands, fingers, limbs, objects, lighting, or focus. Do not overlook a visible change merely because it is subtle.
- Prefer one sentence and fewer than forty words. Never exceed seventy words.
- Use only English words and ordinary English punctuation.
- Do not output digits, timestamps, frame labels, headings, bullets, JSON, or Markdown.
- Return only the caption."""
REPAIR_SYSTEM_PROMPT = ("You are a strict English caption editor. Rewrite the candidate without adding new visual "
                        "information. Return only the corrected caption.")
REPAIR_USER_PROMPT = """Rewrite the candidate caption so it satisfies every rule below while preserving its directly observable visual change.

- Use at most seventy English words.
- Use only English letters, spaces, apostrophes, hyphens, and ordinary sentence punctuation.
- Remove every digit and every non-English character.
- Never quote or transcribe visible text, labels, logos, hashtags, dates, counters, timers, or numbers.
- Generalize written-content changes. For example, say a subtitle changes, a banner disappears, or a countdown decreases.
- Remove speculation.
- Delete every complete sentence or clause that says something remains, stays, is unchanged, is stationary, is still, is the same, or shows no visible change.
- If deleting those clauses leaves no reliable visible change, return exactly: No discernible change is visible.
- Return only the corrected caption, without quotation marks, JSON, Markdown, or explanation.

Candidate caption:
"""
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
ALLOWED_RE = re.compile(r"^[A-Za-z\s.,;:!?()'\-]+$")
BANNED_RE = re.compile(
    r"\b(?:appears? to|seems?|suggests?|as if|likely|possibly|indicat(?:e|es|ing)|implying|unchanged|remains?|stays?|"
    r"stationary|maintain(?:s|ed|ing)?|the same|consistent|speaks?|speaking|talks?|talking|says?|saying|"
    r"no (?:visible )?change|no (?:camera )?movement|camera (?:position )?(?:is )?(?:unchanged|stationary))\b", re.I)
PUNCT = str.maketrans({" ": " ", "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


class Invalid(ValueError):
    def __init__(self, msg, caption=""):
        super().__init__(msg)
        self.caption = caption


def normalize(raw):
    c = raw.translate(PUNCT).strip()
    if c.startswith("```") and c.endswith("```"):
        lines = c.splitlines()
        if len(lines) >= 3:
            c = "\n".join(lines[1:-1]).strip()
    if len(c) >= 2 and c[0] == c[-1] and c[0] in {'"', "'"}:
        c = c[1:-1].strip()
    return " ".join(c.split())


def validate(c):
    if not c:
        raise Invalid("the caption was empty", c)
    if not ALLOWED_RE.fullmatch(c):
        raise Invalid("it contained digits, non-English characters, or unsupported punctuation", c)
    words = WORD_RE.findall(c)
    if not words:
        raise Invalid("it contained no English words", c)
    if len(words) > 70:
        raise Invalid(f"it contained {len(words)} words, exceeding seventy", c)
    if c != NO_CHANGE_CAPTION and BANNED_RE.search(c):
        raise Invalid("it contained speculation or described unchanged content", c)
    return len(words)


def data_uri(path, max_dim, quality):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS, reducing_gap=3.0)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class RateLimiter:
    def __init__(self, per_sec):
        self.n, self.q, self.lock = per_sec, deque(), threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.q and now - self.q[0] >= 1.01:
                    self.q.popleft()
                if len(self.q) < self.n:
                    self.q.append(now)
                    return
                wait_s = 1.01 - (now - self.q[0])
            time.sleep(max(wait_s, 0.001))


def caption_pair(client, limiter, args, before, after):
    content = [{"type": "text", "text": USER_PROMPT}, {"type": "text", "text": "First image, before."},
               {"type": "image_url", "image_url": {"url": data_uri(before, args.max_dimension, args.jpeg_quality)}},
               {"type": "text", "text": "Second image, after."},
               {"type": "image_url", "image_url": {"url": data_uri(after, args.max_dimension, args.jpeg_quality)}}]
    feedback, no_change_retry, attempts, last = None, 0, 0, None
    for attempt in range(1, args.max_retries + 2):
        msg = list(content)
        if feedback:
            msg[0] = {"type": "text", "text": USER_PROMPT + f"\n\nYour prior answer was invalid: {feedback} Please correct it."}
        limiter.acquire()
        attempts += 1
        try:
            out = client.chat.completions.create(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                                                 messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}])
            cap = normalize(out.choices[0].message.content or "")
            if cap == NO_CHANGE_CAPTION and no_change_retry < 1 and attempt <= args.max_retries:
                no_change_retry += 1
                feedback = ("you returned the no-change fallback too quickly. Inspect the two images again for subtle but "
                            "reliable mouth, eye, head, hand, object, lighting, or focus changes. Use the fallback only if none exists.")
                continue
            return cap, validate(cap), attempts
        except Invalid as exc:
            last, feedback = exc, str(exc)
            limiter.acquire()
            attempts += 1
            try:
                rep = client.chat.completions.create(model=args.model, temperature=0, max_tokens=args.max_tokens,
                                                     messages=[{"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                                                               {"role": "user", "content": REPAIR_USER_PROMPT + "\n" + exc.caption}])
                cap = normalize(rep.choices[0].message.content or "")
                return cap, validate(cap), attempts
            except Invalid as rexc:
                last, feedback = rexc, str(rexc)
        except Exception as exc:
            last = exc
            status = getattr(exc, "status_code", None)
            if (status is not None and status not in (408, 409, 429) and status < 500) or attempt > args.max_retries:
                raise
        if attempt <= args.max_retries:
            time.sleep(random.uniform(0.5, min(2 ** attempt, 60)))
    raise last


def main():
    ap = argparse.ArgumentParser()
    add_root_arg(ap)
    ap.add_argument("--openvid-root", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--requests-per-second", type=int, default=30)
    ap.add_argument("--max-dimension", type=int, default=448)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="pairs to process after resume filtering")
    ap.add_argument("--dry-run", action="store_true", help="encode images, no API calls")
    args = ap.parse_args()
    root = args.openvid_root or os.path.join(data_root(args.root), "openvid_raw")
    manifest = os.path.join(root, "sequences.jsonl")
    out_path = os.path.join(root, "outputs", "captions.jsonl")
    err_path = os.path.join(root, "outputs", "captions.errors.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("caption"):
                        done.add(r["pair_id"])
                except json.JSONDecodeError:
                    pass

    def pairs():
        n = 0
        with open(manifest) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                s = json.loads(line)
                frames = s["frames"]
                for u in range(len(frames) - 1):
                    pid = f"{s['id']}:{u:02d}"
                    if pid in done:
                        continue
                    if args.limit is not None and n >= args.limit:
                        return
                    n += 1
                    yield pid, idx, s, u

    if args.dry_run:
        k = 0
        for pid, idx, s, u in pairs():
            data_uri(os.path.join(root, s["frames"][u]), args.max_dimension, args.jpeg_quality)
            data_uri(os.path.join(root, s["frames"][u + 1]), args.max_dimension, args.jpeg_quality)
            k += 1
        print(f"dry run: {k} pairs encoded, 0 API requests")
        return

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set OPENAI_API_KEY (and OPENAI_BASE_URL or --base-url)")
    local = threading.local()

    def client():
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0)
        return c

    limiter = RateLimiter(args.requests_per_second)

    def work(item):
        pid, idx, s, u = item
        cap, wc, attempts = caption_pair(client(), limiter, args, os.path.join(root, s["frames"][u]), os.path.join(root, s["frames"][u + 1]))
        return {"pair_id": pid, "sequence_id": s["id"], "input_index": idx, "pair_index": u, "before_frame_index": u,
                "after_frame_index": u + 1, "before_frame": s["frames"][u], "after_frame": s["frames"][u + 1],
                "caption": cap, "word_count": wc, "model": args.model, "prompt_version": PROMPT_VERSION,
                "source_caption_used": False, "attempts": attempts, "source_video": s.get("video")}

    ok = failed = 0
    with open(out_path, "a") as fo, open(err_path, "a") as fe, ThreadPoolExecutor(args.workers) as ex:
        pending = {}

        def drain():
            nonlocal ok, failed
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                item = pending.pop(fut)
                try:
                    rec = fut.result()
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok += 1
                except Exception as exc:
                    fe.write(json.dumps({"pair_id": item[0], "sequence_id": item[2]["id"], "pair_index": item[3],
                                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                                         "invalid_caption": getattr(exc, "caption", "")}, ensure_ascii=False) + "\n")
                    failed += 1
                fo.flush()
                fe.flush()
                if (ok + failed) % 1000 == 0:
                    print(f"finished={ok + failed} ok={ok} failed={failed}", flush=True)

        for item in pairs():
            pending[ex.submit(work, item)] = item
            if len(pending) >= args.workers * 2:
                drain()
        while pending:
            drain()
    print(f"captioning complete: ok={ok} failed={failed} -> {out_path}")


if __name__ == "__main__":
    main()
