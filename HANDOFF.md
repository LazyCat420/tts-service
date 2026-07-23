# Handoff — tts-service

## ROOT CAUSE FOUND AND FIXED 2026-07-22 — onnxruntime mem-pattern

The 2026-07-20 "session doesn't survive repeated use" model below was
**incomplete**: failures are **text-deterministic**, not random decay. The office
line "Me wait. No buy now. No sell now. Wait for clear path." failed **6/6** even
with a fresh session per attempt (so the retry workaround still leaked 500s —
observed live 2026-07-22 18:53, both attempts failed, office fell back to silent
Web Speech).

Actual cause: **onnxruntime 1.18.1's memory-pattern optimisation corrupts
ScatterND indices on the 2nd+ run of a session.** piper's `synthesize()` runs one
inference **per sentence** on the same session, so any multi-sentence text hits
run 2+ inside a single request. (This is also why lowercase variants "passed" —
espeak sentence splitting differs.)

Fix (in `_load_voice_safe_session`): create the `InferenceSession` ourselves with
`sess_options.enable_mem_pattern = False` and wrap in
`PiperVoice(config=..., session=...)`. Measured in the prod container:
failing text 0/6 stock → **10/10** with mem-pattern off (arena setting
irrelevant). Retry + per-voice lock retained as backstops.

## Intermittent 500s (fixed by workaround 2026-07-20)

Symptom: the browser saw an unbroken wall of 503s from HTML-Notes' `/tts/synthesize`
proxy. That proxy was faithful — the upstream was genuinely returning 500:

```
[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while
running ScatterND node. Name:'/dp/flows.7/ScatterND_9'
Status Message: invalid indice found, indice = -4655178232744876906
```

The index is a float bit-pattern read as an int64, and the node varies
(`ScatterND`, `GatherElements`), always under `/dp/` — the VITS duration predictor.

### What it is NOT

- **Not bad input.** It fails on `"Hello there"`. `main.py` already strips emoji;
  that is unrelated.
- **Not the stochastic duration predictor.** Failures were exactly alternating,
  not random.
- **Not a missing `sid`.** I theorised piper feeds `{"sid": None}` for
  single-speaker voices and shipped `speaker_id=0`. **Wrong** — these graphs have
  no `sid` input at all; onnxruntime SKIPS a None value but rejects a real array
  with `Invalid input name: sid`. It took the failure rate from 50% to **100%**.
  Reverted. Do not re-derive this.

### What it is

**An ONNX session here does not reliably survive repeated use.** A freshly loaded
voice synthesises once; a later run on the same session can fail. Measured: 10/20
failures on identical sequential input, and exactly 3/6 on every voice tried.

The clean `200 500 200 500` alternation was an artifact of the evict-on-error
handler added in the same session: fail → evict → fresh load → 200 → reuse → 500.

### The workaround in place

1. **Retry once on a fresh session** (`synthesize`). The error path evicts the
   voice, so attempt 2 always loads clean. 20/20 sequential pass, and 8 concurrent
   through the HTML-Notes proxy all pass. Costs a ~1.6s reload on the retry path.
2. **Per-voice synthesis lock** (`_synth_lock_for`). Independently correct:
   `_lock` guards only the cache and is released before inference, so two requests
   could otherwise run on ONE session. Verified separately that concurrent load
   used to poison a session permanently (6 concurrent → 2x200 + 4x500, then 500
   forever).

### Still open — the real fix

This is a workaround for a defect below our code. The proper resolution is to
bisect `piper-tts` / `onnxruntime` versions in `requirements.txt` and find the
combination where a session survives repeated use. Until then every other request
pays a model reload.

Reproduce with:

```sh
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "%{http_code} " -X POST \
    http://10.0.0.16:3032/api/v1/tts/synthesize \
    -H 'Content-Type: application/json' -d '{"text":"Third check"}'
done
```

Temporarily disable the retry to see the underlying failure rate.
