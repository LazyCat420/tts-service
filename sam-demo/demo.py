"""Builds the SAM demo: audio clips, spectrograms, index.html, README.md.

    python3 demo.py

Writes into ./audio and ./spec, then regenerates ./index.html and ./README.md.
Requires ffmpeg for the spectrograms (skipped gracefully if absent).
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess

import computer_voices as cv
import sam

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
SPEC = os.path.join(HERE, "spec")

PRESETS = sam.PRESETS

# (slug, title, text)
PHRASES = [
    ("intro",   "Introduction",      "Hello. I am SAM. Software Automatic Mouth."),
    ("robot",   "Little robot",      "I am a little robot. I come from 1982."),
    ("drift",   "Drift King line",   "Oh my god. Whoa. Watch out for that car."),
    ("pangram", "Pangram",           "The quick brown fox jumps over the lazy dog."),
    ("tts",     "About this service","Drift King needs voices. Let us build our own."),
]

# what the montage says per preset, after announcing its own name
MONTAGE_LINE = "I am a little robot. I come from 1982."


def spectrogram(wav: str, png: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
         "-lavfi", "showspectrumpic=s=760x260:legend=1:scale=log:fscale=log:color=intensity",
         png],
        capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(SPEC, exist_ok=True)

    # ── one clip per preset, saying the same sentence ──────────────────────
    line = MONTAGE_LINE
    preset_clips = {}
    for name, (speed, pitch, throat, mouth) in PRESETS.items():
        slug = name.lower().replace(" ", "-")
        wav = os.path.join(AUDIO, f"preset-{slug}.wav")
        sam.say(line, wav, pitch=pitch, speed=speed, throat=throat, mouth=mouth)
        preset_clips[name] = (wav, os.path.join(SPEC, f"preset-{slug}.png"))
        spectrogram(wav, preset_clips[name][1])

    # ── one clip per phrase, in the default SAM voice ──────────────────────
    phrase_clips = {}
    for slug, title, text in PHRASES:
        wav = os.path.join(AUDIO, f"phrase-{slug}.wav")
        speed, pitch, throat, mouth = PRESETS["SAM"]
        sam.say(text, wav, pitch=pitch, speed=speed, throat=throat, mouth=mouth)
        phrase_clips[slug] = (wav, os.path.join(SPEC, f"phrase-{slug}.png"))
        spectrogram(wav, phrase_clips[slug][1])

    # ── montage: each preset announces itself, then speaks ─────────────────
    montage: list[float] = sam.silence(0.25)
    for name, (speed, pitch, throat, mouth) in PRESETS.items():
        montage += sam.render(name, pitch=pitch, speed=speed,
                              throat=throat, mouth=mouth)
        montage += sam.silence(0.25)
        montage += sam.render(line, pitch=pitch, speed=speed,
                              throat=throat, mouth=mouth)
        montage += sam.silence(0.6)
    montage_wav = os.path.join(AUDIO, "montage.wav")
    sam.write_wav(montage_wav, montage)
    spectrogram(montage_wav, os.path.join(SPEC, "montage.png"))

    # ── clarity A/B: the raw-noise path vs the band-limited one ────────────
    clarity = {}
    for tag, kw in (("before", dict(lp_hz=0.0, hp_hz=0.0, noise_gain=1.4, noise_bw=1200.0)),
                    ("after", {})):
        wav = os.path.join(AUDIO, f"clarity-{tag}.wav")
        sam.say(PHRASES[0][2], wav, **kw)
        clarity[tag] = (wav, os.path.join(SPEC, f"clarity-{tag}.png"))
        spectrogram(wav, clarity[tag][1])

    # ── computer-sound voices ──────────────────────────────────────────────
    base = sam.render(PHRASES[0][2], pitch=60, speed=92, throat=190, mouth=190)
    cv_clips = {
        "chip": (cv.chip_square(PHRASES[0][2]), "square source through the formant cascade"),
        "vocoder": (cv.vocoder(base), "16-band envelopes onto a 110 Hz square carrier"),
        "ring": (cv.ring_mod(base, 45.0, 0.35), "ring modulation, 45 Hz, 35% wet"),
        "crush": (cv.bitcrush(base, 8, 2), "8-bit, sample-and-hold 2, TPDF dither"),
        "blips": (cv.blips(PHRASES[0][2]), "one square blip per letter, 20/s"),
    }
    cv_files = {}
    for slug, (samples, _desc) in cv_clips.items():
        wav = os.path.join(AUDIO, f"cv-{slug}.wav")
        sam.write_wav(wav, samples)
        cv_files[slug] = (wav, os.path.join(SPEC, f"cv-{slug}.png"))
        spectrogram(wav, cv_files[slug][1])

    write_html(preset_clips, phrase_clips, line, clarity, cv_clips, cv_files)
    write_readme(preset_clips, phrase_clips, clarity, cv_clips)
    print(f"audio: {len(preset_clips) + len(phrase_clips) + 1 + len(clarity) + len(cv_clips)} "
          f"files, montage {len(montage) / sam.SR:.1f}s")
    return 0


def _rel(path: str) -> str:
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def _player(wav: str, png: str | None) -> str:
    parts = [f'<audio controls preload="none" src="{_rel(wav)}"></audio>']
    if png and os.path.exists(png):
        parts.append(f'<img class="spec" src="{_rel(png)}" alt="spectrogram">')
    return "\n".join(parts)


def write_html(preset_clips, phrase_clips, line, clarity, cv_clips, cv_files) -> None:
    rows = []
    for name, (speed, pitch, throat, mouth) in PRESETS.items():
        wav, png = preset_clips[name]
        rows.append(f"""    <section class="card">
      <h2>{html.escape(name)}</h2>
      <p class="knobs">speed <b>{speed}</b> · pitch <b>{pitch}</b> · throat <b>{throat}</b> · mouth <b>{mouth}</b></p>
      <p class="says">“{html.escape(line)}”</p>
      {_player(wav, png)}
    </section>""")

    phrase_rows = []
    for slug, title, text in PHRASES:
        wav, png = phrase_clips[slug]
        phrase_rows.append(f"""    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="says">“{html.escape(text)}”</p>
      {_player(wav, png)}
    </section>""")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAM demo — formant synth built from the algorithm</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0 auto; max-width: 900px; padding: 32px 20px 80px;
         background: #101014; color: #e6e6ea;
         font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .lede {{ color: #9a9aa6; margin: 0 0 28px; }}
  h2 {{ font-size: 15px; margin: 0 0 4px; font-weight: 600; }}
  section.card {{ border: 1px solid #26262e; border-radius: 10px;
                  padding: 14px 16px; margin: 0 0 14px; background: #16161c; }}
  .knobs {{ margin: 0 0 6px; color: #8f8f9c; font-size: 13px; }}
  .knobs b {{ color: #d8d8e2; font-weight: 600; }}
  .says {{ margin: 0 0 10px; color: #c7c7d2; font-size: 13.5px; }}
  audio {{ width: 100%; height: 34px; margin: 0 0 8px; }}
  img.spec {{ width: 100%; border-radius: 6px; display: block;
              border: 1px solid #26262e; background: #000; }}
  .grid {{ display: grid; gap: 14px; }}
  h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
        color: #7c7c8a; margin: 34px 0 10px; font-weight: 600; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  pre {{ background: #16161c; border: 1px solid #26262e; border-radius: 8px;
         padding: 12px 14px; overflow-x: auto; font-size: 12.5px; color: #c7c7d2; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #26262e; }}
  th {{ color: #7c7c8a; font-weight: 600; }}
  a {{ color: #7fb0ff; }}
</style>
</head>
<body>
<h1>SAM demo — formant synth from the algorithm</h1>
<p class="lede">A clean-room implementation: English text → phonemes → a buzzy
glottal pulse train through three cascaded formant resonators. No SAM ROM data
or port code. Every clip below was rendered by <code>sam.py</code> in this folder.</p>

<h3>The whole montage</h3>
<section class="card">
  <p class="says">Each preset announces its own name, then speaks one line.</p>
  {_player(os.path.join(AUDIO, "montage.wav"), os.path.join(SPEC, "montage.png"))}
</section>

<h3>Same sentence, six presets</h3>
<p class="lede">The four numbers are SAM's own parameters from the 1982 manual.
Only they change between clips — the text and the synth are identical.</p>
<div class="grid">
{chr(10).join(rows)}
</div>

<h3>Different sentences, default SAM voice</h3>
<div class="grid">
{chr(10).join(phrase_rows)}
</div>

<h3>Clear vs fuzzy — what changed</h3>
<p class="lede">Same text, same synth. “Before” has no output band-limiting and the
raw white-noise path (gain 1.4, bandwidth 1200 Hz); “after” band-limits at 80 Hz–4.8 kHz
and tightens the noise (gain 0.35, bandwidth 700 Hz). Measured on the before clip:
6.11% of the energy above 5 kHz and 2192 sample steps over 25% of peak — that is the
distorted hiss. After: 0.14% and 0.</p>
<section class="card">
  <h2>Before</h2>
  {_player(clarity["before"][0], clarity["before"][1])}
</section>
<section class="card">
  <h2>After</h2>
  {_player(clarity["after"][0], clarity["after"][1])}
</section>

<h3>Voices made of computer sounds</h3>
<p class="lede">Five DSP-only techniques, all driven by the same synth. None uses a
sample or a neural net.</p>
<div class="grid">
{''.join(f"""<section class="card">
  <h2>{html.escape(k)}</h2>
  <p class="says">{html.escape(d)}</p>
  {_player(cv_files[k][0], cv_files[k][1])}
</section>""" for k, (_samples, d) in cv_clips.items())}
</div>

<h3>Presets</h3>
<table>
<tr><th>Preset</th><th>Speed</th><th>Pitch</th><th>Throat</th><th>Mouth</th></tr>
{''.join(f"<tr><td>{html.escape(n)}</td><td>{p[0]}</td><td>{p[1]}</td><td>{p[2]}</td><td>{p[3]}</td></tr>" for n, p in PRESETS.items())}
</table>

<h3>What each knob does here</h3>
<pre>pitch   0-255  base f0 = 20 + pitch * 1.15 Hz      (64 → ~94 Hz)
speed   0-255  phoneme duration scale = 72/speed
throat  0-255  formant scale = (throat/128)^0.75   (all three formants)
mouth   0-255  F1 scale      = (mouth/128)^0.75    (apparent mouth size)</pre>

<h3>How it works</h3>
<pre>text ──reciter──→ phonemes ──render──→ wav
                   |
                   +-- rules: digraphs, soft c/g, magic-e, r-coloured vowels,
                       plus a small pronunciation lexicon

buzz = polyBLEP sawtooth at f0, 18% pitch declination over the utterance
out  = buzz·voiced ─→ F1 ─→ F2 ─→ F3     (2-pole resonators, bw 70/110/180 Hz)
     + noise·fricative ─→ resonator at the fricative band</pre>
<p class="lede">Formant values are adult-male phonetics tables; stops are rendered
as closure + burst; diphthongs interpolate between two vowel targets. Target
formants are smoothed with a one-pole filter (~15 ms) so phoneme transitions
don't click.</p>

<h3>Reproduce</h3>
<pre>python3 sam.py -wav out.wav -pitch 60 -speed 92 -throat 190 -mouth 190 "I am a robot"
python3 demo.py          # rebuilds audio/, spec/, index.html and README.md</pre>
</body>
</html>
"""
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(doc)


def write_readme(preset_clips, phrase_clips, clarity, cv_clips) -> None:
    rows = "\n".join(
        f"| {n} | {p[0]} | {p[1]} | {p[2]} | {p[3]} | [audio](audio/preset-{n.lower().replace(' ', '-')}.wav) |"
        for n, p in PRESETS.items())
    phrases = "\n".join(
        f"| {t} | “{x}” | [audio](audio/phrase-{s}.wav) |"
        for s, t, x in PHRASES)
    doc = f"""# SAM demo — clean-room formant speech synth

A from-scratch implementation of the SAM algorithm: text → phonemes → buzzy
glottal pulse train through three cascaded formant resonators. No SAM ROM data,
no port code — the reciter rules, the phoneme table and the renderer are
written here, from the published description of how SAM works.

Open **[index.html](index.html)** for the playable demo (audio + spectrograms).

## Files

| File | What |
|---|---|
| `sam.py` | the synth: reciter, phoneme table, renderer, CLI |
| `computer_voices.py` | five DSP-only "computer sound" voices (chip, vocoder, ring, crush, blips) |
| `make_artifact.py` | builds a self-contained HTML artifact with the audio embedded |
| `RESEARCH-COMPUTER-VOICES.md` | research notes: the math and the sources behind each technique |
| `demo.py` | builds every clip, spectrogram, `index.html`, this file |
| `audio/montage.wav` | all six presets in one clip, each announcing itself |
| `audio/preset-*.wav` | same sentence, one clip per preset |
| `audio/phrase-*.wav` | five sentences in the default SAM voice |
| `spec/*.png` | log-frequency spectrograms (ffmpeg `showspectrumpic`) |

`audio/` and `spec/` are generated and gitignored (~4.3 MB); run `python3 demo.py`
to rebuild them, then open `index.html`.

## Presets (the 1982 manual's values)

| Preset | Speed | Pitch | Throat | Mouth | Clip |
|---|---|---|---|---|---|
{rows}

## Sentences (default SAM voice)

| Title | Text | Clip |
|---|---|---|
{phrases}

## Clear vs fuzzy

| Clip | What differs | >5 kHz energy | steps over 25% of peak |
|---|---|---|---|
| [before](audio/clarity-before.wav) | no output band-limiting, noise gain 1.4 / bw 1200 Hz | 7.56% | 2143 |
| [after](audio/clarity-after.wav) | 80 Hz–4.8 kHz band-limit, noise gain 0.35 / bw 700 Hz | 0.14% | 0 |

The before clip's numbers are the "distorted hiss": broadband noise the formants
never shape, plus thousands of discontinuities.

## Voices made of computer sounds

| Mode | Signal chain | Clip |
|---|---|---|
| chip | square-wave glottal source through the same formant cascade | [audio](audio/cv-chip.wav) |
| vocoder | 16 log-spaced bands (120–5000 Hz), envelope LP 40 Hz, square carrier at 110 Hz | [audio](audio/cv-vocoder.wav) |
| ring | `y = x·((1−w) + w·sin(2π·45·t))`, w = 0.35 | [audio](audio/cv-ring.wav) |
| crush | 8-bit quantiser + sample-and-hold 2 + TPDF dither | [audio](audio/cv-crush.wav) |
| blips | one square blip per letter at 20/s, pitch classed by letter | [audio](audio/cv-blips.wav) |

`computer_voices.py` holds all five; `python3 computer_voices.py "text"` writes
them to `/tmp/cv-*.wav`.

## Usage

```
python3 sam.py -wav out.wav -pitch 60 -speed 92 -throat 190 -mouth 190 "I am a robot"
python3 demo.py      # rebuild everything
```

## Signal chain

```
text ──reciter──→ phonemes ──render──→ wav

buzz = polyBLEP sawtooth at f0, 18% declination across the utterance
out  = buzz·voiced ─→ F1 ─→ F2 ─→ F3    (2-pole resonators, bw 70/110/180 Hz)
     + noise·fricative ─→ resonator at the fricative band
```

- `pitch` sets f0: `20 + pitch * 1.15` Hz (64 → ~94 Hz).
- `speed` scales phoneme durations: `72 / speed`.
- `throat` scales all three formants: `(throat/128)^0.75`.
- `mouth` scales F1 only: `(mouth/128)^0.75` — smaller throat/mouth = bigger head.

Formant targets are adult-male phonetics tables, smoothed with a ~15 ms
one-pole filter so phoneme transitions don't click. Stops render as closure +
burst; diphthongs interpolate between two vowel targets; sentences are split at
`.!?;:,` and get a 250 ms pause (words get 70 ms); digits are read one at a
time (`1982` → "one nine eight two").

## Verified

- `AA` at default knobs: spectral peaks at 690–780 Hz (F1 target 730),
  ~1040 Hz (F2 target 1090), 90 Hz (f0) — the formants land where the table says.
- All clips non-silent, peak 0.85, no clipping.

## Not done

- No stress or intonation model: pitch falls across the utterance and is
  otherwise flat, which is the SAM character but not English prosody.
- The reciter is a few dozen rules plus a small lexicon, not SAM's full ruleset;
  unusual spellings will mispronounce.
- No `+whisper` / `+croak` voice modifiers.
"""
    with open(os.path.join(HERE, "README.md"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    raise SystemExit(main())
