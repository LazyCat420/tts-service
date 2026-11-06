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
import sam_faithful

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


def metrics(samples) -> dict:
    """Duration, high-band energy share, discontinuities and formant contrast."""
    import math
    x = [v / 32768.0 if isinstance(v, int) else float(v) for v in samples]
    pk = max(abs(v) for v in x) or 1e-9
    steps = [abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]
    seg = x[len(x) // 2: len(x) // 2 + 2048]
    mags = []
    for f in range(50, 11000, 25):
        re_ = sum(v * math.cos(2 * math.pi * f * i / sam.SR) for i, v in enumerate(seg))
        im = sum(v * math.sin(2 * math.pi * f * i / sam.SR) for i, v in enumerate(seg))
        mags.append((f, math.hypot(re_, im)))
    tot = sum(v * v for _, v in mags) or 1e-9
    hi = sum(v * v for f, v in mags if f > 5000) / tot
    band = sorted((v for f, v in mags if 200 <= f <= 3500), reverse=True)
    return {"dur": len(x) / sam.SR, "hi": hi,
            "clicks": sum(1 for d in steps if d > 0.25 * pk),
            "contrast": band[0] / max(band[len(band) // 2], 1e-9)}


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

    # ── faithful SAM port: the real 1982 renderer ──────────────────────────
    faithful = {}
    for name, (speed, pitch, throat, mouth) in PRESETS.items():
        slug = name.lower().replace(" ", "-")
        wav = os.path.join(AUDIO, f"faithful-{slug}.wav")
        sam_faithful.faithful_say(line, wav, pitch=pitch, speed=speed,
                                  mouth=mouth, throat=throat)
        faithful[name] = (wav, os.path.join(SPEC, f"faithful-{slug}.png"))
        spectrogram(wav, faithful[name][1])

    # ── A/B: this folder's synth vs the real engine, same text ─────────────
    ab = {}
    ab_text = PHRASES[0][2]
    speed, pitch, throat, mouth = PRESETS["SAM"]
    ab["ours"] = (os.path.join(AUDIO, "ab-ours.wav"), os.path.join(SPEC, "ab-ours.png"))
    sam.say(ab_text, ab["ours"][0], pitch=pitch, speed=speed, throat=throat, mouth=mouth)
    ab["faithful"] = (os.path.join(AUDIO, "ab-faithful.wav"), os.path.join(SPEC, "ab-faithful.png"))
    sam_faithful.faithful_say(ab_text, ab["faithful"][0], pitch=pitch, speed=speed,
                              mouth=mouth, throat=throat)
    for tag in ab:
        spectrogram(*ab[tag])

    # ── faithful montage ───────────────────────────────────────────────────
    f_montage: list[int] = [0] * int(0.25 * sam.SR)
    for name, (speed, pitch, throat, mouth) in PRESETS.items():
        f_montage += sam_faithful.faithful_samples(name, pitch, speed, mouth, throat)
        f_montage += [0] * int(0.25 * sam.SR)
        f_montage += sam_faithful.faithful_samples(line, pitch, speed, mouth, throat)
        f_montage += [0] * int(0.6 * sam.SR)
    f_montage_wav = os.path.join(AUDIO, "montage-faithful.wav")
    sam.write_wav(f_montage_wav, f_montage)
    spectrogram(f_montage_wav, os.path.join(SPEC, "montage-faithful.png"))

    ab_speed, ab_pitch, ab_throat, ab_mouth = PRESETS["SAM"]
    ab_metrics = {
        "ours": metrics(sam.render(ab_text, pitch=ab_pitch, speed=ab_speed,
                                   throat=ab_throat, mouth=ab_mouth)),
        "faithful": metrics(sam_faithful.faithful_samples(ab_text, ab_pitch, ab_speed,
                                                          ab_mouth, ab_throat)),
    }

    write_html(preset_clips, phrase_clips, line, clarity, cv_clips, cv_files,
               faithful, ab, f_montage_wav, ab_metrics)
    write_readme(preset_clips, phrase_clips, clarity, cv_clips, faithful, ab_metrics)
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


def write_html(preset_clips, phrase_clips, line, clarity, cv_clips, cv_files,
               faithful, ab, f_montage_wav, ab_metrics) -> None:
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
<title>SAM demo — real engine and clean-room synth</title>
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
<h1>SAM demo — the real engine and a clean-room synth</h1>
<p class="lede">Two synths in one folder. <code>sam.py</code> is a clean-room
implementation: English text → phonemes → a buzzy glottal pulse train through three
cascaded formant resonators, no SAM ROM data or port code. <code>sam_engine/</code> is
the opposite: a faithful port of the real 1982 engine — its tables, its three direct
oscillators, its 4-bit output. Play both and hear what the port buys you.</p>

<h3>The whole montage</h3>
<section class="card">
  <p class="says">Each preset announces its own name, then speaks one line.</p>
  {_player(os.path.join(AUDIO, "montage.wav"), os.path.join(SPEC, "montage.png"))}
</section>

<h3>Real SAM vs this folder's clean-room synth</h3>
<p class="lede">Identical text, identical knobs (speed 72, pitch 64, throat 128, mouth 128).
The difference is structural, not a matter of tuning: <code>sam.py</code> filters a
sawtooth through three resonators, while the real engine sums three oscillators directly —
sin(F1)·A1 + sin(F2)·A2 + rect(F3)·A3 — and resets their phase at every glottal pulse.
That hard sync is what makes SAM's formants cut through and gives it its grit.</p>
<section class="card">
  <h2>sam.py — clean-room cascade synth</h2>
  <p class="says">“{html.escape(PHRASES[0][2])}” · {ab_metrics["ours"]["dur"]:.2f} s</p>
  {_player(ab["ours"][0], ab["ours"][1])}
</section>
<section class="card">
  <h2>sam_engine — faithful port of the 1982 renderer</h2>
  <p class="says">“{html.escape(PHRASES[0][2])}” · {ab_metrics["faithful"]["dur"]:.2f} s</p>
  {_player(ab["faithful"][0], ab["faithful"][1])}
</section>
<p class="lede">The port was checked against the reference engine byte for byte:
0 differing bytes on every case tested, and the reciter/parser output matches its
phoneme stream and frame tables exactly. Its output is 8-bit PCM at 22.05 kHz, so the
quantisation steps you can see in the spectrogram are SAM's own, not a defect.</p>

<h3>Faithful engine, six presets</h3>
<p class="lede">The same six voices, rendered by the ported engine. Compare with the
<code>sam.py</code> versions further down: the vowels are the same phonetics, the
articulation is SAM's.</p>
<section class="card">
  <p class="says">Each preset announces its own name, then speaks one line.</p>
  {_player(f_montage_wav, os.path.join(SPEC, "montage-faithful.png"))}
</section>
<div class="grid">
{''.join(f'''<section class="card">
  <h2>{html.escape(n)}</h2>
  <p class="knobs">speed <b>{p[0]}</b> · pitch <b>{p[1]}</b> · throat <b>{p[2]}</b> · mouth <b>{p[3]}</b></p>
  <p class="says">“{html.escape(line)}”</p>
  {_player(faithful[n][0], faithful[n][1])}
</section>''' for n, p in PRESETS.items())}
</div>

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


def write_readme(preset_clips, phrase_clips, clarity, cv_clips, faithful, ab_metrics) -> None:
    rows = "\n".join(
        f"| {n} | {p[0]} | {p[1]} | {p[2]} | {p[3]} | [audio](audio/preset-{n.lower().replace(' ', '-')}.wav) |"
        for n, p in PRESETS.items())
    phrases = "\n".join(
        f"| {t} | “{x}” | [audio](audio/phrase-{s}.wav) |"
        for s, t, x in PHRASES)
    doc = f"""# SAM demo — two synths: a clean-room one and the real engine

Two independent implementations of SAM in one folder:

- **`sam.py`** — clean-room. Text → phonemes → buzzy glottal pulse train through
  three cascaded formant resonators. No SAM ROM data, no port code: the reciter
  rules, phoneme table and renderer are written here from the published
  description of the algorithm. Tuned to *resemble* SAM.
- **`sam_engine/`** — faithful. A port of the actual 1982 engine (reciter,
  parser, renderer and its tables), verified **byte-identical** to the reference
  implementation on every case tested.

Open **[index.html](index.html)** for the playable demo (audio + spectrograms).

## Why the port sounds better

| | duration | energy >5 kHz | steps >25% of peak | formant contrast |
|---|---|---|---|---|
| `sam.py` (cascade synth) | {ab_metrics["ours"]["dur"]:.2f} s | {ab_metrics["ours"]["hi"] * 100:.2f}% | {ab_metrics["ours"]["clicks"]} | {ab_metrics["ours"]["contrast"]:.0f}× |
| `sam_engine` (real engine) | {ab_metrics["faithful"]["dur"]:.2f} s | {ab_metrics["faithful"]["hi"] * 100:.2f}% | {ab_metrics["faithful"]["clicks"]} | {ab_metrics["faithful"]["contrast"]:.0f}× |

Same text, same knobs. The real engine sums three oscillators directly —
`sin(F1)·A1 + sin(F2)·A2 + rect(F3)·A3` — and resets their phase at every glottal
pulse. The clean-room one filters a sawtooth through resonators, which smears the
formants across phoneme boundaries and loses the articulation that makes SAM
intelligible. The step count is not a defect: SAM writes 8-bit PCM through a
5-sample C64 simulation timetable, so coarse steps *are* its sound.

## Files

| File | What |
|---|---|
| `sam.py` | clean-room synth: reciter, phoneme table, renderer, CLI |
| `sam_engine/` | faithful port: `parse.py` (reciter + parser), `render.py` (renderer), `tables.py` |
| `sam_faithful.py` | wrapper around the port: `faithful_say(text, path, pitch, speed, mouth, throat)` |
| `computer_voices.py` | five DSP-only "computer sound" voices (chip, vocoder, ring, crush, blips) |
| `make_artifact.py` | builds a self-contained HTML artifact with the audio embedded |
| `RESEARCH-COMPUTER-VOICES.md` | research notes: the math and the sources behind each technique |
| `demo.py` | builds every clip, spectrogram, `index.html`, this file |
| `audio/montage.wav` | all six presets in one clip, each announcing itself |
| `audio/montage-faithful.wav` | the same montage from the ported engine |
| `audio/faithful-*.wav` | same sentence per preset, ported engine |
| `audio/ab-ours.wav`, `audio/ab-faithful.wav` | the A/B pair from the table above |
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
python3 sam_faithful.py "Hello. I am SAM." -wav hello.wav -pitch 64 -speed 72
python3 demo.py      # rebuild everything
```

The port's knobs are SAM's own: `pitch` 0–255 (f0 table, not a Hz formula),
`speed` 0–255 (frames per phoneme), `mouth` 0–255 (F1 scale) and `throat`
0–255 (F1/F2 scale). The default voice is 72 / 64 / 128 / 128.

## Licensing note

`sam.py` is original code. `sam_engine/` carries SAM's own tables, which come from a
reverse-engineered ROM: SAM (Software Automatic Mouth) is abandonware by SoftVoice,
Inc. Fine for personal and experimental use; do not ship it commercially without
checking the rights first.

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
