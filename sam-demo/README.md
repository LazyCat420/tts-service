# SAM demo — two synths: a clean-room one and the real engine

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
| `sam.py` (cascade synth) | 3.32 s | 0.14% | 0 | 133× |
| `sam_engine` (real engine) | 3.63 s | 4.72% | 4144 | 26× |

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
| SAM | 72 | 64 | 128 | 128 | [audio](audio/preset-sam.wav) |
| Little Robot | 92 | 60 | 190 | 190 | [audio](audio/preset-little-robot.wav) |
| Elf | 72 | 64 | 110 | 160 | [audio](audio/preset-elf.wav) |
| Stuffy Guy | 82 | 72 | 110 | 105 | [audio](audio/preset-stuffy-guy.wav) |
| Little Old Lady | 82 | 32 | 145 | 145 | [audio](audio/preset-little-old-lady.wav) |
| Extra-Terrestrial | 100 | 64 | 150 | 200 | [audio](audio/preset-extra-terrestrial.wav) |

## Sentences (default SAM voice)

| Title | Text | Clip |
|---|---|---|
| Introduction | “Hello. I am SAM. Software Automatic Mouth.” | [audio](audio/phrase-intro.wav) |
| Little robot | “I am a little robot. I come from 1982.” | [audio](audio/phrase-robot.wav) |
| Drift King line | “Oh my god. Whoa. Watch out for that car.” | [audio](audio/phrase-drift.wav) |
| Pangram | “The quick brown fox jumps over the lazy dog.” | [audio](audio/phrase-pangram.wav) |
| About this service | “Drift King needs voices. Let us build our own.” | [audio](audio/phrase-tts.wav) |

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
