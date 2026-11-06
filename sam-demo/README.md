# SAM demo — clean-room formant speech synth

A from-scratch implementation of the SAM algorithm: text → phonemes → buzzy
glottal pulse train through three cascaded formant resonators. No SAM ROM data,
no port code — the reciter rules, the phoneme table and the renderer are
written here, from the published description of how SAM works.

Open **[index.html](index.html)** for the playable demo (audio + spectrograms).

## Files

| File | What |
|---|---|
| `sam.py` | the synth: reciter, phoneme table, renderer, CLI |
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
