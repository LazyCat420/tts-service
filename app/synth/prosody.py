"""Prosody planner: phone list -> per-phone durations (s) and frame-rate F0 (Hz).

Implements section 3C of PLAN-VOICE-SYNTHESIS-FROM-SCRATCH.md:
  * duration rule stack (stress, syllable count, cluster, break-index lengthening,
    accent, context) with incompressibility-floored rate compression and a 2.0x cap;
  * Fujisaki command-response F0 (alpha=3, beta=20, gamma=0.9) on a 200 Hz frame grid;
  * microprosody, one-pole pitch slew, and silence on gaps and stop closures.

Phones are duck-typed: each needs ``name``, ``stress``, ``syl``, ``word``, ``boundary``.
Pure numpy + stdlib.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

FRAME_HZ = 200
FRAME_S = 1.0 / FRAME_HZ

# Inherent durations (s). Copied values from sam-demo/sam.py: field 3 of the V
# table, plus the DIPHTHONGS table durations. Nothing is imported from sam.py.
INHERENT_DUR: dict[str, float] = {
    "IY": 0.105, "IH": 0.070, "EH": 0.090, "AE": 0.115, "AA": 0.115,
    "AO": 0.115, "UH": 0.075, "UW": 0.100, "AH": 0.085, "AX": 0.055,
    "ER": 0.100,
    "M": 0.080, "N": 0.075, "NG": 0.080, "L": 0.070, "R": 0.070,
    "W": 0.060, "Y": 0.060,
    "S": 0.090, "SH": 0.095, "F": 0.085, "TH": 0.080, "HH": 0.070,
    "CH": 0.095, "Z": 0.085, "ZH": 0.090, "V": 0.080, "DH": 0.070,
    "JH": 0.085,
    "P": 0.090, "B": 0.085, "T": 0.085, "D": 0.080, "K": 0.090, "G": 0.085,
    "DX": 0.030,  # alveolar flap (American "water", "little")
    # diphthongs (sam.py DIPHTHONGS durations); treated as single vowel nuclei
    "AY": 0.165, "EY": 0.150, "OW": 0.155, "OY": 0.170, "AW": 0.160,
}

# Klatt phone-specific minimum durations (s): the affine duration floor each
# phone is compressed against, so stressed/lengthened phones keep their
# relative shape while short phones never collapse.
MIN_DUR: dict[str, float] = {
    "AX": 0.028, "IH": 0.036, "UH": 0.040, "EH": 0.045, "AH": 0.042,
    "IY": 0.050, "AE": 0.055, "AA": 0.055, "AO": 0.055, "UW": 0.050,
    "ER": 0.050,
    "AY": 0.080, "EY": 0.075, "OW": 0.075, "OY": 0.080, "AW": 0.080,
    "DX": 0.022, "P": 0.048, "B": 0.045, "T": 0.046, "D": 0.045,
    "K": 0.048, "G": 0.045, "M": 0.038, "N": 0.035, "NG": 0.040,
    "L": 0.032, "R": 0.032, "W": 0.030, "Y": 0.030, "S": 0.045,
    "SH": 0.048, "F": 0.040, "TH": 0.038, "HH": 0.032, "CH": 0.048,
    "Z": 0.040, "ZH": 0.042, "V": 0.036, "DH": 0.030, "JH": 0.045,
}

VOWELS = frozenset({
    "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "AX", "ER",
    "AY", "EY", "OW", "OY", "AW",
})
VOICELESS_OBS = frozenset({"P", "T", "K", "F", "TH", "S", "SH", "HH", "CH"})
VOICED_OBS = frozenset({"B", "D", "G", "V", "DH", "Z", "ZH", "JH"})
OBSTRUENTS = VOICELESS_OBS | VOICED_OBS
STOPS = frozenset({"P", "B", "T", "D", "K", "G"})
VOICED_FRICATIVES = frozenset({"Z", "ZH", "V", "DH", "JH"})
VOICED_STOPS = frozenset({"B", "D", "G"})
NASALS = frozenset({"M", "N", "NG"})

WORD_GAP = "_"
SENTENCE_GAP = "__"

# Duration constants
BASE_SPEED = 72.0
WORD_GAP_S = 0.035
SENTENCE_PAUSE_S = {3: 0.110, 4: 0.210}  # by boundary; default is the BI4 value
VOWEL_INCOMP = 0.42
CONSONANT_INCOMP = 0.60
TOTAL_CAP = 2.0

# Fujisaki constants
FUJI_ALPHA = 3.0
FUJI_BETA = 20.0
FUJI_GAMMA = 0.9
AP_INITIAL = 0.35
AP_BOUNDARY = 0.15
AP_FINAL_FALL = -0.20
PHRASE_LEAD_S = 0.200
ACCENT_AA = {2: 0.35, 1: 0.20}  # keyed by vowel stress; 2 = nuclear
ACCENT_SHIFT_S = 0.045
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0

# Microprosody constants
MICRO_WINDOW_S = 0.050
MICRO_VOICELESS = 1.08
MICRO_VOICED = 0.95
CONT_RISE_HZ = 8.0
CONT_WINDOW_S = 0.060
SLEW_TAU_S = 0.025


def _is_gap(name: str) -> bool:
    return name == WORD_GAP or name == SENTENCE_GAP


def _frame(t: float, n_frames: int) -> int:
    return min(max(int(round(t * FRAME_HZ)), 0), n_frames)


def _gp(x: np.ndarray) -> np.ndarray:
    """Fujisaki phrase kernel Gp(t) = a^2 t exp(-a t), zero for t < 0."""
    xc = np.maximum(x, 0.0)
    return np.where(x >= 0.0, FUJI_ALPHA * FUJI_ALPHA * xc * np.exp(-FUJI_ALPHA * xc), 0.0)


def _ga(x: np.ndarray) -> np.ndarray:
    """Fujisaki accent step Ga(t) = min(gamma, 1 - (1 + b t) exp(-b t)), zero for t < 0."""
    xc = np.maximum(x, 0.0)
    val = np.minimum(FUJI_GAMMA, 1.0 - (1.0 + FUJI_BETA * xc) * np.exp(-FUJI_BETA * xc))
    return np.where(x >= 0.0, val, 0.0)


def _validate(phones: Sequence[Any]) -> None:
    for i, p in enumerate(phones):
        if _is_gap(p.name):
            continue
        if p.name not in INHERENT_DUR:
            raise ValueError(f"prosody: unknown phone {p.name!r} at index {i}")
        if p.syl < 0:
            raise ValueError(f"prosody: speech phone {p.name!r} at index {i} has syl={p.syl}")
        if p.name in VOWELS and p.stress not in ACCENT_AA and p.stress != 0:
            raise ValueError(f"prosody: vowel {p.name!r} at index {i} has stress={p.stress}")


def plan(phones: Sequence[Any], voice: Any) -> tuple[list[float], np.ndarray]:
    """Return (per-phone durations in seconds, F0 in Hz at FRAME_HZ).

    ``voice`` needs ``pitch`` (0-255); optional ``speed`` (default 72),
    ``rate_scale`` (default 1.0).
    """
    n = len(phones)
    if n == 0:
        return [], np.zeros(0, dtype=np.float64)
    _validate(phones)

    speed = float(voice.get("speed", BASE_SPEED))
    if speed <= 0:
        raise ValueError(f"prosody: speed must be > 0, got {speed}")
    rate_scale = float(voice.get("rate_scale", 1.0))
    pitch = float(voice["pitch"])
    time_scale = (BASE_SPEED / speed) * rate_scale  # rate = 72/speed (times rate_scale)

    names = [p.name for p in phones]
    speech = [i for i in range(n) if not _is_gap(names[i])]

    # ── syllable / word structure ────────────────────────────────────────
    members: dict[int, list[int]] = defaultdict(list)  # syl -> ordered speech indices
    vowel_of_syl: dict[int, int] = {}
    word_syls: dict[int, set[int]] = defaultdict(set)
    first_vowel_of_word: dict[int, int] = {}
    for i in speech:
        p = phones[i]
        members[p.syl].append(i)
        if names[i] in VOWELS:
            if p.syl in vowel_of_syl:
                raise ValueError(f"prosody: syllable {p.syl} has two vowels")
            vowel_of_syl[p.syl] = i
            word_syls[p.word].add(p.syl)
            first_vowel_of_word.setdefault(p.word, i)
    for s in members:
        if s not in vowel_of_syl:
            raise ValueError(f"prosody: syllable {s} has no vowel")

    # Boundaries: a gap carrying a boundary anchors to the preceding speech phone.
    # final_bi maps the syllable of each BI3/BI4 anchor to its break index.
    final_bi: dict[int, int] = {}
    bounds: list[tuple[int, int]] = []  # (phone index whose end is the boundary, break index)
    for i in range(n):
        b = phones[i].boundary
        if b not in (3, 4):
            continue
        bounds.append((i, b))
        anchor = i
        if _is_gap(names[i]):
            prev = [j for j in speech if j < i]
            if not prev:
                continue
            anchor = prev[-1]
        s = phones[anchor].syl
        final_bi[s] = max(final_bi.get(s, 0), b)

    # ── duration stack ───────────────────────────────────────────────────
    base = np.zeros(n)
    mult = np.ones(n)
    for i in speech:
        base[i] = INHERENT_DUR[names[i]]

    for i in speech:
        p = phones[i]
        is_vowel = names[i] in VOWELS
        syl_stress = phones[vowel_of_syl[p.syl]].stress

        if is_vowel:
            mult[i] *= 1.3 if p.stress > 0 else 0.8                       # stress
        if len(word_syls[p.word]) >= 3:
            mult[i] *= 0.78                                               # polysyllabic
        if not is_vowel:
            fv = first_vowel_of_word.get(p.word)
            if fv is not None and i > fv:
                mult[i] *= 0.7                                            # non-word-initial consonant
            if syl_stress == 0:
                mult[i] *= 0.8                                            # unstressed-syllable consonant
            ms = members[p.syl]
            k = ms.index(i)
            prev_cons = k > 0 and names[ms[k - 1]] not in VOWELS
            next_cons = k + 1 < len(ms) and names[ms[k + 1]] not in VOWELS
            if prev_cons:
                mult[i] *= 0.78                                           # later cluster member
            elif next_cons:
                mult[i] *= 0.88                                           # first cluster member

    for s, b in final_bi.items():                                         # pre-boundary lengthening
        vj = vowel_of_syl[s]
        mult[vj] *= 1.5 if b == 4 else 1.3
        for i in members[s]:
            if names[i] in OBSTRUENTS:
                mult[i] *= 1.2 if b == 4 else 1.1

    for i in speech:                                                      # accent lengthening
        if names[i] in VOWELS and phones[i].stress > 0:
            mult[i] *= 1.25 if phones[i].stress == 2 else 1.15

    for s, vj in vowel_of_syl.items():                                    # final open-syllable vowel
        open_syl = members[s][-1] == vj
        last_in_word = s == max(word_syls[phones[vj].word])
        if open_syl and last_in_word:
            mult[vj] *= 1.2

    for s, vj in vowel_of_syl.items():                                    # vowel context
        if (vj + 1 < n and not _is_gap(names[vj + 1])
                and phones[vj + 1].word == phones[vj].word):
            nxt = names[vj + 1]
            if nxt in VOICED_FRICATIVES:
                mult[vj] *= 1.6
            elif nxt in VOICED_STOPS:
                mult[vj] *= 1.2
            elif nxt in NASALS:
                mult[vj] *= 0.85
            elif nxt in VOICELESS_OBS:
                mult[vj] *= 0.86                                          # pre-fortis clipping

    # Stress-foot compression (Klatt/SAM connect): unstressed syllables
    # between stresses crowd each other; the more of them share a foot, the
    # more each one shortens.
    stressed_syls = {s for s, vj in sorted(vowel_of_syl.items())
                     if phones[vj].stress > 0}
    foot_size: dict[int, int] = {}
    run: list[int] = []
    for s in sorted(vowel_of_syl):
        if s in stressed_syls:
            for u in run:
                foot_size[u] = min(len(run), 4)
            run = []
        else:
            run.append(s)
            if s in final_bi:                                     # clause end
                for u in run:
                    foot_size[u] = min(len(run), 4)
                run = []
    for u in run:
        foot_size[u] = min(len(run), 4)
    for s, cnt in foot_size.items():
        if cnt > 0:
            f = 1.0 / math.sqrt(1.0 + 0.25 * cnt)
            for i in members[s]:
                mult[i] *= f

    for i in speech:                                                      # stop + liquid
        if (names[i] in {"L", "R", "W", "Y"} and i > 0
                and names[i - 1] in STOPS and phones[i].word == phones[i - 1].word):
            mult[i] *= 0.75

    # rate compression with incompressibility floor and Klatt phone-specific
    # minimum duration: d = d_min + (base - d_min)*mult, floored at 75% d_min.
    dur = np.zeros(n)
    for i in speech:
        incomp = VOWEL_INCOMP if names[i] in VOWELS else CONSONANT_INCOMP
        d_min = MIN_DUR.get(names[i], 0.035)
        d = d_min + max(0.0, base[i] - d_min) * mult[i]
        floor = 0.045 if names[i] in STOPS else 0.75 * d_min
        dur[i] = max(floor, ((1.0 - incomp) * d + incomp * d * time_scale))

    speech_idx = np.array(speech, dtype=np.int64)
    speech_total = float(dur[speech_idx].sum())
    cap_total = TOTAL_CAP * float((base[speech_idx] * time_scale).sum())
    if speech_total > cap_total > 0.0:
        dur[speech_idx] *= cap_total / speech_total

    # gaps
    for i in range(n):
        if names[i] == WORD_GAP:
            dur[i] = WORD_GAP_S * time_scale
        elif names[i] == SENTENCE_GAP:
            dur[i] = SENTENCE_PAUSE_S.get(phones[i].boundary, SENTENCE_PAUSE_S[4]) * time_scale

    # Quantize every duration to an exact integer multiple of the frame
    # (at least 1 frame) so frames.build's per-phone frame sums match the
    # total frame count with zero drift.
    dur = np.maximum(1.0, np.round(dur * FRAME_HZ)) / FRAME_HZ

    ends = np.cumsum(dur)
    starts = ends - dur
    total = float(ends[-1])
    n_frames = int(round(total * FRAME_HZ))
    durations = [float(x) for x in dur]

    # ── F0: Fujisaki command-response ────────────────────────────────────
    f_base = 20.0 + 1.15 * pitch
    t = np.arange(n_frames) * FRAME_S

    bound_times = [(float(ends[i]), b) for i, b in bounds]
    phrase: list[tuple[float, float]] = [(0.0, AP_INITIAL)]
    for k, (bt, b) in enumerate(bound_times):
        t0 = max(0.0, bt - PHRASE_LEAD_S)
        if k == len(bound_times) - 1:
            phrase.append((t0, AP_FINAL_FALL))   # final fall at the last boundary
        else:
            phrase.append((t0, AP_BOUNDARY))

    accents: list[tuple[float, float, float]] = []
    for i in speech:
        if names[i] in VOWELS and phones[i].stress > 0:
            aa = ACCENT_AA[phones[i].stress]
            accents.append((float(starts[i]) - ACCENT_SHIFT_S,
                            float(ends[i]) - ACCENT_SHIFT_S, aa))

    log_f = np.full(n_frames, math.log(f_base))
    for t0, ap in phrase:
        log_f += ap * _gp(t - t0)
    for t1, t2, aa in accents:
        log_f += aa * (_ga(t - t1) - _ga(t - t2))
    f0 = np.clip(np.exp(log_f), F0_MIN_HZ, F0_MAX_HZ)

    # ── silence: gap phones, voiceless stops, and voiced-stop closures ───
    if n_frames:
        frame_phone = np.minimum(np.searchsorted(ends, t, side="right"), n - 1)
        silent = np.zeros(n_frames, dtype=bool)
        for k in range(n_frames):
            pi = int(frame_phone[k])
            nm = names[pi]
            if _is_gap(nm) or nm in {"P", "T", "K"}:
                silent[k] = True
            elif nm in VOICED_STOPS:
                # Closure occupies the first 55% of the voiced stop; the
                # remaining 45% is the voiced locus release (av = 0.45).
                if t[k] < starts[pi] + 0.55 * dur[pi] - 1e-9:
                    silent[k] = True
    else:
        silent = np.zeros(0, dtype=bool)

    # ── microprosody ─────────────────────────────────────────────────────
    factor = np.ones(n_frames)
    for i in speech:
        if names[i] in OBSTRUENTS:
            lo = _frame(float(ends[i]), n_frames)
            hi = _frame(float(ends[i]) + MICRO_WINDOW_S, n_frames)
            factor[lo:hi] = MICRO_VOICELESS if names[i] in VOICELESS_OBS else MICRO_VOICED
    f0 = f0 * factor

    for bt, b in bound_times:                                             # continuation rise (BI3)
        if b != 3:
            continue
        lo = _frame(bt - CONT_WINDOW_S, n_frames)
        hi = _frame(bt, n_frames)
        if hi > lo:
            ramp = np.arange(1, hi - lo + 1) / float(hi - lo)
            f0[lo:hi] += CONT_RISE_HZ * ramp

    f0[silent] = 0.0

    # one-pole slew (25 ms) within each voiced run, so no silence smearing
    alpha = 1.0 - math.exp(-FRAME_S / SLEW_TAU_S)
    out = np.zeros(n_frames)
    k = 0
    while k < n_frames:
        if silent[k]:
            k += 1
            continue
        y = f0[k]
        while k < n_frames and not silent[k]:
            y += alpha * (f0[k] - y)
            out[k] = y
            k += 1

    return durations, out
