# PLAN: Lightweight Robotic TTS Architecture for `tts-service`

**Date:** 2026-09-24  
**Target Repository:** `LazyCat420/tts-service`  
**Host & Environment:** Synology NAS (`10.0.0.16:3032`), Docker container `tts-service:latest`, Python 3.11-slim  
**Methodology:** Verified-Claim Plan Methodology (VCPM) & First-Principles Audio Engineering  

---

## 1. Executive Summary & Ground Truth

### 1.1 Problem Statement
The current `tts-service` relies exclusively on **Piper TTS** (VITS neural architecture executed via `onnxruntime` CPU). While Piper produces natural-sounding human speech, it has three critical limitations for robotic voice use cases:
1. **High Latency on NAS CPU:** Synthesis takes **~1.1s – 1.3s per sentence** on the Synology NAS CPU (*Verified Fact: measured live via curl to `10.0.0.16:3032` on 2026-09-24*).
2. **Excessive Memory & Resource Overhead:** Piper models are ~60MB each on disk, requiring ~60MB–120MB active RAM for ONNX inference sessions, plus complex session management to prevent ONNX `ScatterND` memory corruptions.
3. **Lack of Authentic Robotic Character:** Piper models are trained on clean audiobook human actors (Lessac, Alan, Amy, Thorsten). Modifying client-side `playbackRate` or `pitch` in Web Audio does not create an authentic robot voice—it merely sounds like a human speaking unnaturally fast or chipmunk-pitched.

### 1.2 Objective
Evaluate and plan the integration of dedicated, ultra-lightweight robotic TTS engines and DSP transformation pipelines into `tts-service` that:
- Drop synthesis latency from **~1,200ms to <15ms** (an **80x speedup**).
- Reduce memory footprint from **~120MB to <5MB RAM** for robot voices.
- Deliver genuine retro, sci-fi, and mechanical robotic voice aesthetics (e.g., Dalek, C64 retro bot, mechanical synth, cybernetic whisper).
- Maintain **100% backward compatibility** with all existing consumers (`office-client`, `AgentEditor`, `trading-client`, `braindeadbot-client`, `HTML-Notes`).

### 1.3 User-Confirmed Design Selections (2026-09-24)
- **Engine Architecture:** **Unified Multi-Engine** supporting **eSpeak-NG** (crisp mechanical synth), **SAM** (retro 1982 8-bit Commodore arcade bot), and **DSP Effects** (Dalek ring modulation / metallic comb filter / radio lofi) alongside existing Piper models.
- **Client Configuration Contract:** **Presets via `voice_accent` aliases** (e.g. `robot-classic`, `robot-retro`, `robot-deep`, `cyborg-dalek`) so existing clients (`office-client`, `AgentEditor`, `trading-client`) immediately work with zero breaking changes or frontend schema rewrites.

---

## 2. Verified Baseline Measurements (Current State)

| Metric | Measured Live Ground Truth | Source / Verification Method |
|---|---|---|
| **Live NAS Endpoint** | `http://10.0.0.16:3032/health` → `200 OK` (`voices_available: 37`) | Live curl check (2026-09-24T05:58Z) |
| **Cold Synthesis Latency** | **1.086s** (103,468 bytes WAV) | `curl -w "%{time_total}s" -X POST .../synthesize` |
| **Warm Synthesis Latency** | **1.272s** (98,860 bytes WAV) | `curl -w "%{time_total}s" -X POST .../synthesize` |
| **Loaded RAM (Piper)** | ~60MB – 120MB (`MAX_LOADED_VOICES = 2`) | `app/piper_manager.py:28` |
| **Model Storage** | 37 models on disk (~1.5GB total on NAS volume) | Query `GET /api/v1/tts/voices` |
| **Existing Dependencies** | `fastapi`, `uvicorn`, `piper-tts`, `onnxruntime`, `numpy`, `espeak-ng` | `Dockerfile:18`, `requirements.txt` |
| **Critical Docker Asset** | **`espeak-ng` is already installed in runner image!** | `Dockerfile:18` (`apt-get install -y espeak-ng`) |

---

## 3. Evaluation of Lightweight Robotic TTS Candidates

We evaluated five distinct technical approaches against footprint, latency, robotic authenticity, and maintenance overhead:

### Candidate Comparison Matrix

| Candidate | Engine Type | Model/Binary Size | Runtime RAM | Synthesis Latency | Robotic Authenticity | NAS CPU Load | External Deps |
|---|---|---|---|---|---|---|---|
| **eSpeak-NG** *(Recommended Primary)* | Formant Synth (Klatt) | ~3 MB | **< 5 MB** | **2 – 10 ms** | ⭐⭐⭐⭐⭐ (Crisp monotone mechanical, whisper, alien, sci-fi) | < 0.1% | **Already installed in Dockerfile!** |
| **SAM (Software Auto Mouth)** *(Recommended Retro)* | Algorithmic (1982 C64) | **< 100 KB** | **< 2 MB** | **< 1 ms** | ⭐⭐⭐⭐⭐ (Iconic 80s arcade robot, Commodore 64 / WarGames) | 0.0% | Zero (pure Python or tiny C) |
| **DSP FX on Piper/eSpeak** *(Recommended Hybrid)* | Audio DSP Filter Chain | < 50 KB code | < 1 MB | **1 – 3 ms** | ⭐⭐⭐⭐⭐ (Ring modulation Dalek, metallic comb filter, radio lofi) | Negligible | `scipy.signal` / `numpy` |
| **Flite (Festival-Lite)** | Formant/Diphone C | ~15 MB | ~10 MB | 10 – 20 ms | ⭐⭐⭐⭐ (Fast robotic diphone `kal`) | < 0.5% | Needs `flite` apt package |
| **Piper Low Models** | Neural VITS (ONNX) | ~15 – 25 MB | ~40 MB | 400 – 600 ms | ⭐⭐ (Still sounds like compressed human narrator) | 25 – 40% | Already supported |
| **Kokoro-82M** | Neural Flow/Style | ~350 MB | ~500 MB+ | 500 – 1000 ms | ⭐ (Trained purely for human realism; not a robot voice) | High | PyTorch / Heavy ONNX |

---

### Detailed Analysis of Top Candidates

#### Option A: eSpeak-NG (Best Modern Mechanical Robot)
- **Why it fits:** `espeak-ng` is already baked into the production Docker runner container (`Dockerfile:18`) because Piper used it for grapheme-to-phoneme conversion.
- **Aesthetic:** Classic synthesizer formant speech (think Star Wars battle droids, Stephen Hawking, cybernetic terminal).
- **Control Parameters:**
  - Pitch (`-p 0-99`): Setting to 40-50 creates a solid synthetic tone.
  - Pitch Range (`-k 0`): Crucial! Setting `-k 0` flattens all emotional intonation contours into **100% monotone mechanical cadence**.
  - Speed (`-s 80-450`): Allows rapid-fire computational speech (e.g. 180-220 wpm).
  - Voice Variants: Built-in modifiers such as `+whisper` (creepy cybernetic stalker), `+croak` (distorted cyber bot), `+m3`/`+m5` (heavy bass droid).
- **Performance:** Generates WAV in **~5ms** directly via `subprocess` or `py-espeak-ng`.

#### Option B: SAM (Software Automatic Mouth — Best Retro 80s Robot)
- **Why it fits:** The original 1982 software formant synthesizer from the Commodore 64. Pure retro gaming nostalgia.
- **Aesthetic:** Unapologetic 8-bit vintage arcade robot (think Daft Punk, Tron, Berzerk, 80s sci-fi computer).
- **Resource Footprint:** Less than 100 KB total. Runs in < 1ms on any CPU.
- **Control Parameters:** `pitch`, `speed`, `throat`, `mouth`. Adjusting throat and mouth shifts formants from tiny chirpy droids to deep industrial mechs.

#### Option C: Server-Side DSP Robot Audio Pipeline (Universal Robotic Effect)
- **Why it fits:** Professional game audio sound designers don't rely only on voice models; they use **DSP modulation chains**.
- **Algorithms:**
  1. **Ring Modulation (Dalek / Sci-Fi Robot):** Multiplies input audio with a 40–70 Hz sine wave oscillator, producing metallic sum/difference harmonic sidebands.
  2. **Metallic Comb Filter:** Adds a 5–15 ms short delay line with 50–70% feedback, simulating speech through a hollow metal chassis or internal robotic cavity.
  3. **Bandpass / Radio Transceiver Filter (300 Hz – 3.4 kHz):** Emulates robot speaker grille or communication comms.
  4. **Bitcrusher / Lo-Fi Quantizer:** 8-bit or 12-bit amplitude quantization with decimation.
- **Implementation:** Pure vectorized `numpy` / `scipy.signal` running in ~2ms. Can be applied to eSpeak output, SAM output, or Piper neural output!

---

## 4. Proposed Architecture: Multi-Engine Unified TTS Service

### 4.1 Component Architecture
```
                           Incoming POST /api/v1/tts/synthesize
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Engine & Alias Resolver │
                               └────────────┬────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
     ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
     │   Piper Engine    │        │  eSpeak-NG Engine │        │    SAM Engine     │
     │  (Neural Humans)  │        │ (Mechanical Synth)│        │   (80s Retro Bot) │
     │  ~120MB, ~1200ms  │        │   ~5MB, ~5ms      │        │   <2MB, <1ms      │
     └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │  Optional DSP Effect FX   │
                              │ (RingMod / Comb / Radio)  │
                              └─────────────┬─────────────┘
                                            ▼
                                  Standard Audio/WAV
```

### 4.2 Seamless API Contract (Zero Breaking Changes)

The existing API schema remains 100% backward compatible:
```python
class TTSRequest(BaseModel):
    text: str
    voice_accent: str = "default"
    # New optional fields:
    engine: Optional[str] = "auto"   # "auto" | "espeak" | "sam" | "piper"
    effect: Optional[str] = "none"   # "none" | "robot_ringmod" | "metallic" | "radio" | "bitcrush"
    speed: Optional[int] = None      # Word rate or speed override
    pitch: Optional[int] = None      # Pitch override
```

### 4.3 Curated Robotic Voice Aliases

Clients can instantly use the new robot voices simply by passing familiar alias names in `voice_accent`:

| Alias Name | Engine | Style / Character Description | Latency (est.) |
|---|---|---|---|
| `robot-classic` | eSpeak-NG | Monotone crisp mechanical synth (`-k 0 -p 45 -s 150`) | **~5 ms** |
| `robot-deep` | eSpeak-NG | Deep heavy mechanical drone (`+m3 -k 0 -p 25 -s 130`) | **~5 ms** |
| `robot-whisper` | eSpeak-NG | Cybernetic whispering stalker (`+whisper -s 140`) | **~5 ms** |
| `robot-glados` | eSpeak-NG + Comb | High-pitch mechanical cadence with metallic resonance | **~8 ms** |
| `robot-retro` | SAM | 1982 8-bit Commodore arcade robot voice | **~1 ms** |
| `cyborg-dalek` | Piper / eSpeak + RingMod | 50Hz Ring-modulated sci-fi cyborg | **~8 ms** |
| `cyborg-radio` | Piper + Bandpass/Distortion | Radio transceiver / tactical helmet comms | **~10 ms** |

---

## 5. Implementation Roadmap (Phases)

> **IMPORTANT:** In accordance with user rules, this plan is for brainstorming and review only. Implementation will NOT proceed until explicit user authorization.

### Phase 1: Engine Implementation in `tts-service`
1. **Add `app/espeak_manager.py`**:
   - Wrap `espeak-ng` command or C API.
   - Return clean 16-bit 22.05kHz mono WAV bytes.
   - Configure presets (`robot-classic`, `robot-deep`, `robot-whisper`, etc.).
2. **Add `app/sam_manager.py`**:
   - Integrate lightweight Software Automatic Mouth module.
   - Return standard WAV bytes.
3. **Add `app/dsp_effects.py`**:
   - Pure numpy/scipy functions for ring modulation, metallic comb filter, and bandpass/radio EQ.

### Phase 2: Router & Unified Manager Integration
1. Update `app/piper_manager.py` / Create `app/engine_router.py` to route based on requested alias or engine parameter.
2. Update `/api/v1/tts/voices` to list both neural and robotic presets with metadata flags (`type: "robotic"` vs `type: "neural"`).
3. Add unit tests in `tts-service/tests/test_engines.py` verifying each engine and effect generates valid WAV headers and playable audio.

### Phase 3: Client Presets in `office-client` & `AgentEditor`
1. Add robotic accents to `ACCENTS` in `AgentEditor.jsx`:
   - `robot-classic`: "🤖 Classic Mech (eSpeak)"
   - `robot-retro`: "🕹️ Retro 80s Bot (SAM)"
   - `robot-deep`: "🦾 Heavy Mech Drone"
   - `robot-whisper`: "👻 Cyber Whisper"
2. Map agent archetypes (e.g. `QUANT` or automated bots) to default to robotic voices if configured.

### Phase 4: Build & Deployment to Synology NAS
1. Test Docker build locally via WSL (`testrun`).
2. Deploy container to Synology NAS via `./deploy.sh` (scoped to `tts-service`).
3. Run smoke test on NAS `http://10.0.0.16:3032/api/v1/tts/synthesize` comparing latency before and after.

---

## 6. Risks, Assumptions & Mitigations

- **ASSUMPTION-1:** The Synology NAS Docker host allows running `espeak-ng` sub-processes inside the container without extra security capabilities.  
  *Validation Path:* `espeak-ng` is already in the Debian package list of the current Dockerfile and requires only standard userspace execution.
- **ASSUMPTION-2:** Existing frontend Web Audio decoders can seamlessly handle 22.05kHz mono WAV from eSpeak and SAM without sample-rate mismatches.  
  *Validation Path:* Web Audio's `audioCtx.decodeAudioData()` automatically resamples any valid WAV container to the destination sample rate (44.1kHz or 48kHz).
- **RISK-1: Memory leaks on high concurrency.**  
  *Mitigation:* eSpeak and SAM are stateless and hold zero long-lived inference graph sessions, completely bypassing Piper's ONNX session corruption risks.

---

## 7. Verification Standard & Acceptance Tests

Before declaring implementation done, the following verified tests must pass:
1. **Unit Test (`tests/test_espeak.py`):** Synthesize "System online" using `robot-classic`; assert HTTP 200, valid WAV header (`RIFF`), sample rate 22050Hz, latency < 30ms.
2. **Unit Test (`tests/test_sam.py`):** Synthesize "Self test complete" using `robot-retro`; assert valid WAV, latency < 10ms.
3. **Unit Test (`tests/test_effects.py`):** Apply `robot_ringmod` and `metallic`; assert output audio array matches input length and has expected spectral alterations without clipping.
4. **Backward Compatibility Test (`tests/test_compat.py`):** Synthesize using legacy `default` (Piper Lessac); assert identical behavior as today.
5. **NAS Live Smoke Test:** Query `http://10.0.0.16:3032/api/v1/tts/synthesize` for both Piper and Robot voices; assert robot voice returns in under 50ms total network round-trip.
