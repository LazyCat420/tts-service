import os
import io
import wave
import time
import threading
from typing import Dict, List, Optional
from collections import OrderedDict

try:
    import json
    import onnxruntime
    from piper import PiperVoice
    from piper.voice import PiperConfig
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False
    print("WARNING: piper-tts is not installed. TTS generation will fail.")


class PiperTTSManager:
    """
    Manages Piper TTS voice models with LRU eviction to minimize RAM usage.
    Only keeps a limited number of voices loaded in memory at any time.
    """

    # Maximum number of voices to keep loaded in memory simultaneously.
    # Each medium-quality model uses ~60MB RAM, so 2 = ~120MB max.
    MAX_LOADED_VOICES = 2

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.environ.get(
                "PIPER_MODELS_DIR",
                os.path.join(os.path.dirname(__file__), "..", "data", "piper_models"),
            )
        self.models_dir = os.path.abspath(models_dir)

        # LRU cache: OrderedDict preserves insertion order; we move-to-end on access.
        self._cache: OrderedDict[str, "PiperVoice"] = OrderedDict()
        self._lock = threading.Lock()
        # One synthesis lock per voice. `_lock` protects the CACHE only and is
        # released before inference runs, so it cannot serialise synthesis —
        # which is what let concurrent requests corrupt a shared ONNX session.
        self._synth_locks: Dict[str, threading.Lock] = {}

        # ── Accent aliases ────────────────────────────────────────
        # Map friendly accent codes → actual model names on disk.
        self.aliases: Dict[str, str] = {
            # American
            "default":       "en_US-lessac-medium",
            "en-US":         "en_US-lessac-medium",
            "en-US-FEMALE":  "en_US-amy-medium",
            "en-US-MALE":    "en_US-ryan-medium",
            "american":      "en_US-lessac-medium",
            "american-f":    "en_US-amy-medium",
            "american-m":    "en_US-ryan-medium",
            # British
            "en-GB":         "en_GB-alan-medium",
            "british":       "en_GB-alan-medium",
            "british-f":     "en_GB-alba-medium",
            "british-m":     "en_GB-alan-medium",
            # Indian
            "indian":        "hi_IN-pratham-medium",
            "indian-m":      "hi_IN-pratham-medium",
            "indian-f":      "hi_IN-priyamvada-medium",
            "hindi":         "hi_IN-pratham-medium",
            # European
            "french":        "fr_FR-tom-medium",
            "german":        "de_DE-thorsten-medium",
            "european":      "fr_FR-tom-medium",
            # South American
            "brazilian":     "pt_BR-faber-medium",
            "portuguese":    "pt_BR-faber-medium",
            # African
            "african":       "sw_CD-lanfrica-medium",
            "swahili":       "sw_CD-lanfrica-medium",
        }

        print(f"[PiperTTS] Models directory: {self.models_dir}")
        print(f"[PiperTTS] Max voices in memory: {self.MAX_LOADED_VOICES}")
        available = self.list_voices()
        print(f"[PiperTTS] Found {len(available)} voice models on disk")

    def _get_model_path(self, voice_name: str) -> tuple:
        """Resolves alias and returns (onnx_path, resolved_name)."""
        name = self.aliases.get(voice_name, voice_name)

        onnx_path = os.path.join(self.models_dir, f"{name}.onnx")
        if not os.path.exists(onnx_path):
            fallback = self.aliases.get("default", "en_US-lessac-medium")
            print(
                f"[PiperTTS] WARNING: Model '{name}' not found at {onnx_path}, "
                f"falling back to '{fallback}'"
            )
            name = fallback
            onnx_path = os.path.join(self.models_dir, f"{name}.onnx")

        return onnx_path, name

    def _evict_lru(self) -> None:
        """Remove the least-recently-used voice from cache if at capacity."""
        while len(self._cache) >= self.MAX_LOADED_VOICES:
            evicted_name, _ = self._cache.popitem(last=False)
            print(f"[PiperTTS] Evicted '{evicted_name}' from memory (LRU)")

    def load_voice(self, voice_name: str) -> "PiperVoice":
        """Loads a voice into memory with LRU eviction."""
        if not HAS_PIPER:
            raise RuntimeError("piper-tts is not installed.")

        onnx_path, resolved_name = self._get_model_path(voice_name)

        with self._lock:
            # Cache hit — move to end (most recently used)
            if resolved_name in self._cache:
                self._cache.move_to_end(resolved_name)
                return self._cache[resolved_name]

            # Cache miss — evict LRU if needed, then load
            json_path = f"{onnx_path}.json"
            if not os.path.exists(onnx_path) or not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"Missing ONNX or JSON file for {resolved_name} at {self.models_dir}"
                )

            self._evict_lru()

            print(f"[PiperTTS] Loading voice into memory: {resolved_name}...")
            t0 = time.time()
            voice = self._load_voice_safe_session(onnx_path, json_path)
            self._cache[resolved_name] = voice
            elapsed = time.time() - t0
            print(
                f"[PiperTTS] Loaded {resolved_name} in {elapsed:.2f}s "
                f"({len(self._cache)}/{self.MAX_LOADED_VOICES} slots used)"
            )
            return voice

    @staticmethod
    def _load_voice_safe_session(onnx_path: str, json_path: str) -> "PiperVoice":
        """Load a PiperVoice on an InferenceSession with mem-pattern OFF.

        onnxruntime 1.18.1's memory-pattern optimisation corrupts ScatterND
        indices in these graphs on the SECOND AND LATER runs of a session
        (piper runs one inference per sentence, so multi-sentence text fails
        inside a single request too). The symptom is
        "invalid indice found, indice = <garbage int64>" from /dp/ (the
        duration predictor) — deterministic per text, not a race and not
        session decay. Measured in the prod container: the failing text went
        0/6 on a stock session and 10/10 with enable_mem_pattern=False.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            config = PiperConfig.from_dict(json.load(f))
        opts = onnxruntime.SessionOptions()
        opts.enable_mem_pattern = False
        session = onnxruntime.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        return PiperVoice(config=config, session=session)

    def _synth_lock_for(self, resolved_name: str) -> threading.Lock:
        """The per-voice synthesis lock, created on first use."""
        with self._lock:
            lock = self._synth_locks.get(resolved_name)
            if lock is None:
                lock = self._synth_locks[resolved_name] = threading.Lock()
            return lock

    def synthesize(self, text: str, voice_name: str = "default") -> bytes:
        """Synthesizes text to a WAV byte string.

        Synthesis is SERIALISED per voice. `self._lock` guarded only the cache
        lookup, so concurrent requests obtained the same PiperVoice and ran
        inference on its ONNX session simultaneously — which corrupts it. The
        symptom was an ONNXRuntimeError with a garbage index
        ("indice = 4601523038094714111", a float bit-pattern read as an int64)
        from a random node, so it looked like bad input rather than a race.

        Worse, the damage was PERMANENT: the corrupted voice stayed in the LRU
        cache, so every later request failed for the process lifetime. Verified:
        sequential requests all returned 200 after a restart; 6 concurrent gave
        2x200 + 4x500; sequential afterwards were 500 forever.

        Piper inference is CPU-bound and releases the GIL inside ONNX Runtime, so
        serialising costs throughput but not correctness — and a wrong answer at
        higher throughput is worth nothing.
        """
        onnx_path, resolved_name = self._get_model_path(voice_name)
        last_error: Optional[Exception] = None

        # The root cause (onnxruntime mem-pattern corrupting ScatterND on 2nd+
        # runs) is fixed at load time in _load_voice_safe_session. The retry
        # stays as a backstop: on any residual failure the session is evicted
        # and attempt 2 gets a clean load. Note the old "retry-on-fresh-session"
        # theory was incomplete — failures were text-deterministic, so a retry
        # alone could NOT fix them (both attempts failed on the same text).
        for attempt in (1, 2):
            voice = self.load_voice(voice_name)
            try:
                return self._synthesize_once(voice, resolved_name, text)
            except Exception as e:  # noqa: BLE001 — re-raised below
                last_error = e
                with self._lock:
                    if self._cache.pop(resolved_name, None) is not None:
                        print(f"[PiperTTS] Evicted '{resolved_name}' after a synthesis "
                              f"error (attempt {attempt}); reloading for a clean session")
        raise last_error  # type: ignore[misc]

    def _synthesize_once(self, voice: "PiperVoice", resolved_name: str, text: str) -> bytes:
        """One synthesis attempt, serialised against other users of this voice.

        The lock matters independently of the retry above: `self._lock` guards only
        the cache lookup and is released before inference, so without this two
        requests could run on the SAME ONNX session at once, which corrupts it for
        every later caller.
        """
        with self._synth_lock_for(resolved_name):
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                # Do NOT pass speaker_id here. These voices are single-speaker and
                # their graphs have no `sid` input at all: onnxruntime SKIPS a None
                # value in the feed but rejects a real array with
                # "Invalid input name: sid". Tried and measured — it took the
                # failure rate from 50% to 100%.
                voice.synthesize(text, wav_file)
            return wav_io.getvalue()

    def list_voices(self) -> List[str]:
        """Returns a sorted list of available voice model names on disk."""
        voices = []
        if not os.path.isdir(self.models_dir):
            return voices
        for f in os.listdir(self.models_dir):
            if f.endswith(".onnx") and not f.endswith(".onnx.json"):
                json_path = os.path.join(self.models_dir, f + ".json")
                if os.path.exists(json_path):
                    voices.append(f[:-5])  # Strip .onnx extension
        voices.sort()
        return voices

    def list_aliases(self) -> Dict[str, str]:
        """Returns the alias → model name mapping."""
        return dict(self.aliases)


# Singleton instance
tts_manager = PiperTTSManager()
