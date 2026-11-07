import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import traceback
import uvicorn
from app.piper_manager import tts_manager
from app import synth_manager

app = FastAPI(title="TTS Service", description="Standalone Piper TTS Neural Engine")

class TTSRequest(BaseModel):
    text: str
    voice_accent: str = "default"
    # "piper" (neural, default) or "synth" (our from-scratch formant engine).
    # For "synth", voice_accent selects a voice type from synth_manager.
    engine: str = "piper"

@app.post("/api/v1/tts/synthesize")
async def synthesize_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Strip emojis and exotic symbols to prevent ONNX GatherElements out-of-range crashes
        safe_text = re.sub(r'[^\w\s\.,!\?\-\'":;À-ÿ]', '', req.text)
        if not safe_text.strip():
            safe_text = "Audio unavailable."
            
        engine = (req.engine or "piper").strip().lower()
        if engine == "synth":
            voice = req.voice_accent if req.voice_accent in synth_manager.list_voices() else "narrator"
            wav_bytes = synth_manager.synthesize(text=safe_text, voice_name=voice)
        else:
            wav_bytes = tts_manager.synthesize(text=safe_text, voice_name=req.voice_accent)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@app.get("/api/v1/tts/voices")
def list_voices():
    """Returns the list of available Piper voice models on disk and accent aliases."""
    voices = tts_manager.list_voices()
    return {
        "voices": voices,
        "count": len(voices),
        "aliases": tts_manager.list_aliases(),
        "models_dir": tts_manager.models_dir
    }

@app.get("/api/v1/tts/synth-voices")
def list_synth_voices():
    """Voice types of the from-scratch synthesiser (engine="synth")."""
    names = synth_manager.list_voices()
    return {
        "voices": [{"name": n, **synth_manager.voice_info(n)} for n in names],
        "count": len(names),
        "default": "narrator",
        "engine": "synth",
        "sample_rate": synth_manager.SAMPLE_RATE,
    }

@app.get("/health")
def health():
    voice_count = len(tts_manager.list_voices())
    return {
        "status": "ok",
        "service": "tts-service",
        "voices_available": voice_count,
        "synth_voices": len(synth_manager.list_voices()),
        "engines": ["piper", "synth"],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
