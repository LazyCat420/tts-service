from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import traceback
import uvicorn
from app.piper_manager import tts_manager

app = FastAPI(title="TTS Service", description="Standalone Piper TTS Neural Engine")

class TTSRequest(BaseModel):
    text: str
    voice_accent: str = "default"

@app.post("/api/v1/tts/synthesize")
async def synthesize_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        wav_bytes = tts_manager.synthesize(text=req.text, voice_name=req.voice_accent)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "tts-service"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
