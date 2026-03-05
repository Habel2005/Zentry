import json
import base64
import audioop
import numpy as np
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
from session.session_store import SessionStore
from llm.brain import init_globals
# Import your modules
from backend.call_pipeline import CallPipeline
from backend.call_context import CallContext
from db.call_repo import start_call
from backend.stt_worker import MalayalamSTT
from tts.tts_module import TTSModule

# --- GLOBAL MODELS ---
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE when server starts
    print("⏳ Loading AI Models...")
    # Using 'cpu' to save VRAM for LLM as per codebase patterns,
    # but the user mentioned being efficient.
    # STT usually needs GPU for speed, TTS can be CPU/GPU.
    # The existing code used "models/whisper" and "models/tts/tts_mal.onnx"
    models["stt"] = MalayalamSTT("models/whisper")
    # Change from the local folder to the standard Hugging Face ID
    models["tts"] = TTSModule()

    print("🧠 Initializing Session Memory...")
    global_store = SessionStore()
    init_globals(global_store)
    
    print("✅ AI Models Ready!")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    pipeline = None
    stream_sid = None

    # State for upsampling (8k -> 16k)
    upsample_state = None

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                print(f"📞 Incoming Call: {stream_sid}")

                # Setup Context
                caller = data['start']['customParameters'].get('caller', 'unknown')
                ctx = CallContext(uuid=stream_sid, phone=caller)

                # Initialize DB entry
                ctx.call_id, ctx.caller_id = start_call(stream_sid, ctx.phone)

                # Initialize Pipeline with Global Models
                pipeline = CallPipeline(ctx, websocket, models["stt"], models["tts"])
                pipeline.is_twilio = True
                print(f"✅ Pipeline Attached: {stream_sid}")

            elif data['event'] == 'media' and pipeline:
                payload = data['media']['payload']
                chunk_mulaw = base64.b64decode(payload)

                # 1. Decode Mu-law -> PCM 16-bit (still 8000Hz)
                chunk_pcm_8k = audioop.ulaw2lin(chunk_mulaw, 2)

                # 2. UPSAMPLE 8000Hz -> 16000Hz (Required for Whisper)
                # ratecv(fragment, width, channels, in_rate, out_rate, state)
                chunk_pcm_16k, upsample_state = audioop.ratecv(
                    chunk_pcm_8k, 2, 1, 8000, 16000, upsample_state
                )

                # 3. Send clean 16k audio to VAD/STT
                await pipeline.handle_audio(chunk_pcm_16k)

            elif data['event'] == 'stop':
                print(f"❌ Call Ended: {stream_sid}")
                if pipeline: await pipeline.cleanup()
                break

    except Exception as e:
        print(f"⚠️ Twilio WebSocket Error: {e}")
    finally:
        if pipeline:
            await pipeline.cleanup()
