# backend/stt_worker.py
import asyncio
import math
from faster_whisper import WhisperModel

class MalayalamSTT:
    def __init__(self, model_path):
        print(f"⚙️ Loading Whisper Model: {model_path}")
        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16" # Use int8_float16 if VRAM is tight
        )
        self.gpu_lock = asyncio.Semaphore(3)

    async def transcribe(self, audio_bytes, sample_rate=16000):
        async with self.gpu_lock:
            return await asyncio.to_thread(self._sync_transcribe, audio_bytes, sample_rate)

    def _sync_transcribe(self, audio_bytes, sample_rate):
        import numpy as np
        
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        if sample_rate != 16000:
            num_samples = len(audio_array)
            target_num_samples = int(num_samples * 16000 / sample_rate)
            audio_array = np.interp(
                np.linspace(0.0, 1.0, target_num_samples, endpoint=False),
                np.linspace(0.0, 1.0, num_samples, endpoint=False),
                audio_array
            )

        # Retrieve both segments AND info
        segments_gen, info = self.model.transcribe(audio_array, language="ml", beam_size=1)
        
        text = ""
        logprobs = []
        
        for s in segments_gen:
            text += s.text + " "
            logprobs.append(s.avg_logprob)
            
        text = text.strip()
        
        # Calculate a pseudo-confidence score (0.0 to 1.0)
        avg_logprob = sum(logprobs)/len(logprobs) if logprobs else -1.0
        confidence = math.exp(avg_logprob) if avg_logprob < 0 else 1.0
        
        # Now we return text, confidence, and language
        return text, confidence, info.language