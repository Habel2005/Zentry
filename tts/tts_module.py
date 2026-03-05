import numpy as np
import sounddevice as sd
import logging
import torch
import torchaudio
from piper import PiperVoice

class TTSModule:
    def __init__(self, model_path="models/tts/ml_IN-meera-medium.onnx", device="cuda"):
        """
        Initializes the Piper TTS engine. 
        It will attempt to use CUDA via ONNXRuntime-GPU if available, otherwise falls back to CPU.
        """
        self.device = device
        use_cuda = True if device == "cuda" else False
        
        print(f"🚀 Loading Piper TTS Module: {model_path}...")
        try:
            self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
            print(f"✅ Piper TTS Module Loaded (CUDA: {use_cuda})")
        except Exception as e:
            print(f"⚠️ Failed to load with CUDA: {e}. Falling back to CPU...")
            self.voice = PiperVoice.load(model_path, use_cuda=False)

        # Piper's native sample rate for medium models is usually 22050Hz
        self.native_sr = self.voice.config.sample_rate

    def tell(self, text, play=True, sr=8000):
        """
        Synthesizes text to audio.
        Returns a float32 numpy array resampled to the requested target sample rate (sr).
        """
        if not text or not text.strip():
            return np.zeros(sr, dtype=np.float32)

        try:
            audio_bytes = b""
            
            # The official Piper streaming API yields chunks directly in memory
            for chunk in self.voice.synthesize(text):
                audio_bytes += chunk.audio_int16_bytes
            
            # 1. Convert raw bytes to 16-bit PCM numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            
            # 2. Normalize to [-1.0, 1.0] standard audio float representation
            audio_np = audio_np / 32767.0
            
            # 3. Resample to target Twilio frequency if needed (Piper is 22050Hz)
            if self.native_sr != sr:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                resampled = torchaudio.functional.resample(
                    audio_tensor, 
                    orig_freq=self.native_sr, 
                    new_freq=sr
                )
                audio_np = resampled.squeeze().numpy()

            # 4. Smart Normalization / Volume Boosting
            max_val = np.abs(audio_np).max()
            if max_val > 0.01:
                audio_np = audio_np / max_val
                
            mean_val = np.abs(audio_np).mean()
            if mean_val < 0.01:
                print(f"⚠️ TTS WARNING: Audio is very quiet! (Mean: {mean_val:.4f})")

            print(f"🔊 TTS Gen ({sr}Hz): {len(audio_np)} samples, Peak Amp: {max_val:.4f}")

            # 5. Optional Playback
            if play:
                try:
                    sd.play(audio_np, samplerate=sr)
                    sd.wait()
                except Exception as e:
                    pass # Fails cleanly on headless servers

            return audio_np

        except Exception as e:
            logging.error(f"TTS Error: {e}")
            return np.zeros(sr, dtype=np.float32)