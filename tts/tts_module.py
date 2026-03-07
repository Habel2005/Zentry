import numpy as np
import sounddevice as sd
import re
import logging
import torch
import torchaudio
from piper import PiperVoice

import re

def normalize_for_piper(text):
    if not text:
        return text

    # 1. PHONE NUMBERS: Add spaces between sequences of 5+ digits
    # Fixes: "9995043464" -> "9 9 9 5 0 4 3 4 6 4"
    text = re.sub(r'\b(\d{5,})\b', lambda m: ' '.join(m.group(1)), text)

    # 2. URLS & EMAILS (Newly found in TIST RAG Data)
    tist_entities = {
        r"https://tistcochin\.edu\.in": "ടിസ്റ്റ് കൊച്ചിൻ ഡോട്ട് ഇ ഡി യു ഡോട്ട് ഇൻ",
        r"tistcochin\.edu\.in": "ടിസ്റ്റ് കൊച്ചിൻ ഡോട്ട് ഇ ഡി യു ഡോട്ട് ഇൻ",
        r"admission@tistcochin\.edu\.in": "അഡ്മിഷൻ അറ്റ് ടിസ്റ്റ് കൊച്ചിൻ ഡോട്ട് ഇ ഡി യു ഡോട്ട് ഇൻ",
        r"certificate@tistcochin\.edu\.in": "സർട്ടിഫിക്കറ്റ് അറ്റ് ടിസ്റ്റ് കൊച്ചിൻ ഡോട്ട് ഇ ഡി യു ഡോട്ട് ഇൻ",
        r"placement@tistcochin\.edu\.in": "പ്ലേസ്മെൻ്റ് അറ്റ് ടിസ്റ്റ് കൊച്ചിൻ ഡോട്ട് ഇ ഡി യു ഡോട്ട് ഇൻ",
        r"placement\.tist@gmail\.com": "പ്ലേസ്മെൻ്റ് ഡോട്ട് ടിസ്റ്റ് അറ്റ് ജിമെയിൽ ഡോട്ട് കോം",
        r"cee\.kerala\.gov\.in": "സി ഇ ഇ ഡോട്ട് കേരള ഡോട്ട് ഗവ് ഡോട്ട് ഇൻ",
        r"cee-kerala\.org": "സി ഇ ഇ ഹൈഫൺ കേരള ഡോട്ട് ഓർഗ്",
        r"dtekerala\.gov\.in": "ഡി ടി ഇ കേരള ഡോട്ട് ഗവ് ഡോട്ട് ഇൻ",
        r"lbscentre\.in": "എൽ ബി എസ് സെൻ്റർ ഡോട്ട് ഇൻ",
        r"@": " അറ്റ് ", 
    }

    for pattern, phonetic in tist_entities.items():
        text = re.sub(pattern, phonetic, text, flags=re.IGNORECASE)

    # 3. ACRONYMS & SYMBOLS (Newly found in TIST RAG Data)
    acronyms = {
        r"\bToc H\b": "ടോക്ക് എച്ച്",
        r"\bTocH\b": "ടോക്ക് എച്ച്",
        r"\bTIST\b": "ടിസ്റ്റ്",
        r"\bB\.Tech\b": "ബി ടെക്",
        r"\bM\.Tech\b": "എം ടെക്",
        r"\bMBA\b": "എം ബി എ",
        r"\bNAAC\b": "നാക്",
        r"\bNBA\b": "എൻ ബി എ",
        r"\bCUSAT\b": "കുസാറ്റ്",
        r"\bKTU\b": "കെ ടി യു",
        r"\bKEAM\b": "കീം",
        r"\bJEE\b": "ജെ ഇ ഇ",
        r"\bLET\b": "എൽ ഇ ടി",
        r"\bNRI\b": "എൻ ആർ ഐ",
        r"\bAICTE\b": "എ ഐ സി ടി ഇ",
        r"\bUGC\b": "യു ജി സി",
        r"\bDTE\b": "ഡി ടി ഇ",
        r"₹": "രൂപ ",
        r"&": " കൂടാതെ ",
        r"%": " ശതമാനം "
    }

    for pattern, phonetic in acronyms.items():
        text = re.sub(pattern, phonetic, text, flags=re.IGNORECASE)

    return text

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
        
        clean_text = normalize_for_tts(text)

        try:
            audio_bytes = b""
            
            # The official Piper streaming API yields chunks directly in memory
            for chunk in self.voice.synthesize(clean_text):
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
        


