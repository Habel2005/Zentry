import numpy as np
import wave
import os
from tts.tts_module import TTSModule

def generate_all_assets():
    os.makedirs("assets", exist_ok=True)
    
    # Initialize your new Piper TTS
    print("Initializing TTS for Asset Generation...")
    tts = TTSModule(model_path="models/tts/ml_IN-meera-medium.onnx", device="cpu")

    # Define the phrases. Notice we have 3 different "wait" variations!
    phrases = {
        "intro": "ഹലോ, ഞാൻ സെൻട്രി. നിങ്ങളുടെ അഡ്മിഷൻ അസിസ്റ്റന്റ് ആണ്. എനിക്ക് എങ്ങനെ സഹായിക്കാനാകും?",
        "error": "ക്ഷമിക്കണം, നിങ്ങൾ പറഞ്ഞത് എനിക്ക് വ്യക്തമായില്ല. ഒന്നുകൂടി പറയാമോ?",
        "fallback": "ക്ഷമിക്കണം, ആ വിവരം ഇപ്പോൾ എന്റെ കൈവശമില്ല. ദയവായി വെബ്സൈറ്റ് പരിശോധിക്കുക.",
        "wait1": "ഒന്ന് നിൽക്കൂ, ഞാൻ അതൊന്ന് പരിശോധിക്കട്ടെ.",
        "wait2": "ഒരു നിമിഷം, ഞാൻ വിവരങ്ങൾ നോക്കുകയാണ്.",
        "wait3": "ശരി, ഞാൻ അതൊന്ന് നോക്കട്ടെ."
    }

    for name, text in phrases.items():
        print(f"🎙️ Generating {name}.wav...")
        # Get float32 audio at 16000Hz
        audio_fp32 = tts.tell(text, play=False, sr=16000)
        
        # Convert [-1.0, 1.0] float32 to 16-bit PCM integer
        audio_int16 = (audio_fp32 * 32767).astype(np.int16)
        
        # Write to WAV file
        filepath = f"assets/{name}.wav"
        with wave.open(filepath, "w") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 2 bytes = 16-bit
            wav_file.setframerate(16000)  # 16000Hz sample rate
            wav_file.writeframes(audio_int16.tobytes())
            
        print(f"✅ Saved {filepath}")

if __name__ == "__main__":
    generate_all_assets()