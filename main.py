import random
from Phi3.brain import Brain
from translate import Translator
from TTS import TTSModule
# Pre-recorded fallback audios (Malayalam)
FALLBACK_AUDIOS = [
    "audio/fallback_1.wav",
    "audio/fallback_2.wav",
    "audio/fallback_3.wav",
]

# Fallback keywords/phrases (lowercased for detection)
FALLBACK_KEYWORDS = [
    "not available",
    "i don’t have that info",
    "that information isn’t provided",
    "only answer from official admission",
    "no info available"
]

def is_fallback(response: str) -> bool:
    """Check if Phi3 response is a fallback using fuzzy keyword match"""
    resp = response.lower()
    return any(keyword in resp for keyword in FALLBACK_KEYWORDS)

def play_fallback():
    """Play a random prerecorded fallback audio file"""
    file = random.choice(FALLBACK_AUDIOS)
    speech, sr = sf.read(file, dtype="float32")
    print(f"🔊 Played fallback audio: {file}")

    
if __name__ == "__main__":
    # Init models once
    phi3 = Brain(model_path="models/phi3-mini/phi3-mini.gguf")
    translator = Translator()
    tts = TTSModule(model_path="models/tts/tts_mal.onnx")

    print("👩‍🎓 Zentry AI Assistant (Phi3 → Eng-Mal → TTS)")
    while True:
        q = input("\nYou (English): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # 1. English Q → Phi-3 answer in English
        eng_answer = phi3.ask(q)
        print(f"Phi3 (Eng): {eng_answer}")

        if is_fallback(eng_answer):
            print("⚠️ Fallback detected, skipping translation + TTS")
            play_fallback()
            continue

        # 2. English → Malayalam
        mal_text = translator.translate(eng_answer, direction="en-ml")
        print(f"Translated (Mal): {mal_text}")

        # 3. Malayalam → TTS
        print("🔊 Speaking...")
        tts.tell(mal_text)
