# import numpy as np
# import wave
# import os
# from tts.tts_module import TTSModule

# def generate_all_assets():
#     os.makedirs("assets", exist_ok=True)
    
#     # Initialize your new Piper TTS
#     print("Initializing TTS for Asset Generation...")
#     tts = TTSModule(model_path="models/tts/ml_IN-meera-medium.onnx", device="cpu")

#     # Define the phrases. Notice we have 3 different "wait" variations!
#     phrases = {
#         "intro": "ഹലോ, ഞാൻ സെൻട്രി. നിങ്ങളുടെ അഡ്മിഷൻ അസിസ്റ്റന്റ് ആണ്. എനിക്ക് എങ്ങനെ സഹായിക്കാനാകും?",
#         "error": "ക്ഷമിക്കണം, നിങ്ങൾ പറഞ്ഞത് എനിക്ക് വ്യക്തമായില്ല. ഒന്നുകൂടി പറയാമോ?",
#         "fallback": "ക്ഷമിക്കണം, ആ വിവരം ഇപ്പോൾ എന്റെ കൈവശമില്ല. ദയവായി വെബ്സൈറ്റ് പരിശോധിക്കുക.",
#         "wait1": "ഒന്ന് നിൽക്കൂ, ഞാൻ അതൊന്ന് പരിശോധിക്കട്ടെ.",
#         "wait2": "ഒരു നിമിഷം, ഞാൻ വിവരങ്ങൾ നോക്കുകയാണ്.",
#         "wait3": "ശരി, ഞാൻ അതൊന്ന് നോക്കട്ടെ."
#     }

#     for name, text in phrases.items():
#         print(f"🎙️ Generating {name}.wav...")
#         # Get float32 audio at 16000Hz
#         audio_fp32 = tts.tell(text, play=False, sr=16000)
        
#         # Convert [-1.0, 1.0] float32 to 16-bit PCM integer
#         audio_int16 = (audio_fp32 * 32767).astype(np.int16)
        

#         # Write to WAV file
#         filepath = f"assets/{name}.wav"
#         with wave.open(filepath, "w") as wav_file:
#             wav_file.setnchannels(1)      # Mono
#             wav_file.setsampwidth(2)      # 2 bytes = 16-bit
#             wav_file.setframerate(16000)  # 16000Hz sample rate
#             wav_file.writeframes(audio_int16.tobytes())
            
#         print(f"✅ Saved {filepath}")

# if __name__ == "__main__":
#     generate_all_assets()




import asyncio
import websockets
import json
import base64
import wave
import os

API_KEY = "sk_zh6900yz_tkbf3QJMXUUIfeQTZE2rYipJ"  # ഇവിടെ നിങ്ങളുടെ Sarvam API Key നൽകുക

async def generate_sarvam_asset(name, text):
    uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
    
    headers = {
        "Api-Subscription-Key": API_KEY
    }
    
    print(f"🎙️ Generating {name}.wav...")
    
    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # 1. Send Config (16000Hz, linear16 - No Alien Voice!)
        config_payload = {
            "type": "config",
            "data": {
                "target_language_code": "ml-IN",
                "speaker": "ritu",
                "speech_sample_rate": "16000",
                "output_audio_codec": "linear16" 
            }
        }
        await websocket.send(json.dumps(config_payload))
        
        # 2. Send Text
        await websocket.send(json.dumps({
            "type": "text",
            "data": {"text": text}
        }))
        
        # 3. Send Flush
        await websocket.send(json.dumps({"type": "flush"}))
        
        all_audio_bytes = b""
        
        # 4. Receive Audio Chunks
        while True:
            try:
                message = await websocket.recv()
                response = json.loads(message)
                
                if response.get("type") == "audio":
                    audio_base64 = response["data"]["audio"]
                    audio_bytes = base64.b64decode(audio_base64)
                    all_audio_bytes += audio_bytes
                    
                elif response.get("type") == "event" and response.get("data", {}).get("event_type") == "final":
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                break
                
        # 5. Save to WAV file
        filepath = f"assets/{name}.wav"
        with wave.open(filepath, "wb") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit PCM
            wav_file.setframerate(16000)  # 16000Hz
            wav_file.writeframes(all_audio_bytes)
            
        print(f"✅ Saved {filepath} successfully!")

async def generate_all_assets():
    os.makedirs("assets", exist_ok=True)
    
    # പുതിയ വാചകങ്ങൾ 
    phrases = {
        "welcome": "നമസ്കാരം, ടിസ്റ്റിലേക്ക് സ്വാഗതം. ഞാൻ സെൻട്രി. നിങ്ങളുടെ അഡ്മിഷൻ അസിസ്റ്റന്റ് ആണ്. എനിക്ക് എങ്ങനെ സഹായിക്കാനാകും?",
        "intro": "ഹലോ, ഞാൻ ഇവിടെയുണ്ട്. എന്ത് വിവരമാണ് അറിയേണ്ടത്?",
        "error": "ക്ഷമിക്കണം, നിങ്ങൾ പറഞ്ഞത് എനിക്ക് വ്യക്തമായില്ല. ഒന്നുകൂടി പറയാമോ?",
        "fallback": "ക്ഷമിക്കണം, ആ വിവരം ഇപ്പോൾ എന്റെ കൈവശമില്ല. ദയവായി വെബ്സൈറ്റ് പരിശോധിക്കുക.",
        "wait1": "ഒന്ന് നിൽക്കൂ, ഞാൻ അതൊന്ന് പരിശോധിക്കട്ടെ.",
        "wait2": "ഒരു നിമിഷം, ഞാൻ വിവരങ്ങൾ നോക്കുകയാണ്.",
        "wait3": "ശരി, ഞാൻ അതൊന്ന് നോക്കട്ടെ."
    }

    for name, text in phrases.items():
        await generate_sarvam_asset(name, text)
        await asyncio.sleep(1) # API നിരന്തരമായി വിളിക്കാതിരിക്കാൻ ചെറിയൊരു ഇടവേള

if __name__ == "__main__":
    asyncio.run(generate_all_assets())