import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

print("🚀 Starting PURE NATIVE 3080 Ti Test...")

device = "cuda"

# 1. Load the original model in native float16 (No 'quanto' bugs!)
print("📥 Loading Original Model natively...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts", 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
).to(device)

# 2. Load tokenizers directly
print("📖 Loading tokenizers...")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# 3. Persona and Text
text = "നമസ്കാരം! ഞാൻ സെൻട്രിയാണ്. എൻറെ ശബ്ദം ഇപ്പോൾ കൃത്യമായി കേൾക്കാമോ?"
description = (
    "Anjali, a young female speaker, delivers a gentle, empathetic, and warm speech "
    "with a soft, comforting tone perfect for an admission assistant. "
    "The recording is very high quality, very clear, and close-up, with no background noise."
)

desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 4. Generate!
print("\n🗣️ Generating audio natively on Tensor Cores...")
with torch.no_grad():
    generation = model.generate(
        input_ids=desc_inputs.input_ids,
        prompt_input_ids=prompt_inputs.input_ids
    )

audio_arr = generation.cpu().numpy().squeeze()

# If it works, this shape should be around 100,000 to 200,000 (several seconds of audio)
print(f"📊 Array Shape: {audio_arr.shape}") 

output_filename = "zentry_native_success.wav"
sf.write(output_filename, audio_arr, model.config.sampling_rate)

print(f"✅ Success! Check {output_filename}")