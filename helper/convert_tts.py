import torch
from transformers import VitsModel, AutoTokenizer

# Load model + tokenizer
model_name = "facebook/mms-tts-mal"
print(f"🔄 Loading {model_name}...")
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()  # inference mode

# Dummy input for export
text = "ഹലോ, നിങ്ങൾക്ക് സുഖം താനേ?"
inputs = tokenizer(text, return_tensors="pt")

# File path
onnx_path = "tts_mal.onnx"

print("⚙️ Exporting to ONNX...")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),  # model inputs
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["waveform"],
    opset_version=17,   # >= 17 works well for HF models
    dynamic_axes={
        "input_ids": {1: "sequence"},
        "attention_mask": {1: "sequence"},
    },
)

print(f"✅ Exported ONNX model saved at {onnx_path}")
