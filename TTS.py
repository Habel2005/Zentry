# tts_module.py
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from transformers import AutoTokenizer

class TTSModule:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        model_path:path for the onnx tts model,device: default cpu choose 
        """
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")

        # Choose execution providers
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Load ONNX session once
        self.session = ort.InferenceSession(model_path, providers=providers)

    def tell(self, text: str, play: bool = True, sr: int = 16000):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="np")

        # Prepare ONNX input dict
        ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64)}
        if "attention_mask" in inputs:
            ort_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)

        # Run inference
        ort_outputs = self.session.run(None, ort_inputs)

        # Extract + normalize audio
        audio = ort_outputs[0].squeeze().astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.abs(audio).max()

        # Play or just return
        if play:
            sd.play(audio, samplerate=sr)
            sd.wait()

        return audio
