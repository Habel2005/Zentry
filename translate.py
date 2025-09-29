# translator.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Domain-specific keyword mapping
PRE_MAP = {"ബി.ടെക്ക്": "B.Tech", "എം.സിഎ": "MCA", "എം.ടെക്": "M.Tech"}
POST_MAP = {v: k for k, v in PRE_MAP.items()}

class Translator:
    def __init__(self, directions=("ml-en", "en-ml")):
        """
        directions: tuple of supported translation directions
        e.g. ("ml-en", "en-ml")
        """
        self.models = {}
        self.tokenizers = {}
        self.directions = directions
        self.ip = IndicProcessor(inference=True)

        for direction in directions:
            if direction == "ml-en":
                model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
                src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
            elif direction == "en-ml":
                model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
                src_lang, tgt_lang = "eng_Latn", "mal_Mlym"
            else:
                raise ValueError(f"Unsupported direction: {direction}")

            print(f"🔄 Loading {direction} model: {model_name} on {DEVICE} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float16 if DEVICE=="cuda" else torch.float32
            ).to(DEVICE)

            # Compile on GPU only
            if DEVICE == "cuda":
                model = torch.compile(model)

            self.models[direction] = model
            self.tokenizers[direction] = tokenizer
            setattr(self, f"{direction}_src_lang", src_lang)
            setattr(self, f"{direction}_tgt_lang", tgt_lang)

        print("✅ Translator initialized successfully.\n")

    def _pre_map(self, text: str, direction: str) -> str:
        # Only apply pre-map for ML→EN
        if direction == "ml-en":
            for k, v in PRE_MAP.items():
                text = re.sub(rf"{re.escape(k)}[\u0D00-\u0D7F]*", v, text)
        return text

    def _post_map(self, text: str, direction: str) -> str:
        # Only apply post-map for EN→ML
        if direction == "en-ml":
            for k, v in POST_MAP.items():
                text = re.sub(re.escape(k), v, text)
        return re.sub(r"<.*?>", "", text).strip()

    def translate(self, text: str, direction="ml-en") -> str:
        """
        Translate a single sentence in the specified direction.
        direction: "ml-en" or "en-ml"
        """
        if direction not in self.models:
            raise ValueError(f"Direction {direction} not initialized")

        model = self.models[direction]
        tokenizer = self.tokenizers[direction]
        src_lang = getattr(self, f"{direction}_src_lang")
        tgt_lang = getattr(self, f"{direction}_tgt_lang")

        text_pre = self._pre_map(text, direction)
        batch = self.ip.preprocess_batch([text_pre], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                use_cache=False  # safe on CPU
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        translated = self.ip.postprocess_batch(decoded, lang=tgt_lang)[0]
        return self._post_map(translated, direction)


# Test example
if __name__ == "__main__":
    translator = Translator()
    ml_text = "എനിക്ക് ബി.ടെക്കിന്റെ അഡ്മിഷൻ തീയതികൾ അറിയണം."
    en_text = translator.translate(ml_text, direction="ml-en")
    print("ML → EN:", en_text)

    en_input = "The last date for B.Tech admission is August 10."
    ml_output = translator.translate(en_input, direction="en-ml")
    print("EN → ML:", ml_output)
