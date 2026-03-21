# Sambashini (Project Zentry)

**Real-Time, On-Premise Malayalam AI Telephony Assistant**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FreeSwitch](https://img.shields.io/badge/FreeSwitch-000000?style=flat-square&logo=freeswitch&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper_STT-00A67E?style=flat-square&logo=openai&logoColor=white)
![Phi-4](https://img.shields.io/badge/Phi--4_Mini-0078D4?style=flat-square&logo=microsoft&logoColor=white)
![Piper TTS](https://img.shields.io/badge/Piper_TTS-FF4500?style=flat-square)

Sambashini is a fully on-premise, AI-driven telephony assistant designed specifically to handle college admission inquiries for TIST (Toc H Institute). It processes natural spoken Malayalam in real-time, providing accurate, conversational responses to prospective students and parents over standard phone calls.

---

### 🏗 Architecture Overview

![Sambashini Architecture Flow]([./assets/architecture-diagram.png](https://drive.google.com/file/d/1dqbGE94BuyL6ebUeid1CpidQvso7omkY/view?usp=sharing))

The system operates entirely locally to ensure data privacy and minimize latency. The pipeline handles raw SIP audio streams, converts Malayalam speech to text, processes the intent via a local LLM, and synthesizes a natural Malayalam voice response.

---

### ⚙️ Core Tech Stack

*   **Telephony Server:** [FreeSWITCH](https://freeswitch.com/) handles incoming SIP trunks and RTP audio streams.
*   **Speech-to-Text (STT):** [OpenAI Whisper](https://github.com/openai/whisper) for high-accuracy Malayalam audio transcription.
*   **Translation Layer:** [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) for bridging Malayalam transcripts with the core reasoning engine.
*   **Large Language Model (LLM):** **Phi-4 Mini** running locally to evaluate queries, fetch admissions data, and generate context-aware responses.
*   **Text-to-Speech (TTS):** **Piper / Parler TTS** configured with custom acoustic models for natural-sounding Malayalam voice generation.

---

### ✨ Key Features

*   **Real-Time Voice Processing:** Optimized pipeline for sub-second latency from speech endpointing to audio playback.
*   **100% On-Premise:** No dependency on external cloud APIs (like OpenAI or Google Cloud) during runtime, ensuring zero recurring API costs and strict data privacy.
*   **Native Malayalam Support:** Tailored specifically for the linguistic nuances of Malayalam speakers.
*   **Admissions Knowledge Base:** Grounded in TIST-specific data (courses, fees, hostel availability, cutoffs) to prevent hallucination.

---

### 🚀 Getting Started

#### Prerequisites
*   Ubuntu 22.04 LTS (Recommended)
*   Python 3.10+
*   FreeSWITCH installed and configured with SIP trunks.
*   CUDA-compatible GPU (NVIDIA RTX 3000 series or higher recommended for real-time STT/LLM inference).

#### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Habel2005/zentry.git](https://github.com/Habel2005/zentry.git)
   cd zentry
   ```

2.  **Set up the virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Weights:**

      * Run the fetching script to download Whisper, IndicTrans2, Phi-4 Mini, and TTS models to the `models/` directory.

    <!-- end list -->

    ```bash
    python scripts/download_models.py
    ```

5.  **Start the FreeSWITCH Event Socket Layer (ESL) Server:**

    ```bash
    python src/main.py
    ```

-----

### 🤝 Contributing

Contributions are welcome. Please ensure that any pull requests maintaining the on-premise, offline-first philosophy of the project.

### 📄 License

[MIT License](https://www.google.com/search?q=LICENSE)
