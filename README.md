# Zentry: Malayalam AI Telephony Assistant

**Real-Time, AI-Driven Voice Assistant for College Admissions**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=flat-square&logo=twilio&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper_Medium-00A67E?style=flat-square&logo=openai&logoColor=white)
![Phi-4](https://img.shields.io/badge/Phi--4-0078D4?style=flat-square&logo=microsoft&logoColor=white)

Zentry is a real-time AI telephony assistant designed to handle college admission inquiries for TIST (Toc H Institute of Science and Technology). It processes natural spoken Malayalam, retrieves accurate admissions data, and responds contextually over a standard phone call.

---

### 🏗 Architecture & Flow

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1dqbGE94BuyL6ebUeid1CpidQvso7omkY" alt="Zentry Architecture Diagram" width="800">
</p>

The system connects callers via a cloud telephony gateway to a local inference engine. Audio streams are transcribed, translated, processed for intent, and synthesized back into Malayalam speech with sub-second latency targets.

---

### ⚙️ Core Tech Stack

* **Telephony Gateway:** **Twilio** handles incoming calls, bridging the SIP/voice traffic to the backend processing server.
* **Speech-to-Text (STT):** **Whisper Medium (Fine-tuned)** using the custom Malayalam weights trained by *thennal* for superior dialect recognition and accuracy.
* **Translation Layer:** **IndicTrans2** bridges the Malayalam audio transcripts with the English-centric reasoning engine.
* **Reasoning Engine (LLM):** **Phi-4** evaluates queries, fetches TIST-specific admissions data, and constructs the response.
* **Text-to-Speech (TTS):** A hybrid approach utilizing optimized TTS models (incorporating frameworks like Piper and Parler) to generate natural, real-time Malayalam audio.

---

### 🚀 Getting Started

#### Prerequisites
* Ubuntu 22.04 LTS (Recommended) / Windows with WSL2
* Python 3.10+
* Twilio Account (SID, Auth Token, and active phone number)
* CUDA-compatible GPU for local model inference

#### Installation (try to use the "new" branch)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Habel2005/zentry.git](https://github.com/Habel2005/zentry.git)
   cd zentry
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file and add your Twilio credentials and server configurations.

5. **Start the Application:**
   ```bash
   python -m backend.main_server
   ```

---

### 📖 The Journey: Building Zentry

Building an AI that speaks native Malayalam and operates over a phone line required navigating a complex landscape of telecom protocols and rapidly evolving open-source models. Here is the story of how the current stack came to be:

#### The Telephony Struggle: Asterisk -> FreeSWITCH -> Twilio
The initial vision was a completely on-premise PBX system. The journey started with **Asterisk**, but the configuration and SIP trunking complexities proved to be a heavy bottleneck. The next logical step was **FreeSWITCH**, which offered better documentation for modern application integration. However, managing RTP audio streams, compiling modules, and battling firewall NAT issues took focus away from the AI logic. Ultimately, the architecture pivoted to **Twilio**. Offloading the telecom infrastructure to Twilio's reliable cloud APIs allowed for a streamlined focus purely on the conversational AI and low-latency websocket streaming.

#### The LLM Dilemma: Native Models vs. Translation
Finding an LLM that could "think" and "speak" Malayalam accurately was the biggest hurdle. Extensive testing was done in Google Colab, heavily evaluating various open-weight models using custom prompts. 
* **Native Fine-tunes:** Models like Sarvam, and various Malayalam fine-tunes of Llama and Gemma were tested. While promising, they often hallucinated, struggled with complex reasoning regarding college data, or lacked the inference speed needed for real-time voice.
* **The Pivot:** The solution was a translation bridge. By utilizing **IndicTrans2**, Malayalam input is seamlessly translated to English, processed by the highly capable and fast **Phi-4** model, and then translated back. This guaranteed high-quality reasoning without sacrificing linguistic accuracy.

#### Solving the Speech Pipeline (STT & TTS)
* **Hearing (STT):** Standard Whisper models struggled with the specific intonations and speed of conversational Malayalam. The breakthrough came by integrating a **Whisper Medium model fine-tuned by *thennal***, which drastically improved transcription accuracy.
* **Speaking (TTS):** Finding a natural Malayalam voice was an iterative grind. The project cycled through almost every open-source TTS framework available—testing Coqui, exploring MMS (Massively Multilingual Speech), and experimenting with Parler. The final TTS pipeline leverages a tailored configuration (often relying on Piper's efficiency) to balance realistic voice inflection with the strict latency requirements of a live phone call.

Zentry is the result of continuous prototyping, testing, and pivoting to find the perfect balance between local AI inference and reliable telecom infrastructure.



<img src="https://static.scarf.sh/a.png?x-pxid=0b994c4e-62ce-47f6-8af6-27235e610eec" width="0" height="0" alt="" />


<img src="https://omni-dash-five.vercel.app/api/track?project=Zentry&source=github-readme" width="0" height="0" alt="" />