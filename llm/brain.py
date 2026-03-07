# llm/brain.py
import time
import re
from llm.intent import detect_intent, detector
from llm.engine import PhiEngine
from llm.scheduler import gpu_scheduler, cpu_scheduler
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.rag.retriever import RAGRetriever
from llm.rag.embedder import embedder_instance 
from llm.translate import ml_to_en, en_to_ml
from db.call_repo import log_message
from db.ai_repo import log_processing_step, log_intent
from db.snapshot_repo import get_snapshot

import json
from db.client import init_supabase

async def extract_and_learn(call_id, phone):
# Safety check: If memory isn't loaded, skip learning
    if not session_store:
        print("⚠️ session_store is None. Skipping extract_and_learn.")
        return

    # 1. Fetch the full conversation history from your session store
    session = session_store.get_session(phone)
    history = session.get("history", [])
    
    if len(history) < 2:
        return # Conversation was too short to learn anything
        
    transcript = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history])
    
    # 2. The Extraction Prompt
    # --- UPDATE THE PROMPT ---
    extraction_prompt = f"""
    Read the following conversation transcript. Your goal is to extract facts about the user.
    Look for:
    1. The user's name.
    2. Their true underlying interest regarding college courses.
    3. Did they agree to schedule a consultation?
    4. Their 12th standard PCM (Physics, Chemistry, Maths) marks/percentage, if mentioned.
    
    Transcript:
    {transcript}
    
    Output ONLY a raw JSON object with this exact structure:
    {{
        "user_name": "extracted name or null",
        "course_interest": "the program code they want or null",
        "motivation_or_notes": "A brief sentence explaining WHY they want it.",
        "wants_consultation": true or false,
        "pcm_marks": "extracted marks/percentage or null"
    }}
    """
    
    raw_output = await gpu_scheduler.run(engine.model, extraction_prompt, max_tokens=150)
    raw_json_str = raw_output["choices"][0]["text"]
    
    try:
        # --- NEW ROBUST JSON PARSER ---
        import re
        
        # 1. Search for everything between the first '{' and the last '}'
        match = re.search(r'\{.*\}', raw_json_str, re.DOTALL)
        
        if not match:
            raise ValueError("No JSON object found in LLM output.")
            
        clean_json = match.group(0)
        extracted_data = json.loads(clean_json)
        # ------------------------------
        
        # 4. Update the Database!
        sb = init_supabase()
        
        # Update Name in caller_profiles
        if extracted_data.get("user_name"):
            print(f"🧠 Learned Name: {extracted_data['user_name']}")
            # Update the caller profile table with the new name
            
        # Update Nuanced Interest in interest_signals
        if extracted_data.get("course_interest"):
            print(f"🧠 Learned Interest: {extracted_data['course_interest']} ({extracted_data.get('motivation_or_notes', '')})")
            
        # --- UPDATE CONSULTATION LOGIC ---
        if extracted_data.get("wants_consultation") is True:
            name_to_save = extracted_data.get("user_name") or "Unknown"
            marks_to_save = extracted_data.get("pcm_marks")
            
            print(f"📅 Consultation: {name_to_save} (Marks: {marks_to_save})")
            
            sb.table("consultation_requests").insert({
                "call_id": call_id,
                "caller_phone": phone,
                "caller_name": name_to_save,
                "pcm_marks": marks_to_save, # NEW: Save the marks!
                "status": "pending"
            }).execute()
            
    except Exception as e:
        print(f"⚠️ Failed to parse LLM extraction: {e}")
            
    except Exception as e:
        print(f"⚠️ Failed to parse LLM extraction: {e}")
        print(f"Raw Output Was: {raw_json_str}") # Add this so you can see what it hallucinated!

# Singletons
engine = PhiEngine("models/llm/Phi-4-mini-instruct-Q5_K_M.gguf")
rag = RAGRetriever(embedder_instance=embedder_instance)
session_store = None 

INTENT_TO_TOPIC = {
    "seat": "seats",
    "fee": "fees",
    "placement": "placements",
    "eligibility": "requirements",
    "courses": "seats" 
}

def init_globals(store_instance):
    global session_store
    session_store = store_instance

def fix_translation_errors(text):
    """
    Hardcoded fixes for common STT/Translation mistakes specific to TIST.
    """
    text = text.lower()
    
    # 1. Fix College Name Mishearings
    # "Tharkkichil" (Dispute) -> "TIST"
    # "Hokich" -> "TIST"
    replacements = {
        "dispute": "TIST college",
        "argument": "TIST college",
        "hokich": "TIST",
        "touch": "Toc H",
        "pranches": "branches",
        "benches": "branches",
        "course is": "courses",
        "fe": "fee"
    }
    
    for wrong, right in replacements.items():
        if wrong in text:
            print(f"🔧 FIX: Replaced '{wrong}' -> '{right}'")
            text = text.replace(wrong, right)
            
    return text

async def handle_llm(call_id, caller_id, phone, text_ml) -> tuple:
    print(f"\n--- 🗣️ DETECTED (ML): {text_ml} ---")

    # 1. REFLEX
    greetings = ["ഹലോ", "നമസ്കാരം", "hello", "hi", "hey"]
    cleaned = text_ml.lower().strip().replace("!", "")
    if cleaned in greetings or len(cleaned) < 3:
        log_message(call_id, "ai", "Hello! (Reflex)") 
        return ("reflex", "intro", "Hello! (Reflex)")

    # 2. TRANSLATION
    t0 = time.time()
    try:
        text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    except:
        text_en = text_ml
    
    # [CRITICAL FIX] Apply Dictionary Patch
    text_en_clean = fix_translation_errors(text_en)
    
    print(f"🔍 DEBUG [Trans]: '{text_en}' -> '{text_en_clean}'") 
    log_processing_step(call_id, "translate_ml_en", text_ml, text_en_clean, latency_ms=int((time.time()-t0)*1000))

    if not text_en_clean or len(text_en_clean) < 2:
        return ("reflex", "error", "Audio Unclear")

    # 3. INTENT
    intent = await cpu_scheduler.run(detect_intent, text_en_clean)
    print(f"🔍 DEBUG [Intent]: {intent}")
    log_intent(call_id, intent)
    
    if intent in ["seat", "fee", "admission", "courses"]:
        init_supabase().table("interest_signals").insert({
            "call_id": call_id, "caller_id": caller_id, "quota_type": "general", "strength": "HIGH"
        }).execute()

    # 4. SMART RAG
    rag_topic = INTENT_TO_TOPIC.get(intent, None)
    search_query = text_en_clean
    
    # Trigger "Smart Search" if we see branch/course keywords
    if "branch" in text_en_clean or "course" in text_en_clean or intent == "courses":
        print("🧠 SMART RAG: Rewriting query -> 'B.Tech Programs Offered'")
        search_query = "List of B.Tech Engineering Programs and Departments offered"

    rag_results = []
    if rag_topic:
        print(f"🔍 DEBUG [RAG]: Searching '{search_query}' in topic '{rag_topic}'...")
        rag_results = await cpu_scheduler.run(rag.retrieve, search_query, rag_topic)
    
    if not rag_results:
        print(f"🔍 DEBUG [RAG]: Fallback global search...")
        rag_results = await cpu_scheduler.run(rag.retrieve, search_query, None)

    # Debug RAG
    if rag_results:
        print(f"\n📥 DEBUG [RAG] Found {len(rag_results)} chunks.")
        print(f"📄 Top Chunk: {rag_results[0]['text'][:100]}...\n")
    else:
        print("❌ DEBUG [RAG]: NO DOCUMENTS FOUND.")

    rag_docs = [item['text'] for item in rag_results]

    # Guardrail
    if not rag_docs and intent not in ["general", "eligibility"]:
        log_message(call_id, "ai", "I don't have that info. (Reflex)")
        return ("reflex", "fallback", "I don't have that info.")

    # 5. LLM
    snapshot = get_snapshot(caller_id, intent)
    history = []
    if session_store:
        session = session_store.get_session(phone)
        history = session.get("history", [])[-4:]

    prompt = build_prompt(text_en_clean, rag_docs, history, snapshot)
    response_en = await gpu_scheduler.run(engine.generate, prompt)
    
    # [FIX] Handle Empty LLM Response (Latency timeout or model failure)
    if not response_en or not response_en.strip():
        print("⚠️ LLM RETURNED EMPTY STRING. Using fallback.")
        response_en = "I'm having trouble retrieving that specific detail correctly. Could you please check our website?"

    print(f"🔍 DEBUG [LLM Raw]: {response_en}")

    safety_response = apply_guardrails(response_en, intent, rag_docs, detector)
    final_en = safety_response if safety_response else response_en

    # 6. TRANSLATE BACK
    reply_ml = await cpu_scheduler.run(en_to_ml, final_en)
    
    if session_store:
        new_history = history + [{"role": "user", "text": text_en_clean}, {"role": "ai", "text": final_en}]
        session_store.update_session(phone, {"history": new_history})
        session_store.persist_later(phone)

    log_message(call_id, "ai", reply_ml)
    
    return ("text", reply_ml, reply_ml)