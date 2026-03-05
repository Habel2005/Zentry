# db/call_repo.py
import hashlib
from db.client import init_supabase

def _hash_phone(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()

def start_call(freeswitch_uuid: str, phone: str):
    sb = init_supabase()
    phone_hash = _hash_phone(phone)

    # 1. Get or create caller
    caller = sb.table("caller_profiles") \
        .select("*") \
        .eq("phone_hash", phone_hash) \
        .execute()

    if caller.data:
        caller_id = caller.data[0]["id"]
        sb.table("caller_profiles") \
          .update({"last_seen": "now()", "total_calls": caller.data[0]["total_calls"] + 1}) \
          .eq("id", caller_id) \
          .execute()
    else:
        res = sb.table("caller_profiles").insert({
            "phone_hash": phone_hash,
            "total_calls": 1
        }).execute()
        caller_id = res.data[0]["id"]

    # 2. Create call session (ADDED DEFAULT FIELDS)
    call = sb.table("call_sessions").insert({
        "freeswitch_uuid": freeswitch_uuid,
        "phone_hash": phone_hash,
        "status": "ongoing",
        "handled_by": "ai",          # <-- Added
        "language_detected": "ml"    # <-- Added default Malayalam
    }).execute()

    call_id = call.data[0]["id"]
    return call_id, caller_id

# NEW FUNCTION: Updates the call session with STT quality after we get audio
def update_call_metrics(call_id: str, language: str, stt_quality: str):
    sb = init_supabase()
    sb.table("call_sessions") \
      .update({
          "language_detected": language,
          "stt_quality": stt_quality
      }) \
      .eq("id", call_id) \
      .execute()

def end_call(call_id: str, status: str = "completed"):
    sb = init_supabase()
    sb.table("call_sessions") \
      .update({"status": status, "ended_at": "now()"}) \
      .eq("id", call_id) \
      .execute()

def log_message(call_id: str, speaker: str, raw_text: str, normalized_text=None, confidence=None):
    sb = init_supabase()
    sb.table("call_messages").insert({
        "call_id": call_id,
        "speaker": speaker,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "confidence": confidence
    }).execute()