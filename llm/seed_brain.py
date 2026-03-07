# llm/seed_brain.py
import os
import shutil
from llm.rag.ingest_docs import ingest_document
from llm.rag.ingest_qa import ingest_qa_file

FILES = [
    # The New Master File (Primary Source)
    ("data/tist_rag_data.txt", "official_summary", "general"), 
    
    # Keep the Q&A doc as it handles specific conversational FAQs well
    #("data/ragQ&A.docx", "counselling_faq", "faq")
]

def train():
    print("🧠 Starting Brain Training...")

    if os.path.exists("rag_db"):
        shutil.rmtree("rag_db")
        print("🗑️  Old memory wiped.")

    for filename, source, topic in FILES:
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue

        print(f"📥 Ingesting {filename}...")
        if "Q&A" in filename:
            ingest_qa_file(filename, source)
        else:
            ingest_document(filename, source, topic)

if __name__ == "__main__":
    train()