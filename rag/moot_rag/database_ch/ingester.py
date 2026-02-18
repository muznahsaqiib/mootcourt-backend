import json
import os
from rag.moot_rag.database_ch.chroma_client import collection
from rag.moot_rag.embeddings.embedder import embed_fn

# Updated list of files to ingest (samples.jsonl removed)
FILES_TO_INGEST = [
    "constitution_clean.jsonl",
    "CPC_fixed.jsonl",
    "CrPC_fixed.jsonl",
    "PPC_fixed.jsonl",
    "lhc_chunks.jsonl",
    "metadata_rag.jsonl",
    "cases.jsonl",
    "Qanoon-e-Shahadat_fixed.jsonl"
]

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data/")

def prepare_text(data):
    """Prepare text for embedding from messages or text field"""
    if "text" in data:
        return data["text"].strip()
    elif "messages" in data:
        texts = [msg.get("content", "").strip() for msg in data["messages"]]
        return "\n".join([t for t in texts if t])
    else:
        return None

def ensure_metadata(data, file_name):
    """Ensure each chunk has all RAG-friendly metadata"""
    fname = file_name.lower()
    
    # Default source_type
    if "cases" in fname or "metadata" in fname:
        data["source_type"] = "case"
    elif "constitution" in fname:
        data["source_type"] = "constitution"
        data["case_key"] = "constitution"
    elif "cpc" in fname:
        data["source_type"] = "law"
        data["case_key"] = "cpc"
    elif "crpc" in fname:
        data["source_type"] = "law"
        data["case_key"] = "crpc"
    elif "ppc" in fname:
        data["source_type"] = "law"
        data["case_key"] = "ppc"
    elif "qanoon-e-shahadat" in fname:
        data["source_type"] = "law"
        data["case_key"] = "qes"
    elif "lhc" in fname:
        import re
        chunk_id = data.get("id", "")
        match = re.match(r"(\d{4})([A-Z]+)(\d+)_\d+", chunk_id)
        if match:
            year, court, number = match.groups()
            data["case_key"] = f"{court}-{year}-{number}"
        else:
            data["case_key"] = data.get("case_key", "LHC")
        data["source_type"] = "case"
    else:
        # fallback for misc files
        data["case_key"] = data.get("case_key", "misc")
        data["source_type"] = data.get("source_type", "misc")
    
    # Ensure RAG-friendly metadata exists
    data.setdefault("case_type", "Unknown")
    data.setdefault("case_title", "")
    data.setdefault("parties", [])
    data.setdefault("judgment_date", "")
    
    # Use source_pdf if available for case_key fallback
    if "case_key" not in data and "source_pdf" in data:
        data["case_key"] = data["source_pdf"].replace(".pdf", "")
    
    return data

for file_name in FILES_TO_INGEST:
    file_path = os.path.join(DATA_FOLDER, file_name)
    if not os.path.exists(file_path):
        print(f"[WARN] File not found, skipping: {file_path}")
        continue

    print(f"\n📥 Ingesting {file_name} ...")
    total_lines = sum(1 for _ in open(file_path, encoding="utf-8"))
    print(f"Total lines to process: {total_lines}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {line_no}")
                continue

            data = ensure_metadata(data, file_name)
            text_to_embed = prepare_text(data)

            if not text_to_embed:
                print(f"[WARN] Skipping line {line_no}: no text/messages")
                continue

            embeddings = embed_fn([text_to_embed])

            try:
                collection.add(
                    documents=[text_to_embed],
                    metadatas=[{
                        "case_key": data["case_key"],
                        "source_type": data["source_type"],
                        "case_type": data.get("case_type", ""),
                        "case_title": data.get("case_title", ""),
                        "parties": ", ".join(data.get("parties", [])),  # <-- convert list to string
                        "judgment_date": data.get("judgment_date", "")
                    }],
                    ids=[data.get("id", f"{file_name}_{line_no}")],
                    embeddings=embeddings
                )

            except Exception as e:
                print(f"[ERROR] Failed to add line {line_no}: {e}")
                continue

            # Console progress every 50 lines
            if line_no % 50 == 0:
                print(f"Processed {line_no}/{total_lines} lines...")

    print(f"✅ Finished ingesting {file_name}")

print("\n🎯 All files ingested successfully!")
