import json
import os
from eval_rag.embeddings.embedder import embed
from eval_rag.db.chroma_client import evaluated_col, legal_col, judge_col

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ✅ FIX: Chunking function to split long legal texts
def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# =====================================================
# EVALUATED ARGUMENTS
# =====================================================
eval_path = os.path.join(BASE_DIR, "data", "evaluated_arguments.jsonl")
print("\n📥 Loading evaluated arguments...")
count_eval = 0

with open(eval_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        count_eval += 1
        doc = json.loads(line)
        print(f"➡ Ingesting evaluated argument #{count_eval} | ID: {doc['id']}")

        # ✅ FIX: Store ONLY the argument text as document
        # Scores and justification moved to metadata — prevents score leakage into LLM context
        argument_text = doc["user_input"]

        # Flatten all metadata values to strings to satisfy ChromaDB
        evaluated_col.add(
            documents=[argument_text],
            embeddings=[embed(argument_text)],
            metadatas=[{
                "case_type": str(doc.get("case_type", "")),
                "round_type": str(doc.get("round_type", "")),
                "scores": json.dumps(doc["scores"]) if isinstance(doc["scores"], (dict, list)) else str(doc.get("scores", "")),
                "justification": str(doc.get("justification", "")),
                "final_comment": str(doc.get("final_comment", ""))
            }],
            ids=[doc["id"]]
        )

print(f"✔ Finished evaluated arguments: {count_eval} records\n")


# =====================================================
# LEGAL SOURCES
# =====================================================
legal_folder = os.path.join(BASE_DIR, "data", "legal_sources")
print("📚 Loading legal sources...")

total_chunks = 0

for filename in os.listdir(legal_folder):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(legal_folder, filename)
        source_name = os.path.splitext(filename)[0].upper()

        print(f"\n📄 Processing file: {filename}")
        file_count = 0

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                file_count += 1
                law = json.loads(line)

                # ✅ FIX: Chunk long legal texts before embedding
                chunks = chunk_text(law["text"], size=400, overlap=50)

                for chunk_idx, chunk in enumerate(chunks):
                    total_chunks += 1
                    chunk_id = f"{law['id']}_chunk_{chunk_idx}"

                    legal_col.add(
                        documents=[chunk],
                        embeddings=[embed(chunk)],
                        metadatas=[{
                            "source": source_name,
                            "parent_id": law["id"],
                            "chunk_index": chunk_idx
                        }],
                        ids=[chunk_id]
                    )

                print(f"   ➡ Law #{file_count} | {len(chunks)} chunk(s) | Total chunks: {total_chunks}")

        print(f"✔ Finished {filename}: {file_count} laws → {total_chunks} chunks")

print(f"\n✔ Total legal chunks ingested: {total_chunks}\n")


# =====================================================
# JUDGE QUESTIONS
# =====================================================
judge_path = os.path.join(BASE_DIR, "data", "judge_questions.json")
print("⚖ Loading judge questions...")

with open(judge_path, encoding="utf-8") as f:
    judge_data = json.load(f)

judge_count = 0
for d, m, i in zip(judge_data["documents"], judge_data["metadatas"], judge_data["ids"]):
    judge_count += 1
    print(f"➡ Ingesting judge question #{judge_count} | ID: {i}")
    judge_col.add(
        documents=[d],
        embeddings=[embed(d)],
        metadatas=[m],
        ids=[i]
    )

print(f"✔ Finished judge questions: {judge_count} records\n")
print("🎉 INGESTION COMPLETED SUCCESSFULLY")