import json
import os
from eval_rag.embeddings.embedder import embed
from eval_rag.db.chroma_client import evaluated_col, legal_col, judge_col


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

        text = f"""
CASE TYPE: {doc['case_type']}

ARGUMENT:
{doc['user_input']}

SCORES:
{doc['scores']}

JUSTIFICATION:
{doc['justification']}

FINAL COMMENT:
{doc['final_comment']}
"""

        evaluated_col.add(
            documents=[text],
            embeddings=[embed(text)],
            metadatas=[{
                "case_type": doc["case_type"],
                "round_type": doc["round_type"]
            }],
            ids=[doc["id"]]
        )

print(f"✔ Finished evaluated arguments: {count_eval} records\n")


# =====================================================
# LEGAL SOURCES
# =====================================================

legal_folder = os.path.join(BASE_DIR, "data", "legal_sources")

print("📚 Loading legal sources...")

total_laws = 0

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
                total_laws += 1

                law = json.loads(line)

                print(f"   ➡ Law #{file_count} | Global #{total_laws}")

                legal_col.add(
                    documents=[law["text"]],
                    embeddings=[embed(law["text"])],
                    metadatas=[{"source": source_name}],
                    ids=[law["id"]]
                )

        print(f"✔ Finished {filename}: {file_count} records")

print(f"\n✔ Total legal records ingested: {total_laws}\n")


# =====================================================
# JUDGE QUESTIONS
# =====================================================

judge_path = os.path.join(BASE_DIR, "data", "judge_questions.json")

print("⚖ Loading judge questions...")

with open(judge_path, encoding="utf-8") as f:
    judge_data = json.load(f)

judge_count = 0

for d, m, i in zip(
    judge_data["documents"],
    judge_data["metadatas"],
    judge_data["ids"]
):
    judge_count += 1

    print(f"➡ Ingesting judge question #{judge_count} | ID: {i}")

    judge_col.add(
        documents=[d],
        embeddings=[embed(d)],
        metadatas=[m],
        ids=[i]
    )

print(f"✔ Finished judge questions: {judge_count} records\n")


# =====================================================
print("🎉 INGESTION COMPLETED SUCCESSFULLY")
print("Chroma DB saved to ./eval_rag/chroma_db")
