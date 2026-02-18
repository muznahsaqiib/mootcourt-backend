import os
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)

evaluated_col = client.get_or_create_collection("evaluated_arguments")
legal_col = client.get_or_create_collection("legal_content")
judge_col = client.get_or_create_collection("judge_questions")
