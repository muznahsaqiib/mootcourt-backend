import chromadb
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../../vector_db")
os.makedirs(DB_PATH, exist_ok=True)  # ensure folder exists

COLLECTION_NAME = "juris_collection"

# Initialize persistent Chroma client
client = chromadb.PersistentClient(path=DB_PATH)

# Get or create collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
