# embeddings/embedder.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Legal BERT embeddings
model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

def embed_fn(texts):
    """
    Embed a list of texts using legal BERT.
    Returns a list of embedding vectors.
    """
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()
