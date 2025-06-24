from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")  # nhẹ, nhanh

def build_faiss_index(data_path="data/knowledge_base.csv", output_path="embeddings"):
    df = pd.read_csv(data_path)
    
    documents = df['text'].tolist()

    embeddings = model.encode(documents, show_progress_bar=True)
    dim = embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(output_path, exist_ok=True)
    faiss.write_index(index, f"{output_path}/vector_index.faiss")

    with open(f"{output_path}/docs.pkl", "wb") as f:
        pickle.dump(documents, f)

if __name__ == "__main__":    
    build_faiss_index()
    print(f"✅ Vector database successfully created!")
