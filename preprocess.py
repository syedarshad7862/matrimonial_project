# preprocess.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Load data
df = pd.read_csv("data/users_data.csv")

# Combine relevant columns for embeddings (e.g., gender + age + education + profession + location)
# df["text"] = df["age"] + " " + df["gender"] + " " + df['education'] + " " + df['profession'] +" "+ df['location'] # Combine Film and Genre for richer context
# df["text"] = df['name']+ ' ' + df["education"] + " " + df["profession"] + " " + df['loaction'] # Combine Film and Genre for richer context
# texts = df["text"].tolist()

# Fill missing values with an empty string or placeholder
df.fillna("unknown", inplace=True)

# Combine columns into a single text column
df["text"] = (
    df["name"].astype(str) + " " +
    df["age"].astype(str) + " " +
    df["education"].astype(str) + " " +
    df["profession"].astype(str) + " " +
    df["location"].astype(str)
)

# Convert the combined text to a list
texts = df["text"].tolist()

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save index
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/index.faiss")
print("Vector store created and saved!")