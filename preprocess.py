# preprocess.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from pymongo import MongoClient
# Load data
# df = pd.read_csv("data/users_data.csv")

def create_vector(mongodb_uri,db_name="matrimonial",collection_name="users"):
    connect = MongoClient(mongodb_uri)
    # database name
    db = connect[db_name]

    # collection name or table
    collection = db[collection_name]

    # fetch data from Mongodb
    data = list(collection.find({}, {"_id":0}))

    # convert to Dataframe
    df = pd.DataFrame(data)

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
    
    return df,data, texts

# Generate embeddings
def generate_embeddings(texts):
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