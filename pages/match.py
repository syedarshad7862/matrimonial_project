# pages/2_Search.py
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pdb

st.title("Match Your Partner")

# Load movie dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/users_data.csv")

df = load_data()

# Load FAISS index
@st.cache_resource
def load_index():
    return faiss.read_index("vectorstore/index.faiss")

index = load_index()

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Search function
def search(query: str, top_k: int = 5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    # pdb.set_trace()
    return indices[0], distances[0]

# User input
query = st.sidebar.selectbox('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"18","25"] )
# query = st.sidebar.multiselect('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"25"] )
# query = st.sidebar.selectbox('Select according', ['education','profession'] )
if query:
    indices, distances = search(query)
    st.write("Top Results:")
    for idx, distance in zip(indices, distances):
        st.write(f"**Name:** {df.iloc[idx]['name']}")
        st.write(f"**Age:** {df.iloc[idx]['age']}")
        st.write(f"**Education:** {df.iloc[idx]['education']}")
        st.write(f"**Profession:** {df.iloc[idx]['profession']}")
        st.write(f"**Location:** {df.iloc[idx]['location']}")
        # st.write(f"**Distance:** {distance}")
        st.write("---")

"""
1. remove preprocess file and convert into module or function.
2. add user profile field.
3. learn pandas filtering (order by, group by)
"""