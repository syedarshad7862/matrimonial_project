# pages/2_Search.py
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pdb
# import the preprocess file
from preprocess import create_vector,generate_embeddings

st.title("Match Your Partner")


# @st.cache_data
# def load_data():
#     return pd.read_csv("data/users_data.csv")

MONGO_URI = st.secrets["mongo"]["uri"]

# function help in load from db and create vectors
df,data, texts = create_vector(MONGO_URI)
# # convert to Dataframe
# df = pd.DataFrame(data)

# generate embeddings and create the index file inside the vectorstore folder
generate_embeddings(texts)
# pdb.set_trace()
# Load FAISS index
@st.cache_resource
def load_index():
    try:
        index = faiss.read_index("vectorstore/index.faiss")
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

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

def search_similar_profiles(df,selected_user, top_k=5):
    """Find similar profiles based on FAISS similarity search."""
    
        # Clear cache to prevent duplicate search results
    st.cache_data.clear()
    
    user_index = df[df["name"] == selected_user].index[0]
    
    # Encode selected user's text into an embedding
    query_embedding = model.encode([df.iloc[user_index]["text"]])
    
    # Search similar users in FAISS
    distances, indices = index.search(query_embedding, top_k + 1)  # +1 to exclude the user itself

    # Filter results: exclude the selected user and filter by gender
    selected_gender = df.iloc[user_index]["gender"]
    selected_age = int(df.iloc[user_index]["age"])  # Convert age to integer
    opposite_gender = "Female" if selected_gender == "Male" else "Male"

    similar_profiles = [
        i for i in indices[0] if df.iloc[i]["name"] != selected_user and df.iloc[i]["gender"] == opposite_gender
    ]
    
    # Convert selected users into a DataFrame
    # similar_df = df.iloc[similar_profiles].copy()

    # Convert ages to integer for comparison
    # similar_df["age"] = similar_df["age"].astype(int)

    # # **Age-Based Matching Logic**
    # if selected_gender == "Male":
    #     # Male users: Prefer younger or same age females, else go for older ones
    #     filtered_df = similar_df[similar_df["age"] <= selected_age]
    #     if filtered_df.empty:
    #         filtered_df = similar_df[similar_df["age"] > selected_age]

    # else:  # Female users
    #     # Female users: Prefer older males, else go for younger ones
    #     filtered_df = similar_df[similar_df["age"] > selected_age]
    #     if filtered_df.empty:
    #         filtered_df = similar_df[similar_df["age"] <= selected_age]

    # Step 3: Sort by age (ascending for males, descending for females)
    # if selected_gender == "Male":
    #     filtered_df = filtered_df.sort_values(by="age", ascending=True)  # Younger first
    # else:
    #     filtered_df = filtered_df.sort_values(by="age", ascending=False)  # Older first

    # return filtered_df.head(top_k)  # Return top_k matches
    return df.iloc[similar_profiles]  # Return top_k matches

# User input
query = st.sidebar.selectbox('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"18","25"] )

if df.empty:
    st.error("No user data found.")
else:
    # User dropdown for selecting profiles
    selected_user = st.sidebar.selectbox("Select a User Profile", df["name"])
        # Display selected user profile
    user_profile = df[df["name"] == selected_user].iloc[0]
    st.write("### Selected User Profile:")
    st.dataframe(user_profile)
    
    if selected_user:
       similar_profiles = search_similar_profiles(df, selected_user)
    # pdb.set_trace()
    if not similar_profiles.empty:
        st.write("### Matching Profiles:")
        st.table(similar_profiles)  # Display results in tabular format
    else:
        st.write("No suitable matches found.")


# query = st.sidebar.multiselect('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"25"] )


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

# """
# 1. remove preprocess file and convert into module or function.
# 2. add user profile field.
# 3. learn pandas filtering (order by, group by)
# """