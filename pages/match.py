# pages/2_Search.py
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pdb
# import the preprocess file
from preprocess import create_vector,generate_embeddings
# Import the card module
# from streamlit_card import card # type: ignore
st.title("Match Your Partner")


# @st.cache_data
# def load_data():
#     return pd.read_csv("data/users_data.csv")

MONGO_URI = st.secrets["mongo"]["uri"]

# function help in load from db and create vectors
df,data, texts = create_vector(MONGO_URI)
new_df =  pd.DataFrame(data)
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
# query = st.sidebar.selectbox('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"18","25"] )

multi = st.sidebar.multiselect('Filter', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"18","25"] )
# Button to trigger search
if st.sidebar.button("Find Matches"):
    if multi:
        query_text = " ".join(multi)  # Combine selected filters into a single query string
        indices, distances = search(query_text)

        # Display Matching Results
        # st.write("### Top Matching Profiles:")
        # for idx, distance in zip(indices, distances):
            # if idx < len(df):  # Prevent index errors
            #     st.write(f"**Name:** {df.iloc[idx]['name']}")
            #     st.write(f"**Age:** {df.iloc[idx]['age']}")
            #     st.write(f"**Education:** {df.iloc[idx]['education']}")
            #     st.write(f"**Profession:** {df.iloc[idx]['profession']}")
            #     st.write(f"**Location:** {df.iloc[idx]['location']}")
            #     st.write("---")
        matched_profiles = []
        for idx, distance in zip(indices, distances):
            if idx < len(new_df):  # Prevent index errors
                matched_profiles.append(new_df.iloc[idx])

        # Convert matched profiles into a DataFrame
        if matched_profiles:
            st.write("### Top Matching Profiles:")
            result_df = pd.DataFrame(matched_profiles)
            result_df.columns = ["Name", "Age", "Gender","Education","Profession","Location","Preference","texts"]
            # st.dataframe(result_df)  # Display as a DataFrame
            st.markdown(result_df.to_html(index=False, escape=False), unsafe_allow_html=True)
        else:
            st.warning("No matching profiles found.")
    else:
        st.warning("Please select at least one filter!")


# Custom CSS for cards
card_style = """
<style>
.card {
    background-color: #ffffff;  /* White background */
    border-radius: 10px;       /* Rounded corners */
    padding: 20px;             /* Padding inside the card */
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2); /* Soft shadow */
    margin-bottom: 20px;       /* Space between cards */
    color: #333;               /* Dark text color */
    font-size: 16px;           /* Text size */
    max-width: 500px;           /* Ensure card doesn't overflow */
    width: 100%;               /* Full width by default */
    transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth hover effect */
    text-align: center;
}

/* Card hover effect */
.card:hover {
    transform: translateY(-5px); /* Slight lift on hover */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* Stronger shadow on hover */
}

/* Card title styling */
.card-title {
    color: #e91e63;            /* Pink color for title (matrimonial theme) */
    font-size: 24px;           /* Larger font size for title */
    font-weight: bold;         /* Bold title */
    margin-bottom: 15px;       /* Space below title */
    text-align: center;        /* Center-align title */
}
</style>
"""

# Inject custom CSS
st.markdown(card_style, unsafe_allow_html=True)


# Main app logic
if df.empty:
    st.error("No user data found.")
else:
    # User dropdown for selecting profiles
    selected_user = st.sidebar.selectbox("Select a User Profile", new_df["name"])
    top_k = st.sidebar.number_input("Top",5,30)
    print(top_k,type(top_k))
    # Display selected user profile
    user_profile = df[df["name"] == selected_user].iloc[0]
    st.write("### Selected User Profile:")
    st.dataframe(user_profile)

    if st.sidebar.button("Show Profile"):
        similar_profiles = search_similar_profiles(df, selected_user,top_k)

        if not similar_profiles.empty:
            st.write("### Matching Profiles:")
            for _, profile in similar_profiles.iterrows():
                # Create a card for each profile
                card_content = f"""
                <div class="card">
                <div class="card-title">{profile["name"]}</div>
                <div class="card-content">
                    <div><strong>Age:</strong> {profile["age"]}</div>
                    <div><strong>Gender:</strong> {profile["gender"]}</div>
                    <div><strong>Education:</strong> {profile["education"]}</div>
                    <div><strong>Profession:</strong> {profile["profession"]}</div>
                    <div><strong>Location:</strong> {profile["location"]}</div>
                    <div><strong>Preference:</strong> {profile["preference"]}</div>
                </div>
                </div>
                """
                st.markdown(card_content, unsafe_allow_html=True)
        else:
            st.write("No suitable matches found.")


# query = st.sidebar.multiselect('Select according', [None,'Graduate','Under Graduate', 'Student','Saudi Arabia','19',"25"] )


# """
# 1. remove preprocess file and convert into module or function.
# 2. add user profile field.
# 3. learn pandas filtering (order by, group by)
# """