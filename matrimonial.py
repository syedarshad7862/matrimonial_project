import streamlit as st
from pymongo import MongoClient  
import pandas as pd
# it convert text to embedding
from sentence_transformers import SentenceTransformer
# Store and search vector embeddings.
import faiss


st.set_page_config(page_title="Matrimonial", page_icon=":material/edit:")

# upload = st.Page("pages/bio_form.py", title="New User", icon=":material/add_circle:")
# search = st.Page("pages/match.py", title="Match", icon=":material/delete:")
# about = st.Page("pages/3_About.py", title="About App", icon=":material/delete:")

# pg = st.navigation([upload, search])
MONGO_URI = st.secrets["mongo"]["uri"]
connect = MongoClient(MONGO_URI)
# database name
db = connect["matrimonial"]

# collection name or table
collection = db['users']

# fetch data from Mongodb
data = list(collection.find({}, {"_id":0}))

# convert to Dataframe
df = pd.DataFrame(data)

# save to excel
# df.to_csv("data/users_data.csv",index=False)

st.title("WelCome to Matrimonial AI Web App")

st.markdown("# All Users")

st.dataframe(df)

        
# """
# 1. fetch data from db
# 2. convert dictionary into dataframe and pass to interface.
# """


