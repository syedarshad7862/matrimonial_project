import streamlit as st
from pymongo import MongoClient  
import pandas as pd
# it convert text to embedding
from sentence_transformers import SentenceTransformer
# Store and search vector embeddings.
import faiss


connect = MongoClient('mongodb://localhost:27017/')
# database name
db = connect["matrimonial"]

# collection name
collection = db['users']

# fetch data from Mongodb
data = list(collection.find({}, {"_id":0}))

st.markdown("## Enter Your Details")
users = {}
with st.form("my_form"):
    name = st.text_input("Enter Your Name:")
    age = st.text_input("Enter Your Age:")
    gender = st.selectbox("Select Your Gender:",["Male","Female"])
    education = st.selectbox("Select Your Education:",["Postgraduate","Graduate","Under Graduate", "10th", "below 10th"])
    profession = st.selectbox("Select Your Profession:",["Employee","Student"])
    location = st.text_input("Enter Your Location:")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        # st.write("name:", name, "age:", age, "email:", email)
        print(f"name = {name}")
        # users.update({
        #     "name": name,
        #     "age": age,
        #     "email": email
        #     })
        connect.matrimonial.users.insert_one({
            "name": name,
            "age": age,
            "gender": gender,
            "education": education,
            "profession": profession,
            "location": location
        })
        # fetch data from Mongodb
        # data = list(collection.find({}, {"_id":0}))

        # convert to Dataframe
        df = pd.DataFrame(data)
        st.success("Add form submitted successfully.")
        
"""
1. only save data in db.
2. no csv
"""
            
# fetch data from mongodb
data = list()

print(users)
