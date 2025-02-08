import streamlit as st
from pymongo import MongoClient  
import pandas as pd
# it convert text to embedding
from sentence_transformers import SentenceTransformer
# Store and search vector embeddings.
import faiss
# import the preprocess file
from preprocess import create_vector

MONGO_URI = st.secrets["mongo"]["uri"]
connect = MongoClient(MONGO_URI)
# database name
db = connect["matrimonial"]

# collection name
collection = db['users']

# fetch data from Mongodb
data = list(collection.find({}, {"_id":0}))

st.markdown("## Enter Your Details")
users = {}
with st.form("my_form",clear_on_submit=True):
    name = st.text_input("Enter Your Name:")
    age = st.text_input("Enter Your Age:")
    gender = st.selectbox("Select Your Gender:",["Male","Female"])
    education = st.selectbox("Select Your Education:",["Postgraduate","Graduate","Under Graduate", "10th", "below 10th",
                                                       "Bachelor of Commerce (BCom)","Bachelor of Arts (BA)", "Bachelor of Science (BSc)",
                                                       "Bachelor of Technology (BTech)","Bachelor of Engineering (BE)", "Bachelor of Medicine, Bachelor of Surgery (MBBS)",
                                                       "Bachelor of Pharmacy (BPharm)", "Bachelor of Law (LLB)","Master of Arts (MA)", "Master of Science (MSc)", "Master of Commerce (MCom)",
                                                        "Master of Business Administration (MBA)", "Master of Technology (MTech)",
                                                       
                                                       ])
    profession = st.selectbox("Select Your Profession:",
                              ["Employee","Student","Software Developer", "Full Stack Developer",
                                "Data Scientist", "UI/UX Designer","Doctor", "Nurse","Civil Engineer",
                                "Mechanical Engineer", "Electrical Engineer", "School Teacher", "College Professor",
                                "Hotel Manager", "Chef", "Air Host/Hostess", "Event Planner","Beautician", "Tailor",
                                 "Electrician", "Plumber", "Mechanic"
                                ])
    location = st.text_input("Enter Your Location:")
    preference = st.text_area("Enter Your Preference:")

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
            "location": location,
            "preference": preference
        })
        # fetch data from Mongodb
        # data = list(collection.find({}, {"_id":0}))

        # convert to Dataframe
        df = pd.DataFrame(data)
        st.success("Form Submitted Successfully.")
        
# """
# 1. only save data in db.
# 2. no csv
# """