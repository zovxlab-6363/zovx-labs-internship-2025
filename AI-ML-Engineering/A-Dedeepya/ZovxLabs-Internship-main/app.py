import streamlit as st
import torch
import torch.nn.functional as F
import faiss
from model import TwoTowerModel  # import your model class
from util import build_faiss_index, get_user_vector, recommend_top_k

# Load model
embedding_dim = 32  # same as your model
num_users = 100
num_times = 4
num_devices = 3
num_items = 1000
num_categories = 10

model = TwoTowerModel(
    num_users=num_users,
    num_times=num_times,
    num_devices=num_devices,
    num_items=num_items,
    num_categories=num_categories
)
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

# Load item vectors and build FAISS index
item_vectors, item_ids = build_faiss_index(model, num_items, num_categories, embedding_dim)

# UI
st.title("🎥 YouTube-style Video Recommender")
st.markdown("Get personalized video recommendations based on your profile")

user_id = st.slider("Select User ID", 0, num_users - 1, 0)
time_of_day = st.selectbox("Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
device_type = st.selectbox("Select Device", ["Mobile", "Desktop", "Tablet"])

if st.button("Recommend"):
    user_vector = get_user_vector(model, user_id, time_of_day, device_type)
    recommended_ids, scores = recommend_top_k(user_vector, item_vectors, item_ids, top_k=10)
    
    st.success("✅ Recommendations Generated!")
    for i, (item_id, score) in enumerate(zip(recommended_ids, scores)):
        st.write(f"{i+1}. Item ID: {item_id} | Score: {score:.4f}")
