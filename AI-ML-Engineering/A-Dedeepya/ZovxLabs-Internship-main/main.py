import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from googleapiclient.discovery import build

# --- Set up YouTube API ---
API_KEY = "AIzaSyCICFyY1wfWBblFsxLtIFrOtNmrzcHP4RA"  # Replace with your actual API key
youtube = build("youtube", "v3", developerKey=API_KEY)

# --- Two-Tower Model Definition ---
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

    def forward(self, user_input, item_input):
        user_vec = self.user_tower(user_input)
        item_vec = self.item_tower(item_input)
        return torch.sigmoid((user_vec * item_vec).sum(dim=1))

@st.cache_data(show_spinner=False)
def fetch_videos():
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode="IN",
        maxResults=50
    )
    response = request.execute()
    data = []
    for item in response["items"]:
        data.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "category": item["snippet"]["categoryId"],
            "channel": item["snippet"]["channelTitle"]
        })
    return pd.DataFrame(data)

def simulate_users(df):
    df["user_id"] = np.random.randint(0, 1000, size=len(df))
    df["device"] = np.random.choice(["mobile", "desktop"], size=len(df))
    df["time"] = np.random.choice(["morning", "afternoon", "evening"], size=len(df))
    df["activity"] = np.random.choice(["browse", "search", "watch"], size=len(df))
    df["engagement"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    return df

def prepare_data(df):
    le_device = LabelEncoder()
    le_time = LabelEncoder()
    le_activity = LabelEncoder()
    le_category = LabelEncoder()
    le_video = LabelEncoder()

    user_features = pd.DataFrame({
        "user_id": df["user_id"],
        "device": le_device.fit_transform(df["device"]),
        "time": le_time.fit_transform(df["time"]),
        "activity": le_activity.fit_transform(df["activity"])
    })
    item_features = pd.DataFrame({
        "video_id": le_video.fit_transform(df["video_id"]),
        "category": le_category.fit_transform(df["category"])
    })
    scaler_u, scaler_i = StandardScaler(), StandardScaler()
    X_user = scaler_u.fit_transform(user_features)
    X_item = scaler_i.fit_transform(item_features)
    y = torch.tensor(df["engagement"].values, dtype=torch.float32)
    return torch.tensor(X_user, dtype=torch.float32), torch.tensor(X_item, dtype=torch.float32), y, df

def train_model(model, X_user, X_item, y):
    dataset = torch.utils.data.TensorDataset(X_user, X_item, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for epoch in range(5):
        for u, i, label in loader:
            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

def get_recommendations(model, X_user, X_item, df):
    with torch.no_grad():
        preds = model(X_user, X_item).numpy()
    df["score"] = preds
    return df.sort_values(by="score", ascending=False).head(10)

# --- Streamlit App ---
st.set_page_config(page_title="YouTube Recommender", layout="wide")
st.title("🎬 YouTube-Style Recommendation Engine (Two-Tower Model)")

with st.spinner("Fetching YouTube trending videos and training model..."):
    df = simulate_users(fetch_videos())
    X_user, X_item, y, df_raw = prepare_data(df)
    model = TwoTowerModel(X_user.shape[1], X_item.shape[1])
    train_model(model, X_user, X_item, y)
    top_videos = get_recommendations(model, X_user, X_item, df_raw)

st.success("Model trained. Here are top video recommendations:")

for _, row in top_videos.iterrows():
    st.write(f"**{row['title']}** — {row['channel']} (Category ID: {row['category']}, Score: {row['score']:.2f})")
