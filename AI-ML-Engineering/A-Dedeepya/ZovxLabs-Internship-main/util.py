import torch
import torch.nn.functional as F
import faiss
import numpy as np

# Map categorical inputs to integers
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
device_map = {"Mobile": 0, "Desktop": 1, "Tablet": 2}

def build_faiss_index(model, num_items, num_categories, embedding_dim):
    item_ids = torch.arange(num_items)
    categories = torch.randint(0, num_categories, (num_items,))
    with torch.no_grad():
        item_emb = model.item_embedding(item_ids)
        cat_emb = model.category_embedding(categories)
        item_vec = model.fc_item(torch.cat([item_emb, cat_emb], dim=1))
        item_vec = F.normalize(item_vec, p=2, dim=1)

    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(item_vec.numpy())
    return faiss_index, item_ids.numpy()

def get_user_vector(model, user_id, time_str, device_str):
    user = torch.tensor([user_id])
    time = torch.tensor([time_map[time_str]])
    device = torch.tensor([device_map[device_str]])
    with torch.no_grad():
        user_emb = model.user_embedding(user)
        time_emb = model.time_embedding(time)
        device_emb = model.device_embedding(device)
        user_vec = model.fc_user(torch.cat([user_emb, time_emb, device_emb], dim=1))
        user_vec = F.normalize(user_vec, p=2, dim=1)
    return user_vec.numpy()

def recommend_top_k(user_vector, faiss_index, item_ids, top_k=10):
    D, I = faiss_index.search(user_vector, top_k)
    return item_ids[I[0]], D[0]
