import torch.nn as nn
import torch

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_times, num_devices, num_items, num_categories, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.time_embedding = nn.Embedding(num_times, embedding_dim)
        self.device_embedding = nn.Embedding(num_devices, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        self.fc_user = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc_item = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_id, time, device, item_id, category):
        user_vec = torch.cat([
            self.user_embedding(user_id),
            self.time_embedding(time),
            self.device_embedding(device)
        ], dim=1)
        item_vec = torch.cat([
            self.item_embedding(item_id),
            self.category_embedding(category)
        ], dim=1)
        user_vec = self.fc_user(user_vec)
        item_vec = self.fc_item(item_vec)
        return (user_vec * item_vec).sum(dim=1)  # dot product
