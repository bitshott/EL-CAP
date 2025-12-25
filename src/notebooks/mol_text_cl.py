#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.data import Data, Batch
from torch_geometric.utils.smiles import from_smiles

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm
import joblib 

from src.configs.experiments_configs import cl_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, pooling: str = "cls", device: str = "cuda"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.device = device
        self.to(device)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        outputs = self.encoder(**inputs)
        hidden = outputs.last_hidden_state  

        if self.pooling == "cls":
            emb = hidden[:, 0, :] 

        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb


# In[3]:


class GINBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINBlock, self).__init__()
        self.gin = GINConv(nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=out_channels), 
                    nn.LeakyReLU(), 
                    nn.Linear(in_features=out_channels,out_features=out_channels)
                    ))
        self.bn = BatchNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.gin(x, edge_index)
        x = self.bn(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(x)

        return x


# In[ ]:


class GINEncoder(nn.Module):
    def __init__(self, hidden_dims: list ):
        super(GINEncoder, self).__init__()

        self.gins = nn.ModuleList()
       
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            gin_block = GINBlock(in_channels=in_dim, out_channels=out_dim)
            self.gins.append(gin_block)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for block in self.gins:
            x = block(x, edge_index)
            
        out = global_mean_pool(x, batch) 
        return out


# In[5]:


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)


# In[6]:


class SmilesTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, smiles_col="smiles", text_col="passage"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mol_graph: Data = from_smiles(row[self.smiles_col])
        mol_graph.y = torch.tensor([idx])

        passage = row[self.text_col]

        return mol_graph, passage


# In[7]:


def collate_fn(batch):
    graphs, texts = zip(*batch)
    graphs = Batch.from_data_list(graphs)

    return graphs, list(texts)


# In[8]:


class ContrastiveModel(nn.Module):
    def __init__(self, text_encoder, graph_encoder, text_dim=768, graph_dim=512, proj_dim=256):
        super().__init__()
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder

        self.text_proj = ProjectionHead(text_dim, proj_dim)
        self.graph_proj = ProjectionHead(graph_dim, proj_dim)

    def forward(self, batch_graphs, passages):
        g_emb = self.graph_encoder(batch_graphs)             
        g_emb = self.graph_proj(g_emb)                      

        t_emb = self.text_encoder(passages)                   
        t_emb = self.text_proj(t_emb)                         

        return g_emb, t_emb



# In[9]:


def contrastive_loss(g_emb, t_emb, temperature=0.07):
    """
    Симметричная InfoNCE loss:
    - Максимизируем похожесть между (g, t) для положительных пар
    - Минимизируем для отрицательных (других в батче)
    """
    batch_size = g_emb.size(0)

    sim_matrix = torch.matmul(g_emb, t_emb.T) / temperature 

    labels = torch.arange(batch_size, device=g_emb.device)

    loss_t2g = F.cross_entropy(sim_matrix, labels)           
    loss_g2t = F.cross_entropy(sim_matrix.T, labels)        

    return (loss_t2g + loss_g2t) / 2


# In[10]:


def train_one_epoch(model, loader, optimizer, device="cuda"):
    model.train()
    total_loss = 0
    for batch_graphs, passages in tqdm(loader, desc='Train'):
        batch_graphs = batch_graphs.to(device)

        optimizer.zero_grad()
        g_emb, t_emb = model(batch_graphs, passages)
        loss = contrastive_loss(g_emb, t_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# In[11]:


def evaluate(model, loader, device="cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_graphs, passages in tqdm(loader, desc='Evaluation'):
            batch_graphs = batch_graphs.to(device)
            g_emb, t_emb = model(batch_graphs, passages)
            loss = contrastive_loss(g_emb, t_emb)
            total_loss += loss.item()
    return total_loss / len(loader)


# In[12]:


text_encoder = TextEncoder(cl_config.TEXT_ENCODER, pooling="cls", device=cl_config.DEVICE)
graph_encoder = GINEncoder(hidden_dims=cl_config.GNN_HIDDEN_DIMS).to(cl_config.DEVICE)

model = ContrastiveModel(text_encoder, graph_encoder, text_dim=768, graph_dim=256, proj_dim=256).to(cl_config.DEVICE)


# In[13]:


optimizer = torch.optim.AdamW(model.parameters(), lr=cl_config.LR, weight_decay=cl_config.L2_NORM)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                         factor=0.7, threshold=0.05, patience=10,
#                                         min_lr=1e-6)


# In[14]:


def set_seed(SEED):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(cl_config.RANDOM_SEED)

generator=torch.Generator().manual_seed(cl_config.RANDOM_SEED)

df = pd.read_csv('src/data/chembl_35_activity_new.csv')
df = df.sample(n=100000, random_state=cl_config.RANDOM_SEED)

train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_size = len(df) - train_size - val_size

train_df, val_df, test_df = random_split(
    df, [train_size, val_size, test_size]
)

train_dataset = SmilesTextDataset(train_df.dataset.iloc[train_df.indices])
val_dataset   = SmilesTextDataset(val_df.dataset.iloc[val_df.indices])
test_dataset  = SmilesTextDataset(test_df.dataset.iloc[test_df.indices])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, worker_init_fn=seed_worker, generator=generator)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, worker_init_fn=seed_worker, generator=generator)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, worker_init_fn=seed_worker, generator=generator)


# In[ ]:


import mlflow


run_description = """
class CLConfig():
    TEXT_ENCODER: str = 'bitshott/scibert_scivocab_chembl_passages_v1'
    DEVICE: str = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    EPOCHES: int = 5
    LR: float = 2e-5
    L2_NORM: float = 1e-4
    GNN_HIDDEN_DIMS: list[int] = [9, 32, 64, 128, 256, 512]

    RANDOM_SEED: int = 42
"""
tags = {
        'mlflow.note.content': run_description
    }
timestamp = str(datetime.now())
mlflow.set_experiment('GIN_SCIBERT_CASED_CONTRASTIVE_LEARN')
with mlflow.start_run(run_name=f'no_scheduler_{cl_config.GNN_HIDDEN_DIMS}_{cl_config.LR}_{timestamp}_{len(df)}',tags=tags):   

    for epoch in range(cl_config.EPOCHES):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=cl_config.DEVICE)
        val_loss = evaluate(model, val_loader, device=cl_config.DEVICE)
        mlflow.log_metric(f'Loss/Train_InfoNCE', train_loss, epoch)
        mlflow.log_metric(f'Loss/Val_InfoNCE', val_loss, epoch)
        
        torch.save(model.state_dict(), f'src/models/clmodel_dataset_{len(df)}_{timestamp}_epoch_{epoch}.pt')
        torch.save(text_encoder.state_dict(), f'src/models/textencoder_dataset_{len(df)}_{timestamp}_epoch_{epoch}.pt')
        torch.save(graph_encoder.state_dict(), f'src/models/graphencoder_dataset_{len(df)}_{timestamp}_epoch_{epoch}.pt')

        # scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")


# In[21]:




# In[ ]:




