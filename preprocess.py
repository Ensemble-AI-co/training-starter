from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch

def create_dataloader_for_test(df, batch_size=64):
   df = df.drop(['10'], axis=1)
   X = df.astype(float)
   X = torch.tensor(X.values, dtype=torch.float32)
   dataset = TensorDataset(X)
   batch_size = 32
   loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
   torch.save(loader, 'dataloader.pth')
