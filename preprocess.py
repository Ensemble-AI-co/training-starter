from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch

def create_dataloader_for_test(csv_path, batch_size=64):
   df = pd.read_csv(csv_path)
   df = df.drop(['10'], axis=1)
   X = df.astype(float)
   X = torch.tensor(X.values, dtype=torch.float32)
   dataset = TensorDataset(X)
   batch_size = 32
   loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
   torch.save(loader, 'dataloader.pth')

import sys
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    create_dataloader_for_test(csv_path)