import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_data(df: pd.DataFrame,noise : float = 0.0):
    test_set_percent = 0.1
    noise_factor = noise

    df = df.div(df.sum(axis=1), axis=0)

    x_test = df.sample(frac=test_set_percent)
    x_train = df.drop(x_test.index)
    x_train_noisy = x_train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_train.shape
    )
    x_test_noisy = x_test + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_test.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0001, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0001, 1.0)

    return x_train, x_train_noisy, x_test, x_test_noisy


class DeepMS(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(DeepMS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, original_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class flipped(nn.Module):
    def __init__(self, original_dim,encoding_dim):
        super(DeepMS, self).__init__()
        self.encoder1 = nn.Linear(original_dim,encoding_dim)
        self.encoder2 = nn.ReLU()
        self.decoder1 = nn.Linear(encoding_dim,original_dim)
        self.decoder2 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x

    def decode(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x

class WeightClipper(object):

    def __init__(self):
        pass
    #     self.frequency = frequency

    def __call__(self, module):
    # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            # print("weight enter")
            w = module.weight.data
            w = w.clamp(0,1)
            module.weight.data = w

# Create dataset from pandas dataframe
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df1, df2):
        self.x1 = torch.tensor(df1.values, dtype=torch.float32)
        self.x2 = torch.tensor(df2.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx]

#TODO correctly need to do apply constrain more dynamic or split up to multiple files
def train_model(
    epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    flipped = False
):
    if flipped == True:
        clipper = WeightClipper()
    model.share_memory()
    total_train_loss = []
    total_test_loss = []
    # ttotal = time.time()
    for e in range(epochs):
        train_losses = []
        # t1 = time.time()
        for x_n, x_o in train_loader:
            x_n = x_n.to(device)
            x_o = x_o.to(device)
            # print(x_n)
            # print(x_o)
            optimizer.zero_grad()
            output = model(x_n)
            loss = criterion(output, x_o)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if flipped == True:
                model.apply(clipper)
        total_train_loss.append(np.mean(train_losses))
        # print(f"Epoch: {e + 1}/{epochs}...", 
        #       f"Train Loss: {np.mean(train_losses)}... ",
        #       f"Time: {time.time()-t1}...")
    
    
        if __name__ == "__main__":
            with torch.no_grad():
                test_losses = []
                for x_n, x_o in test_loader:
                    x_n = x_n.to(device)
                    x_o = x_o.to(device)
                    test_output = model(x_n)
                    test_loss = criterion(test_output, x_o)
                    test_losses.append(test_loss.item())
            total_test_loss.append(np.mean(test_losses))
            print(
                f"Epoch: {e + 1}/{epochs}...",
                f"Train Loss: {np.mean(train_losses)}... ",
                f"Test Loss: {np.mean(test_losses)}...",
            )

    # if __name__ == "__main__":
    #     import matplotlib.pyplot as plt

    #     plt.plot(total_train_loss, label="Training loss")
    #     plt.plot(total_test_loss, label="Testing loss")
    #     plt.legend(frameon=False)
    #     plt.show()
    # print(f"Total Time: {time.time()-ttotal}...")
    return total_train_loss[-1]


def _AE(df: pd.DataFrame, components: int = 200, criterion=nn.MSELoss(),noise : float = 0.0):
    batch_size = 8
    epochs = 500
    learning_rate = 1e-3
    original_dim = df.shape[1]
    
    x_train, x_train_noisy, x_test, x_test_noisy = prepare_data(df,noise)

    train_dataset = Dataset(x_train_noisy, x_train)   
    
    test_dataset = Dataset(x_test_noisy, x_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True
    )
    model = DeepMS(original_dim, components).to(device)
    # model.apply(constrain)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss = train_model(epochs, model, optimizer, criterion, train_loader, test_loader)

    # get all latents
    latents : np.ndarray = (
        model.encode(torch.tensor(df.values, dtype=torch.float32).to(device))
        .cpu()
        .detach()
        .numpy()
    )

    # get all weights
    weights : np.ndarray = (
        [x.weight.data for i, x in enumerate(model.encoder.modules()) if i == 1][0]
        .cpu()
        .detach()
        .numpy()
    )

    latents =np.array([np.array([i/np.sum(x) for i in x]) for x in latents])
    return latents, weights, loss


def _flippedAE(df: pd.DataFrame, components: int = 200, criterion=nn.MSELoss(),noise : float = 0.0):
    batch_size = 8
    epochs = 500
    learning_rate = 1e-3
    original_dim = df.shape[1]
    # print(original_dim,components)
    
    x_train, x_train_noisy, x_test, x_test_noisy = prepare_data(df,noise)

    train_dataset = Dataset(x_train_noisy, x_train)
    
    test_dataset = Dataset(x_test_noisy, x_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True
    )
    model = flipped(original_dim, components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = train_model(epochs, model, optimizer, criterion, train_loader, test_loader,True)

    # get all latents
    latents : np.ndarray = (
        model.encode(torch.tensor(df.values, dtype=torch.float32).to(device))
        .cpu()
        .detach()
        .numpy()
    )
    weights = model.encoder1.weight.detach().numpy()
    return latents, weights, loss


def mseAE(df: pd.DataFrame, components: int = 200,noise: float = 0.0):

    return _AE(df, components, nn.MSELoss(),noise)

def klAE(df: pd.DataFrame, components: int = 200,noise:float = 0.0):
    return _AE(df, components, nn.KLDivLoss(),noise)

def flipped_mseAE(df : pd.DataFrame,components: int=50,noise:float = 0.0):
    if df.shape[1]<components:
        components = int(df.shape[1]*0.75)
    sig,weights,loss= _flippedAE(df.T,components,nn.MSELoss(),noise)
    return weights.T,sig.T,loss





if __name__ == "__main__":
    df = pd.read_csv(r"datasets\england\catalogues_Breast_SBS\catalogues_Breast_SBS.tsv", sep="\t", index_col=0)
    # input()
    
    t1 = time.time()
    sig, weights,loss = mseAE(df, 100,0)
    print(time.time()-t1)
  