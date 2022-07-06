import math
import numpy as np

import os
import csv
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 114514,
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epochs': 3000,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 200,
    'save_path': './models/model.ckpt'
}


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class C19Dataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class C19Model(nn.Module):
    def __init__(self, input_dim):
        super(C19Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train = train_data[:, -1]
    y_valid = valid_data[:, -1]
    raw_x_train = train_data[:, :-1]
    raw_x_valid = valid_data[:, :-1]
    raw_x_test = test_data

    if select_all:
        feat_item = list(range(raw_x_train.shape[1]))
    else:
        feat_item = [0, 1, 2, 3, 4]

    return raw_x_train[:, feat_item], raw_x_valid[:, feat_item], raw_x_test[:, feat_item], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # writer = SummaryWriter()

    if not os.path.isdir('./models'):
        os.mkdir("./models")

    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()

        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step = step + 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1} / {n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        # writer.add_scalar('loss-train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1} / {n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('loss-valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model. Loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving!')
            return


same_seed(config['seed'])

train_data = pd.read_csv('./dataset/covid.train_new.csv').values
test_data = pd.read_csv('./dataset/covid.test_un.csv').values

train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

print(f'number of features: {x_train.shape[1]}')

train_dataset = C19Dataset(x_train, y_train)
valid_dataset = C19Dataset(x_valid, y_valid)
test_dataset = C19Dataset(x_test)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# print(x_train.shape[1])

model = C19Model(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)

def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# model.load_state_dict(torch.load(config['save_path']))
# preds = predict(test_loader, model, device)
# save_pred(preds, 'pred.csv')
