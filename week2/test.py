import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LibriDataset import LibriDataset
from MyModel import Classifier
from main import preprocess_data, batch_size, concat_nframes, model_path, input_dim, hidden_layers, hidden_dim, device

test_X = preprocess_data(split='test', feat_dir='./Dataset/libriphone/libriphone/feat',
                         phone_path='./Dataset/libriphone/libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
