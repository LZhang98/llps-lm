import torch
from encoder import Encoder
from dataset import CustomDataset
from torch.utils.data import DataLoader

# path = '/cluster/projects/kumargroup/luke/output/trained_models/fold-0.pth'
# path = '/cluster/projects/kumargroup/luke/output/new_models/model_no_pdb_no_batch.pth'
path = '/cluster/projects/kumargroup/luke/output/new_models/model_no-batch_full-set.pth'
num_layers = 5
model_dim = 1280
num_heads = 4
ff_dim = 1280
batch_size = 50

print(f'path: {path}')
print(f'num_layers: {num_layers}')
print(f'model_dim: {model_dim}')
print(f'num_heads: {num_heads}')
print(f'ff_dim: {ff_dim}')
print(f'batch_size: {batch_size}')

my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)

my_model.load_state_dict(torch.load(path))
my_model.eval()

# Test 1: subsetted data (downsampled negative, no PDB*)

# annotation1 = '/cluster/projects/kumargroup/luke/deephase_annotations_no_pdb.csv'
data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
# dataset1 = CustomDataset(annotations_file=annotation1, data_dir=data_dir)
# dataloader1 = DataLoader(dataset1, batch_size=batch_size)

# correct, total = 0, 0
# for i, data in enumerate(dataloader1):
#     inputs, targets = data
#     inputs = inputs.squeeze().float()
#     targets = targets.unsqueeze(1).float()

#     outputs = my_model(inputs)
#     predicted = (outputs > 0.5).int()
#     total += targets.size(0)
#     correct += (predicted == targets).sum().item()

# print(f'Accuracy: {correct/total}')

# Test 2: full dataset
annotation2 = '/cluster/projects/kumargroup/luke/deephase_annotations_uniq.csv'
dataset2 = CustomDataset(annotations_file=annotation2, data_dir=data_dir)
dataloader2 = DataLoader(dataset2, batch_size=batch_size)

correct, total = 0, 0
for i, data in enumerate(dataloader2):
    inputs, targets = data
    inputs = inputs.squeeze().float()
    targets = targets.unsqueeze(1).float()

    outputs = my_model(inputs)
    predicted = (outputs > 0.5).int()
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f'Accuracy: {correct/total}. {correct}/{total}')