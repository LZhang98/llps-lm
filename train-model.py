from encoder import Encoder
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
import numpy as np
import sys

if __name__ == '__main__':

    # Hyperparams
    num_epochs = 50
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-2
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280
    random_seed = 69
    batch_size = 100
    model_name = f'model_lr{int(np.log10(learning_rate))}_bs{batch_size}_e{num_epochs}'

    print(f'num_epochs: {num_epochs}')
    print(f'learning_rate: {learning_rate}')
    print(f'num_layers: {num_layers}')
    print(f'model_dim: {model_dim}')
    print(f'num_heads: {num_heads}')
    print(f'ff_dim: {ff_dim}')
    print(f'batch_size: {batch_size}')
    print(f'model_name: {model_name}')
    sys.stderr.write(f'learning_rate: {learning_rate}\n')
    sys.stderr.write(f'batch_size: {batch_size}\n')
 
    # fixed seed
    torch.manual_seed(random_seed)

    logfile = '/cluster/projects/kumargroup/luke/' + model_name + '_log.csv'

    with open(logfile, 'a') as f:
        f.write('epoch\ttraining_loss\tvalidation_loss\taccuracy\tprediction\n')

    # Loading the dataset
    annotation = '/cluster/projects/kumargroup/luke/deephase_annotations_uniq.csv'
    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    dataset = CustomDataset(annotations_file=annotation, data_dir=data_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    # model and optimizer
    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # training loop

    for epoch in range(num_epochs):

        print('Starting epoch', epoch)
        my_model.train()
        running_t_loss = 0
        # i = batch index
        for i, data in enumerate(dataloader):
            inputs, targets = data
            inputs = inputs.squeeze().float()
            targets = targets.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = my_model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_t_loss += batch_loss * inputs.size(0)
            print(f'Loss: {batch_loss}')

        # write
        with open(logfile, 'a') as f:
            f.write(f'{epoch}\t{batch_loss}\n')

    # Process is complete.
    print('Training complete. Saving...')
    # Saving the model
    save_path = f'/cluster/projects/kumargroup/luke/output/new_models/{model_name}.pth'
    torch.save(my_model.state_dict(), save_path)
    print(save_path)