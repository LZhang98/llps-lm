from encoder import Encoder
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import CustomDataset
import numpy as np
import sys

if __name__ == '__main__':

    # Hyperparams
    num_epochs = 50
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-4
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280
    validation_split = 0.2
    random_seed = 69
    batch_size = 50

    # fixed seed
    torch.manual_seed(random_seed)
    
    model_name = 'model_no_pdb_no_batch'
    logfile = '/cluster/projects/kumargroup/luke/' + model_name + '_log.csv'

    with open(logfile, 'a') as f:
        f.write('epoch\ttraining_loss\tvalidation_loss\taccuracy\tprediction\n')

    # Loading the dataset
    annotation = '/cluster/projects/kumargroup/luke/deephase_annotations_no_pdb.csv'
    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    dataset = CustomDataset(annotations_file=annotation, data_dir=data_dir)

    # Split into train and valid
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split * len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]

    # data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validationloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    # model and optimizer
    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # training loop

    for epoch in range(num_epochs):

        print('Starting epoch', epoch)
        my_model.train()
        running_t_loss = 0
        running_v_loss = 0
        # i = batch index
        for i, data in enumerate(trainloader):
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
            print(f'Loss after mini-batch {i+1}: {batch_loss}')
        
        # stats for epoch
        training_loss = running_t_loss / len(train_indices)

        # evaluation
        my_model.eval()
        correct, total = 0, 0
        predictions = []
        with torch.no_grad():
        # Iterate over the validation data and generate predictions
            for i, data in enumerate(validationloader, 0):
                # Get inputs
                inputs, targets = data
                inputs = inputs.squeeze()
                targets = targets.unsqueeze(1)

                # Generate outputs
                outputs = my_model(inputs)

                # validation loss
                loss = loss_function(outputs, targets)

                # Set total and correct
                predicted = (outputs > 0.5).int()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # loss tabulation
                batch_loss = loss.item()
                running_v_loss += batch_loss * inputs.size(0)

                # prediction tabulation
                predicted = predicted.squeeze()
                for i in range(predicted.size(0)):
                    predictions += [predicted[i].item()]
            
            # epoch statistics
            validation_loss = running_v_loss / len(validation_indices)
            accuracy = correct / total

            # write
            with open(logfile, 'a') as f:
                f.write(f'{epoch}\t{training_loss}\t{validation_loss}\t{accuracy}\t{str(predictions)}\n')

    # Process is complete.
    print('Training complete. Saving...')
    # Saving the model
    # save_path = f'/cluster/projects/kumargroup/luke/output/trained_models/fold-{fold}.pth'
    save_path = f'/cluster/projects/kumargroup/luke/output/new_models/{model_name}.pth'
    torch.save(my_model.state_dict(), save_path)

