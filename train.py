from encoder import Encoder
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
import os
from sklearn.model_selection import KFold

if __name__ == '__main__':

    # Hyperparams
    k_folds = 5
    num_epochs = 10
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-4
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280

    results = {}
    # fixed seed
    torch.manual_seed(69)

    # Loading the dataset
    annotation = '/cluster/projects/kumargroup/luke/deephase_annotations_uniq.csv'
    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    dataset = CustomDataset(annotations_file=annotation, data_dir=data_dir)

    # Defining the K-fold Cross Validator to generate the folds.
    kfold = KFold(n_splits=k_folds, shuffle=True)
    print('---------------------------------------')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('FOLD', fold)
        print('---------------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=32, sampler=test_subsampler)

        my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
        optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):

            print('Starting epoch', epoch+1)
            current_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                input, targets = data
                input = input.squeeze().float()
                targets = targets.unsqueeze(1).float()
                optimizer.zero_grad()
                outputs = my_model(input)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                #Stats
                current_loss += loss.item()
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 100))
                current_loss = 0
        
        # Process is complete.
        print('Training complete.')
        # Saving the model
        save_path = f'/cluster/projects/kumargroup/luke/output/trained_models/fold-{fold}.pth'
        torch.save(my_model.state_dict(), save_path)

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs, targets = data
                inputs = inputs.squeeze()
                targets = targets.unsqueeze(1)

                # Generate outputs
                outputs = my_model(inputs)

                # Set total and correct
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

    # After training for every fold, we evaluate the performance for that fold.
    # Finally, we perform performance evaluation for the model - across the folds.
