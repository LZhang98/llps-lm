import model
import encoder
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
import os
from sklearn.model_selection import KFold

if __name__ == '__main__':

    # The preparatory steps.
    k_folds = 5
    num_epochs = 10
    loss_function = torch.nn.CrossEntropyLoss()
    results = {}
    # fixed seed
    torch.manual_seed(69)

    # Loading the dataset
    annotation = '/cluster/projects/kumargroup/luke/deephase_annotations_uniq.csv'
    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'

    dataset = CustomDataset(annotations_file=annotation, data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Defining the K-fold Cross Validator to generate the folds.
    kfold = KFold(n_splits=k_folds, shuffle=True)
    print('---------------------------------------')

    # Then, generating the splits that we can actually use for training the model, which we also do - once for every fold.

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('FOLD', fold)
        print('---------------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=10, sampler=test_subsampler)

        my_model = model.ESM()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):

            print('Starting epoch', epoch+1)
            current_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                input, targets = data
                optimizer.zero_grad()
                outputs = my_model(input)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                #Stats
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                    current_loss = 0.0

    # After training for every fold, we evaluate the performance for that fold.
    # Finally, we perform performance evaluation for the model - across the folds.