from encoder import Encoder
import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader

def test1():
    # Hyperparams
    k_folds = 5
    num_epochs = 10
    # loss_function1 = torch.nn.CrossEntropyLoss()
    loss_function2 = torch.nn.MSELoss()
    learning_rate = 1e-4
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280

    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)

    input = torch.randn(10, 1336, 1280)
    print(input.size())

    output = my_model(input)
    print(output)
    print(output.size())
    print(output.type())

    targets = torch.randint(low=0, high=2, size=(10, 1)).float()
    print(targets)
    
    loss = loss_function2(output, targets)
    print(loss)

def test2():
    # Hyperparams
    k_folds = 5
    num_epochs = 30
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-2
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280

    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # Loading the dataset
    annotation = '/cluster/projects/kumargroup/luke/deephase_annotations_uniq.csv'
    data_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings-padded/'
    dataset = CustomDataset(annotations_file=annotation, data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=50)

    inputs, targets = next(iter(dataloader))
    inputs = inputs.squeeze().float()
    targets = targets.unsqueeze(1).float()
    # outputs = my_model(inputs)
    # print(outputs)
    # print(targets)
    # loss = loss_function(outputs, targets)
    # print(loss)
    # predicted = (outputs > 0.5).float()
    # print(predicted)

    for epoch in range(num_epochs):
        my_model.train()
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        with open('/cluster/projects/kumargroup/luke/test_log.csv', 'a') as f:
            print(f'{epoch}:\t{loss.item()}')
            f.write(f'{epoch}:\t{loss.item()}\n')
    
    predictions = (outputs > 0.5).float()
    print(predictions)
    return None

def test3():
    # Hyperparams
    k_folds = 5
    num_epochs = 50
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-4
    num_layers = 5
    model_dim = 1280
    num_heads = 4
    ff_dim = 1280
    torch.manual_seed(69)

    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    inputs = torch.randn(32, 1336, 1280)
    targets = torch.randint(low=0, high=2, size=(32, 1)).float()

    open('/cluster/projects/kumargroup/luke/test3_log.csv', 'w').close()

    for epoch in range(num_epochs):
        my_model.train()
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        predictions = (outputs > 0.5).float()
        accuracy = ((predictions == targets).sum().item()) / 32

        with open('/cluster/projects/kumargroup/luke/test3_log.csv', 'a') as f:
            print(f'{epoch}:\t{loss.item()}\t{accuracy}\t{predictions.reshape(32).tolist()}')
            f.write(f'{epoch}:\t{loss.item()}\t{accuracy}\t{predictions.reshape(32).tolist()}\n')
    
    return None

def test4():
    # Hyperparams
    embedding_dim = 240

    k_folds = 5
    num_epochs = 50
    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-4
    num_layers = 5
    model_dim = embedding_dim
    num_heads = 4
    ff_dim = embedding_dim
    torch.manual_seed(69)
    dataset_size = 32

    my_model = Encoder(num_layers, model_dim, num_heads, ff_dim)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    inputs = torch.randn(dataset_size, 1336, embedding_dim)
    targets = torch.randint(low=0, high=2, size=(dataset_size, 1)).float()

    open('/cluster/projects/kumargroup/luke/test4_log.csv', 'w').close()

    for epoch in range(num_epochs):
        my_model.train()
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        predictions = (outputs > 0.5).float()
        accuracy = ((predictions == targets).sum().item()) / 32

        with open('/cluster/projects/kumargroup/luke/test4_log.csv', 'a') as f:
            print(f'{epoch}:\t{loss.item()}\t{accuracy}\t{predictions.reshape(32).tolist()}')
            f.write(f'{epoch}:\t{loss.item()}\t{accuracy}\t{predictions.reshape(32).tolist()}\n')
    
    return None

if __name__ == '__main__':
    test3()
