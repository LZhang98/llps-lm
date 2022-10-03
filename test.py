from encoder import Encoder
import torch

if __name__ == '__main__':

    # Hyperparams
    k_folds = 5
    num_epochs = 10
    loss_function1 = torch.nn.CrossEntropyLoss()
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