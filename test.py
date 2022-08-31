from processing import deephase_data
import model
import encoder

if __name__ == '__main__':
    print('test')
    data, categories = deephase_data()
    esm = model.ESM()
    num_layers = 1
    model_dim = 12
    num_heads = 3
    ff_dim = 12
    encoder = encoder.Encoder(num_layers, model_dim, num_heads, ff_dim)
    print(data[0])
    test_output = esm.get_representation([data[0]])
    print(test_output)