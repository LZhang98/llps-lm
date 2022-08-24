from processing import deephase_data
import model
import encoder

if __name__ == '__main__':
    data = deephase_data()
    esm = model.ESM()
    num_layers = 1
    model_dim = 12
    num_heads = 3
    ff_dim = 12

    print('test')
    encoder = encoder.Encoder(num_layers, model_dim, num_heads, ff_dim)
    print(data)
