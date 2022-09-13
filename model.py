import torch
import esm.pretrained
from encoder import Encoder

class ESM():

    def __init__(self, eval=True):
            
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        if (eval):
            self.model.eval() # don't forget to disable for training

    def get_representation(self, x):

        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts = True)
        token_representations = results['representations'][33]

        return token_representations
        
if __name__ == '__main__':
    print('model/model.py')
