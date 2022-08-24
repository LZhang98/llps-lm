import torch
import esm.pretrained
from encoder import Encoder

class ESM():

    def __init__(self, eval=True):
            
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

        if (eval):
            self.model.eval() # don't forget to disable for training

    def get_representation(self, data):

        representations = []
        for x in data:
            label, sequence = x
            batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts = True)
            token_representations = results['representations'][33][0][1:len(sequence)+1]

        representations.append(label, token_representations)
        
if __name__ == '__main__':
    print('model/model.py')

    # data = [(label, seq)]