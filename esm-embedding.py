import model
import torch
import sys

args = sys.argv
if len(args) > 2:
    print('Usage: python3 esm-embedding.py input_fasta')
input_fasta = args[1]

my_esm = model.ESM()
output_dir = '/cluster/projects/kumargroup/luke/output/esm-embeddings/'

with open(input_fasta, 'r') as f_in:
    content = f_in.read()
    label = content.split('\n')[0].split(';')[0][1:]
    sequence = content.split('\n')[1]
    x = [(label, sequence)]
    representation = my_esm.get_representation(x)
    torch.save(representation, output_dir + label + '.pt')
    print(label + ':' + str(list(representation.size())))

    # test
    test = torch.load(output_dir + label + '.pt')
    print(torch.equal(representation, test))