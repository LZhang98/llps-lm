from processing import deephase_data
import model
import torch

print('test.py')
out_dir = 'script-output/'
data, categories = deephase_data()
# test_data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]
esm = model.ESM()
# test_output = esm.get_representation(test_data)
test_output = esm.get_representation([data[0]])
print(test_output.size())
torch.save(test_output, out_dir + 'esm_embedding.pt')