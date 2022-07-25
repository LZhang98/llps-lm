import torch

class Model():

    def __init__(self):
        self.conv1 = torch.nn.Conv1d()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        return(x)
        
if __name__ == '__main__':
    print('model/model.py')