import torch
import torch.nn as nn
from config import Config

class Model(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        if Config.model_version == 0:
            self.lstm1 = nn.LSTM(input_size=5, hidden_size=256, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)

            self.fc1 = nn.Sequential(
                nn.Linear(in_features=5, out_features=256),
                nn.ReLU())

            self.fc2 = nn.Sequential(
                nn.Linear(in_features=512, out_features=128),
                nn.ReLU())

            print(action_dims)
            self.multi_output = nn.ModuleList([nn.Linear(in_features=128, out_features=dim) for dim in action_dims]) 

            # self.dueling = nn.Sequential(
            #     nn.Linear(in_features=512, out_channels=1),
            #     nn.ReLU())
        
    def forward(self, observation):
        # Shape of observation: (batch, 15, 10) (batch, seq, input_size)
        if Config.model_version == 0:
            lstm1_out = self.lstm1(torch.transpose(observation[:,0:5,:],1,2))         # output shape(batch, seq, input_size) as batch_first is used, (batch, 15, 5)
            fc1_out = self.fc1(torch.squeeze(observation[:,5:, -1]))             # output shape(batch, hidden_size) as batch_first is used, (batch, 256)   
            fc2_out = self.fc_2(torch.cat((lstm1_out.view(-1, 15*5), fc1_out), 1))      #fc2_out shape: (batch, 128)
            advantages=[]
            for count, linear in self.multi_output:
                branch_out = linear(fc2_out)                            # Advantage of each branch shape (batch, 6/7)
                advantages.append(branch_out)
        return advantages                                           # set of y_d 
    
    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
            
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
