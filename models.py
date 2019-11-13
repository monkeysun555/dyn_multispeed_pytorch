import torch
import torch.nn as nn
from config import Config

class Model(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        if Config.model_version == 0:
            self.lstm1 = nn.LSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)

            self.fc1 = nn.Sequential(
                nn.Linear(in_features=5, out_features=32),
                nn.ReLU())

            self.fc2 = nn.Sequential(
                nn.Linear(in_features=992, out_features=128),
                nn.ReLU())

            print(action_dims)
            self.multi_output = nn.ModuleList([nn.Linear(in_features=128, out_features=dim) for dim in action_dims]) 

            # self.dueling = nn.Sequential(
            #     nn.Linear(in_features=512, out_channels=1),
            #     nn.ReLU())
        
    def forward(self, observation):
        # Shape of observation: (batch, 15, 10) (batch, seq, input_size)
        h0 = torch.randn(2*2, 1, 32).cuda()
        c0 = torch.randn(2*2, 1, 32).cuda()
        if Config.model_version == 0:
            lstm1_out, (hn, cn) = self.lstm1(torch.transpose(observation[0:5,:],0,1).unsqueeze(0), (h0,c0))         # input: (5, 15) to (1,15,5) , output: (1,15, 2*32)
            fc1_out = self.fc1(observation[5:,-1].unsqueeze(0))                            # input: (1,5) output: (1, 32)
            # print(fc1_out.size())         
            fc2_out = self.fc2(torch.cat((torch.flatten(lstm1_out, start_dim=1), fc1_out), 1))                      # flatten: (1,15,2*32) to (1,-1) and cat with (1,32)
            advantages=[]
            for linear in self.multi_output:
                branch_out = linear(fc2_out)                           
                advantages.append(branch_out)
        return advantages

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
