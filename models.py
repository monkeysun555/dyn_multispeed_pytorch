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

            # print(action_dims)
            self.multi_output_1 = nn.Linear(in_features=128, out_features=action_dims[0]) 
            self.multi_output_2 = nn.Linear(in_features=128, out_features=action_dims[1]) 
            # self.dueling = nn.Sequential(
            #     nn.Linear(in_features=512, out_channels=1),
            #     nn.ReLU())
        elif Config.model_version == 1:
            self.lstm1 = nn.LSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=5, out_features=32),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=992, out_features=128),
                nn.ReLU())
            self.output = nn.Linear(in_features=128, out_features=action_dims[0]*action_dims[1])

    def forward(self, observation):
        # Shape of observation: (batch, 15, 10) (batch, seq, input_size)
        h0 = torch.randn(2*2, len(observation), 32).cuda()
        c0 = torch.randn(2*2, len(observation), 32).cuda()
        if Config.model_version == 0:
            lstm1_out, (hn, cn) = self.lstm1(torch.transpose(observation[:, 0:5,:], 1, 2), (h0,c0))         # input: (5, 15) to (1,15,5) , output: (1,15, 2*32)
            fc1_out = self.fc1(observation[:, 5:, -1])                                                     # input: (1,5) output: (1, 32)
            fc2_out = self.fc2(torch.cat((torch.flatten(lstm1_out, start_dim=1), fc1_out), 1))                      # flatten: (1,15,2*32) to (1,-1) and cat with (1,32)
            advantages=[self.multi_output_1(fc2_out), self.multi_output_2(fc2_out)]
            return advantages
        elif Config.model_version == 1:
            lstm1_out, (hn, cn) = self.lstm1(torch.transpose(observation[:, 0:5,:], 1, 2), (h0,c0))         # input: (5, 15) to (1,15,5) , output: (1,15, 2*32)
            fc1_out = self.fc1(observation[:, 5:, -1])
            fc2_out = self.fc2(torch.cat((torch.flatten(lstm1_out, start_dim=1), fc1_out), 1))                      # flatten: (1,15,2*32) to (1,-1) and cat with (1,32)
            advantage = self.output(fc2_out)
            return advantage

    def save(self, path, step, optimizers):
        if Config.model_version == 0:
            torch.save({
                'step': step,
                'state_dict': self.state_dict(),
                'optimizer_1': optimizers[0].state_dict(),
                'optimizer_2': optimizers[1].state_dict(),
            }, path)
        elif Config.model_version == 1:
            torch.save({
                'step': step,
                'state_dict': self.state_dict(),
                'optimizer': optimizers.state_dict()
            }, path)
            
    def load(self, checkpoint_path, optimizers=None, version=Config.model_version):
        if version == 0:
            checkpoint = torch.load(checkpoint_path)
            step = checkpoint['step']
            self.load_state_dict(checkpoint['state_dict'])
            if not optimizers:
                optimizers[0].load_state_dict(checkpoint['optimizer_1'])
                optimizers[1].load_state_dict(checkpoint['optimizer_2'])
        elif version == 1:
            checkpoint = torch.load(checkpoint_path)
            step = checkpoint['step']
            self.load_state_dict(checkpoint['state_dict'])
            if not optimizers:
                optimizer.load_state_dict(checkpoint['optimizer'])
