import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
import math
from config import Config
from models import Model

class Agent:
    def __init__(self, action_dims, random_seed=Config.random_seed):
        np.random.seed(random_seed)
        # self.action_num = action_num
        self.action_dims = action_dims
        self.epsilon = Config.initial_epsilon
        self.epsilon_final = Config.epsilon_final
        self.epsilon_start = Config.epsilon_start
        self.epsilon_decay = Config.epsilon_decay
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_dims).cuda()
        self.target_network = Model(self.action_dims).cuda()
        # Change learning rate for commen net !!!! Start from here
        if Config.model_version == 0:
            self.optimizers = [optim.Adam([
                {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
                {'params': self.Q_network.fc2.parameters()},
                {'params': self.Q_network.fc1.parameters()},
                {'params': self.Q_network.lstm1.parameters()},
                ], lr=0.5*Config.lr), 
                optim.Adam([
                {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
                {'params': self.Q_network.fc2.parameters()},
                {'params': self.Q_network.fc1.parameters()},
                {'params': self.Q_network.lstm1.parameters()},
                ], lr=0.5*Config.lr)]
        elif Config.model_version == 1:
            self.optimizers = optim.Adam(self.Q_network.parameters(), lr=Config.lr)
    
    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network_v0(self, state, action_1, action_2, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action_1 = torch.from_numpy(action_1).float()
        action_2 = torch.from_numpy(action_2).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state).cuda()
        action_1 = Variable(action_1).cuda()                 
        action_2 = Variable(action_2).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        # actions_new = self.Q_network.forward(state_new).max(dim=2)[1].cpu().data.view(-1, 1)        # To be modified
        # actions_new = self.Q_network.forward(state_new).max(dim=2)[1].cpu().data.view(-1, 1)        # To be modified
        actions_new = [torch.max(q_value, 1)[1].cpu().data.view(-1, 1) for q_value in self.Q_network.forward(state_new)] 
        actions_new_onehot = [torch.zeros(Config.sampling_batch_size, action_dim) for action_dim in self.action_dims]
        actions_new_onehot = [Variable(actions_new_onehot[action_idx].scatter_(1, actions_new[action_idx], 1.0)).cuda() for action_idx in range(len(self.action_dims))]
        
        # Different loss and object
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        
        actions = [action_1, action_2]
        pre_y = self.target_network.forward(state_new)
        y = []
        for new_q_idx in range(len(pre_y)):
            y.append(reward + torch.mul(((pre_y[new_q_idx]*actions_new_onehot[new_q_idx]).sum(dim=1)*terminal),Config.discount_factor))
        self.Q_network.train()
        pre_Q = self.Q_network.forward(state)
        Q = []
        for q_idx in range(len(pre_Q)):
            Q.append((pre_Q[q_idx]*actions[q_idx]).sum(dim=1))
        losses = []
        for action_idx in range(len(self.action_dims)):
            # y = reward + torch.mul(((self.target_network.forward(state_new)[action_idx]*actions_new_onehot[action_idx]).sum(dim=1)*terminal), Config.discount_factor)
            # Q = (self.Q_network.forward(state)[action_idx]*actions[action_idx]).sum(dim=1)
            loss = mse_loss(input=Q[action_idx], target=y[action_idx].detach())
            self.optimizers[action_idx].zero_grad()
            if action_idx < len(self.action_dims)-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.optimizers[action_idx].step()
            losses.append(loss.item())
        # print(losses)
        return losses

    def update_Q_network_v1(self, state, action, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state).cuda()
        action = Variable(action).cuda()                  # shape (batch, 6*7)
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        # actions_new = self.Q_network.forward(state_new).max(dim=2)[1].cpu().data.view(-1, 1)        # To be modified
        # actions_new = self.Q_network.forward(state_new).max(dim=2)[1].cpu().data.view(-1, 1)        # To be modified
        actions_new = self.Q_network.forward(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
        actions_new_onehot = torch.zeros(Config.sampling_batch_size, self.action_dims[0]*self.action_dims[1]) 
        actions_new_onehot = Variable(actions_new_onehot.scatter_(1, actions_new, 1.0)).cuda()
        
        # Different loss and object
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = reward + torch.mul(((self.target_network.forward(state_new)*actions_new_onehot).sum(dim=1)*terminal),Config.discount_factor)
        self.Q_network.train()
        Q = (self.Q_network.forward(state)*action).sum(dim=1)
        losses = []
        # y = reward + torch.mul(((self.target_network.forward(state_new)[action_idx]*actions_new_onehot[action_idx]).sum(dim=1)*terminal), Config.discount_factor)
        # Q = (self.Q_network.forward(state)[action_idx]*actions[action_idx]).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())
        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()
        losses.append(loss.item())
        return losses

    def take_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state).cuda()
        self.Q_network.eval()
        if Config.model_version == 0:
            estimate = [torch.max(q_value, 1)[1].data[0] for q_value in self.Q_network.forward(state)] 
            # with epsilon prob to choose random action else choose argmax Q estimate action
            if random.random() < self.epsilon:
                return [random.randint(0, self.action_dims[action_idx]-1) for action_idx in range(len(self.action_dims))]
            else:
                return estimate
        elif Config.model_version == 1:
            estimate = torch.max(self.Q_network.forward(state), 1)[1].data[0]
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dims[0]*self.action_dims[1]-1)
            else:
                return estimate

    def update_epsilon_by_epoch(self, epoch):
        self.epsilon = self.epsilon_final+(self.epsilon_start - self.epsilon_final) * math.exp(-1.*epoch/self.epsilon_decay)       
    def set_epsilon_for_testing(self):
        self.epsilon = 0.0
    
    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > Config.maximum_model - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step, optimizers=self.optimizers)
        print('=> Save {}' .format(logs_path)) 
    
    def restore(self, logs_path, version=Config.model_version):
        if Config.model_version == 0:
            self.Q_network.load(logs_path, self.optimizers, version)
            self.target_network.load(logs_path, self.optimizers, version)
            print('=> Restore {}' .format(logs_path)) 
            # self.optimizers = [optim.Adam([
            #     {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
            #     {'params': self.Q_network.fc2.parameters()},
            #     {'params': self.Q_network.fc1.parameters()},
            #     {'params': self.Q_network.lstm1.parameters()},
            #     ], lr=0.5*Config.lr), 
            #     optim.Adam([
            #     {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
            #     {'params': self.Q_network.fc2.parameters()},
            #     {'params': self.Q_network.fc1.parameters()},
            #     {'params': self.Q_network.lstm1.parameters()},
            #     ], lr=0.5*Config.lr)]
        elif Config.model_version == 1:
            self.Q_network.load(logs_path, self.optimizers, version)
            self.target_network.load(logs_path, self.optimizers, version)
            print('=> Restore {}' .format(logs_path))

