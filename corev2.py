import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
from torch.distributions import Categorical
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        # self.lstm1 = nn.LSTMCell(hidden_dim, hidden_dim)
       
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.4)
      

    def forward(self, x, Test=True):
        
            # pdb.set_trace()
        x = x.float()
        # x = self.l_obs(x)
            
            # Initialize hidden state with zeros
            # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
    
            # Initialize cell state
            # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        hx, cx = self.lstm(x)
        out = self.relu(self.fc(hx))
        out = self.fc2(out)
        return out
#%%

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)




def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()      
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.sp = nn.Softplus()
        self.tan = nn.Tanh()
        
    def forward(self, obs):
        # First layer out.
        # pdb.set_trace()
        net_out = self.net(obs)
        # mu_out = self.mu_layer(net_out).clamp(max = 5.0, min= -5.0)
        mu_out = self.mu_layer(net_out)
        # mu_out = self.sp(mu_out).clamp(max = 5.0, min= 1e-20)
        # mu_out = 3*self.tan(mu_out)
        # pdb.set_trace()
        # Action layer.
        # if obs.shape[0] == 1:
        #     mu = F.softmax(mu_out, dim=0).clamp(max = 1 - 1e-20, min= 1e-20)
        # else:
            # mu = F.softmax(mu_out, dim=1).clamp(max = 1 - 1e-20, min= 1e-20)
        # mu = F.gumbel_softmax(logits=mu_out, tau=1, dim=1).clamp(max = 1 - 1e-20, min= 1e-20)
        
        mu = F.softmax(mu_out, dim=1).clamp(max = 1 - 1e-20, min= 1e-20)
        # mu = F.softmax(mu_out, dim=1)
        # mu_log = F.log_softmax(mu, dim=1)
        # if obs.shape[0] > 1:
        #     mu = F.softmax(mu, dim=1).clamp(max = 1 - 1e-20)
        # else:
        #     mu = F.softmax(mu, dim=0).clamp(max = 1 - 1e-20)
        # pdb.set_trace()
        
        m = Categorical(mu)
        action = m.sample()
        # print(action)
        return mu, action, mu.log()
     

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        
    def forward(self, obs):
        pdb.set_trace()
        q = self.q(obs)
        # pdb.set_trace()
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    
class estimate_q(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # pdb.set_trace()
        self.lstm = nn.LSTMCell(obs_dim, hidden_sizes[0])
        self.fc = nn.Linear(hidden_sizes[0], act_dim)
        # self.est_q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.sg = nn.Sigmoid()

    def forward(self, obs,hx=None,cx=None):
        # pdb.set_trace()
        if hx==None:    
            hx,cx = self.lstm(obs)
            q = self.fc(hx).clamp(min=0.0)
        else:
            hx,cx = self.lstm(obs,(hx,cx))
            q = self.fc(hx)
        q = self.sg(q)
        # pdb.set_trace()
        return torch.squeeze(q, -1),hx,cx # Critical to ensure q has right shape.
    

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space,  action_space, hidden_sizes=(256,256),
                 activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space #22
        act_dim = action_space

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
       

    def act(self, obs):
        with torch.no_grad():
            # pdb.set_trace()
            action_probs, a, loga = self.pi(obs.reshape(1,len(obs)).to(device))
            # print(a)
            return a, loga, action_probs
        
    
       

