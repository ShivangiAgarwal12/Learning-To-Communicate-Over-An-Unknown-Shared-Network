# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 22:25:46 2022

@author: Shivangi A
"""
from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
# import corev2 as core
# import corev3 as core
import corev5 as core
import pdb

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#%% Algorithms

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN (policy) and LSTM.
    """

    def __init__(self, state_dim, obs_dim, size):
        self.mse_input = np.zeros(core.combined_shape(size, state_dim), dtype=np.float32)
        self.mse_output = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
      

    def store(self, mse_input, mse_output):
        # pdb.set_trace()
        self.mse_input[self.ptr] = mse_input 
        self.mse_output[self.ptr] = mse_output
        
      
    
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(MSE_input=self.mse_input[idxs],
                     MSE_output=self.mse_output[idxs])
      
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    
class ReplayBuffer_nstep:
    """
    A simple FIFO experience replay buffer for DQN (policy) and LSTM.
    """

    def __init__(self, state_dim, obs_dim, obs_dim_policy, act_dim_policy, log_pi_dim, size, n_step_agent):
        self.mse_input = np.zeros((size, n_step_agent, state_dim), dtype=np.float32)
        self.mse_output = np.zeros((size, n_step_agent, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        #for policy
        self.obs_buf_policy = np.zeros(core.combined_shape(size, obs_dim_policy), dtype=np.float32)
        self.last_lstm_in = np.zeros(core.combined_shape(size, state_dim), dtype=np.float32)
        self.obs2_buf_policy = np.zeros(core.combined_shape(size, obs_dim_policy), dtype=np.float32)
        self.curr_lstm_in = np.zeros(core.combined_shape(size, state_dim), dtype=np.float32)
        self.act_buf_policy = np.zeros(core.combined_shape(size, act_dim_policy), dtype=np.float32)
        self.logpi_buf = np.zeros(core.combined_shape(size, log_pi_dim), dtype=np.float32)
        self.dsac_alpha = np.zeros(core.combined_shape(size, 1), dtype = np.float32)
        self.n_step_agent = np.zeros(core.combined_shape(size, 1), dtype = np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        

    def store(self, mse_input, mse_output, lstm_in, lstm_next, obs_policy, next_obs_policy,  act_policy, rew, log_act, done, alpha, n_steps):
        # pdb.set_trace()
        self.mse_input[self.ptr] = np.array(mse_input).reshape(self.mse_input[self.ptr].shape)
        self.mse_output[self.ptr] = np.array(mse_output).reshape(self.mse_output[self.ptr].shape)
        
        self.last_lstm_in[self.ptr] = lstm_in
        self.curr_lstm_in[self.ptr] = lstm_next
        self.obs_buf_policy[self.ptr] = obs_policy
        self.obs2_buf_policy[self.ptr] = next_obs_policy
        self.act_buf_policy[self.ptr] = act_policy
        self.rew_buf[self.ptr] = rew
        self.logpi_buf[self.ptr] = log_act
        self.done_buf[self.ptr] = done
        self.dsac_alpha[self.ptr] = alpha
        self.n_step_agent[self.ptr] = n_steps

        
    
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(MSE_input=self.mse_input[idxs],
                     MSE_output=self.mse_output[idxs])
        # pdb.set_trace()
        batch_policy = dict(obs=self.obs_buf_policy[idxs],
                     obs2=self.obs2_buf_policy[idxs],
                     lstm_in = self.last_lstm_in[idxs],
                     curr_lstm = self.curr_lstm_in[idxs],
                     act=self.act_buf_policy[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     logpi=self.logpi_buf[idxs],
                     alpha = self.dsac_alpha[idxs],
                     n_steps = self.n_step_agent[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}, \
            {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch_policy.items()}

class ReplayBuffer_service:
    """
    A simple FIFO experience replay buffer for DQN (policy) and LSTM.
    """

    def __init__(self, size,input_size,hid_size):
        self.input = np.zeros(core.combined_shape(size, input_size), dtype=np.float32)
        self.output = np.zeros(core.combined_shape(size, 1), dtype=np.float32)
        self.hidden = np.zeros(core.combined_shape(size, hid_size), dtype=np.float32)
        self.cell = np.zeros(core.combined_shape(size, hid_size), dtype=np.float32)
        
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
      

    def store(self, input_service, output_service, hidden, cell):
        # pdb.set_trace()
        self.input[self.ptr] = input_service 
        self.output[self.ptr] = output_service
        self.hidden[self.ptr] = hidden 
        self.cell[self.ptr] = cell 
        
      
    
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(input_service=self.input[idxs],
                     output_service=self.output[idxs],
                     hidden=self.hidden[idxs],
                     cell=self.cell[idxs])
      
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    

class Dsac:
    def __init__(self,env,actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
            replay_size=int(3e3), gamma=0.99, 
            polyak=0.995, lr=1e-3, alpha=0.0, batch_size=100,
            env_info = {}, n_steps_agent=[1,1,1], age_hist = 0, lstm_hid=None):
    
        # Discount factor.
        self.gamma = gamma
        # N steps.
        self.n_steps_agent = n_steps_agent
        # Learning rate.
        self.lr = lr
        # Polay 
        self.polyak = polyak
        # Batch size
        self.batch_size = batch_size
    
    
        input_dim = env.input_space
        mse_dim_buffer = 4
        
        obs_dim_policy = env.observation_space_policy.shape
        act_dim_policy = env.action_space_policy.shape[0]
        # pdb.set_trace()
        # Create actor-critic module and target networks
        self.ac = actor_critic(obs_dim_policy[0]+age_hist, act_dim_policy,\
                               **ac_kwargs).to(device)
        self.ac_targ = deepcopy(self.ac)
        
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q1_params =  itertools.chain(self.ac.q1.parameters())
        self.q2_params =  itertools.chain(self.ac.q2.parameters())
        self.pi_params = itertools.chain(self.ac.pi.parameters())
        # Experience buffer
        # pdb.set_trace()
        self.replay_buffer_mse = ReplayBuffer(state_dim = input_dim, obs_dim= mse_dim_buffer,
                                              size = replay_size)
            
        
        # self.replay_buffer_rl = ReplayBuffer_nstep(state_dim = input_dim, obs_dim= mse_dim_buffer,\
        #                                   obs_dim_policy = obs_dim_policy[0]+age_hist, act_dim_policy=act_dim_policy-1,\
        #                                       size = replay_size, log_pi_dim = act_dim_policy, n_step = n_steps)
        
        self.replay_buffer_rl = {}
        for x in range(0,3):
            self.replay_buffer_rl["buffer{0}".format(x)] = ReplayBuffer_nstep(state_dim = input_dim, obs_dim= mse_dim_buffer,\
                                              obs_dim_policy = obs_dim_policy[0]+age_hist, act_dim_policy=act_dim_policy-1,\
                                                  size = replay_size, log_pi_dim = act_dim_policy,\
                                                      n_step_agent = n_steps_agent[x])
            
        # pdb.set_trace()
        self.replay_buffer_service = ReplayBuffer_service(size = replay_size, input_size = age_hist+3, hid_size=lstm_hid)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
      
        #self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        # self.q_optimizer = Adam(self.q_params, lr= self.lr, eps=1e-4)
        self.q1_optimizer = Adam(self.q1_params, lr= self.lr)
        self.q2_optimizer = Adam(self.q2_params, lr= self.lr)
        # Alpha optimizer.
        # self.log_alpha = torch.tensor(-0.9, requires_grad=True)
        # Config for q=0.1
        # self.log_alpha = torch.tensor(1.0, requires_grad=True)
        # self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        self.log_alpha = torch.tensor(1.0, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        
        # Target alpha.
        self.t_alpha = 0.98 * -np.log(1 / act_dim_policy)
        # Confiq q = 0.1
        # self.t_alpha = 0.1
        # self.t_alpha = 0.4
        # self.t_alpha = 0.09
        # Alpha
        # pdb.set_trace()
        self.alpha = self.log_alpha.exp().item()
        # self.alpha = 0.1
        # Average reward.
        self.avg_r = torch.tensor(0.0)
        # Loss objective.
        self.criterion = nn.MSELoss()
        

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2, n_steps, _ = data['obs'].to(device),\
            data['act'].to(device), data['rew'].to(device), \
                data['obs2'].to(device), data['n_steps'].to(device), data['done'].to(device)
        # pdb.set_trace()
        q1 = self.ac.q1(o).gather(1, a.long())
        q2 = self.ac.q2(o).gather(1, a.long())
                                      
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # pdb.set_trace()
            pi_a2, a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2)
            q2_pi_targ = self.ac_targ.q2(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # pdb.set_trace()
            Val_next = pi_a2*(q_pi_targ - self.alpha * logp_a2)
            Val_next = Val_next.sum(dim=1)
            backup = r + self.gamma**(int(n_steps[0].item()))*Val_next
            
        # pdb.set_trace()
        backup = backup.reshape(Val_next.shape[0],1)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        
        # loss_q = loss_q1 + loss_q2 
        # pdb.set_trace()
        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q1,loss_q2,q_info

   

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        # pdb.set_trace()
        o,a = data['obs'].to(device),data['act'].to(device)
        
        # Given the observation, the policy network gives the 
        # distribution of the discrete set of actions.
        pi, a, logp_pi = self.ac.pi(o)
        
        # Given the observation the critic network provides the Q values
        # for each action.
        q1_pi = self.ac.q1(o)
        q2_pi = self.ac.q2(o)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        inside_terms = self.alpha * logp_pi - q_pi 
        loss_pi = (pi*inside_terms).sum(dim=1).mean()
        # pdb.set_trace()
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
       
        return  loss_pi, pi_info
    
    def alpha_loss(self, data):
        # pdb.set_trace()
        o = data['obs'].to(device)
        alpha = data['alpha'][0].to(device)
        # pdb.set_trace()
        #mu is the probability, action is final to pick, alpha is h backward
        with torch.no_grad():
            pi,a, logp_pi = self.ac.pi(o)
        
        # pdb.set_trace()
        log_probs = pi*logp_pi
        log_probs = log_probs.sum(dim=1)
        # alpha_loss = -(self.log_alpha*(log_probs.detach() + self.t_alpha)).mean()
        alpha_loss = -(self.log_alpha*(log_probs.detach() + alpha.item())).mean()
        # loss_al = pi*(-(self.log_alpha*(logp_pi  + self.t_alpha)))
        # loss_al = loss_al.sum(dim=1).mean()
        # pdb.set_trace()
        return alpha_loss
        

    def update(self,data):
        # First run one gradient descent step for Q1 and Q2
        # pdb.set_trace()
        
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        # pdb.set_trace()
        loss_q1,loss_q2, q_info = self.compute_loss_q(data)
        loss_q1.backward()
        loss_q2.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q1_params:
            p.requires_grad = False
        for p in self.q2_params:
            p.requires_grad = False

        
        for p in self.pi_params:
            p.requires_grad = True
        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        
        # Optimize alpha.
        self.alpha_optimizer.zero_grad()
        loss_al = self.alpha_loss(data)
        loss_al.backward()
        self.alpha_optimizer.step()
        # # Set alpha.
        self.alpha = self.log_alpha.exp().item()
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q1_params:
            p.requires_grad = True
        for p in self.q2_params:
            p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "a dd", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
      
        return loss_q1.item(), loss_q2.item(), loss_pi.item()
    
    def update_stability(self,data):
        # First run one gradient descent step for Q1 and Q2
        # pdb.set_trace()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        # pdb.set_trace()
        loss_q1,loss_q2, q_info = self.compute_loss_q(data)
        loss_q1.backward()
        loss_q2.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        
        # # Optimize alpha.
        # self.alpha_optimizer.zero_grad()
        # loss_al = self.alpha_loss(data)
        # loss_al.backward()
        # self.alpha_optimizer.step()
        # # Set alpha.
        # self.alpha = self.log_alpha.exp().item()
        
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q1_params:
            p.requires_grad = True
        for p in self.q2_params:
            p.requires_grad = True
        
        
        # pdb.set_trace()
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for q1,q2,q1_targ,q2_targ in zip(self.ac.q1.parameters(),self.ac.q2.parameters(), self.ac_targ.q1.parameters(), self.ac_targ.q2.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "a dd", which would make new tensors.
                q1_targ.data.mul_(self.polyak)
                q1_targ.data.add_((1 - self.polyak) * q1.data)
                q2_targ.data.mul_(self.polyak)
                q2_targ.data.add_((1 - self.polyak) * q2.data)
                
      
        return loss_q1.item(), loss_q2.item()
    
    def update_avgr(self,data):
        # First run one gradient descent step for Q1 and Q2
        
        self.q_optimizer.zero_grad()
       
        loss_q, q_info = self.compute_loss_q_avgr(data)
        # pdb.set_trace()
        loss_q.backward()
        self.q_optimizer.step()
        # self.q_scheduler.step()


        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        # pdb.set_trace()
        self.pi_optimizer.zero_grad()
        #self.lstm_optimer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        act_loss = loss_pi
        loss_pi.backward()
      
        #torch.nn.utils.clip_grad_norm_(self.ac.pi.net[0].parameters(), 40.0)
        self.pi_optimizer.step()
        # self.scheduler.step()
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        
        for p in self.pi_params:
            p.requires_grad = True
        

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "a dd", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
      
        return act_loss

    def get_action(self,o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32))
