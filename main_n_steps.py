# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 22:22:56 2022

@author: Shivangi A
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shivangi_nocntrl_noterm as SVS
import corev2 as core
import torch
import DSAC_RL as RL
import pdb
import pickle
import torch.nn as nn
import torch.nn.functional as F
import config
use_cuda = torch.cuda.is_available()
from collections import deque
criterion = nn.MSELoss()
import torch.optim as optim
device   = torch.device("cuda:0" if use_cuda else "cpu")
# pdb.set_trace()

#%% Agent and parameter
# Initial state.
x_init = random.randrange(-1000,1000)
# Initial velocity.
v_init = random.uniform(-10,10)
#initial control
c_init = random.uniform(-3, 3)
# Destination.
x_des = 0
# Card id.
car_id = 0
# Position threshold.
pos_th = 1000
vel_th = 10
#%%
# Call the environment and lstm model
env = SVS.Vehicle_model(car_id,x_init,x_init, v_init,v_init, c_init, c_init)
#%% RL agent initialize.
output_dim = env.output_dim

# Actions.
num_action = env.action_space_policy.shape[0]

# Batch size.
batch_size = 256
batch_size_update = 64
# Discount factor.
gamma = 0.99
# Update target per epsiode.
TARGET_UPDATE = 1
# Update every episode.
update_every = 2
#size of replay buffer
replay_size = int(1e6)
# N step.
n_steps = 10
# DQN_CE agent.
agent_DSAC = RL.Dsac(env, replay_size = replay_size, batch_size = batch_size, gamma=gamma,\
               lr= 3*1e-4, n_steps=n_steps)
    
# 0.5*1e-3
# Number of episodes.
episodes = 1000
# episodes = 100
# Number of time steps in an episode.
t_eps = 1000+n_steps
# t_eps = 1000
# Polay averaging.
polyak = 0.995
d = 1
d_int = 1
#%%LSTM initialisations
input_dim = env.input_space
hidden_dim = 64
layer_dim = 5  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = env.output_dim

# Actions.
num_action = env.action_space_policy.shape[0]

agent_lstm  = core.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(agent_lstm.parameters(), lr=learning_rate,  \
                             weight_decay= 1e-5) 
criterion_DQN = nn.MSELoss(reduction='none')
#%% Saving 
# Number of states
num_state = env.observation_space.shape[0]
state_str = np.zeros((episodes,t_eps+1,2))
state_str_t = np.zeros((episodes,t_eps+1,5))
#camera state
camera_str_t = np.zeros((episodes, t_eps+1, 4))

est_str_t = np.zeros((episodes,t_eps+1,4))
loss_train = np.zeros((episodes, t_eps,1))
control = np.zeros((episodes, t_eps+1,2))
process_noise = np.zeros((episodes, t_eps+1,4))
obs_noise = np.zeros((episodes, t_eps+1,4))
arrival_rate = np.zeros((episodes, t_eps+1,4))
service_rate = np.zeros((episodes))
reward_catch = np.zeros((episodes, t_eps+1, 1))
action_probs_catch = np.zeros((episodes, t_eps+1, 2))

# LSTM input and output.
MSE_in_queue = deque(maxlen=(n_steps))
MSE_op_queue = deque(maxlen=(n_steps))

MSE_in_queue_lstm = deque(maxlen=(n_steps))
MSE_op_queue_lstm = deque(maxlen=(n_steps))
# Input to LTSM
state_LSTM_input = np.zeros((2,1))

# Loss 
loss_str = []
#loss_lstm
loss_lstm = []
# Reward global
r_g_DQN_CE = []
# Intermediate update. 
int_up = 20
# Saving duration.
save_eps = 5
# Age performance measure.
age_per = np.inf
# MSE performance measure.
mse_per = np.inf
# Saving frequency.
save_f = 10
# Saving the test performance.
age_test_save = []
mse_test_save = []
p_test_save = []
pol_lr = []
crit_lr = []
al_str = []
pi_loss_str = []
q1_loss = []
q2_loss = []
# agent_DSAC.alpha = 0.01
#%% Counters and optimizer.
# Global counter.
steps_done = 0
ep = 0
#%% Training loop.
# Set the Actor network train.
agent_DSAC.ac.train()
# Set the LSTM to train.
agent_lstm.train()
    
    
for ep in range(max(ep,0),episodes):
    x_init = random.uniform(-100,100)
    x_init_y = random.uniform(-100, 100)
    
    config.q = 0.5
    # x_init = 999
    # x_init_y = -999
    
    v_init = random.uniform(-10,10)
    v_init_y = random.uniform(-10,10)
    
    c_init = random.uniform(-3, 3)
    c_init_y = random.uniform(-3, 3)
    env = SVS.Vehicle_model(car_id,x_init,x_init_y, v_init,v_init_y, c_init, c_init_y)
    # Save the actions (inner loop).
    act_save_il = []
    
    
    # Initialize the environment and state.
    state, state_policy = env.env_start(agent_lstm)
    
    # Inner loop.
    # Accumulate reward.
    rew_acc = []
    
    state_str_t[ep,0,0] = env.car[0,-1]
    state_str_t[ep,0,1] = env.car[1,-1]
    state_str_t[ep, 0, 2] = env.car[2,-1]
    state_str_t[ep, 0,3] = env.car[3,-1]
    
    camera_str_t[ep,0,0] = env.camera[0,-1]
    camera_str_t[ep,0,1] = env.camera[1,-1]
    camera_str_t[ep, 0, 2] = env.camera[2,-1]
    camera_str_t[ep, 0,3] = env.camera[3,-1]
    
    
    state_str_t[ep,0,4] = env.age[-1]
    est_str_t[ep,0,0] = env.car[0,-1]
    est_str_t[ep,0,1] = env.car[1,-1]
    est_str_t[ep,0,2] = env.car[2,-1]
    est_str_t[ep,0,3] = env.car[3,-1]
    control[ep, 0,0] = env.control[0].cpu().detach().numpy().item()
    control[ep, 0, 1] = env.control[1].cpu().detach().numpy().item()
        
    process_noise[ep,0,:] = env.W[:,-1].reshape(4)
    obs_noise[ep,0,:] = env.O[:,-1].reshape(4)
    arrival_rate[ep, 0, 0] = 1
    service_rate[ep] = config.q
    tau = 0
    state_n_str = deque(maxlen=(n_steps))
    action_n_str = deque(maxlen=(n_steps))
    rew_n_str = deque(maxlen=(n_steps))
    for t in range(t_eps-n_steps):
        # Next state based on the action taken.
        
        new_state_policy , MSE_input_batch, MSE_output, \
            reward, done, action, loga, action_probs =  env.env_step(agent_DSAC, t,  agent_lstm, 1.0, \
                                                     steps_done)
        
        # Normalize the reward. 
        # reward = reward/1e6
        # Update tau.
        tau = t - n_steps
        
        # Accumulate reward.
        rew_acc.append(reward.detach().cpu().numpy())
        
        #store lstm input and output
        agent_DSAC.replay_buffer_mse.store(MSE_input_batch.detach().cpu().numpy(),\
                                           MSE_output.T.detach().cpu().numpy())
        
        # Store 
        # pdb.set_trace()
        if t == 0:
            # Store the LSTM input for stoing in the LSTM.
            MSE_input_batch_pre = np.copy(MSE_input_batch.detach().cpu().numpy())
            MSE_output_pre = np.copy(MSE_output.T.detach().cpu().numpy())
            # Store the state information for the DQN. 
            state_policy = np.copy(new_state_policy)
            # Store action. 
            action_pre = int(np.copy(action))
            reward_pred = float(np.copy(reward.detach().cpu().numpy()))
            
            # Store the state. 
            state_n_str.append(state_policy)
           
            # Store the action. 
            action_n_str.append(action_pre)
            
            # Store reward. 
            # rew_n_str.append(reward)
            #rew_n_str.append(1-env.age[0]/t_eps)
        # t_eps-env.age[0]
            
        #correct for rewards
        MSE_in_queue.append(MSE_input_batch_pre)
        MSE_op_queue.append(MSE_output_pre)
        
        # pdb.set_trace()
        MSE_in_queue_lstm.append(MSE_input_batch.detach().cpu().numpy())
        MSE_op_queue_lstm.append(MSE_output.T.detach().cpu().numpy())
        
        if t >=1:
            # rew_n_str.append(1-env.age[0].item()/t_eps)
            rew_n_str.append(1+reward.item()/80000)
            
        
        #####################################
        # Store experience to replay buffer
        #####################################
        if tau >=0:
            G = 0
            # Accumulate the return.
            cnt = 0
            for j in range(tau+2,min(tau+n_steps+2,t_eps)):
                G = G + gamma**(j-tau-2)*rew_n_str[cnt]
                cnt = cnt + 1
            # pdb.set_trace()
            agent_DSAC.replay_buffer_rl.store(MSE_in_queue, \
                                             MSE_op_queue, \
                                             state_n_str[0], \
                                                 new_state_policy, \
                                                     action_n_str[0], np.copy(G), \
                                                         np.copy(loga.cpu().numpy().flatten()), \
                                                             done)
            
        if t >= 1:   
            # Update the state.
            state_policy = np.copy(new_state_policy)
            # Store the action.
            action_pre = int(np.copy(action))
            # Store the reward.
            reward_pred = float(np.copy(reward.detach().cpu().numpy()))
            
            # Store the state. 
            state_n_str.append(state_policy)
            # Store reward. 
            # rew_n_str.append(reward)
            # Store the action. 
            action_n_str.append(action)

            # Update the LSTM for next time step.
            MSE_input_batch_pre = np.copy(MSE_input_batch.detach().cpu().numpy())
            MSE_output_pre = np.copy(MSE_output.T.detach().cpu().numpy())
           
            # rew_n_str.append(1-env.age_policy[0].item()/t_eps)
            
         
        # Update loop.
        if ep >= save_eps and ep % update_every == 0:
            for j in range(update_every):
                
                ########################################
                # Update the LSTM every int_up episodes
                ########################################
                batch = agent_DSAC.replay_buffer_mse.sample_batch(batch_size)
                batch_input = batch['MSE_input'].to(device)
                MSE_output = batch['MSE_output'].to(device)
                
                MSE_input = agent_lstm(batch_input)
                optimizer.zero_grad()
                
                loss = criterion(MSE_input,\
                                  MSE_output)
                # pdb.set_trace()               
                # Getting gradients w.r.t. parameters
                loss.backward()
                  # Updating parameters
                optimizer.step()
                loss_lstm.append(loss.item())
                    
            ###########################################
            # Update the RL.
            ###########################################
            
            #when reward is MSE
        if ep>=int_up:
            # pdb.set_trace()
            batch_MSE, batch_policy = agent_DSAC.replay_buffer_rl.sample_batch(batch_size)
            MSE_seq_in = batch_MSE['MSE_input'].to(device)
            MSE_seq_out = batch_MSE['MSE_output']
            
            
            # Recompute the sequence. 
            MSE_out_new = torch.zeros(batch_size,n_steps,MSE_output_pre.shape[1])
            with torch.no_grad():
                for n_itr in range(n_steps):
                    MSE_out_new[:,n_itr,:] = agent_lstm(MSE_seq_in[:,n_itr,:])
            # pdb.set_trace()
            # Update DQN.
            r_new = criterion_DQN(MSE_out_new,MSE_seq_out)
            r_new = r_new.sum(dim=2)/r_new.shape[2]
            r_new = (1-(r_new.detach().cpu())/80000)
            # 
            # Reward sum.
            r_sum = torch.zeros(batch_size,)
            for r_itr in range(n_steps):
                r_sum = r_sum + gamma**(r_itr)*r_new[:,r_itr]
            
            # pdb.set_trace()
            # r_sum = (1+ r_sum/80000)
            batch_policy['obs'] = batch_policy['obs']
            batch_policy['obs2'] = batch_policy['obs2']
            batch_policy['act'] = batch_policy['act']
            #when reward is age
            # batch_policy['rew'] = batch_policy['rew']
            #when reward is mse
            batch_policy['rew'] = r_sum
            batch_policy['done'] = batch_policy['done']
            # pdb.set_trace()
            # Update the main network. 
            if ep <= d or t%d_int!=0:
                rl_loss_1, rl_loss_2 = agent_DSAC.update_stability(data=batch_policy)
            else:
                rl_loss_1, rl_loss_2, pi_loss = agent_DSAC.update(data=batch_policy)
                pi_loss_str.append(pi_loss)
            q1_loss.append(rl_loss_1) 
            q2_loss.append(rl_loss_2)
           
           # Soft update.
           # Store the alpha.
            al_str.append(agent_DSAC.alpha)
            
        state_str_t[ep,t+1,0] = env.car[0,-1]
        state_str_t[ep,t+1,1] = env.car[1,-1]
        state_str_t[ep, t+1,2] = env.car[2,-1]
        state_str_t[ep, t+1, 3] = env.car[3,-1]
        
        camera_str_t[ep,t+1,0] = env.camera[0,-1]
        camera_str_t[ep,t+1,1] = env.camera[1,-1]
        camera_str_t[ep, t+1,2] = env.camera[2,-1]
        camera_str_t[ep, t+1, 3] = env.camera[3,-1]
        
        state_str_t[ep,t+1,4] = env.age[-1]
       
        control[ep, t+1, 0] = env.control[0].cpu().detach().numpy().item()
        control[ep, t+1, 1] = env.control[1].cpu().detach().numpy().item()
        
        process_noise[ep,t+1,:] = env.W[:,-1].reshape(4)
        obs_noise[ep,t+1,:] = env.O[:,-1].reshape(4)
        arrival_rate[ep, t+1, 0] = action
        
        est_str_t[ep,t+1,:] = env.state_learn.cpu().detach().numpy()
       
        reward_catch[ep, t+1, 0] = float(np.copy(reward.detach().cpu().numpy()))
        action_probs_catch[ep, t+1, 0] = action_probs[0][0]
        action_probs_catch[ep, t+1, 1] = action_probs[0][1]
    # pdb.set_trace()
    print('Training, episode {}, reward {}'.format(ep,np.mean(state_str_t[ep,0:t_eps,4])))
    print('Training, episode {}, reward {}'.format(ep,np.sum(rew_acc)))
    
    # pdb.set_trace()
    r_g_DQN_CE.append(np.sum(rew_acc))
pdb.set_trace()

#%%
# from matplotlib import rc,rcParams
FIG_SIZE = 50
legend_size = 20
FONT_SIZE = 20
AXIS_SIZE = 20
TICK_SIZE = 20
lw = 8.0
ms = 20
mew = 3.5
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=AXIS_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=FIG_SIZE) 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


line_plt = ['b','orange','g','y','k','c','m','w']

marker_list = ['o','v','D','H','P','h','d','p','P','*','h','H','+','x','X','D','d','|','_']
lines = ['--', '-.', ':']
#%%
fig = plt.gcf()
fig.set_size_inches(9,7)
temp = arrival_rate[:,:,0]
temp = np.mean(temp, 1)
plt.plot(temp)
plt.xlabel('Episodes')
plt.ylabel('Probability of Querying')
plt.title('Trained Episodes')
#%%probability of last few episodes
start = len(temp) - 200
end = len(temp)
x_labels = np.linspace(start, end, end-start)

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, temp[-(end-start):], '-*')
plt.ylabel('Probability of Querying')
plt.title('Last 200 Trained Episodes')
#%%
fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
temp = state_str_t[:,:,4]
temp = np.mean(temp, 1)
plt.plot(temp)
plt.xlabel('Episodes')
plt.ylabel('Average Age')
#%%probability of last few episodes
start = len(temp) - 200
end = len(temp)
x_labels = np.linspace(start, end, end-start)

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, temp[-(end-start):], '-*')
plt.ylabel('Average Age')
plt.title('Last 200 Trained Episodes')
#%%trained episodes
id_plot = 800


plt.figure()
plt.plot(action_probs_catch[id_plot,1:t,0],  '-*')
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('No query')


plt.figure()
plt.plot(state_str_t[id_plot,1:t,4])
plt.xlabel('Steps')
plt.ylabel('Age')

plt.figure()
plt.plot(reward_catch[id_plot,2:t,0], '-*')
plt.xlabel('Steps')
plt.ylabel('Reward')
#%%plot losses

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(pi_loss_str, '-*')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Steps')
plt.title('Actor Loss')


fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(q1_loss, '-*')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Steps')
plt.title('Critic Loss - N/w 1')


fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(q2_loss, '-*')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Steps')
plt.title('Critic Loss - N/w 2')
#%%plot losses
start = len(pi_loss_str) - 1000
end = len(pi_loss_str)
x_labels = np.linspace(start, end, end-start)

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, pi_loss_str[-(end-start):], '-*')
plt.xlabel('Steps')
plt.title('Actor Loss')

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, q1_loss[-(end-start):], '-*')
plt.xlabel('Steps')
plt.title('Critic Loss')

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, q2_loss[-(end-start):], '-*')
plt.xlabel('Steps')
plt.title('Critic Loss - N/w 2')

#%%
id_arr = [80, 200, 450, 800, 999]
prob_change = np.zeros((len(id_arr),1001,1))
for i in range(len(id_arr)):
    for j in range(1001):
        if arrival_rate[id_arr[i],j,0] == 0:
            prob_change[i,j,0]  = action_probs_catch[id_arr[i],j,0]
        else:
            prob_change[i,j,0]  = action_probs_catch[id_arr[i],j,1]
        
p1, p2 = sorted(prob_change[0,:,0]), \
    np.arange(len(prob_change[0,:,0])) / len(prob_change[0,:,0])
p3, p4 = sorted(prob_change[1,:,0]), \
    np.arange(len(prob_change[1,:,0])) / len(prob_change[1,:,0])
p5, p6 = sorted(prob_change[2,:,0]), \
    np.arange(len(prob_change[2,:,0])) / len(prob_change[2,:,0])
p7,p8 = sorted(prob_change[3,:,0]), \
    np.arange(len(prob_change[3,:,0])) / len(prob_change[3,:,0])
p9, p10 = sorted(prob_change[4,:,0]), \
    np.arange(len(prob_change[4,:,0])) / len(prob_change[4,:,0])
# p11, p12 = sorted(prob_change[5,:,0]), \
#     np.arange(len(prob_change[5,:,0])) / len(prob_change[5,:,0])
plt.figure()
fig = plt.gcf()
fig.set_size_inches(9,7)

a = id_arr[0]
plt.plot(p1,p2, '-*', label = 'Ep {}'.format(a))

a = id_arr[1]
plt.plot(p3,p4, '-*', label  = 'Ep {}'.format(a))

a = id_arr[2]
plt.plot(p5,p6, '-*', label = 'Ep {}'.format(a))

a = id_arr[3]
plt.plot(p7,p8, '-*', label = 'Ep {}'.format(a))

a = id_arr[4]
plt.plot(p9,p10, '-*', label = 'Ep {}'.format(a))

# a = id_arr[5]
# plt.plot(p11,p12, '-*', label = 'Ep {}'.format(a))

plt.legend()
plt.title('CDF for change in probabilities')
#%% Testing the network
agent_DSAC.ac.train()
# Number of episodes.
ep_test = 100
# Number of time steps.
traj_itr = 40000

# State true.
true_state = np.zeros((ep_test,traj_itr+1,5))
#camera state
camera_state = np.zeros((ep_test, traj_itr+1, 5))
# State predict.
pred_state = np.zeros((ep_test,traj_itr+1,4))
arrival_rate_test =np.zeros((ep_test,traj_itr+1,1))
service_rate_test = np.zeros((ep_test))
reward_catch_test = np.zeros((ep_test,traj_itr+1,1))
action_probs_catch_test = np.zeros((ep_test,traj_itr+1,2))
steps_done_test = steps_done

#%%
# Set the Actor network train.
agent_DSAC.ac.eval()
# Set the LSTM to train.
agent_lstm.eval()

# Reward vector.
traj_reward = []
with torch.no_grad():
    for episode_test in range(ep_test):
        print('Episode {}'.format(episode_test))
        # Initial state.
       
        x_init = random.uniform(-100,100)
        x_init_y = random.uniform(-100, 100)
         
          
        v_init = random.uniform(-10,10)
        v_init_y = random.uniform(-10,10)
        
        c_init = random.uniform(-3, 3)
        c_init_y = random.uniform(-3, 3)
        config.q = 0.5
      
        # Environment.
        env = SVS.Vehicle_model(car_id,x_init,x_init_y, v_init,v_init_y, c_init, c_init_y)
        # Reset the environment.
        # Initialize the environment and state.
        state, state_policy = env.env_start(agent_lstm)
        true_state[episode_test,0,0] = env.car[0,-1]
        true_state[episode_test,0,1] = env.car[1,-1]
        true_state[episode_test, 0, 2] = env.car[2,-1]
        true_state[episode_test, 0,3] = env.car[3,-1]
        
        camera_state[episode_test,0,0] = env.camera[0,-1]
        camera_state[episode_test,0,1] = env.camera[1,-1]
        camera_state[episode_test, 0, 2] = env.camera[2,-1]
        camera_state[episode_test, 0,3] = env.camera[3,-1]
        
        true_state[episode_test,0,4] = env.age[-1]
        
        pred_state[episode_test,0,0] = env.car[0,-1]
        pred_state[episode_test,0,1] = env.car[1,-1]
        pred_state[episode_test,0,2] = env.car[2,-1]
        pred_state[episode_test,0,3] = env.car[3,-1]
        
        
        episode_reward = []
        # Start the time trajectory loop.
        for step in range(traj_itr):
         
        
            new_state_policy , MSE_input_batch, MSE_out, \
                reward, done, action, loga, action_probs_test = \
                    env.env_step(agent_DSAC, step,  agent_lstm, 1.0, \
                                                         2000000)
          

            # Save the trajectory.
            true_state[episode_test,step+1,0] = env.car[0,-1]
            true_state[episode_test,step+1,1] = env.car[1,-1]
            true_state[episode_test,step+1,2] = env.car[2,-1]
            true_state[episode_test,step+1,3] = env.car[3,-1]
            
            camera_state[episode_test,step+1,0] = env.camera[0,-1]
            camera_state[episode_test,step+1,1] = env.camera[1,-1]
            camera_state[episode_test, step+1, 2] = env.camera[2,-1]
            camera_state[episode_test, step+1,3] = env.camera[3,-1]
            
            
            true_state[episode_test, step+1, 4] = env.age[-1]
            arrival_rate_test[episode_test, step+1, 0] = action
            reward_catch_test[episode_test, step+1, 0] = reward
            action_probs_catch_test[episode_test, step+1, 0] = action_probs_test[0][0]
            action_probs_catch_test[episode_test, step+1, 1] = action_probs_test[0][1]
            
            
            # Predicted state.
            m_out = agent_lstm(MSE_input_batch,False)
            pred_state[episode_test,step+1,:] = m_out.cpu().numpy()
        print(np.mean(true_state[0,:,4]))
#%%
# from matplotlib import rc,rcParams
FIG_SIZE = 50
legend_size = 20
FONT_SIZE = 20
AXIS_SIZE = 20
TICK_SIZE = 20
lw = 8.0
ms = 20
mew = 3.5
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=AXIS_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=legend_size)    # legend fontsize
plt.rc('figure', titlesize=FIG_SIZE) 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


line_plt = ['b','orange','g','y','k','c','m','w']

marker_list = ['o','v','D','H','P','h','d','p','P','*','h','H','+','x','X','D','d','|','_']
lines = ['--', '-.', ':']
#%%
fig = plt.gcf()
fig.set_size_inches(9,7)
temp = arrival_rate[0:ep-1,:,0]
temp = np.mean(temp, 1)
plt.plot(temp[150:300])
plt.xlabel('Episodes')
plt.ylabel('Probability of Querying')
plt.title('Trained Episodes')
#%%plots
fig = plt.gcf()
fig.set_size_inches(9,7)
temp = arrival_rate_test[:,:,0]
temp = np.mean(temp, 1)
plt.plot(temp, '-*')
plt.xlabel('Episodes')
plt.ylabel('Probability of Querying')

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
temp = true_state[:,:,4]
temp = np.mean(temp, 1)
plt.plot(temp, '-*')
plt.xlabel('Episodes')
plt.ylabel('Average Age')
#%%tested episodes
id_plot = 66

plt.figure()
plt.plot(action_probs_catch_test[id_plot,1:,0],  '-*')
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('Probability of No Query')


plt.figure()
plt.plot(true_state[id_plot,1:,4])
plt.xlabel('Steps')
plt.ylabel('Age')



plt.figure()
plt.plot(reward_catch_test[id_plot,1:,0])
plt.xlabel('Steps')
plt.ylabel('Reward')

plt.figure()
plt.plot(true_state[id_plot,1:,0], label = 'X-Position')
plt.plot(pred_state[id_plot,1:,0], label = 'Estimate')
plt.xlabel('Steps')
#%%
start = 7500
end = 7800
x_labels = np.linspace(start, end, end-start)

fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, true_state[id_plot,start:end,4], '-*')
plt.xlabel('Steps')
plt.title('Age')


fig = plt.gcf()
fig.set_size_inches(9,7)
plt.figure()
plt.plot(x_labels, reward_catch_test[id_plot,start:end,0], '-*')
plt.xlabel('Steps')
plt.title('Reward')
#%%
state_sqr = np.mean(np.square((true_state[:,:,0] - pred_state[:,:,0]) + \
                    (true_state[:,:,1] - pred_state[:,:,1]) + \
                        (true_state[:,:,2] - pred_state[:,:,2]) + \
                            (true_state[:,:,3] - pred_state[:,:,3])), axis = 1)
#%%
plt.plot(state_sqr, '-*')
plt.xlabel('Episodes')
plt.ylabel('MSE')
plt.title('MSE for q = $0.3$')
#%%
id_delete = 42
true_state = np.delete(true_state, id_delete, axis = 0)
pred_state = np.delete(pred_state, id_delete, axis = 0)
arrival_rate_test = np.delete(arrival_rate_test, id_delete, axis = 0)
#%%plot square root of mse each episode
state_sqrt = np.mean(np.sqrt((np.square(true_state[:,:,0] - pred_state[:,:,0])) + \
                    (np.square(true_state[:,:,1] - pred_state[:,:,1])) + \
                        (np.square(true_state[:,:,2] - pred_state[:,:,2])) + \
                            (np.square(true_state[:,:,3] - pred_state[:,:,3]))), axis = 1)
#%%
plt.plot(state_sqrt, '-*')
plt.xlabel('Episodes')
plt.ylabel('Square Root MSE')
plt.title('Euclidean norm for q = $0.5$')
#%% Saving 

path = os.getcwd()
env_name = 'Policy_MSE_DSAC_LSTM/MSE_loss'
dir_save = path + '/' +str(env_name) + '/'  + \
            str(agent_DSAC.t_alpha) + '_' + str(0.5) + '_' + str(n_steps) + '/Train' 
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
#%%
model_DSAC = dir_save + '/Model_DSAC'
# Saving the weights.
torch.save(agent_DSAC.ac.state_dict(), model_DSAC)

model_LSTM = dir_save + '/Model_LSTM'
torch.save(agent_lstm.state_dict(), model_LSTM)

#%% Saving training
# Save the reward, state, action.
with open(dir_save + '/state_true.pickle', 'wb') as handle:
    pickle.dump(state_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/est_state.pickle', 'wb') as handle:
    pickle.dump(est_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_save + '/control.pickle', 'wb') as handle:
#     pickle.dump(control, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_save + '/process_noise.pickle', 'wb') as handle:
#     pickle.dump(process_noise, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_save + '/obs_noise.pickle', 'wb') as handle:
#     pickle.dump(obs_noise, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(dir_save + '/arrival_rate.pickle', 'wb') as handle:
#     pickle.dump(arrival_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/reward.pickle', 'wb') as handle:
    pickle.dump(r_g_DQN_CE, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/alpha_change.pickle', 'wb') as handle:
    pickle.dump(al_str, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/critic_loss.pickle', 'wb') as handle:
    pickle.dump(loss_str, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/actor_loss.pickle', 'wb') as handle:
    pickle.dump(pi_loss_str, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/rew_ep.pickle', 'wb') as handle:
    pickle.dump(reward_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/action_probs.pickle', 'wb') as handle:
    pickle.dump(action_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/lstm_loss.pickle', 'wb') as handle:
    pickle.dump(loss_lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%saving test data
path = os.getcwd()
dir_save = path + '/Policy_MSE_DSAC_LSTM/MSE_loss/0.5_0.5_10/Test'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

#%% Saving data.
# Save the reward, state, action.
with open(dir_save + '/true_state.pickle', 'wb') as handle:
    pickle.dump(true_state, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dir_save + '/pred_state.pickle', 'wb') as handle:
    pickle.dump(pred_state, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open(dir_save + '/arrival_rate.pickle', 'wb') as handle:
    pickle.dump(arrival_rate_test, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open(dir_save + '/reward_ep.pickle', 'wb') as handle:
    pickle.dump(reward_catch_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/action_catch.pickle', 'wb') as handle:
    pickle.dump(action_probs_catch_test, handle, protocol=pickle.HIGHEST_PROTOCOL)