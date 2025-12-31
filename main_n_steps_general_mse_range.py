# -*- coding: utf-8 -*-
"""
age as loss with multiple buffers 
for initial values for t alpha, file is in DSAC_RL under folder name actor_critic_td0/Discrete SAC with kalman
on 192.168.30.243 
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
# import env_with_q_backlog_v2 as SVS
# import env_with_q_backlog_expo as SVS
# import env_with_q_backlog_nohist as SVS
# import env_with_q_backlog_lstm as SVS
# import env_with_q_backlog_range as SVS
import env_rtt as SVS
# import env_with_q_backlog_range_age_sec as SVS
# import corev2 as core
# import corev3 as core
import corev5 as core
import torch
import DSAC_RL_v2 as RL
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
import scipy.stats as stats

#%% Agent and parameter
# Initial state.
x_init = random.randrange(-100,100)
# Initial velocity.
v_init = random.uniform(-10,10)
# Destination.
x_des = 0
# Card id.
car_id = 0
# Position threshold.
pos_th = 1000
vel_th = 10
#%%
# Call the environment and lstm model
env = SVS.Vehicle_model(car_id,x_init,x_init, v_init,v_init, v_init, v_init)
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
update_every = 1
#size of replay buffer
replay_size = int(2e6)
# N step.
n_steps_agent = [60,20,5]
# Age history.
age_hist = 0
hidden_dim_service = (64,)
# pdb.set_trace()
# DQN_CE agent.
agent_DSAC = RL.Dsac(env, replay_size = replay_size, batch_size = batch_size, gamma=gamma,\
               lr= 2.5*1e-4, n_steps_agent=n_steps_agent,age_hist=age_hist,lstm_hid=hidden_dim_service[0])
    
# 0.5*1e-3
# Number of episodes.
episodes = 10000
# episodes = 100
# Number of time steps in an episode.
t_eps = 1000+50
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
criterion_s = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(agent_lstm.parameters(), lr=learning_rate,  \
                             weight_decay= 1e-5) 
criterion_DQN = nn.MSELoss(reduction='none')
# pdb.set_trace()
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
service_est_t = np.zeros((episodes,t_eps+1,1))
reward_catch = np.zeros((episodes, t_eps+1, 1))
action_probs_catch = np.zeros((episodes, t_eps+1, 2))



# Input to LTSM
state_LSTM_input = np.zeros((2,1))

# Loss 
loss_str = []
#loss_lstm
loss_lstm = []
#loss service
loss_service_rate = []
# Reward global
r_g_DQN_CE = []
# Intermediate update. 
# int_up = 5
# Saving duration.
save_eps = 1
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
#%%build the range
a = 5
b = 1
range_1 = np.linspace(b, a, a-b+1)
range_1 = range_1/10
range_1 = np.insert(range_1, 0, 0.05)
range_1 = np.delete(range_1, 4, axis = 0)
#%% Training loop.
# Set the Actor network train.
agent_DSAC.ac.train()
# Set the LSTM to train.
agent_lstm.train()
    
lower, upper, scale = 0.1, 0.3, 1/10.5
q_sample_low = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
data_l = q_sample_low.rvs(episodes)

lower, upper, scale = 0.1, 0.3, 1/4.0
q_sample_high = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
data_r = q_sample_high.rvs(int(episodes - episodes/2))
data_r = q_sample_high.rvs(int(episodes))

t_up = 30
count = 10
flag_LSTM = 1
r_scale = 3.0
#%%
for ep in range(max(ep,0),episodes):
    if ep % 50 == 0 and ep > save_eps:
        flag_LSTM = 1 - flag_LSTM
    x_init = random.uniform(-100,100)
    x_init_y = random.uniform(-100, 100)
    
    
    v_init = random.uniform(-10,10)
    v_init_y = random.uniform(-10,10)

    c_init = random.uniform(-3,3)
    c_init_y = random.uniform(-3,3)
    
    #keeping same buffer from 0.3 to 0.5
    prob_range = 1/(len(range_1)-1)
    choice_service = np.random.choice(range_1, 1)[0]
    # pick_constant = np.random.choice([0.05,0.08], 1)[0]
    pick_constant = np.random.choice([0.6,0.8], 1)[0]
    choice_service = pick_constant
    if choice_service == 0.05 or choice_service == 0.08:
        config.q = np.random.uniform(choice_service, choice_service + 0.03)
        lower, upper, scale = choice_service, choice_service + 0.03, 1/4.0
        q_sample_high = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
        config.q = q_sample_high.rvs(1)
        agent_DSAC.t_alpha = 0.09
        choice_buffer = 0
        n_steps = 60
        
    if choice_service == 0.6 or choice_service == 0.8:
        config.q = np.random.uniform(choice_service, choice_service + 0.1)
        # agent_DSAC.t_alpha = 0.8
        # lower, upper, scale = choice_service, choice_service + 0.2, 1/4.0
        # q_sample_high = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
        # config.q = q_sample_high.rvs(1)
        agent_DSAC.t_alpha = 0.8
        choice_buffer = 2
        n_steps = 5

    # if choice_service == 0.6 or choice_service == 0.8:
    #     config.q = np.random.uniform(choice_service, choice_service + 0.1)
    #     # agent_DSAC.t_alpha = 0.8
    #     # lower, upper, scale = choice_service, choice_service + 0.2, 1/4.0
    #     # q_sample_high = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
    #     # config.q = q_sample_high.rvs(1)
    #     agent_DSAC.t_alpha = 0.8
    #     choice_buffer = 2
    #     n_steps = 5
        
    if choice_service == 0.1 or choice_service == 0.2:
        config.q = np.random.uniform(choice_service, choice_service + 0.1)
        # lower, upper, scale = choice_service, choice_service + 0.1, 1/4.0
        # q_sample_high = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
        # config.q = q_sample_high.rvs(1)
        agent_DSAC.t_alpha = 0.2
        choice_buffer = 1
        n_steps = 20
    
    
    service_rate[ep] = config.q
    
    env = SVS.Vehicle_model(car_id,x_init,x_init_y, v_init,v_init_y, c_init, c_init_y, age_hist=0)
    if ep<=config.int_up and ep%2==0:
        env.r_st = True
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
    lstm_in_n_str = deque(maxlen=(n_steps))
    
    # LSTM input and output.
    MSE_in_queue = deque(maxlen=(n_steps))
    MSE_op_queue = deque(maxlen=(n_steps))

    MSE_in_queue_lstm = deque(maxlen=(n_steps))
    MSE_op_queue_lstm = deque(maxlen=(n_steps))
    
    
    t_eps = 1000+n_steps
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

            # Sotre the last lstm intput.
            lstm_in_n_str.append(np.copy(env.last_lstm_input.cpu().numpy().flatten()))
           
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
            # rew_n_str.append(1-env.age[0].item()/(t_eps - n_steps))
            # rew_n_str.append(r_scale+r_scale*(reward.item()/80000))
            #range 1000
            rew_n_str.append(r_scale+r_scale*(reward.item()/80000.0))
            
        
        #####################################
        # Store experience to replay buffer
        #####################################
        if tau >=0 and ep >= (config.int_up-100):
            G = 0
            # Accumulate the return.
            cnt = 0
            for j in range(tau+2,min(tau+n_steps+2,t_eps)):
                G = G + gamma**(j-tau-2)*rew_n_str[cnt]
                cnt = cnt + 1 

            buffer_store = list(agent_DSAC.replay_buffer_rl.keys())[choice_buffer]
            agent_DSAC.replay_buffer_rl[buffer_store].store(MSE_in_queue, \
	                                         MSE_op_queue, \
	                                         lstm_in_n_str[0],\
	                                         np.copy(env.last_lstm_input.cpu().numpy().flatten()),\
	                                         state_n_str[0], \
	                                             new_state_policy, \
	                                                 action_n_str[0], np.copy(G), \
	                                                     np.copy(loga.cpu().numpy().flatten()), \
	                                                         done,\
	                                                         agent_DSAC.t_alpha, n_steps)
           
            
        if t >= 1:   
            # Update the state.
            state_policy = np.copy(new_state_policy)
            # Store the action.
            action_pre = int(np.copy(action))
            # Store the reward.
            reward_pred = float(np.copy(reward.detach().cpu().numpy()))
            
            # Store the state. 
            state_n_str.append(state_policy)

            # Sotre the last lstm intput.
            lstm_in_n_str.append(np.copy(env.last_lstm_input.cpu().numpy().flatten()))
            # Store reward. 
            # rew_n_str.append(reward)
            # Store the action. 
            action_n_str.append(action)

            # Update the LSTM for next time step.
            MSE_input_batch_pre = np.copy(MSE_input_batch.detach().cpu().numpy())
            MSE_output_pre = np.copy(MSE_output.T.detach().cpu().numpy())
           
            # rew_n_str.append(1-env.age_policy[0].item()/t_eps)
            
         
        # Update loop.
        if ep >= save_eps and ep % update_every == 0 :
            
            for j in range(update_every):
                
                ########################################
                # Update the LSTM every int_up episodes
                ########################################
                batch = agent_DSAC.replay_buffer_mse.sample_batch(batch_size=256)
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
        
        if ep>=config.int_up and t%t_up==0:
            t_up = max(t_up-10,1)
        # if ep>=int_up:
        if ep>=config.int_up and t%t_up == 0 :
            # print('Training Started')
            # pdb.set_trace()
            buffer_pick = np.random.choice(np.arange(0,3))
            buffer_pick = 2
            if buffer_pick ==0:
                agent_DSAC.t_alpha = 0.09
            elif buffer_pick==1:
                agent_DSAC.t_alpha = 0.2
            else:
                agent_DSAC.t_alpha = 0.6

            # pdb.set_trace()
            # buffer_pick = np.random.choice(np.arange(0,3))
            buffer_pick_rl = list(agent_DSAC.replay_buffer_rl)[buffer_pick]
            batch_MSE, batch_policy = agent_DSAC.replay_buffer_rl[buffer_pick_rl].sample_batch(batch_size)
            MSE_seq_in = batch_MSE['MSE_input'].to(device)
            MSE_seq_out = batch_MSE['MSE_output']
            #pdb.set_trace()
            
            # Recompute the sequence.
            batch_policy['n_steps'] = batch_policy['n_steps']
            n_steps_compute =  int(batch_policy['n_steps'][0].item())

            
            MSE_out_new = torch.zeros(batch_size,n_steps_compute,MSE_output_pre.shape[1])
            with torch.no_grad():
                for n_itr in range(n_steps_compute):
                    MSE_out_new[:,n_itr,:] = agent_lstm(MSE_seq_in[:,n_itr,:])
            # pdb.set_trace()
            # Update DQN.
            r_new = criterion_DQN(MSE_out_new,MSE_seq_out)
            r_new = r_new.sum(dim=2)/r_new.shape[2]
            r_new = (r_scale-r_scale*(r_new.detach().cpu()/80000.0))
            # r_new = (r_scale-r_scale*(r_new.detach().cpu()/8000000.0))
            
            # 
            # Reward sum.
            r_sum = torch.zeros(batch_size,)
            for r_itr in range(n_steps_compute):
                r_sum = r_sum + gamma**(r_itr)*r_new[:,r_itr]
            
            # pdb.set_trace()
            with torch.no_grad():
                obs_state_learn = agent_lstm(batch_policy['lstm_in'].to(device))
                obs2_state_learn = agent_lstm(batch_policy['curr_lstm'].to(device))
            
            # pdb.set_trace()
            # r_sum = (1+ r_sum/80000)
            batch_policy['obs'] = batch_policy['obs']
            # Change the input of obs.
            batch_policy['obs'][:,0:output_dim] = obs_state_learn

            batch_policy['obs2'] = batch_policy['obs2']
            # Change the input of obs2.
            batch_policy['obs2'][:,0:output_dim] = obs2_state_learn

            batch_policy['act'] = batch_policy['act']
            #when reward is age
            # batch_policy['rew'] = batch_policy['rew']
            #when reward is mse
            batch_policy['rew'] = r_sum
            batch_policy['done'] = batch_policy['done']
            batch_policy['alpha'] = batch_policy['alpha']
            batch_policy['n_steps'] = batch_policy['n_steps']
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
    # if ep>=int_up and ep%50==0:
    #     t_up = max(t_up-10,30)
    # pdb.set_trace()
    print('Training, episode {}, Age {}'.format(ep,np.mean(state_str_t[ep,0:t_eps,4])))
    # print('Training, episode {}, reward {}'.format(ep,np.sum(rew_acc)))
    print('Training, episode {}, Service rate {}'.format(ep, service_rate[ep]))
    print('Training, episode {}, Query rate {}'.format(ep, np.mean(arrival_rate[ep,0:t_eps,0])))
    if ep >= save_eps:
        print('MSE error {}'.format(loss.item()))

    # if ep>=int_up and ep%50==0:
    #     t_up = max(t_up-10,1)
    r_g_DQN_CE.append(np.sum(rew_acc))
    if ep%500 == 0 and ep > 0:
      path = os.getcwd()
      env_name = 'Policy_MSE'
      # dir_save = path + '/' +str(env_name) + '_DSAC_LSTM/Multiple_buffers/range_0.3_tries/Train_'  + str(ep) + '/' + \
      #              str(agent_DSAC.t_alpha) + '_'  + str(n_steps)
      dir_save = path + '/' +str(env_name) + '_DSAC_LSTM/RTT_based'
      if not os.path.exists(dir_save):
          os.makedirs(dir_save)
      model_DSAC = dir_save + '/Model_DSAC'
      model_lstm  =dir_save + '/Model_LSTM'
      # Saving the weights.
      torch.save(agent_DSAC.ac.state_dict(), model_DSAC)
      torch.save(agent_lstm.state_dict(), model_lstm)

    if ep % 5000 == 0 and ep >0:
        path = os.getcwd()
        env_name = 'Policy_MSE'
        dir_save = path + '/' +str(env_name) + '_DSAC_LSTM/Multiple_buffers/range_0.3_tries/Train_'  +  str(ep) + '/' + \
                     str(agent_DSAC.t_alpha) + '_'  + str(n_steps)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        model_DSAC = dir_save + '/Model_DSAC'
        model_lstm  =dir_save + '/Model_LSTM'
        # Saving the weights.
        torch.save(agent_DSAC.ac.state_dict(), model_DSAC)
        torch.save(agent_lstm.state_dict(), model_lstm)

        #%% Saving data.
        # Save the reward, state, action.
        with open(dir_save + '/state_true.pickle', 'wb') as handle:
            pickle.dump(state_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(dir_save + '/estimate.pickle', 'wb') as handle:
            pickle.dump(est_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/control.pickle', 'wb') as handle:
            pickle.dump(control, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(dir_save + '/arrival_rate.pickle', 'wb') as handle:
            pickle.dump(arrival_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/reward.pickle', 'wb') as handle:
            pickle.dump(r_g_DQN_CE, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/alpha_change.pickle', 'wb') as handle:
            pickle.dump(al_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/critic_loss_1.pickle', 'wb') as handle:
            pickle.dump(q1_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/critic_loss_2.pickle', 'wb') as handle:
            pickle.dump(q2_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/actor_loss.pickle', 'wb') as handle:
            pickle.dump(pi_loss_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/lstm_loss.pickle', 'wb') as handle:
            pickle.dump(loss_lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/reward_ep.pickle', 'wb') as handle:
            pickle.dump(reward_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/action_catch.pickle', 'wb') as handle:
            pickle.dump(action_probs_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(dir_save + '/service_rate.pickle', 'wb') as handle:
            pickle.dump(service_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)




path = os.getcwd()
env_name = 'Policy_MSE'
dir_save = path + '/' +str(env_name) + '_DSAC_LSTM/Multiple_buffers/range_tries/Train_'  + \
             str(agent_DSAC.t_alpha) + '_'  + str(n_steps)
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
model_DSAC = dir_save + '/Model_DSAC'
model_lstm  =dir_save + '/Model_LSTM'
# Saving the weights.
torch.save(agent_DSAC.ac.state_dict(), model_DSAC)
torch.save(agent_lstm.state_dict(), model_lstm)

#%% Saving data.
# Save the reward, state, action.
with open(dir_save + '/state_true.pickle', 'wb') as handle:
    pickle.dump(state_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dir_save + '/estimate.pickle', 'wb') as handle:
    pickle.dump(est_str_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/control.pickle', 'wb') as handle:
    pickle.dump(control, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(dir_save + '/arrival_rate.pickle', 'wb') as handle:
    pickle.dump(arrival_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/reward.pickle', 'wb') as handle:
    pickle.dump(r_g_DQN_CE, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/alpha_change.pickle', 'wb') as handle:
    pickle.dump(al_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/critic_loss_1.pickle', 'wb') as handle:
    pickle.dump(q1_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/critic_loss_2.pickle', 'wb') as handle:
    pickle.dump(q2_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/actor_loss.pickle', 'wb') as handle:
    pickle.dump(pi_loss_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/lstm_loss.pickle', 'wb') as handle:
    pickle.dump(loss_lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/reward_ep.pickle', 'wb') as handle:
    pickle.dump(reward_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/action_catch.pickle', 'wb') as handle:
    pickle.dump(action_probs_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dir_save + '/service_rate.pickle', 'wb') as handle:
    pickle.dump(service_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
  
