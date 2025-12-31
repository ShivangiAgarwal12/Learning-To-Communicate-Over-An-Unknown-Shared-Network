# -*- coding: utf-8 -*-
"""
Configuration file
@author: War Horses
"""
import random
#%%
#service probability of packet:geometric
q = 0.1
#random.uniform(0,1)

#arrival probability of packet: bernoulli
# p = 0.01
#random.uniform(0, q)


#mode to save files 
mode = '0.2q'

#number of cars in environment
t_car = 1

#penalising parameters
#state penalty
Q = 1

#control penalty
R = 1

#RTT alpha smoothing parameter
alpha_rtt = 0.125

#pre-training parameter 
int_up = 50

#batch size 
batch_size = 256
batch_size_update = 64

#saving episode
save_f = 10
#for lstm, when does the lstm train
save_eps = 1
# Update every episode.
update_every = 1

#for stable updates to policy 
d = 1
d_int = 1

t_up = 30
count = 10
flag_LSTM = 1
r_scale = 3.0

#no. of episodes
episodes = 10000

#drop attempts
drop_attempts = 3

#threshold 
threshold = 25