    # -*- coding: utf-8 -*-
"""
Vehicular model for ego vehicle 0, packet from the queue model
Vehicular model, x(k+1) = Ax(k)+Bu(k),
y(k) = Cx(k), using kalman filter due to packet delays
Kinematics equations
s = s0+ut+1/2at^2
v = u+at
Corrected cost for current time step J = (x(k)TQx(k)+ u(k)TRu)
@author: War Horses
"""
#%%import libraries
import numpy as np
import math
import queue_model_mod as qm
import config
import random
import scipy.linalg
from scipy.spatial import distance
import pdb
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import gc
from gym import spaces
from sklearn import preprocessing
# import addcopyfighandler
import matplotlib.ticker as mtick
import torch
import itertools
import torch.nn as nn
# from corev2 import select_action
import DSAC_RL as RL
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda:0" if use_cuda else "cpu")
from numpy import linalg as LA
criterion = nn.MSELoss()
from collections import deque
#%%
class Vehicle_model():
   """
   Vehicular environment, define state, action and A,B,C
   State: (position,velocity)
   Action: Acceleration
   """
   
   def __init__(self, v, px, py, vx, vy):
      """
      State Parameters
      ----------
      v : car_id
      px : x position
      vx : x velocity

      Returns
      -------
      State of the car, to fix the length of packet
      """
      
      self.car_id = v
      #x and y position of car
      self.x_pos = px
      self.y_pos = py
      #x and y velocity of acr
      self.x_vel = vx
      self.y_vel = vy
      
      
      # Initialise state for 2d, first 2 for position and next 2 for vel
      self.state = np.ones((4,1))
      self.state[0,0] = px
      self.state[1,0] = py
      self.state[2,0] = vx
      self.state[3,0] = vy
      
      # LQR parameters.
      self.row_Q = 4
      self.column_Q = 4
      self.row_R = 2
      self.column_R = 2
      self.Q = np.zeros((self.row_Q,self.column_Q))
      self.R = np.zeros((self.row_R, self.column_R))
      
      # System dynamic parameters.
      dt = 0.1
      self.A = [[1, 0 , dt, 0],\
                [0, 1, 0, dt],\
                    [0, 0, 1, 0],\
                        [0, 0, 0, 1]]
          
      self.B = [[(math.pow(dt, 2)/2), 0],\
                [0, math.pow(dt, 2)/2],\
                    [dt, 0],\
                        [0,dt]]
          
      self.A = np.asmatrix(self.A)
      self.B = np.asmatrix(self.B)
      self.C = np.asmatrix(np.eye(4))
      
      # Packet initialization.
      self.packet = {'car_id': self.car_id,
                     'Position': [self.x_pos, self.y_pos],
                     'Velocity' : [self.x_vel, self.y_vel]
         }
      self.queue = []
      
      self.v_temp = {}
      # Kalman filter parameters.
      row = 4
      self.w = 0.001*np.eye((row))
      self.o = 0.001*np.eye((row))
      self.age  = []
      self.age.append(1)
      
      self.age_policy = []
      self.age_policy.append(1)
      
      
      self.packet_rcvd = []
      self.trial_W = []
      self.trial_O = []
      
     
      #catching previous outputs from lstm
      self.previous = []
      #store the received packet, updated only when received new one
      self.store_packet = []
      
      #store state_learn for actor_critic
      self.state_learn = []
      
      # Store the LSTM output.
      self.LSTM_output = {}
      
      #store lstm inputs
      self.LSTM_input = {}
      
      #store policy inputs
      self.policy_inputs = {}
      
      #store last policy actions
      self.policy  = []
      
      #store last control actions
      self.policy_last = []

      
      self.history = 1
      history_packet = self.history*4
      history_age = self.history*2
      history_rep = 4
      history_control = self.history*2
      self.output_dim = history_rep
      self.control_actions = 2
      #send the estimated q as well
      history_policy = history_packet + self.control_actions
        
      self.input_space = history_packet + history_age + history_rep + history_control
      self.input_space_policy = history_policy 
      self.input_space_service = self.control_actions + self.control_actions
      self.output_dim_service = self.control_actions
      
      low = np.zeros((self.input_space,), dtype=np.float32)
      high = 50*np.ones((self.input_space,), dtype = np.float32)
      
      low_policy = np.zeros((self.input_space_policy,), dtype=np.float32)
      high_policy = 50*np.ones((self.input_space_policy,), dtype = np.float32)
      
      low_service = np.zeros((self.input_space_service,), dtype=np.float32)
      high_service = 50*np.ones((self.input_space_service,), dtype = np.float32)
      
      self.observation_space = spaces.Box(
           low  = low,
           high = high
           )
      
      
      self.observation_space_policy = spaces.Box(
          low = low_policy,
          high = high_policy
          )  
     
      self.observation_space_service = spaces.Box(
          low = low_service,
          high = high_service)
         
        #initialise the observtaion space
        #low state being [Position, Velocity, Age], to check upper value of velocity and age
      low  = np.array([0, 0, 0, 0], dtype = np.float32)
      high = np.array([10, 10, 50, 50], dtype = np.float32)
      self.observation_space = spaces.Box(
           low  = low,
           high = high,
           )
        
        #initialise the action space
        #low being action as acceleration
      self.action_space = spaces.Box(
           low = -20,
           high = 20, 
           shape = (1,),
           )
      
      self.action_space_policy = spaces.Box(
          low = 0,
          high = 1,
          shape = (2,),
          )
      self.count_x = 0
      self.count_y = 0
      self.epsilon_decay = []
      self.wait = {'time':99999,
                   'packet_service':False}
      self.serve = 0
      self.lp = []
      self.reset_age = []
      # Nuber of history.
      # hist_n = 10
      # self.reset_age = deque(maxlen=(hist_n))
      # for i in range(hist_n):
      #     self.reset_age.append(1)
      
      
   def generate_other(self,k):
      
      """
      Parameters
      ----------
      k : current time to catch time stamp
      Returns
      -------
      packets of other cars
      """
      
      self.v_temp = {}
      # Packet from the camera.
    
      v, x_pos, y_pos, x_vel, y_vel = 0, self.camera[:,k][0],\
          self.camera[:,k][1], self.camera[:,k][2], self.camera[:,k][3]
        
      vehicle = v, x_pos, y_pos, x_vel, y_vel
      
      # Build the packet.
      self.packet['car_id'],self.packet['Position'], self.packet['Velocity'] = \
         vehicle[0], [vehicle[1], vehicle[2]], [vehicle[3], vehicle[4]]
      
      # Assign packet p.
      p = self.packet
      self.v_temp[k-1] = p
      
      
      return self.v_temp
  
   def generate_initial(self, new_state):
       initial = {}
       # Packet from the camera.
     
       v, x_pos, y_pos, x_vel, y_vel = 0, new_state[0,0],\
           new_state[1,0], new_state[2,0], new_state[3,0]
         
       vehicle = v, x_pos, y_pos, x_vel, y_vel
       
       # Build the packet.
       self.packet_initial = {'car_id': self.car_id,
                      'Position': [self.x_pos, self.y_pos],
                      'Velocity' : [self.x_vel, self.y_vel]
          }
       self.packet_initial['car_id'],self.packet['Position'], self.packet['Velocity'] = \
          vehicle[0], [vehicle[1], vehicle[2]], [vehicle[3], vehicle[4]]
       
       # Assign packet p.
       p = self.packet_initial
       initial[0] = p
       
       
       return initial
       
   def wrapped_position(self, x_curr):
       flag = True
       if (x_curr[2,0] >= 10) or (x_curr[2,0] <= -10):
          
           x_curr[2,0] = np.sign(x_curr[2,0])*10
           flag = False
           
       if (x_curr[3,0] >= 10) or (x_curr[3,0] <= -10):
         
           x_curr[3,0] = np.sign(x_curr[3,0])*10
           flag = False
       return x_curr  
      
   def noise(self):
      """
      Process and observation noise

      Returns
      -------
      Process and Observation noise
      """
      #process noise
      W = np.random.default_rng().multivariate_normal(np.zeros((4)),self.w)
      W = np.asmatrix(W).T
      
      #observation noise
      O = np.random.default_rng().multivariate_normal(np.zeros((4)),self.o)
      O = np.asmatrix(O).T
      self.trial_W.append(W)
      self.trial_O.append(O)
      return W,O
  
   
   
 
   def new_action(self, x_curr, action):
         
       
         action_x = action[0,0]
         action_y = action[1,0]
         # pdb.set_trace()
         # action_x_orig = orig_cntrl[0]
         # action_y_orig = orig_cntrl[1]
        
         # Dimension x is activated.
         if self.flag_x_act:
             if (x_curr[0,0] >= 870/10.0 or x_curr[0,0] <= -870/10.0):
                    action_x = -np.sign(x_curr[0,0])*3
         else:
             action_x = action[0,0]
             
         if self.flag_y_act:
             if (x_curr[1,0] >= 870/10.0 or x_curr[1,0] <= -870/10.0):
                    action_y = -np.sign(x_curr[1,0])*3
         else:
             action_y = action[1,0]
         
        
         return np.array((action_x,action_y))
     
  
   def env_start(self, model):
      """
      at start of environnment, assuming car has last packet of (k-1)
      for now the packet received becomes the state at the car
      Returns
      -------
      Initial state of the system

      """      
      
      # Initial state.
      self.init_state = self.state
      
      # Append on the vehicle state (plant).
      self.car = self.init_state
      
      # Process noise and observation noise
      self.W,self.O = self.noise()
      
      
      # Control save.
      self.control = []
      
      
      # Measurement (camera).
      
      self.camera = np.asarray(self.C@self.init_state + self.O)
     


      # Save the vehicle position and velocity.
      self.x_pos = self.car[0,0]
      self.y_pos = self.car[1,0]
      self.x_vel = self.car[2,0]
      self.y_vel = self.car[3,0]
      
      new_state = np.array([[self.car[:,-1][0]], [self.car[:,-1][1]], [self.car[:,-1][2]], [self.car[:,-1][3]] ,\
                            [self.age[-1]], [0]])
      
      temp = 0
      for i in itertools.cycle([[self.car[:,-1][0], self.car[:,-1][1], self.car[:,-1][2], self.car[:,-1][3]]]):
          if temp > self.history-2:
              break
          else:
              new_state  = np.insert(new_state, 2, i)             
              temp = temp+1
      
      new_state = np.insert(new_state, 4*self.history, np.ones((2*self.history-1)))
      new_state = np.insert(new_state, 6*self.history, np.zeros((2*self.history-1)))
     # new_state = new_state.reshape((3, 1, self.history))
      
     
      self.new_state = new_state.reshape(len(new_state))
      new_state = new_state.reshape((len(new_state),1))
      
      # Initial input for LSTM to get output for estimate for next step
     
      self.previous.append(new_state[0:4])
      
      self.store_packet.append(new_state[0:4*self.history])
      policy_inputs = self.generate_initial(new_state)
      
         
      self.previous = torch.tensor(self.previous).float()
      self.store_packet = torch.tensor(self.store_packet).float()
      self.previous = self.previous.to(device)
      self.store_packet = self.store_packet.to(device)
     
      
      
      #age = [float(self.age[-1])]
      age = self.new_state[4*self.history:6*self.history]
      self.age = age
      age = torch.as_tensor(age)
      age  = age.to(device)
      age.unsqueeze_(1)
      
      
      #action_add = torch.as_tensor([[0]])
      action_add = self.new_state[6*self.history:]
      action_add = torch.as_tensor(action_add)
      action_add = action_add.to(device)
      action_add.unsqueeze_(1)
      self.control = action_add
            
      action_x = random.uniform(-1, 1)
      action_y = random.uniform(-1,1)
      action = torch.tensor([action_x, action_y])
      action = action.unsqueeze_(1).to(device)
      
     
      self.previous = torch.stack((self.previous[-1], \
                                            self.previous[-1]))
      
            
      """
      new_state for actor for getting action, 120*1  , remains just next state 
      at starting
      """
      #new_state = torch.cat((self.state_learn[0].unsqueeze_(1), age[0:]),0)
      
      self.control = self.control[2:]
      self.control = torch.cat((self.control, action),0)
      
     
      new_state = torch.cat((self.store_packet[-1], age[0:], \
                             self.previous[-1], self.control), 0)
      
      
      policy_query = torch.as_tensor([1])
      policy_query  = policy_query.to(device)
      policy_query.unsqueeze_(1)
      new_state_policy = torch.cat((self.store_packet[-1], age[0:], \
                             self.previous[-1], self.control, policy_query), 0)
          
          
      new_state = new_state.cpu().detach().numpy()
      self.new_state = new_state.reshape(len(new_state))
      
      new_state_policy = new_state_policy.cpu().detach().numpy()
      self.new_state_policy = new_state_policy.reshape(len(new_state_policy))
         
      # Flag x
      self.flag_x_act = False
      self.flag_y_act = False
      self.policy.append(1)
      self.policy_inputs[0] = policy_inputs
      self.age_policy = age
      return self.new_state, self.new_state_policy
      
  
   
   def state_evolution(self,k):
       """
       State evolution of car to find packet
       J(k+1) is a function of x(k) and u(k)
       Penalising fact from environment to the car based on true state and control 
       input which in turn is a function of estimated state
       To Do: 
       Parameters
       ----------
       k : current time
       packet_received: if packet recieved or not, 0 denotes no packet

       Returns
       -------
       Next state, current control
       """
       # #take action to add to new state
       action_x = random.uniform(-3, 3)
       action_y = random.uniform(-3,3)
       action = torch.tensor([action_x, action_y])
       action = action.unsqueeze_(1).to(device)
       self.control = self.control[2:]
       self.control = torch.cat((self.control, action),0)
       # Process noise and observation noise
       self.W,self.O = self.noise()
       if k==0:
     
         cntrl = self.control[-2:].cpu().numpy()
         cntrl = np.asarray(cntrl)
         cntrl= cntrl.reshape((2,1))
         x_next = self.A @ self.car[:,k:k+1] + self.B@(cntrl) + self.W
         
         
         x_next = self.wrapped_position(x_next)
         # Flag for each dimension.
         if  (x_next[0,0] >= 970/10.0 or x_next[0,0] <= -970/10.0) :
             self.flag_x_act = True
             self.count_x = self.count_x + 1
         elif (x_next[0,0] <= 870/10.0 or x_next[0,0] >= -870/10.0):
             self.flag_x_act = False
          
         if (x_next[1,0] >= 970/10.0 or x_next[1,0] <= -970/10.0):
             self.flag_y_act = True
             self.count_y = self.count_y + 1
         elif (x_next[1,0] <= 870/10.0 or x_next[1,0] >= -870/10.0):
             self.flag_y_act = False
            
         if self.flag_x_act or self.flag_y_act:
             act_new = self.new_action(x_next, cntrl)
             self.control = self.control[2:]
             cntrl = act_new
             act_new =  torch.tensor(act_new)
             act_new = act_new.unsqueeze_(1)
            
             act_new = act_new.to(device)
             self.control = torch.cat((self.control,act_new ),0)
             cntrl = cntrl.reshape(2,1)
             x_next = self.A @ self.car[:,k:k+1] + self.B@(cntrl) + self.W
     
       else:          
      
         cntrl = self.control[-2:].cpu().numpy()
         cntrl = np.asarray(cntrl)
         cntrl= cntrl.reshape((2,1))
         x_next = self.A @ self.car[:,k:k+1] + self.B @ cntrl + self.W
         x_next = self.wrapped_position(x_next)
         
         if  (x_next[0,0] >= 970/10.0 or x_next[0,0] <= -970/10.0):
             self.flag_x_act = True
             self.count_x = self.count_x+ 1
         elif (x_next[0,0] <= 870/10.0 or x_next[0,0] >= -870/10.0):
             self.flag_x_act = False
             
         if (x_next[1,0] >= 970/10.0 or x_next[1,0] <= -970/10.0) :
             self.flag_y_act = True
             self.count_y = self.count_y + 1
         elif (x_next[1,0] <= 870/10.0 or x_next[1,0] >= -870/10.0):
             self.flag_y_act = False
            
         if self.flag_x_act or self.flag_y_act:
             act_new = self.new_action(x_next, cntrl)
             # pdb.set_trace()
             self.control = self.control[2:]
             cntrl = act_new
             act_new =  torch.tensor(act_new)
             act_new = act_new.unsqueeze_(1)
            
             act_new = act_new.to(device)
             self.control = torch.cat((self.control,act_new ),0)
             cntrl = cntrl.reshape(2,1)
             x_next = self.A @ self.car[:,k:k+1] + self.B@(cntrl) + self.W
             x_next = self.wrapped_position(x_next)
       
       # Plant (car), no evolution, use x_next as it is to move,true position
       
       self.car = np.append(self.car,np.reshape(x_next,(4,1)),axis=1)
       # if x_next[2,0] > 10 or x_next[2,0] < -10:
       #     pdb.set_trace()
      
       # Measurement (camera)
       current_camera = np.asarray(x_next + self.O)
       current_camera = np.reshape(current_camera,(4,1))
       self.camera = np.append(self.camera, current_camera, axis=1)
      
       current_control = cntrl
       
     
       return self.car[:,-1], current_control
    #self.car[:,k+1], current_control
   
   
  
               
   def env_step(self, agent_AC, k, model, eps_threshold, steps_done):
      """
      Parameters
      ----------  
      k: current time, to calculate age of the packet
      post: to append posterior estimates at each time, empty matrix
      prior: to append prior estimates at each time, empty matrix
      Returns
      -------
      J_estimated, error, position, velocity, acceleration
      """   
        # append the current state to generate packet
      x_next, current_control = self.state_evolution(k)
   
     
      # Create the packet for the current instant.
      self.v_temp = self.generate_other(k+1)
      
      #build policy input at each step
      policy_query = torch.as_tensor([self.policy[-1]])
      policy_query  = policy_query.to(device)
      policy_query.unsqueeze_(1)
      # pdb.set_trace()
     
      if k == 0:
          self.policy_inputs[1] = self.policy_inputs[0]
          packet = self.policy_inputs[1]
      
          [pkt_pos, pkt_y_pos] = self.policy_inputs[k+1][0]['Position']
          [pkt_vel, pkt_y_vel] = self.policy_inputs[k+1][0]['Velocity']
          
          pkt_y = np.array(([pkt_pos, pkt_y_pos, pkt_vel, pkt_y_vel]))
          pkt_y = torch.tensor(pkt_y).to(device)
          packet = pkt_y.unsqueeze(0).T
          self.age_policy = self.age_policy[:] + 1
      else:
          # pdb.set_trace()
          key = list(self.policy_inputs[k+1].keys())[0]
          [pkt_pos, pkt_y_pos] = self.policy_inputs[k+1][key]['Position']
          [pkt_vel, pkt_y_vel] = self.policy_inputs[k+1][key]['Velocity']
          
          pkt_y = np.array(([pkt_pos, pkt_y_pos, pkt_vel, pkt_y_vel]))
          pkt_y = torch.tensor(pkt_y).to(device)
          packet = pkt_y.unsqueeze(0).T
      
      with torch.no_grad():
          #for usual update without packet
          temp_lstm = torch.cat((packet, self.age_policy,\
                                  self.previous[-1], self.control),0)
              
          state_learn = model(temp_lstm.T)
      
      #calculate age for current time step
      # pdb.set_trace()
      if k == 0:
           self.last_control = np.random.choice([0, 1], size=(self.control_actions,), p=[0.5, 0.5])
           self.last_control = torch.tensor(self.last_control)
           # pdb.set_trace()
           self.last_age = self.age_policy[-1].item()*torch.ones(self.control_actions)
          
           # pdb.set_trace()
           # temp_policy = torch.cat((state_learn.T, \
           #                          self.last_age.unsqueeze(0).T.to(device),\
           #                              torch.tensor([self.policy[-1]]).unsqueeze(0).T.to(device)),0)
          
            
           temp_policy = torch.cat((state_learn.T, \
                                     self.last_age.unsqueeze(0).T.to(device)),0)
           temp_policy = temp_policy.cpu().detach().numpy()
           temp_policy = temp_policy.reshape(len(temp_policy))
           # pdb.set_trace()
           p, lp, action_probs = agent_AC.get_action(temp_policy)
           # action_probs = action_probs.cpu().detach().numpy()
           p = p.cpu().numpy()[0]
           self.lp.append(lp)
           # self.policy_last.append(p)
      else:
           self.last_control = self.last_control[1:]
           self.last_control = np.append(self.last_control, self.policy[-1])
           self.last_control = torch.tensor(self.last_control)
           
           self.last_age = self.last_age[1:]
           self.last_age = np.append(self.last_age, self.age_policy[-1].item())
           self.last_age = torch.tensor(self.last_age)
           
           # pdb.set_trace()
           
           # temp_policy = torch.cat((state_learn.T, \
           #                          self.last_age.unsqueeze(0).T.to(device),\
           #                              torch.tensor([self.policy[-1]]).unsqueeze(0).T.to(device)),0)
           
           temp_policy = torch.cat((state_learn.T, \
                                     self.last_age.unsqueeze(0).T.to(device)),0)
           
           
          
           temp_policy = temp_policy.cpu().detach().numpy()
           temp_policy = temp_policy.reshape(len(temp_policy))
           p, lp, action_probs = agent_AC.get_action(temp_policy)
           # action_probs = action_probs.cpu().detach().numpy()
           p = p.cpu().numpy()[0]
           self.lp.append(lp)
     
      self.policy.append(p)   
    
      # Send the packet through a FCFS queue.
      if k == 0:
         self.queue,transit_pkt, packet_policy = qm.queueing(p, config.q, self.v_temp, self.queue,reset=True)
      else:
         
         self.queue,transit_pkt, packet_policy = qm.queueing(p, config.q, self.v_temp, self.queue)
      # self.queue_len.append(len(self.queue))
      
      # build policy input to be used at next time step
      # pdb.set_trace()
      
     
      
             
      
      if packet_policy:
          self.policy_inputs[k+2] = packet_policy
          
          pkt_time = list(packet_policy.keys())[0]
          new_age = (k+1) - pkt_time
          self.age_policy[:] = self.age_policy[:] + 1
          self.age_policy = self.age_policy[2:]
          self.age_policy = new_age*np.ones(2)
          self.age_policy = torch.tensor([self.age_policy])
          self.age_policy  = self.age_policy.to(device)
          self.age_policy = self.age_policy.T
          self.reset_age.append(new_age)
      else:
          self.policy_inputs[k+2] = self.policy_inputs[k+1]
          self.age_policy = self.age_policy + 1
          
      
      # Empty control list.
      self.control_estimate = []
    
      # Check if car received any packet, returns True for empty dictionary,
      # No packet recieved
      packet_received = not bool(transit_pkt)
     
      #no packet received, update only prior and make posterior = prior
      if packet_received == True:
         # Increment age  
         self.age[:] = self.age[:] + 1
         age = torch.tensor([self.age])
         age  = age.to(device)
         age = age.T
         
         # Stack on store packet 
         self.store_packet = torch.stack((self.store_packet[-1], \
                                          self.store_packet[-1]))
             
         packet = self.store_packet[-1][4:]
        
         pkt_y = np.array(([self.camera[:,-1][0], self.camera[:,-1][1], self.camera[:,-1][2], \
                            self.camera[:,-1][3]]))
         pkt_y = torch.tensor(pkt_y).to(device)
         pkt_y = pkt_y.unsqueeze(0).T
         packet = torch.cat((packet, pkt_y),0)
         
       
         # Estimate the next state.
         with torch.no_grad():
             #for usual update without packet
             temp_lstm = torch.cat((self.store_packet[-1], age[0:],\
                                    self.previous[-1], self.control),0)
             
             #storing the lstm inputs
             
             # self.LSTM_input[k] = torch.cat((self.store_packet[-1], age[0:], self.previous[-1], \
             #                                 self.control), 0).requires_grad_(True)
             self.LSTM_input[k] = torch.cat((self.store_packet[-1], age[0:], self.previous[-1], \
                                             self.control), 0)
            
             self.state_learn = model(temp_lstm.T)
             
         # Store output.
         self.LSTM_output[k] = self.state_learn
            
         #append in previous to be fed back at next time step  
         self.previous = torch.stack((self.previous[-1], \
                                          self.state_learn[0,:].unsqueeze(1).detach()))
         
         
        
         #next state that goes inside the replay buffer
         new_state = torch.cat((self.store_packet[-1],age[0:], \
                                self.state_learn[0,-2*self.history:].unsqueeze(1).detach(), self.control),0)
         new_state = new_state.cpu().detach().numpy()
         self.new_state = new_state.reshape(len(new_state))
         
         state = temp_lstm
         state = state.cpu().detach().numpy()
         self.buffer_state = state.reshape(len(state))
        
         # MSE input.
         MSE_input_batch = self.LSTM_input[k].T
         
         # MSE output
         MSE_output = pkt_y
         
         # pdb.set_trace()
        
         J_est = []
         loss = criterion(pkt_y, self.LSTM_output[k].T)
         J_est.append(loss)
       
      
      #packet recieved, track back to the time when packet was received, update posterior
      #and prior
      else:
        
         # We will start with the current time stamp of the packet.
         pkt_time = list(transit_pkt.keys())[0]
         
         # Current time.
         cur_time = k
       
         #caching the packet
         self.packet_rcvd.append([k,pkt_time])
         
         #calculate age of received packets
         new_age = cur_time - pkt_time
         self.age[:] = self.age[:] + 1
         self.age = self.age[2:]
         self.age = np.append(self.age, new_age*np.ones(2))
         age = torch.tensor([self.age])
         age  = age.to(device)
         age = age.T
         
         
         # Extract the measurement of pkt_time.
      
         packet = self.store_packet[-1][4:]
         [pkt_pos, pkt_y_pos] = transit_pkt[pkt_time]['Position']
         [pkt_vel, pkt_y_vel] = transit_pkt[pkt_time]['Velocity']
         
         pkt_y = np.array(([pkt_pos, pkt_y_pos, pkt_vel, pkt_y_vel]))
         pkt_y = torch.tensor(pkt_y).to(device)
         pkt_y = pkt_y.unsqueeze(0).T
         packet = torch.cat((packet, pkt_y),0)
         
         
         #save lstm inputs for future using current measurements
         pkt_true = self.store_packet[-1][4:]
       
         pkt_t = np.array(([self.camera[:,-1][0], self.camera[:,-1][1], self.camera[:,-1][2], self.camera[:,-1][3]]))
         pkt_t = torch.tensor(pkt_t).to(device)
         pkt_t = pkt_t.unsqueeze(0).T
         pkt_true = torch.cat((pkt_true, pkt_t),0)
                        
         with torch.no_grad():
            #appending the packet as it is
            temp_lstm = torch.cat((packet, age[0:], self.previous[-1], self.control),0)
            
            self.LSTM_input[k] = torch.cat((packet, age[0:], self.previous[-1], self.control),0)
                
            self.state_learn = model(temp_lstm.T)
 
         # Input for batch update.
         #MSE_input_batch = self.LSTM_input[pkt_time].T
         MSE_input_batch = self.LSTM_input[k].T
           
         # Update the stored packets.
         self.store_packet = torch.stack((self.store_packet[-1], \
                                          packet))
         # Store the LSTM output.
         self.LSTM_output[k] = self.state_learn
         
         
         # MSE loss.
        
         #MSE_input = self.LSTM_output[pkt_time][0,-2:].requires_grad_(True)
         # MSE output.
         MSE_output = pkt_t
         #MSE_output = model(self.LSTM_input[pkt_time].T).detach()
         
         self.previous = torch.stack((self.previous[-1], \
                                          self.state_learn[0,:].unsqueeze(1).detach()))
          
      
        
         #next state that goes inside the replay buffer
         new_state = torch.cat((packet, age[0:],\
                                self.state_learn[0,-2*self.history:].unsqueeze(1).detach(), self.control),0)
         new_state = new_state.cpu().detach().numpy()
         
         #new state at current time, only has updated estimates
         self.new_state = new_state.reshape(len(new_state))
         
       
         J_est = []
         loss = criterion(pkt_t, self.LSTM_output[k].T)
         J_est.append(loss)
         
      state = temp_lstm
      state_policy = temp_policy
        
      state = state.cpu().detach().numpy()
        
      # state_policy = state_policy.cpu().detach().numpy()
      # state_policy = state_policy.reshape(len(state_policy))
      #state that went inside lstm at current time step
      self.buffer_state = state.reshape(len(state))
    
      status = False
      # # eps_term = False
      # print(MSE_input_batch[0][0:4].cpu().detach().numpy())
      # print(state_policy[0:4])
      # print(MSE_input_batch[0][5].cpu().detach().numpy())
      # print(state_policy[5])
      # print(MSE_input_batch[0][6:10].cpu().detach().numpy())
      # print(state_policy[6:10])
      # print(MSE_input_batch[0][-2:].cpu().detach().numpy())
      # print(state_policy[-3:-1])
      # pdb.set_trace()
     
      return state_policy, MSE_input_batch, MSE_output,  -J_est[0], False, p,\
         lp, action_probs
          
#%%