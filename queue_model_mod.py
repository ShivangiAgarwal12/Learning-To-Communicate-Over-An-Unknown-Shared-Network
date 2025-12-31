# -*- coding: utf-8 -*-
"""
functions for channel queueing model, FCFS, bernoulli and geometric
Fix: LCFS queue
@author: War Horses
"""
#%%import libraries
import numpy as np
import config
import pdb
import random
from copy import deepcopy
#%%global variable
#to check if any packet is in service or not, True if it is in service
packet_service = False
time_remain = -1
buffer = 0
pkt_0 = {}
#%%
def queueing(packet_arrived, q,packet,queue,reset=False):
   """
   Queuing model for the channel
   q: for service rate of the packet to the vehicle
   length: (later), if queue has a finite length
   packet: from class Vehicle with position, velocity and TS
   queue: for not passing an empty queue at each time step (see for an alternative)
   """
   global packet_service
   global time_remain
   global buffer
   global pkt_0
   
   if reset==True:
      packet_service = False
      time_remain = -1
      buffer = 0
      pkt_0 = {}
  
   q = config.q
   # Buffer.
   # pdb.set_trace()
   # packet_arrived = arrival(config.p)
   if packet_arrived == 1:
      
      #if packet arrives, append for ego vehicle also
      #queue[first_element][time_stamp][packet_of_car]
      queue.append(deepcopy(packet))
      buffer+= 1
   
      #if any packet is not in service
      if packet_service != True:
         #reduce buffer size and send last packet from queue for service
         buffer = buffer - 1
         slots = np.random.geometric(q)
         
#         pdb.set_trace()
         pkt_0 = queue.pop(0)
         
#         print('Packet in transit',pkt_0)
         time_remain = transmission(slots)
         packet_service, packet_remain = check_transmission(time_remain)
         
         #if packet is to be recieved at next time step, get the packet out for DQN
         if time_remain == 0:
             return queue, {}, pkt_0
         else:
             return queue,{}, {}
         
         
      #if any packet is already in service
      else:
         #move to next time  step but check for completion of transmission,
         #this is fine as if time remain here is 0, means packet is already delivered 
#         pdb.set_trace()
         if time_remain >=0:     
            # pdb.set_trace()
            time_remain = time_remain - 1
            
         packet_service, packet_remain = check_transmission(time_remain)
         #if packet service has ended, return the packets to be delivered at car,
         #catched by transit_pkt and start another transmission
         
         if packet_service == False:                       
            buffer = buffer - 1
            slots = np.random.geometric(q)
            
            #make a deep copy of packet to car           
            pkt_1 = deepcopy(pkt_0)
            #pop the element and send for transmission
            pkt_0 = queue.pop(0)
            time_remain = transmission(slots)
            
            #again check if time remain is just 1 slot to be delivered
            if time_remain == 0:
                return queue, pkt_1, pkt_0
            else:
            #only return the receievd packet and nothing for estimator
                return queue,pkt_1, {}
            
         else:
             #there is no packet to receive but check whose one slot is lift
            if packet_remain == False:
                pkt_1 = deepcopy(pkt_0)
                return queue,{}, pkt_1
            else:
                return queue,{}, {}
   
   #no packet arrived  
   else:
      #packet is not in transit
      if packet_service != True:
         
         #check if buffer is empty, if empty
         if not buffer:
            return queue,{}, {}
         
      #buffer is not empty 
         else:
            slots = np.random.geometric(q)
            pkt_0 = queue.pop(0)
            buffer = buffer - 1
            time_remain = transmission(slots)
            if time_remain == 0:
                return queue, {}, pkt_0
            else:
                return queue,{}, {}
            
      #packet is in transit, just check for transmission
      else:
         if time_remain >= 0:
            time_remain = time_remain - 1
         packet_service, packet_remain = check_transmission(time_remain)
         
         
         #if packet service has ended, then return packets to be sent to car,send packet 
         #for transmission to send if buffer not empty
         if packet_service == False:
            if queue:
               buffer = buffer - 1
               slots = np.random.geometric(q)
               pkt_1 = deepcopy(pkt_0)
               pkt_0 = queue.pop(0)
 #              pdb.set_trace()
#               print('Packet in transit',pkt_0)
               time_remain = transmission(slots)
               if time_remain == 0:
                   return queue, pkt_1, pkt_0
               else:
                   return queue, pkt_1, {}
            else: 
               return queue, pkt_0, {}
         else:
            if packet_remain == False:
                pkt_1 = deepcopy(pkt_0)
                return queue,{}, pkt_1
            else:
                return queue,{}, {}
         
      #move to next time step, check for service packet again
   
#%%transmission to the car
def transmission(time_service):
   """   
   Parameters
   ----------
   time_service : time slots for successfull transmission of packet to the car

   Returns
   -------
   time_service : remaining slot

   """   
   global time_remain
   global packet_service
   packet_service = True
   time_remain = time_service - 1
   return time_remain
#%%check completed transmission
def check_transmission(time_remain):
   """
  Parameters
   ----------
   time_remain : time slots remaining

   Returns
   -------
   packet_service
   """
   global packet_service
   global packet_remain
   if time_remain == -1:
      packet_service = False
   else:
      packet_service = True
    
   if time_remain == 0:
       #packet will be received at the estimator a step later
       packet_remain  = False
   else:
       packet_remain = True
   return packet_service, packet_remain
#%%packet arrival at the channel
def arrival(p):
   """
   Parameters
   ----------
   p : packet arrival rate

   Returns
   -------
   packet_arrive : whether packet has been arrived or not

   """
   p = config.p
   #corrected with binomial distribution
   packet_arrive = np.random.binomial(1,p)
   #packet_arrive = bernoulli.rvs(p)
   return packet_arrive
#%%generate random velocity and position
def generate(v = 0):
   """
   Parameters
   ----------
   v : car_id

   Returns
   -------
   Position and velocity

   """
   if v == None:
      #car id, random number between 1-10
      v = random.randint(1,9)
   #position between 0-5 f0r x and 0 for y for now
   px = random.randint(0,5)
  
   #velocity between 0-1 for x and 0 for y
   vx = round(random.uniform(0,1),2)
   
   return v, px, vx
#%%queueing for LCFS
def queueing_lcfs(q,packet,queue,reset=False):
   """
   LCFS Queuing model for the channel
   q: for service rate of the packet to the vehicle
   length: (later), if queue has a finite length
   packet: from class Vehicle with position, velocity and TS
   queue: for not passing an empty queue at each time step (see for an alternative)
   """
   global packet_service
   global time_remain
   global buffer
   global pkt_0

   if reset==True:
      packet_service = False
      time_remain = -1
      buffer = 0
      pkt_0 = {}
   
   q = config.q
   # Buffer.
#   pdb.set_trace()
   packet_arrived = arrival(config.p)
   if packet_arrived == 1:
      
      #if packet arrives, append for ego vehicle also
      #queue[first_element][time_stamp][packet_of_car]
      queue.append(deepcopy(packet))
      buffer+= 1
   
      #if any packet is not in service
      if packet_service != True:
         #reduce buffer size and send last packet from queue for service
         buffer = buffer - 1
         slots = np.random.geometric(q)
        
         #pop last added packet in the queue
         pkt_0 = queue.pop(-1)
#         pdb.set_trace()
#         print('Packet in transit',pkt_0)
         time_remain = transmission(slots)
         return queue,{}
                  
      #if any packet is already in service
      else:
         #move to next time  step but check for completion of transmission
#         pdb.set_trace()
         if time_remain >=0:
            time_remain = time_remain - 1
         packet_service = check_transmission(time_remain)
                  
         #if packet service has ended, return the packets to be delivered at car,
         #catched by transit_pkt and start another transmission
         
         if packet_service == False:
                        
            buffer = buffer - 1
            slots = np.random.geometric(q)
            
            #make a deep copy of packet to car           
            pkt_1 = deepcopy(pkt_0)
            #pop the element and send for transmission
            pkt_0 = queue.pop(-1)
            time_remain = transmission(slots)
#            print('Packet in transit',pkt_0)

            return queue,pkt_1
         else:
            return queue,{}
   
   #no packet arrived  
   else:
      #packet is not in transit
      if packet_service != True:
        
         #check if buffer is empty, if empty
         if not buffer:
            return queue,{}
         
      #buffer is not empty 
         else:
            slots = np.random.geometric(q)
            pkt_0 = queue.pop(-1)
            buffer = buffer - 1
            time_remain = transmission(slots)
            
            return queue,{}
      #packet is in transit, just check for transmission
      else:
         if time_remain >= 0:
            time_remain = time_remain - 1
         packet_service = check_transmission(time_remain)
         
         
         #if packet service has ended, then return packets to be sent to car,send packet 
         #for transmission to send if buffer not empty
         if packet_service == False:
            if queue:
               buffer = buffer - 1
               slots = np.random.geometric(q)
               pkt_1 = deepcopy(pkt_0)
               pkt_0 = queue.pop(-1)
 #              pdb.set_trace()
#               print('Packet in transit',pkt_0)
               time_remain = transmission(slots)
               return queue, pkt_1
            else:
               return queue, pkt_0
         else:
            return queue,{}
         
      #move to next time step, check for service packet again
#%%
  

  
